"""Stand or step: what the policy does for a ball by where it is and how early it is announced (grasping container).

    /opt/drone_venv/bin/python tools/stroke_kinds.py <model_N.pt> <extra bank .npz>

The actor is not told which stroke is meant (the 110-D frame has the target, its velocity and the time to the
strike; main's Runner passes no reach level to this policy). Training asks two kinds of question:

* clip questions: step forehand | -0.58 | stand forehand | split | stand backhand | +0.10 | step backhand by
  lateral position, announced 0.46-1.2 s ahead;
* direct questions (the Planner's targets, ``task.extra_bank``): stand strokes at every position, 0.2-0.4 s ahead.

Right of -0.58 m and left of +0.10 m the two overlap in position and differ only in notice. This probe asks both
kinds over the whole notice range 0.30-0.60 s (clip questions down to 0.40 s, direct ones up to 0.56 s) and prints,
per zone, question kind and notice: the pelvis shift asked and made at the strike, the share of strokes with a
foot lifted more than 2 cm, and the racket's distance from the target. A clean policy steps 0% or 100%.
"""

import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import from_dict, apply_physics
from supermimic_hope import a3
from supermimic_hope.core.trainer import TrainCfg, Trainer
from supermimic_hope.core.task import STRIKE_FRAME
path = Path(sys.argv[1]); payload = torch.load(path, map_location='cpu', weights_only=False)
cfg = from_dict(TrainCfg, payload['config']); cfg.device = 'cuda:0'; cfg.num_envs = 1024; cfg.seed = 23; cfg.buffer_steps, cfg.capture = 4, False
t = cfg.task
t.extra_bank, t.extra_probability = sys.argv[2], 0.5
t.short_notice_min_s = 0.40; t.short_notice_probability = (0.6, 0.6, 0.6, 0.6); t.minimum_lead = (0.40, 0.40, 0.40, 0.40)
t.extra_notice_stretch = (0.0, 0.0)  # the probe sets the notices itself
t.mocap_dropout_rate = 0.0; t.push_rate = 0.0; t.midstroke_probability = 0.0
apply_physics(cfg, dict(contact='elliptic', handoff='1'))
tr = Trainer(cfg); tr.load(path); tr.actor.load_state_dict(payload['actor_ema']); tr.actor.eval()
task = tr.task
e = task.extra
task.q_lead[e] = torch.round((0.30 + 0.26 * torch.rand(len(e), device=task.q_lead.device)) / task.dt) * task.dt  # direct questions announced 0.30-0.56 s ahead
feet = [task.kin.names.index(n) for n in a3.FEET]
rows, lift = [], torch.zeros(cfg.num_envs, 2, device='cuda:0')
orig = task.observe
def observe(st, tape, reward, info, terminated, truncated):
    pos = task.kin(st)[0]
    new = (task.k <= STRIKE_FRAME - torch.round(task.notice / task.dt).long() + 1) & (task.hold == 0)
    lift[:] = torch.where(new[:, None], torch.zeros_like(lift), torch.maximum(lift, pos[:, feet, 2]))
    strike = (task.k == STRIKE_FRAME) & (task.hold == 0) & ~terminated
    if strike.any():
        i = strike.nonzero().flatten()
        rp, rv, rn = task.racket(task.kin(st[i]), tape['sign'][i])
        rows.append(torch.cat((task.family[i, None].float(), tape['tpos'][i, 1:2], st[i, 1:2], tape['ref'][i, 1:2], lift[i], (rp - tape['tpos'][i]).norm(dim=-1, keepdim=True),
                               task.direct[i, None].float(), task.notice[i, None]), -1).cpu())
    return orig(st, tape, reward, info, terminated, truncated)
task.observe = observe
tr.collect(250, noise=0.0, learn=False); rows.clear(); tr.collect(2500, noise=0.0, learn=False)
a = torch.cat(rows).numpy()
fam, ty, py, ry, lz, rz, perr, direct, notice = a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7] > 0.5, a[:, 8]
b0, b1 = np.median(lz), np.median(rz)
stepped = (lz > b0 + 0.02) | (rz > b1 + 0.02)
print(path.name, len(a), 'strikes; notice of clip questions q1/50/99: %s, of direct questions: %s' % (np.quantile(notice[~direct], [.01, .5, .99]).round(2), np.quantile(notice[direct], [.01, .5, .99]).round(2)))
right, split, left = t.side_bounds  # the checkpoint's own stand | step boundaries
for zone, zl, zh in (('right of %+.2f (step forehand)' % right, -9, right), ('left of %+.2f (step backhand)' % left, left, 9), ('between (stand zone)', right, left)):
    print(' zone %s' % zone)
    for name, sel in (('clip question, step stroke', ~direct & (fam >= 2)), ('clip question, stand stroke', ~direct & (fam < 2)),
                      ('direct question, step stroke', direct & (fam >= 2)), ('direct question, stand stroke', direct & (fam < 2))):
        for nl, nh in ((0.29, 0.37), (0.37, 0.41), (0.41, 0.45), (0.45, 0.49), (0.49, 0.53), (0.53, 0.60), (0.60, 2.0)):
            m = sel & (ty >= zl) & (ty < zh) & (notice >= nl) & (notice < nh)
            if m.sum() < 8: continue
            print('   %-29s notice %.2f-%.2f s: n %4d | asked pelvis shift %+5.1f cm | policy %+5.1f cm (sd %.1f) | stepped %3.0f%% | racket to target %4.1f cm' % (
                name, nl, nh, m.sum(), 100 * ry[m].mean(), 100 * py[m].mean(), 100 * py[m].std(), 100 * stepped[m].mean(), 100 * np.median(perr[m])))
