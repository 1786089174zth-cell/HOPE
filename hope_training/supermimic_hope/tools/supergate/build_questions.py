"""Training questions from Supergate's offline Planner.

    /opt/drone_venv/bin/python tools/supergate/build_questions.py \
        --runs <supergate output dir> [...] --datasets <suite dir> [...] --out assets/<name>.npz     (grasping container)

Each launch of the given runs (``make_launches.py`` suites, not the frozen test suites) that the Planner
committed becomes one question: the racket target in station axes (position, velocity; the face normal is
the velocity direction, as in the training bank), the ball it is for (velocity at the strike, spin from the
suite's ledger), the notice the policy got, the side (forehand or backhand face, by the policy's own lateral
split) and a strike pose for the reference.

The bank's reference is a recorded clip with a right-arm correction solved at the clip's strike frame. That
does not carry over: with the torso where the stand clips have it, Supergate's backhand targets are out of
the arm's reach (median residual 18 cm; they are flat strokes 0.58 m ahead of the station, up to 1.42 m high
and 0.25 m left of it), and with 0.22-0.40 s of notice there is no time for a clip's backswing anyway. So the
strike pose is solved from the HOLD pose with the waist (yaw, roll, pitch) and the seven right-arm joints:
damped least squares for the racket point and face, started from the stand clip's strike arm (or from
``--seed-pose``: the redundant arm has many poses for one racket target, and a reference far from the pose the
policy already strikes with makes it unlearn a working stroke first), inside the soft joint limits and a bounded
waist; then the joint velocity of least weighted norm for the racket velocity. The
task turns it into a direct stroke: HOLD -> strike pose -> HOLD (``core/task.py``, ``_direct``).
A row is kept when the racket is within ``--position`` of the target and the face within ``--normal``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from supermimic_hope import a3  # noqa: E402
from supermimic_hope.core.fk import Kinematics  # noqa: E402

ASSETS = HERE.parents[2] / "whole_body_tracking/assets/wide_stroke_v26_urdf2_plane058_20261110"
HOME_Y = -0.7625
SPLIT = -0.275  # the bank's forehand | backhand boundary, station axes
WAIST, ARM = (2, 5, 8), (13, 18, 22, 24, 26, 28, 30)
# Waist yaw, roll, pitch about the HOLD pose, rad. The yaw is wide on purpose: with the strike plane 0.58 m ahead
# of the station a ball 0.1-0.35 m right of the centre line is straight in front of the hitting shoulder, the
# forearm points along the ball's path and the blade is edge-on unless the wrist yaws far or the trunk turns.
# With the wrist kept near the policy's own use, 49-65% of such targets have a strike pose at +-0.6 rad of waist
# yaw and 87-92% at +-1.0 (the same targets 0.42 m ahead: 95-99% at +-0.6).
WAIST_BOX = ((-1.00, 1.00), (-0.25, 0.25), (-0.15, 0.40))
# The two wrist joints at the blade (pitch, yaw) are the robot's weakest and slowest: action scale 0.07 rad (the
# other arm joints 0.2-0.38), stiffness 20 (30-40). A solve that treats the arm joints alike turns the face with
# the wrist yaw (+-1.2 rad, up to 10 rad/s at the strike), and a policy trained on such a reference follows every
# joint to 0.1-0.2 rad except that one (0.4-0.7 rad short: the face 30-50 deg off, docs/supergate_20261119.md).
# So they cost as much as the waist, turn slowly, and with ``--seed-pose`` stay inside WRIST_BOX of the seed.
SPEED = (2.2, 1.5, 1.5, 8.0, 8.0, 8.0, 7.0, 6.0, 2.0, 2.0)  # rad/s at the strike, waist then arm
# What a unit of each joint's motion costs, for the pose and for the strike velocity. The elbow is the cheapest
# lever for the racket, and at equal arm costs the forehand came out with the elbow bent to 1.44 rad (limit 1.61)
# and flexing at up to 11 rad/s at the strike; the recorded forehand strikes with the elbow at 0.25 rad, extending
# at 4 rad/s, the shoulder pitching at 3 rad/s. Trained on that, the racket reached the strike at two thirds of
# the commanded speed. The trunk turns to place the arm (waist yaw up to 1 rad) but gives little of the racket
# speed, as in the recordings (0.3 rad/s at the strike): a trunk swinging at 3 rad/s is a lot of angular momentum
# for the feet to hold. Waist roll and pitch tilt the trunk and are kept small.
COST = (4.0, 6.0, 6.0, 1.0, 1.0, 1.0, 2.5, 2.0, 5.0, 5.0)
SPEED_COST = (5.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.2, 3.0, 8.0, 8.0)
WRIST_BOX = (0.25, 0.40)  # wrist pitch, yaw about the seed pose, rad
# ... and the wrist roll within 0.9 rad of it: without that a "forehand" is also solved as a backhand arm with
# the wrist rolled half a turn. Shoulder and elbow are free inside their limits (boxed like the bank's adapter,
# about one mean pose, no ball above 1.2 m in the crossover zone has a strike pose).
ARM_BOX = (9.0, 9.0, 9.0, 9.0, 0.9)  # shoulder pitch, roll, yaw, elbow, wrist roll about the seed pose, rad
# How far a strike pose stays inside the commanded-position limits (lower, upper), waist then arm, rad. A PD joint
# holds its load kp * (command - position) away from the command, and the command stops at the limit: the arm
# raised sideways weighs about 10 N m on the shoulder roll (kp 40), so the joint settles 0.2-0.25 rad short of a
# command at the limit. With 0.03 rad everywhere the high backhands 0.15-0.34 m right of the centre line were given
# a shoulder roll 0.12 rad from its limit (-2.36 of -2.48); every checkpoint arrived 0.22 rad short of it, the racket
# 5 cm under the target, whatever the reward did (ft27, ft29).
LIMIT_MARGIN = ((0.03, 0.03), (0.03, 0.03), (0.03, 0.03), (0.10, 0.10), (0.28, 0.03), (0.05, 0.05), (0.05, 0.10), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05))
# ... and that margin is where the joint settles, not where it is 0.4 s after the announcement: with the pose at
# -2.20 every checkpoint struck at -2.04..-2.08 (ft30-ft37), also with the reference ahead of its cubic and the
# command free to jump (ft36, ft37), and the racket stayed 5.4-6.2 cm low on the high backhands 0.15-0.34 m
# right of the centre line. The trunk has the reach instead: with the waist roll command at its limit over the
# stroke the same policy (ft37 model_99000) had the racket 0.8 cm from the target. The first solve hardly asks
# for it (-0.06 rad there: the waist costs 6 in COST). Rows whose pose has the shoulder roll beyond ROLL_REACH
# are solved again from that pose, with the shoulder roll kept inside ROLL_REACH and ROLL_COST: the waist roll
# makes up for it, the other joints stay about where they were (solved afresh with a cheap waist roll alone,
# a third of these rows went to another arm configuration 3 cm from the target).
ROLL_REACH = 0.42  # rad from the shoulder roll's lower command limit: -2.06
ROLL_COST = (4.0, 1.0, 6.0, 3.0, 1.0, 3.0, 4.0, 4.0, 8.0, 8.0)


class Racket:
    """Racket point, face normal and velocity of full states, as the task computes them."""

    def __init__(self):
        model, _ = a3.build_model()
        self.kin = Kinematics(model, a3.JOINTS)
        self.wrist = self.kin.names.index("right_wrist_yaw_Link")
        self.offset = torch.tensor(model.site(a3.RACKET_SITE).pos, dtype=torch.float32)

    def __call__(self, s, sign):
        pos, rot, omega, vel = self.kin(s)
        R = rot[:, self.wrist]
        arm = (R @ self.offset[..., None])[..., 0]
        return pos[:, self.wrist] + arm, R[:, :, 1] * sign[:, None], vel[:, self.wrist] + torch.cross(omega[:, self.wrist], arm, dim=-1)


def jacobian(fn, q, eps=1e-4):
    """Central-difference Jacobian of ``fn(q) -> (B, m)`` over the columns of ``q`` (B, n): (B, m, n)."""
    cols = []
    for j in range(q.shape[1]):
        d = torch.zeros_like(q)
        d[:, j] = eps
        cols.append((fn(q + d) - fn(q - d)) / (2 * eps))
    return torch.stack(cols, -1)


@torch.no_grad()
def solve(pos, vel, side, assets=ASSETS, iterations=150, device="cpu", seed_pose=None, step=None, margin=LIMIT_MARGIN, cost=COST):
    """Strike pose and joint velocity as offsets from HOLD, (N, 31) each, and the residuals
    (position m, face deg, velocity m/s, largest joint offset rad). Rows with ``step`` are solved with the
    pelvis where the side's step clip has it at the strike (12 cm right and 8 cm down for the forehand,
    10 cm left and 5 cm down for the backhand, slightly turned): the task takes root and legs from that clip."""
    torch.set_default_device(device)  # everything below is built on the solver's device
    racket = Racket()
    racket.kin.to(device)
    clips = [np.load(assets / f"{name}.npz") for name in ("stand_forehand", "stand_backhand")]
    hold = torch.zeros(75)
    hold[0:3], hold[3:7] = torch.tensor(clips[1]["body_pos_w"][0, 0]).float(), torch.tensor(clips[1]["body_quat_w"][0, 0]).float()
    hold[13:44] = torch.tensor(clips[1]["joint_pos"][0]).float()
    cols = torch.tensor(WAIST + ARM)
    pos, vel = torch.tensor(pos, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)
    side = torch.tensor(side)  # 0 forehand, 1 backhand
    root = hold[0:7].repeat(len(pos), 1)
    if step is not None and np.any(step):
        steps = [np.load(assets / f"{name}.npz") for name in ("step_forehand", "step_backhand")]
        at_strike = torch.tensor(np.stack([np.concatenate((c["body_pos_w"][60, 0], c["body_quat_w"][60, 0])) for c in steps]), dtype=torch.float32)
        root = torch.where(torch.tensor(np.asarray(step, bool))[:, None], at_strike[side], root)
    sign = torch.where(side == 0, 1.0, -1.0)
    normal = vel / vel.norm(dim=-1, keepdim=True)
    n = len(pos)
    seed = hold[13:44][cols].repeat(n, 1)
    seed[:, 3:] = torch.tensor(np.stack([c["joint_pos"][60][list(ARM)] for c in clips]), dtype=torch.float32)[side]
    if isinstance(seed_pose, str):  # "hold": start from and stay near the HOLD pose itself
        seed = hold[13:44][cols].repeat(n, 1)
    elif seed_pose is not None:  # (2, 10): waist and arm to start from and to stay near, per side; (N, 10): per row
        seed = torch.tensor(seed_pose, dtype=torch.float32)
        seed = seed[side] if seed.shape[0] == 2 and n != 2 else seed
    margin = torch.tensor(margin)
    lo, hi = torch.tensor(a3.QDES_LO, dtype=torch.float32)[cols] + margin[:, 0], torch.tensor(a3.QDES_HI, dtype=torch.float32)[cols] - margin[:, 1]
    box = torch.tensor(WAIST_BOX)
    lo[:3], hi[:3] = torch.maximum(lo[:3], hold[13:44][cols[:3]] + box[:, 0]), torch.minimum(hi[:3], hold[13:44][cols[:3]] + box[:, 1])
    cost = torch.tensor(cost)
    lo, hi = lo.repeat(n, 1), hi.repeat(n, 1)
    if seed_pose is not None:
        box = torch.tensor(ARM_BOX + WRIST_BOX)
        lo[:, 3:], hi[:, 3:] = torch.maximum(lo[:, 3:], seed[:, 3:] - box), torch.minimum(hi[:, 3:], seed[:, 3:] + box)

    def state(q, dq=None):
        s = hold.repeat(n, 1)
        s[:, 0:7] = root
        s[:, 13 + cols] = q
        if dq is not None:
            s[:, 44 + cols] = dq
        return s

    def task(q):
        point, face, _ = racket(state(q), sign)
        return torch.cat((pos - point, 0.20 * (normal - face)), -1)

    q = torch.minimum(torch.maximum(seed, lo), hi)
    eye6, eye10 = torch.eye(6), torch.eye(10)
    for _ in range(iterations):
        err = task(q)
        J = -jacobian(task, q) / cost  # in cost-scaled joint coordinates
        inv = torch.linalg.solve(J @ J.transpose(1, 2) + 0.0003 * eye6, torch.cat((err[:, :, None], J), -1))
        pseudo = J.transpose(1, 2) @ inv
        change = pseudo[:, :, 0] + 0.025 * ((eye10 - pseudo[:, :, 1:]) @ ((seed - q) * cost)[:, :, None])[:, :, 0]
        q = torch.minimum(torch.maximum(q + (change / cost).clamp(-0.08, 0.08), lo), hi)
    point, face, _ = racket(state(q), sign)
    speed_cost = torch.tensor(SPEED_COST)
    Jv = jacobian(lambda dq: racket(state(q, dq), sign)[2], torch.zeros_like(q), eps=1e-2) / speed_cost
    limit = 0.9 * torch.tensor(SPEED)
    dq, free = torch.zeros_like(q), torch.ones_like(q)
    for _ in range(4):  # what a joint at its speed limit cannot give is asked of the others
        rest = vel - (Jv * speed_cost @ dq[:, :, None])[:, :, 0]
        Jf = Jv * free[:, None, :]
        dq = dq + (Jf.transpose(1, 2) @ torch.linalg.solve(Jf @ Jf.transpose(1, 2) + 1e-4 * torch.eye(3), rest[:, :, None]))[:, :, 0] / speed_cost
        free = free * (dq.abs() < limit).float()
        dq = torch.minimum(torch.maximum(dq, -limit), limit)
    got = racket(state(q, dq), sign)[2]
    errors = torch.stack(((point - pos).norm(dim=-1), torch.rad2deg(torch.acos((face * normal).sum(-1).clamp(-1, 1))),
                          (got - vel).norm(dim=-1), (q - hold[13:44][cols]).abs().amax(-1)), -1)
    full_q, full_dq = torch.zeros(n, 31), torch.zeros(n, 31)
    full_q[:, cols], full_dq[:, cols] = q - hold[13:44][cols], dq
    torch.set_default_device("cpu")
    return full_q.cpu().numpy(), full_dq.cpu().numpy(), errors.cpu().numpy()


def gravity_load(strike_q, side, step, assets=ASSETS):
    """How far past each strike pose a PD joint has to be commanded to hold it: gravity's generalized force on
    the joint over its stiffness, rad, (N, 31) in the policy's joint order, and the same for HOLD (31,).
    The pelvis is where ``solve`` had it (HOLD, or the step clip's strike frame)."""
    import mujoco

    model, _ = a3.build_model()
    data, ix = mujoco.MjData(model), a3.indices(model)
    clips = {name: np.load(assets / f"{name}.npz") for name in ("stand_backhand", "step_forehand", "step_backhand")}
    hold_root = np.concatenate((clips["stand_backhand"]["body_pos_w"][0, 0], clips["stand_backhand"]["body_quat_w"][0, 0]))
    hold_q = clips["stand_backhand"]["joint_pos"][0]
    roots = [np.concatenate((clips[n]["body_pos_w"][60, 0], clips[n]["body_quat_w"][60, 0])) for n in ("step_forehand", "step_backhand")]
    kp = np.asarray(a3.KP, dtype=np.float64)

    def load(root, q):
        data.qpos[:] = model.qpos0
        data.qpos[0:7] = root
        data.qpos[ix["qpos"]] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return data.qfrc_bias[ix["dof"]] / kp

    out = np.stack([load(roots[int(side[i])] if step[i] else hold_root, hold_q + strike_q[i]) for i in range(len(strike_q))])
    return out.astype(np.float32), load(hold_root, hold_q).astype(np.float32)


def collect(run: Path, dataset: Path):
    rows = []
    for shard in range(6):
        ledger = json.loads((dataset / f"shard-{shard + 1:02d}-ledger.json").read_text())
        res = {int(r["shot"]): r for r in csv.DictReader((run / f"env-{shard}-results.csv").open())}
        for k in csv.DictReader((run / f"env-{shard}-kinematics.csv").open()):
            shot = int(k["shot"])
            r = res.get(shot)
            if k["event"] != "planned" or r is None or int(r["valid_plans"]) < 1 or shot > len(ledger):
                continue
            led = ledger[shot - 1]
            rows.append([float(k[f"target_p_{a}"]) for a in "xyz"] + [float(k[f"target_v_{a}"]) for a in "xyz"]
                        + [float(r[f"strike_v{a}"]) for a in "xyz"] + list(led["predicted_contact_spin"])
                        + [float(r["lead"]), float(r["side"]), float(r["legal"]), float(int(r["contacts"]) > 0)])
    return rows


def tuple_rows(path: Path, home_x, plane, tolerance, window, count, notice, seed):
    """Questions of a physical tuple bank (main_urdf_2's stationary_reach_v5_coverage: target position from the
    station, racket velocity along the face normal, incoming velocity and spin) as rows like ``collect``'s.
    Kept: targets within ``tolerance`` of the strike plane and inside the lateral ``window``; ``count`` of
    them, drawn evenly over lateral position (eight bands) x incoming speed (< 2.5, < 4.0, faster) x height
    (< 1.0, < 1.2, higher). The bank has no announcement time: drawn from ``notice``, what the Planner gives a
    rally ball. Its strokes are another law than main's Planner (more lift: 0.74 m/s upwards for 0.06)."""
    z = np.load(path)
    pos, vel = z["target_position_offset"].astype(np.float64), z["target_velocity"].astype(np.float64)
    keep = np.flatnonzero((np.abs(pos[:, 0] - plane) <= tolerance) & (pos[:, 1] >= window[0]) & (pos[:, 1] < window[1]))
    rng = np.random.default_rng(seed)
    lateral = np.digitize(pos[keep, 1], np.linspace(window[0], window[1], 9)[1:-1])
    speed = np.digitize(np.linalg.norm(z["incoming_velocity"][keep], axis=1), (2.5, 4.0))
    height = np.digitize(pos[keep, 2], (1.0, 1.2))
    key = lateral * 9 + speed * 3 + height
    per_cell = int(np.ceil(count / 72))
    chosen = np.concatenate([rng.permutation(keep[key == c])[:per_cell] for c in range(72)])
    chosen = rng.permutation(chosen)[:count]
    n = len(chosen)
    world = pos[chosen] + np.array([home_x, HOME_Y, 0.0])
    lead = rng.uniform(notice[0], notice[1], n)
    return np.concatenate((world, vel[chosen], z["incoming_velocity"][chosen], z["incoming_spin"][chosen], lead[:, None],
                           np.zeros((n, 1)), np.ones((n, 2))), 1), len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", default=[])
    ap.add_argument("--datasets", type=Path, nargs="+", default=[])
    ap.add_argument("--rows", type=Path, nargs="+", default=[], help="Planner targets computed without a simulation "
                    "(planner_targets.py --out): npz files with 'rows' in the layout of a run's rows")
    ap.add_argument("--tuple-bank", type=Path, help="also take questions from a physical tuple bank (see tuple_rows)")
    ap.add_argument("--tuple-count", type=int, default=120000)
    ap.add_argument("--tuple-tolerance", type=float, default=0.03, help="largest distance of a tuple target from the strike plane, m")
    ap.add_argument("--tuple-notice", type=float, nargs=2, default=(0.34, 0.40))
    ap.add_argument("--plane", type=float, default=0.58, help="strike plane ahead of the station, m (tuple bank rows)")
    ap.add_argument("--chunk", type=int, default=30000, help="rows solved at a time")
    ap.add_argument("--no-poses", action="store_true", help="the questions alone: no strike pose is solved (zero offsets, every row "
                    "kept). For training with task.extra_free=1.0, where a direct question is asked without a pose")
    ap.add_argument("--assets", type=Path, default=ASSETS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--position", type=float, default=0.03, help="largest racket position residual of a kept row, m")
    ap.add_argument("--normal", type=float, default=10.0, help="largest face residual of a kept row, deg")
    ap.add_argument("--velocity", type=float, default=0.15, help="largest racket velocity residual of a kept row, m/s")
    ap.add_argument("--either-side", action="store_true", help="solve the forehand and the backhand for every target and take the other "
                    "side where the bank's (lateral split at -0.275 m) has no strike pose. Not for training banks: the Planner "
                    "asks by the split alone, and the task moves a question whose side is not the split's to the other side "
                    "of it, away from the target its pose was solved for. With it, 70-80%% of the forehands above 1.15 m and "
                    "0.275-0.35 m right of the centre line became backhands at the split, and the policy lost those balls")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--step-beyond", type=float, nargs=2, metavar=("RIGHT", "LEFT"), help="targets further right than RIGHT "
                    "(forehands) or further left than LEFT (backhands), m in station axes, are step strokes (family 2, 3): where "
                    "a stand stroke runs out of reach (forehand 51%% of targets at 0.70-0.80 m right, backhand 61%% at 0.52-0.58 m "
                    "left; with the step 80%% and 95%%). Must be task.side_bounds[0] and [2] of the training (-0.68 0.44)")
    ap.add_argument("--split", type=float, default=SPLIT, help="forehand | backhand boundary, m in station axes: the side each "
                    "question is solved and asked on. With the strike 0.58 m ahead a backhand (trunk turned 45 deg) has a strike "
                    "pose for 95%% of the targets 0.275-0.36 m right of the centre line, a forehand for 45-52%% (5-19%% above "
                    "1.15 m), so the split the policy is trained and deployed with since ft27 is -0.34 (task.side_bounds)")
    ap.add_argument("--seed-pose", type=Path, help="npz with 'forehand' and 'backhand' (10: waist yaw/roll/pitch, 7 arm joints): the pose "
                    "the solve starts from and stays near, e.g. the policy's own mean strike pose (default: HOLD waist, the clip's strike arm)")
    ap.add_argument("--roll-reach", type=float, default=ROLL_REACH, help="rows whose strike pose has the right shoulder roll closer "
                    "than this to its lower command limit are solved again with the shoulder roll inside it and the trunk's roll "
                    "cheap (ROLL_COST); 0 keeps the first pose")
    ap.add_argument("--home-x", type=float, default=-0.58, help="HOME x the runs were made with (SUPERGATE_HOME_X; the protocol's is -0.5)")
    args = ap.parse_args()
    if len(args.runs) != len(args.datasets):
        ap.error("one dataset per run")
    if not args.runs and not args.tuple_bank and not args.rows:
        ap.error("--runs/--datasets, --rows or --tuple-bank")
    rows = np.array([x for run, data in zip(args.runs, args.datasets) for x in collect(run, data)]).reshape(-1, 16)
    source = np.zeros(len(rows), np.int8)  # 0: a Planner target of a run, 1: a tuple bank row, 2: a computed Planner target
    for path in args.rows:
        extra = np.load(path)["rows"]
        rows, source = np.concatenate((rows, extra)), np.concatenate((source, np.full(len(extra), 2, np.int8)))
    if args.tuple_bank:
        window = (args.step_beyond[0], args.step_beyond[1]) if args.step_beyond else (-0.68, 0.44)  # stand strokes only
        extra, near = tuple_rows(args.tuple_bank, args.home_x, args.plane, args.tuple_tolerance, window, args.tuple_count,
                                 args.tuple_notice, seed=20261121)
        print(f"{args.tuple_bank}: {near} rows within {args.tuple_tolerance} m of the plane and {window[0]:+.2f}..{window[1]:+.2f} m, {len(extra)} taken")
        rows, source = np.concatenate((rows, extra)), np.concatenate((source, np.ones(len(extra), np.int8)))
    finite = np.isfinite(rows).all(1)
    rows, source = rows[finite], source[finite]
    pos, vel = rows[:, 0:3] - np.array([args.home_x, HOME_Y, 0.0]), rows[:, 3:6]
    side = (pos[:, 1] >= args.split).astype(np.int64)
    step = np.zeros(len(pos), bool)
    if args.step_beyond:
        step = (pos[:, 1] < args.step_beyond[0]) | (pos[:, 1] > args.step_beyond[1])
    seed_pose = None
    if args.seed_pose:
        z = np.load(args.seed_pose)
        seed_pose = np.stack((z["forehand"], z["backhand"]))
    good = lambda e: (e[:, 0] <= args.position) & (e[:, 1] <= args.normal) & (e[:, 2] <= args.velocity)  # noqa: E731

    def solve_all(pos, vel, side, step, seed, **kw):  # in chunks: the solver holds ten Jacobian columns per row
        parts = [solve(pos[i:i + args.chunk], vel[i:i + args.chunk], side[i:i + args.chunk], args.assets, device=args.device,
                       seed_pose=seed if seed is None or np.ndim(seed) == 2 and len(seed) == 2 else seed[i:i + args.chunk],
                       step=step[i:i + args.chunk], **kw) for i in range(0, len(pos), args.chunk)]
        return tuple(np.concatenate([part[j] for part in parts]) for j in range(3))
    if args.no_poses:
        strike_q, strike_dq = np.zeros((len(pos), 31), np.float32), np.zeros((len(pos), 31), np.float32)
        errors = np.zeros((len(pos), 4), np.float32)
    elif args.either_side:
        # Both strokes for every target: the bank's side (split at -0.275 m) where it has a strike pose, the
        # other side where only that one has.
        fh, bh = (solve(pos, vel, np.full(len(pos), k), args.assets, device=args.device, seed_pose=seed_pose) for k in (0, 1))
        ok = [(r[2][:, 0] <= 0.03) & (r[2][:, 1] <= 10.0) for r in (fh, bh)]  # a strike pose: racket point and face
        use_bh = np.where(ok[0] ^ ok[1], ok[1], side == 1)
        side = use_bh.astype(np.int64)
        strike_q, strike_dq, errors = (np.where(use_bh[:, None], b, a) for a, b in zip(fh, bh))
    else:
        strike_q, strike_dq, errors = solve_all(pos, vel, side, step, seed_pose)
    rolled = np.zeros(len(pos), bool)
    if args.roll_reach > 0 and not args.either_side and not args.no_poses:
        roll = ARM[1]
        hold_roll = float(np.load(args.assets / "stand_backhand.npz")["joint_pos"][0][roll])
        beyond = hold_roll + strike_q[:, roll] < a3.QDES_LO[roll] + args.roll_reach - 1e-3
        if beyond.any():
            margin = [list(m) for m in LIMIT_MARGIN]
            margin[3 + 1][0] = args.roll_reach
            first = np.load(args.assets / "stand_backhand.npz")["joint_pos"][0][list(WAIST + ARM)] + strike_q[beyond][:, list(WAIST + ARM)]
            again = solve_all(pos[beyond], vel[beyond], side[beyond], step[beyond], first, margin=margin, cost=ROLL_COST)
            take = good(again[2]) | ~good(errors[beyond])  # the second pose unless only the first is a strike pose
            rows_again = np.flatnonzero(beyond)[take]
            strike_q[rows_again], strike_dq[rows_again], errors[rows_again] = again[0][take], again[1][take], again[2][take]
            rolled[rows_again] = True
            print(f"shoulder roll beyond {a3.QDES_LO[roll] + args.roll_reach:.2f} rad: {int(beyond.sum())} rows, solved again with the trunk's roll; "
                  f"{int(take.sum())} taken, {int(good(again[2]).sum())} strike poses; "
                  f"waist roll offset median {np.median(again[0][take][:, WAIST[1]]):+.2f}, 10-90% "
                  f"{np.quantile(again[0][take][:, WAIST[1]], 0.1):+.2f}..{np.quantile(again[0][take][:, WAIST[1]], 0.9):+.2f} rad; "
                  f"position residual median {100 * np.median(again[2][take][:, 0]):.1f} cm")
    keep = good(errors)
    if args.no_poses:  # HOLD's load for every row: nothing uses a pose's
        _, hold_load = gravity_load(strike_q[:1], side[:1], step[:1], args.assets)
        strike_load = np.tile(hold_load, (len(pos), 1))
    else:
        strike_load, hold_load = gravity_load(strike_q, side, step, args.assets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, target_position_offset=pos.astype(np.float32), target_velocity=vel.astype(np.float32),
                        incoming_velocity=rows[:, 6:9].astype(np.float32), incoming_spin=rows[:, 9:12].astype(np.float32),
                        time_to_strike=rows[:, 12].astype(np.float32), planner_side=rows[:, 13].astype(np.int8),
                        family=(side + 2 * step).astype(np.int32), strike_q=strike_q.astype(np.float32), strike_dq=strike_dq.astype(np.float32),
                        errors=errors.astype(np.float32), qualified=keep, strike_load=strike_load, hold_load=hold_load,
                        source_legal=rows[:, 14].astype(bool), source_contact=rows[:, 15].astype(bool), source=source)
    print(f"{args.out}: {len(rows)} planned launches, {int(keep.sum())} kept (position <= {args.position} m, face <= {args.normal} deg)")
    q = lambda x: f"{np.median(x):.2f}/{np.quantile(x, 0.9):.2f}"  # noqa: E731
    for f, name in enumerate(("forehand", "backhand", "step forehand", "step backhand")):
        m = (side + 2 * step) == f
        if not m.any():
            continue
        e = errors[m]
        print(f"  {name:13s} {int(m.sum()):5d} rows, kept {int((keep & m).sum()):5d}; residual median/p90: position "
              f"{100 * np.median(e[:, 0]):.1f}/{100 * np.quantile(e[:, 0], 0.9):.1f} cm, face {q(e[:, 1])} deg, velocity {q(e[:, 2])} m/s; "
              f"waist yaw/roll/pitch offsets p90 {np.quantile(np.abs(strike_q[m][:, list(WAIST)]), 0.9, axis=0).round(2).tolist()} rad; notice {np.median(rows[m, 12]):.2f} s")
    cols = list(WAIST + ARM)
    print("  gravity load at the strike pose (command past the pose, rad; waist yaw/roll/pitch, shoulder pitch/roll/yaw, elbow, wrist "
          f"roll/pitch/yaw): median {np.median(strike_load[:, cols], 0).round(2).tolist()}, HOLD {hold_load[cols].round(2).tolist()}")
    for lo, hi in ((0.8, 1.0), (1.0, 1.2), (1.2, 1.5)):
        m = (pos[:, 2] >= lo) & (pos[:, 2] < hi)
        print(f"  height {lo:.1f}-{hi:.1f} m: {int(m.sum()):5d} rows, kept {int((keep & m).sum()):5d}")
    speed, height = np.digitize(np.linalg.norm(rows[:, 6:9], axis=1), (2.5, 4.0)), np.digitize(pos[:, 2], (1.0, 1.2))
    print("  kept rows per stroke, incoming speed (< 2.5, < 4.0, faster) x height (< 1.0, < 1.2, higher); Planner targets + tuple bank rows:")
    for f, name in enumerate(("forehand", "backhand", "step forehand", "step backhand")):
        m = keep & ((side + 2 * step) == f)
        print(f"    {name:13s} " + "  ".join("[" + " ".join(f"{int((m & (source != 1) & (speed == i) & (height == j)).sum())}+{int((m & (source == 1) & (speed == i) & (height == j)).sum())}"
                                                             for j in range(3)) + "]" for i in range(3)))
    for lo, hi in ((-1.0, -0.8), (-0.8, -0.68), (-0.68, -0.5), (-0.5, -0.34), (-0.34, -0.1), (-0.1, 0.1), (0.1, 0.3), (0.3, 0.44), (0.44, 0.56), (0.56, 0.7)):
        m = (pos[:, 1] >= lo) & (pos[:, 1] < hi)
        print(f"  lateral {lo:+.2f}..{hi:+.2f} m: {int(m.sum()):5d} rows, kept {int((keep & m).sum()):5d}")


if __name__ == "__main__":
    main()
