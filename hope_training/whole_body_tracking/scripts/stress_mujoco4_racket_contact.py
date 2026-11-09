"""One-step high-speed ball sweeps through the published A3 racket mesh.

This isolates collision detection from the GRU, serve bank, and rally scorer.
It deliberately changes timestep and speed beyond the published evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from mujoco_eval_recurrent import Plant


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE = ROOT / 'a3_deploy/a3_deploy_example/models/model_19000/policy/exported'
SPEEDS = (5., 15., 25., 40., 60.)
TIMESTEPS = (.001, .002, .005, .016)
START_OFFSET_M = .05
REFERENCE_DT = .00025


def racket_contacts(plant):
    mj, model, data, ball = plant.mj, plant.model, plant.data, plant.ball_physics
    rows = []
    for index, contact in enumerate(data.contact):
        if {int(contact.geom1), int(contact.geom2)} != {ball.gid, ball.racket}:
            continue
        force = np.zeros(6)
        mj.mj_contactForce(model, data, index, force)
        rows.append({'normal_force_n': float(force[0]),
                     'solver_distance_m': float(contact.dist)})
    return rows


def position_ball(plant, speed, side, timestep):
    mj, model, data, ball = plant.mj, plant.model, plant.data, plant.ball_physics
    plant.reset(0)
    model.opt.timestep = timestep
    site = data.site_xpos[plant.racket].copy()
    normal = data.site_xmat[plant.racket].reshape(3, 3)[:, 1].copy()
    data.qpos[ball.q:ball.q + 7] = [*(site + side * START_OFFSET_M * normal), 1, 0, 0, 0]
    data.qvel[ball.v:ball.v + 6] = [*(-side * speed * normal), 0, 0, 0]
    mj.mj_forward(model, data)
    return site, normal


def geometric_gap(plant):
    mj, model, data, ball = plant.mj, plant.model, plant.data, plant.ball_physics
    return float(mj.mj_geomDistance(model, data, ball.gid, ball.racket, 1., np.zeros(6)))


def reference_contact_time(plant, speed, side):
    position_ball(plant, speed, side, REFERENCE_DT)
    for step in range(80):
        plant.mj.mj_step(plant.model, plant.data)
        if any(row['normal_force_n'] > 1e-8 for row in racket_contacts(plant)):
            return (step + 1) * REFERENCE_DT
    return None


def one_step(plant, speed, side, timestep, reference_time):
    site, normal = position_ball(plant, speed, side, timestep)
    ball = plant.ball_physics
    data = plant.data
    gap_before = geometric_gap(plant)
    plant.mj.mj_step(plant.model, data)
    contacts = racket_contacts(plant)
    force_contact = any(row['normal_force_n'] > 1e-8 for row in contacts)
    ball_position = data.qpos[ball.q:ball.q + 3].copy()
    ball_velocity = data.qvel[ball.v:ball.v + 3].copy()
    plant.mj.mj_forward(plant.model, data)
    gap_after = geometric_gap(plant)
    end_side_m = float(side * np.dot(ball_position - site, normal))
    # The fine-step solver reports contact at the end of the step whose starting
    # pose first overlaps. Physical impact lies in the preceding reference tick.
    contact_due = reference_time is not None and reference_time - REFERENCE_DT <= timestep + 1e-12
    return {'speed_mps': speed, 'side': side, 'timestep_s': timestep,
            'reference_contact_s': reference_time, 'reference_contact_due': contact_due,
            'reference_contact_window_s': ([reference_time - REFERENCE_DT, reference_time]
                                           if reference_time is not None else None),
            'gap_before_m': gap_before, 'gap_after_m': gap_after,
            'end_side_m': end_side_m, 'end_normal_velocity_mps': float(side * np.dot(ball_velocity, normal)),
            'solver_contacts': contacts, 'positive_force_contact': force_contact,
            'missed_due_contact': bool(contact_due and not force_contact),
            'clear_tunnel': bool(contact_due and not force_contact and
                                 gap_after > 0 and end_side_m < -START_OFFSET_M)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speculative-contacts', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    manifest_path = args.bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    plant = Plant(manifest, physical=True, speculative_contacts=args.speculative_contacts)
    reference_plant = (Plant(manifest, physical=True, speculative_contacts=False)
                       if args.speculative_contacts else plant)
    bit = getattr(plant.mj.mjtEnableBit, 'mjENBL_SPECULATIVE', None)
    enabled = bool(bit is not None and plant.model.opt.enableflags & int(bit))
    assert enabled == args.speculative_contacts
    pairs = [(speed, side) for side in (1, -1) for speed in SPEEDS]
    reference = {(speed, side): reference_contact_time(reference_plant, speed, side)
                 for speed, side in pairs}
    rows = [one_step(plant, speed, side, timestep, reference[speed, side])
            for speed, side in pairs for timestep in TIMESTEPS]
    due = [row for row in rows if row['reference_contact_due']]
    result = {'kind': 'one_step_racket_mesh_stress_not_rally_or_gate3',
              'mujoco_version': plant.mj.__version__,
              'speculative_contacts': enabled, 'checkpoint_sha256': manifest['checkpoint_sha256'],
              'onnx_sha256': manifest['onnx_sha256'],
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'start_offset_m': START_OFFSET_M, 'reference_timestep_s': REFERENCE_DT,
              'reference_speculative_contacts': False,
              'directions': 'both sides of the real right_racket_collision mesh',
              'ball_radius_m': plant.ball_physics.radius,
              'racket_geom_name': 'right_racket_collision',
              'racket_pair': plant.ball_physics.racket_pair,
              'mujoco_integrator': str(plant.model.opt.integrator),
              'due_trials': len(due),
              'positive_force_due_trials': sum(row['positive_force_contact'] for row in due),
              'missed_due_trials': sum(row['missed_due_contact'] for row in due),
              'clear_tunnels': sum(row['clear_tunnel'] for row in due), 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('mujoco_version', 'speculative_contacts',
          'due_trials', 'positive_force_due_trials', 'missed_due_trials', 'clear_tunnels')}))


if __name__ == '__main__':
    sys.exit(main())
