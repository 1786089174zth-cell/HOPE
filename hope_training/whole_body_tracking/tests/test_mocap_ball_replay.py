"""Causality and original bank reconstruction checks."""

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch


TRAIN = Path(__file__).resolve().parents[1]
MDP = TRAIN / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, MDP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def constant_velocity(p, v, w, dt, params):
    return p + v * dt, v


def elastic_table_contact(v, w, params):
    return v * v.new_tensor([1, 1, -1]), -w


replay = load_module("mocap_ball_replay")
paddle = load_module("mocap_paddle_contact")
preparation = load_module("mocap_preparation")


class MocapReplayTests(unittest.TestCase):
    def setUp(self):
        self.time = torch.tensor([[0.0, 0.02, 0.04, 0.10], [0.0, 0.02, 0.04, torch.inf]])
        self.pos = torch.tensor([[[2.0, -0.7, 0.2], [1.9, -0.7, 0.2],
                                  [1.8, -0.7, 0.2], [1.5, -0.7, 0.2]],
                                 [[2.5, -0.5, 0.3], [2.4, -0.5, 0.3],
                                  [2.3, -0.5, 0.3], [0.0, 0.0, 0.0]]])
        self.batch = replay.ReplayBatch(self.time, self.pos)

    def test_actor_never_sees_unreceived_samples_or_padded_rows(self):
        history = self.batch.observe(torch.tensor([0, 1]), torch.tensor([0.051, 0.20]),
                                     period_s=0.02, delay_s=0.015, max_age_s=0.05).reshape(2, 4, 5)
        # The 40 ms sample would only arrive at 55 ms.
        torch.testing.assert_close(history[0, 0], torch.tensor([1.9, -0.7, 0.2, 0.031, 1.0]))
        torch.testing.assert_close(history[0, 1], torch.tensor([2.0, -0.7, 0.2, 0.031, 1.0]))
        self.assertFalse(history[0, 2:].any().item())
        self.assertFalse(history[1].any().item())

    def test_contact_path_samples_only_valid_intervals(self):
        position, velocity, valid = self.batch.physical_sample(
            torch.tensor([0, 0, 1]), torch.tensor([0.01, 0.07, 0.055]), max_gap_s=0.025,
        )
        self.assertEqual(valid.tolist(), [True, False, False])
        torch.testing.assert_close(position[0], torch.tensor([1.95, -0.7, 0.2]))
        torch.testing.assert_close(velocity[0], torch.tensor([-5., 0., 0.]))
        self.assertFalse(position[1:].any().item())

    def test_contact_path_includes_padded_row_endpoint_only(self):
        _, _, valid = self.batch.physical_sample(
            torch.tensor([1, 1]), torch.tensor([0.04, 0.041]), max_gap_s=0.025,
        )
        self.assertEqual(valid.tolist(), [True, False])

    def test_bank_replay_ends_at_original_strike_state(self):
        offset = (0.5, 0.7625, 0.76)
        target = np.array([[0.60, -0.30, 1.0], [0.55, 0.30, 1.05]])
        velocity = np.array([[-3.0, 0., 1.0], [-2.0, 0., 1.0]])
        with TemporaryDirectory() as directory:
            bank = Path(directory) / "test.npz"
            np.savez(bank, clip=[0, 1], station_xy=np.zeros((2, 2)),
                     target_position_offset=target, incoming_velocity=velocity,
                     incoming_spin=np.zeros((2, 3)))
            sha = hashlib.sha256(bank.read_bytes()).hexdigest()

            params = SimpleNamespace(ball_radius=0.02)
            flights, rows = replay.reconstruct_bank_flights(
                bank, sha, 1, 0, offset, constant_velocity, params,
            )
            extended, spins = replay.extend_bank_flights(
                flights, bank, rows, 0.120, offset, constant_velocity, params, elastic_table_contact, None,
            )
        self.assertEqual(rows, [0, 1])
        for row, (time, xyz) in enumerate(flights):
            np.testing.assert_allclose(xyz[-1] + offset, target[row])
            np.testing.assert_allclose(xyz[0] + offset, target[row] - time[-1] * velocity[row])
            self.assertGreater(xyz[0, 2], 0.025)
            self.assertLess(time[-1], 0.60)
            extended_time, extended_pos = extended[row]
            np.testing.assert_array_equal(extended_time[:len(time)], time)
            np.testing.assert_array_equal(extended_pos[:len(xyz)], xyz)
            self.assertEqual(len(extended_time) - len(time), 120)
            self.assertAlmostEqual(extended_time[-1] - time[-1], 0.120)
            np.testing.assert_allclose(extended_pos[-1] + offset, target[row] + 0.120 * velocity[row])
        self.assertEqual(spins.shape, (2, 121, 3))

    def test_table_step_rebounds_only_above_table_and_advances_remaining_time(self):
        position = torch.tensor([[1., 0., 0.79], [1., 0.90, 0.79]])
        velocity = torch.tensor([[2., 0., -2.], [2., 0., -2.]])
        spin = torch.ones_like(velocity)
        p, v, w, bounce, falling, impact = replay.step_with_table(
            position, velocity, spin, 0.020, constant_velocity, SimpleNamespace(ball_radius=0.02),
            elastic_table_contact, None, 0.5, 0.76,
        )
        self.assertEqual(falling.tolist(), [True, True])
        self.assertEqual(bounce.tolist(), [True, False])
        torch.testing.assert_close(impact[:, 0], torch.tensor([1.01, 1.01]))
        torch.testing.assert_close(p[:, 2], torch.tensor([0.81, 0.75]))
        torch.testing.assert_close(v[:, 2], torch.tensor([2., -2.]))
        torch.testing.assert_close(w[:, 0], torch.tensor([-1., 1.]))

    def test_table_step_accepts_a_partial_physics_step_per_ball(self):
        p = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
        v = torch.tensor([[2., 0., 0.], [2., 0., 0.]])
        new_p, *_ = replay.step_with_table(
            p, v, torch.zeros_like(v), torch.tensor([[0.002], [0.004]]), constant_velocity,
            SimpleNamespace(ball_radius=0.02), elastic_table_contact, None, 0.5, 0.76,
        )
        torch.testing.assert_close(new_p[:, 0], torch.tensor([1.004, 1.008]))


class PaddleSweepTests(unittest.TestCase):
    def test_sweep_catches_interframe_face_and_rim_only_at_ball_radius(self):
        start = torch.tensor([[0., 0., 0.04], [0.09, 0., 0.04], [0.096, 0., 0.04], [0., 0., 0.03]])
        end = torch.tensor([[0., 0., -0.04], [0.09, 0., -0.04], [0.096, 0., -0.04], [0., 0., 0.023]])
        normal = torch.tensor([0., 0., 1.]).expand_as(start)
        hit, t, point, outward, *_ = paddle.sweep_ball_against_disc(start, end, normal, normal, 0.075, 0.02)
        self.assertEqual(hit.tolist(), [True, True, False, False])
        expected_rim_height = math.sqrt(0.02**2 - 0.015**2)
        torch.testing.assert_close(t[:2], torch.tensor([0.25, (0.04 - expected_rim_height) / 0.08]), atol=3e-5, rtol=0)
        torch.testing.assert_close(point[:2, 0], torch.tensor([0., 0.075]))
        torch.testing.assert_close(outward[1], torch.tensor([0.75, 0., expected_rim_height / 0.02]), atol=1e-4, rtol=0)

    def test_sweep_handles_rotation_and_rejects_separating_overlap(self):
        start = torch.tensor([[0., 0.06, 0.06], [0., 0., 0.015]])
        end = torch.tensor([[0., 0.06, 0.06], [0., 0., 0.04]])
        n0 = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
        n1 = torch.tensor([[0., -1., 0.], [0., 0., 1.]])
        hit, t, _, normal, *_ = paddle.sweep_ball_against_disc(start, end, n0, n1, 0.075, 0.02)
        self.assertEqual(hit.tolist(), [True, False])
        self.assertGreater(t[0].item(), 0.0)
        self.assertLess(t[0].item(), 0.5)
        torch.testing.assert_close(torch.linalg.vector_norm(normal[0], dim=-1), torch.tensor(1.0))


class PreparationRewardTests(unittest.TestCase):
    def test_preparation_credits_only_new_progress_toward_ready(self):
        best = torch.tensor([-1.0])
        paid = []
        for distance in (0.30, 0.15, 0.15, 0.30, 0.15, 0.0):
            delta, best = preparation.advance_preparation_progress(
                best, torch.tensor([distance**2]), torch.tensor([0.20]),
                torch.tensor([True]), torch.tensor([True]), (0.10, 0.30), 0.15,
            )
            paid.append(delta.item())
        self.assertEqual([x > 0 for x in paid], [False, True, False, False, False, True])
        self.assertAlmostEqual(sum(paid), 1.0 - math.exp(-4.0), places=6)

    def test_preparation_checks_timing_ball_availability_and_uprightness(self):
        prev = torch.full((5,), 0.25)
        progress, best = preparation.advance_preparation_progress(
            prev, torch.zeros(5), torch.tensor([0.31, 0.20, 0.09, 0.20, 0.20]),
            torch.tensor([True, True, True, False, True]),
            torch.tensor([True, False, True, True, True]), (0.10, 0.30), 0.15,
        )
        torch.testing.assert_close(progress, torch.tensor([0., 0., 0., 0., 0.75]))
        # Returning upright without moving the racket cannot cash in an unstable advance.
        recovered, _ = preparation.advance_preparation_progress(
            best[1:2], torch.zeros(1), torch.tensor([0.18]),
            torch.tensor([True]), torch.tensor([True]), (0.10, 0.30), 0.15,
        )
        self.assertEqual(recovered.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
