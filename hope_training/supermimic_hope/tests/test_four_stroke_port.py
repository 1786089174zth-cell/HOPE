"""CPU checks for the pieces added when SuperMimic moved to the v26 four-stroke task."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope.analytic.noise import SensorNoisePenalty, parse_channel  # noqa: E402
from supermimic_hope.environment import scale_schedule_clock  # noqa: E402

OmegaConf = pytest.importorskip("omegaconf").OmegaConf


def test_schedule_clock_scales_learning_schedules_only():
    task = OmegaConf.create({
        "domain_rand": {"push": {"ramp_steps": 96000, "interval_s": [5.0, 8.0]}},
        "motion": {
            "stroke_coverage_floor_start_steps": 48000, "stroke_coverage_floor_ramp_steps": 240000,
            "stroke_coverage_ramp_steps": 64000, "rally_timing_ramp_steps": 0,
            # evidence windows and physical durations: real time, never scaled
            "stroke_stage_check_steps": 200, "stroke_stage_max_age_steps": 2000,
            "hold_long_steps_range": [500, 1000], "stand_start_min_hold": 200,
        },
        "rewards": {"full_body_quiet_start_steps": 128000, "full_body_quiet_ramp_steps": 128000,
                    "qdes_reversal_ramp_steps": 32000, "full_body_hold_start_steps": 0},
        "racket": {"vel_range_ramp_steps": 96000, "ability_gate_dwell_steps": 250,
                   "ability_quality_max_age_steps": 250, "base_mocap_delay_steps": 1},
    })
    changed = scale_schedule_clock(task, 5.0)
    assert task.motion.stroke_coverage_floor_start_steps == 9600
    assert task.motion.stroke_coverage_floor_ramp_steps == 48000
    assert task.rewards.full_body_quiet_start_steps == 25600
    assert task.domain_rand.push.ramp_steps == 19200
    assert task.rewards.full_body_hold_start_steps == 0 and task.motion.rally_timing_ramp_steps == 0
    for section, key, value in (("motion", "stroke_stage_check_steps", 200), ("motion", "stroke_stage_max_age_steps", 2000),
                                ("motion", "stand_start_min_hold", 200), ("racket", "ability_gate_dwell_steps", 250),
                                ("racket", "ability_quality_max_age_steps", 250), ("racket", "base_mocap_delay_steps", 1)):
        assert task[section][key] == value
    assert list(task.motion.hold_long_steps_range) == [500, 1000]
    assert changed["rewards.qdes_reversal_ramp_steps"] == [32000, 6400]
    assert len(changed) == 8


def test_schedule_clock_identity_and_validation():
    task = OmegaConf.create({"rewards": {"a_ramp_steps": 7}})
    assert scale_schedule_clock(task, 1.0) == {} and task.rewards.a_ramp_steps == 7
    with pytest.raises(ValueError):
        scale_schedule_clock(task, 0.5)
    with pytest.raises(ValueError):
        scale_schedule_clock(OmegaConf.create({"rewards": {"a_ramp_steps": [1, 2]}}), 2.0)
    assert scale_schedule_clock(OmegaConf.create({"rewards": {"a_ramp_steps": 3}}), 100.0) == {"rewards.a_ramp_steps": [3, 1]}


def test_sensor_noise_penalty_measures_weighted_actor_gain():
    torch.manual_seed(0)
    gain = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 3.0], [5.0, 5.0, 5.0, 5.0]])
    actor = lambda x: x @ gain.T  # noqa: E731 -- linear actor: output noise is exactly gain @ n
    channels = ({"dim": 1, "distribution": "uniform", "scale": 0.3, "targets": [[0, 1, 0, 1]]},
                {"dim": 1, "distribution": "gaussian", "scale": [0.01], "targets": [[2, 3, 0, -1], [3, 4, 0, -1]]})
    penalty = SensorNoisePenalty(actor, 2.0, channels, output_weights=(1.0, 0.5, 0.0), samples=200000)
    value = float(penalty(torch.randn(200000, 4)))
    # uniform half-width 0.3 -> variance 0.03; one mocap draw moves both targets with sign -1.
    expected = 2.0 * (4.0 * 0.03 + 0.25 * (6.0 * 0.01) ** 2)
    assert value == pytest.approx(expected, rel=0.02)
    assert float(SensorNoisePenalty(actor, 0.0, channels, ())(torch.randn(8, 4))) == 0.0


def test_sensor_noise_channel_validation():
    assert parse_channel({"dim": 3, "scale": 0.1, "targets": [[0, 3, 0, 1]]})[1] is False
    for bad in ({"dim": 0, "scale": 0.1, "targets": [[0, 1, 0, 1]]},
                {"dim": 2, "scale": [0.1], "targets": [[0, 2, 0, 1]]},
                {"dim": 2, "scale": 0.1, "targets": [[0, 3, 0, 1]]},
                {"dim": 2, "scale": 0.1, "targets": [[0, 2, 0, 2]]},
                {"dim": 2, "scale": 0.1, "targets": []}):
        with pytest.raises(ValueError):
            parse_channel(bad)
