from pathlib import Path
import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_hope.logging import WandbLogger, remote_metrics
from supermimic_hope.telemetry import EpisodeStats, scalar_snapshot


def test_remote_names_preserve_event_denominators_and_episode_semantics():
    row = {
        "iteration": 20,
        "wm_loss": 1.0,
        "actor_loss": 2.0,
        "imagined_return": 3.0,
        "task/Episode_Reward/full_body_hold": 0.5,
        "task/Metrics/racket_target/virtual_hit_rate": 0.2,
        "Live/racket_target/virtual_hit_rate": 0.8,
        "Live/racket_target/qdes_step_rms_rad_hold": math.pi / 180,
        "StandFH/hit_rate": float("nan"),
        "StandFH/strike_trials_recent": 0.0,
        "StandBH/hit_rate": 0.5,
        "StandBH/strike_trials_recent": 4.0,
    }
    result = remote_metrics(row)
    assert result["WorldModel/train_loss"] == 1.0
    assert result["Policy/loss"] == 2.0
    assert result["Imagination/predicted_return"] == 3.0
    assert result["EpisodeMetrics/racket_target/virtual_hit_rate"] == 0.2
    assert result["Return/contact_rate"] == 0.8
    assert result["Action/qdes_step_rms_deg_hold"] == pytest.approx(1.0)
    assert "StandFH/hit_rate" not in result
    assert result["StandFH/strike_trials_recent"] == 0.0
    assert result["StandBH/hit_rate"] == 0.5
    assert not any(key.startswith("PPO/") for key in result)


def test_scalar_snapshot_is_read_only_rng_neutral_and_keeps_missing_values():
    x = torch.tensor([1.0, 3.0])
    original = x.clone()
    rng = torch.random.get_rng_state().clone()
    values = scalar_snapshot(
        {"mean": x, "empty": torch.empty(0), "missing": torch.tensor(float("nan"))}
    )
    assert values == {"mean": 2.0, "missing": None}
    assert torch.equal(x, original) and torch.equal(rng, torch.random.get_rng_state())
    json.dumps(values, allow_nan=False)


def test_episode_bookkeeping_counts_true_completions_without_carrying_reset_rewards():
    stats = EpisodeStats(2, "cpu", 0.02)
    assert "Overview/episode_return_mean" not in stats.scalars()
    stats.update(torch.tensor([1.0, 2.0]), torch.tensor([False, True]))
    stats.update(torch.tensor([3.0, 5.0]), torch.tensor([True, False]))
    scalars = stats.scalars()
    assert scalars["Overview/completed_episodes"] == 2
    assert scalars["Overview/episode_return_mean"] == 3.0
    assert scalars["Overview/episode_seconds_mean"] == 0.03
    torch.testing.assert_close(stats.returns, torch.tensor([0.0, 5.0]))


class FakeRun:
    id, url, project, entity, dir = (
        "fake-id",
        "https://example.invalid/run",
        "test",
        "test",
        "/tmp",
    )

    def __init__(self):
        self.summary, self.records, self.definitions, self.files = {}, [], [], []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def save(self, *args, **kwargs):
        self.files.append((args, kwargs))

    def log(self, values, step):
        self.records.append((step, values))

    def finish(self, exit_code):
        self.exit_code = exit_code


def make_logger(tmp_path):
    run = FakeRun()
    sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
    cfg = {
        "supermimic": {
            "backend": "mujoco_warp",
            "wandb": {
                "project": "test",
                "entity": "test",
                "mode": "offline",
                "max_metric_keys": 2048,
            },
        }
    }
    return WandbLogger(tmp_path, cfg, {}, sdk=sdk), run


def test_one_log_per_iteration_has_explicit_axis_and_does_not_upload_checkpoints(
    tmp_path,
):
    (tmp_path / "model_500.pt").write_bytes(b"local weights")
    logger, run = make_logger(tmp_path)
    logger.log({"iteration": 0, "wm_loss": 1.0})
    logger.log({"iteration": 10, "wm_loss": 0.5, "StandFH/hit_rate": 0.2})
    with pytest.raises(ValueError):
        logger.log({"iteration": 10})
    logger.finish("completed")
    assert [step for step, _ in run.records] == [0, 10]
    assert run.records[-1][1]["Progress/iteration"] == 10
    assert any(
        kw.get("step_metric") == "Progress/iteration" for _, kw in run.definitions
    )
    assert not run.files
    status = json.loads((tmp_path / "wandb_run.json").read_text())
    assert status["last_queued_iteration"] == 10 and status["state"] == "completed"


def test_existing_full_task_row_is_not_silently_trimmed_by_ppo_profile():
    row = {"iteration": 0}
    row.update({f"task/Metrics/racket_target/detail_{i}": float(i) for i in range(875)})
    result = remote_metrics(row)
    assert len([k for k in result if k.startswith("EpisodeMetrics/")]) == 875


def test_sdk_history_cannot_overwrite_final_checkpoint_summary(tmp_path, monkeypatch):
    sdk = pytest.importorskip("wandb")
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = {
        "supermimic": {
            "backend": "mujoco_warp",
            "wandb": {
                "project": "hope-logging-test",
                "mode": "offline",
                "max_metric_keys": 2048,
            },
        }
    }
    logger = WandbLogger(tmp_path, cfg, {}, sdk=sdk)
    logger.log({"iteration": 0, "Checkpoint/last_saved_iteration": -1})
    logger.checkpoint_saved(1, tmp_path / "model_1.pt")
    logger.finish("completed")
    store = DataStore()
    store.open_for_scan(str(next(tmp_path.glob("wandb/offline-run*/run-*.wandb"))))
    summary = {}
    try:
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") == "summary":
                summary.update(
                    {
                        item.key or "/".join(item.nested_key): json.loads(
                            item.value_json
                        )
                        for item in record.summary.update
                    }
                )
    finally:
        store.close()
    assert summary["Checkpoint/latest_saved_iteration"] == 1
    assert summary["Checkpoint/last_saved_iteration"] == -1
    assert summary["Training/status"] == "completed"
