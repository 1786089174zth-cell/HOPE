"""W&B scalar logging for SuperMimic; no PPO or simulator imports here."""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path


@lru_cache(maxsize=1)
def task_metric_mapper():
    # The existing mapper is deliberately standalone. Importing its package
    # would initialize Isaac tasks in CPU-only logging tests.
    path = (
        Path(__file__).resolve().parents[2] / "whole_body_tracking/source/"
        "whole_body_tracking/whole_body_tracking/utils/wandb_logging.py"
    )
    spec = importlib.util.spec_from_file_location("_hope_dashboard_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stroke_v22_metric


SCALAR_NAMES = {
    "iteration": "Progress/iteration",
    "sim_vector_steps": "Progress/real_vector_steps",
    "sim_transitions": "Progress/real_transitions",
    "real_mean_reward": "Overview/real_step_reward",
    "real_terminations": "Safety/real_terminations",
    "wm_loss": "WorldModel/train_loss",
    "wm_validation_loss": "WorldModel/validation_loss",
    "wm_validation_state_mse": "WorldModel/validation_state_normalized_mse",
    "wm_validation_valid_fraction": "WorldModel/validation_target_valid_fraction",
    "wm_validation_reward_mse": "WorldModel/validation_residual_reward_normalized_mse",
    "wm_validation_continuation_bce": "WorldModel/validation_failure_model_loss",
    "wm_validation_actor_obs_mse": "WorldModel/validation_actor_obs_normalized_mse",
    "wm_validation_privileged_obs_mse": "WorldModel/validation_privileged_obs_normalized_mse",
    "wm_grad_norm": "WorldModel/gradient_norm_before_clip",
    "wm_valid_fraction": "Replay/dynamics_target_valid_fraction",
    "actor_loss": "Policy/loss",
    "actor_grad_norm": "Policy/gradient_norm_before_clip",
    "value_loss": "Value/loss",
    "value_grad_norm": "Value/gradient_norm_before_clip",
    "sensor_noise_weighted": "Policy/sensor_noise_weighted_loss",
    "imagined_return": "Imagination/predicted_return",
    "imagined_disagreement": "Imagination/ensemble_disagreement",
    "imagined_survival": "Imagination/predicted_final_survival",
    "imagined_projection_fraction": "Imagination/projected_action_fraction",
    "collection_seconds": "Performance/collection_seconds",
    "learning_seconds": "Performance/learning_seconds",
    "iteration_seconds": "Performance/iteration_seconds",
}


def remote_metrics(row):
    """Preserve episode semantics; never relabel reset-cohort means as live data."""
    mapped = {}
    missing = 0
    mapper = task_metric_mapper()
    for name, value in row.items():
        if value is None or (
            isinstance(value, (float, int)) and not math.isfinite(value)
        ):
            missing += 1
            continue  # Missing denominators are gaps, never fabricated zero rates.
        if not isinstance(value, (int, float)):
            continue
        scale = 1.0
        if name in SCALAR_NAMES:
            target = SCALAR_NAMES[name]
        elif name.startswith("task/Episode_Reward/"):
            target = "Reward/" + name.removeprefix("task/Episode_Reward/")
        elif name.startswith("task/Metrics/"):
            target = "EpisodeMetrics/" + name.removeprefix("task/Metrics/")
        elif name.startswith("task/Episode_Termination/"):
            target = "Safety/episode_" + name.removeprefix("task/Episode_Termination/")
        elif name.startswith("task/"):
            target = "Task/" + name.removeprefix("task/")
        elif name.startswith("Instrumentation/qdes_safety/"):
            item = mapper(name)
            if item is None:
                target = "ActionSafety/" + name.removeprefix(
                    "Instrumentation/qdes_safety/"
                )
            else:
                target, scale = item
        elif name.startswith(("Live/", "Instrumentation/")):
            item = mapper(name)
            if item is None:
                continue  # Full detail remains in local JSONL/TensorBoard.
            target, scale = item
        else:
            target = name  # Explicit Progress/Replay/StandFH/etc. groups.
        mapped[target] = value * scale
    mapped["Logging/missing_scalar_count"] = missing
    return mapped


class WandbLogger:
    """One native W&B run per training directory, with one log call per update."""

    def __init__(self, out, config, manifest, *, sdk=None):
        self.out = Path(out)
        settings = config["supermimic"]["wandb"]
        self.max_keys = int(settings["max_metric_keys"])
        self.keys = set()
        self.last_iteration = -1
        if sdk is None:
            import wandb as sdk
        self.run = sdk.init(
            project=settings["project"],
            entity=settings.get("entity"),
            mode=settings["mode"],
            name=self.out.name,
            dir=str(self.out),
            config={"resolved": config, "training_manifest": manifest},
            tags=["supermimic", str(config["supermimic"]["backend"]), "four-stroke"],
            job_type="train",
            save_code=False,
            sync_tensorboard=False,
            settings=sdk.Settings(console="off", init_timeout=45),
        )
        self.run.define_metric("Progress/iteration")
        self.run.define_metric("*", step_metric="Progress/iteration", step_sync=True)
        # Kept available for queries without flooding the default dashboard.
        self.run.define_metric("EpisodeMetrics/*", hidden=True)
        self.run.summary["Logging/semantics"] = (
            "StandFH/StandBH/StepFH/StepBH are event-ledger rates with denominators; "
            "EpisodeMetrics are reset-cohort means; Imagination is predicted, not evaluated."
        )
        self._status("running")
        for name in (
            "params/task_resolved.yaml",
            "physics_receipt.json",
            "training_manifest.json",
        ):
            path = self.out / name
            if path.exists():
                self.run.save(str(path), base_path=str(self.out), policy="now")
        print(f"[supermimic] W&B: {self.run.url or self.run.dir}", flush=True)

    def _status(self, state):
        payload = dict(
            state=state,
            run_id=self.run.id,
            url=self.run.url,
            project=self.run.project,
            entity=self.run.entity,
            last_queued_iteration=self.last_iteration,
            distinct_metrics=len(self.keys),
        )
        path = self.out / "wandb_run.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(path)

    def log(self, row):
        iteration = int(row["iteration"])
        if iteration <= self.last_iteration:
            raise ValueError(
                "W&B iteration must increase; refusing duplicate/backward history"
            )
        payload = remote_metrics(row)
        keys = self.keys | set(payload)
        if self.max_keys and len(keys) > self.max_keys:
            raise RuntimeError(
                f"W&B metric budget exceeded: {len(keys)} > {self.max_keys}"
            )
        self.run.log(payload, step=iteration)
        self.keys = keys
        self.last_iteration = iteration
        if iteration % 10 == 0:
            self._status("running")

    def finish(self, status):
        self.run.summary["Training/status"] = status
        self.run.summary["Logging/last_iteration"] = self.last_iteration
        self.run.summary["Logging/distinct_metrics"] = len(self.keys)
        self.run.finish(exit_code=0 if status == "completed" else 1)
        self._status(status)

    def checkpoint_saved(self, iteration, path):
        # History is processed asynchronously. Its automatic summary must not
        # overwrite metadata written after the final successful checkpoint.
        self.run.summary["Checkpoint/latest_saved_iteration"] = int(iteration)
        self.run.summary["Checkpoint/local_path"] = str(Path(path).resolve())
