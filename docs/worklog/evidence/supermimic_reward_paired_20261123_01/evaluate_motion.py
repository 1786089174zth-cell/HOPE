"""Run evaluate.py with exact per-motion strike, racket-position and fall counters.

Falls are assigned to the action that led into termination. Waiting at startup and
between strokes has its own exposure. The scored-strike convention is evaluate.py's.
"""
import math
import sys
from pathlib import Path
from unittest.mock import patch

# Accept either a copy beside experiment evidence (launched from the package)
# or an installation under supermimic_hope/tools.
package = Path(__file__).resolve().parents[1]
if not (package / "evaluate.py").is_file():
    package = Path.cwd()
sys.path.insert(0, str(package))
import evaluate
import torch
from supermimic_hope.core.task import FourStrokeTask


def main(argv=None):
    advance = FourStrokeTask.advance
    observe = FourStrokeTask.observe
    metrics = FourStrokeTask.metrics
    totals = None
    groups = None

    def record_action(task):
        nonlocal groups
        tape = task.tape()
        groups = torch.stack((
            (tape["in_hold"] == 0) & (tape["family"] < 2),
            (tape["in_hold"] == 0) & (tape["family"] >= 2),
            tape["in_hold"] != 0,
        ))
        return advance(task)

    def record_outcome(task, state, tape, reward, info, terminated, truncated):
        nonlocal totals
        counts = torch.stack((groups.sum(dim=1), (groups & terminated).sum(dim=1)))
        if totals is None:
            totals = counts
        else:
            totals.add_(counts)
        return observe(task, state, tape, reward, info, terminated, truncated)

    def motion_metrics(task):
        nonlocal totals
        out = {}
        for i, name in enumerate(("stand", "step", "hold")):
            out[f"motion/{name}/ticks"] = totals[0, i].item()
            out[f"motion/{name}/falls"] = totals[1, i].item()
            if name != "hold":
                families = slice(2 * i, 2 * i + 2)
                for src, label in (("comp", "hits"), ("strikes", "strikes"), ("pos_err", "position_error_sum_m")):
                    out[f"motion/{name}/{label}"] = task._m[src][families].sum().item()
        total_ticks = task.N * task._m["steps"]
        if totals[0].sum().item() != total_ticks or totals[1].sum().item() != task._m["falls"].item():
            raise ValueError("Motion fall counts or exposure disagree with the standard evaluator")
        totals = None
        result = metrics(task)
        strikes = out["motion/stand/strikes"] + out["motion/step/strikes"]
        hits = out["motion/stand/hits"] + out["motion/step/hits"]
        error = out["motion/stand/position_error_sum_m"] + out["motion/step/position_error_sum_m"]
        if strikes != result["task/strikes"].item() or not math.isclose(
            hits / max(strikes, 1), result["task/hit_rate"].item(), abs_tol=1e-6
        ) or not math.isclose(error / max(strikes, 1), result["task/strike_pos_err_m"].item(), abs_tol=1e-6):
            raise ValueError("Motion strike aggregates disagree with the standard evaluator")
        return result | out

    with patch.object(FourStrokeTask, "advance", record_action), patch.object(
        FourStrokeTask, "observe", record_outcome
    ), patch.object(FourStrokeTask, "metrics", motion_metrics):
        evaluate.main(argv)


if __name__ == "__main__":
    main()
