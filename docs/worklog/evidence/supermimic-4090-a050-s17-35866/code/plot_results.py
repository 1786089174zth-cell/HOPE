"""Render training curves and measured Gate3 landings from the archived evidence."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

root = Path(__file__).resolve().parents[1]
with (root / "training/metrics.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))
iteration = np.array([int(row["iteration"]) for row in rows])


def values(key):
    return np.array([float(row[key]) if row[key] else np.nan for row in rows])


def curve(ax, key, label, color, scale=1):
    data = values(key) * scale
    ok = np.isfinite(data)
    x, y = iteration[ok], data[ok]
    ax.plot(x, y, alpha=.2, color=color, lw=.8)
    window = 20
    ax.plot(x[window - 1:], np.convolve(y, np.ones(window)/window, mode="valid"),
            color=color, lw=1.8, label=label)
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax.set_xlabel("Training iteration")


fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
curve(axes[0,0], "task/hit_rate", "Hit rate (%)", "#007b83", 100)
curve(axes[0,1], "collect/reward_per_s", "Collection reward / s", "#4e5fbf")
curve(axes[1,0], "wm/loss", "Dynamics loss", "#007b83")
curve(axes[1,0], "policy/value_loss", "Critic loss", "#bf6b28")
axes[1,0].set_yscale("log")
curve(axes[1,1], "policy/upper_grad_norm", "Upper body", "#4e5fbf")
curve(axes[1,1], "policy/lower_grad_norm", "Lower body", "#bf6b28")
axes[1,1].set_ylabel("Gradient norm before clipping")
fig.suptitle("SuperMimic training · α = 0.5 · seed 17\nFaint lines are logged samples. Solid lines average 20 samples.", fontsize=13)
fig.savefig(root / "figures/training-curves.png", dpi=180)
plt.close(fig)

result = json.loads((root / "gate3/fixed-home-100/pp_physical_ball_report.json").read_text())
points = [row["landing_event"]["position_world"][:2]
          for row in result["rows"] if row["landing_pass"]]
x, y = np.asarray(points).T
fig, ax = plt.subplots(figsize=(8.2, 4.1), constrained_layout=True)
ax.add_patch(Rectangle((0, -1.525), 2.74, 1.525, color="#eaf0f7"))
ax.axvline(1.37, color="#687786", lw=1.2, label="Net")
ax.scatter(x, y, color="#007b83", alpha=.54, s=24, label=f"Legal landings ({len(points)})")
ax.scatter([2.055], [-.7625], marker="+", s=130, color="#bf6b28", label="Planner target")
ax.set(xlim=(-.05,2.79), ylim=(-1.575,.05), xlabel="Along the table (m)",
       ylabel="Across the table (m)", title="Fixed-HOME 100-ball test · measured landing positions")
ax.set_aspect("equal")
ax.legend(loc="lower left", frameon=False)
fig.savefig(root / "figures/fixed-home-100-landings.png", dpi=180)
