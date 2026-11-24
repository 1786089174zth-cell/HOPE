"""Plot every seed and the equally weighted alpha means from learning_curves.csv."""
import csv
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
if len(sys.argv) == 2:
    sys.path.insert(0, sys.argv[1])
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def main():
    with (ROOT / "learning_curves.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True, sharey="row", layout="constrained")
    colors = {"1.0": "#196d91", "0.5": "#b6511a"}
    fields = (("hit_pct", "Hit rate (%)"), ("error_cm", "Mean racket position error (cm)"),
              ("falls_per_1000", "Falls per 1,000 active robot ticks"))
    iters = sorted({int(row["iteration"]) for row in rows})
    for j, mode in enumerate(("stand", "step")):
        axes[0, j].set_title("Standing strokes" if mode == "stand" else "Stepping strokes", loc="left")
        for i, (field, label) in enumerate(fields):
            ax = axes[i,j]
            for alpha, color in colors.items():
                data = [r for r in rows if r["mode"] == mode and r["alpha"] == alpha]
                for seed in sorted({r["seed"] for r in data}):
                    series = sorted((r for r in data if r["seed"] == seed), key=lambda r: int(r["iteration"]))
                    ax.plot(iters, [float(r[field]) for r in series], color=color, alpha=.32, linewidth=.9)
                ax.plot(iters, [statistics.mean(float(r[field]) for r in data if int(r["iteration"]) == k)
                                for k in iters], color=color, marker="o", markersize=3.5, linewidth=2)
            if i == 0:
                ax.axhline(20, color="#555555", linestyle="--", linewidth=.8)
            ax.grid(axis="y", alpha=.22)
            ax.spines[["right","top"]].set_visible(False)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_ylim(bottom=0)
            if j == 0:
                ax.set_ylabel(label)
            if i == 2:
                ax.set_xticks(iters, [f"{k/1000:g}k" for k in iters])
                ax.set_xlabel("Training iteration")
    fig.legend([Line2D([], [], color=c, linewidth=2, marker="o", markersize=4) for c in colors.values()],
               ["α = 1", "α = 0.5"], loc="outside upper right", ncols=2, frameon=False)
    fig.suptitle("SuperMimic  |  Three paired training seeds", x=.03, ha="left")
    fig.savefig(ROOT / "learning_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
