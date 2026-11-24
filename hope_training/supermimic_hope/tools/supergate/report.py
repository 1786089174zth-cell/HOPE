"""Supergate results by kind of ball, per policy (stdlib only).

    python3 tools/supergate/report.py <supergate output dir> [--dataset <suite dir>]

Joins ``env-*-results.csv`` (per shot: contact, legal return, lead, executed side, reason) and
``env-*-kinematics.csv`` (racket vs target at the planned instant and at contact) with the suite's
ledger (which cell each launch was built for: lateral position, incoming speed, height, horizontal
angle, ascending or descending at contact). Prints legal-return rates along each axis, the reasons
of the shots that had no swing, the notice the policy got, and the stroke errors; writes ``report.json``.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

LATERAL = ("right  y<-0.32", "middle -0.32..-0.15", "left  y>-0.15")
SPEED = ("slow <2.5 m/s", "medium 2.5-4", "fast >4")
HEIGHT = ("low 0.8-1.0 m", "mid 1.0-1.2", "high 1.2-1.45")
ANGLE = ("from right <-5 deg", "straight", "from left >5 deg")


def load(out: Path, dataset: Path):
    run = json.loads((out / "run.json").read_text())
    labels = [g["label"] for g in run["policy_groups"]]
    ledgers = [json.loads((dataset / f"shard-{i:02d}-ledger.json").read_text()) for i in range(1, 7)]
    shots = {label: [] for label in labels}
    for g, label in enumerate(labels):
        for shard in range(6):
            env = 6 * g + shard
            rows = list(csv.DictReader((out / f"env-{env}-results.csv").open()))
            kin = {}
            path = out / f"env-{env}-kinematics.csv"
            if path.is_file():
                for k in csv.DictReader(path.open()):
                    kin[(int(k["shot"]), k["event"])] = k
            for r in rows:
                i = int(r["shot"]) - 1
                if i >= len(ledgers[shard]):
                    continue
                led = cell(ledgers[shard][i])
                c, p = kin.get((i + 1, "contact")), kin.get((i + 1, "planned"))
                f = lambda k, key: float(k[key]) if k and k[key] not in ("", "nan") else float("nan")  # noqa: E731
                shots[label].append(dict(
                    shard=shard + 1, shot=i + 1, section=led.get("section"), lateral=led.get("lateral_bin"), speed=led.get("speed_bin"),
                    height=led.get("height_bin"), angle=led.get("angle_bin"), y=led.get("target_relative_y"),
                    ball_speed=led.get("predicted_contact_speed", led.get("target_speed")), ball_height=led.get("target_height"),
                    ascending=(led.get("predicted_contact") or [0] * 6)[5] > 0,
                    plan=int(r["valid_plans"]) > 0, contact=int(r["contacts"]) > 0, legal=int(r["legal"]) > 0,
                    lead=float(r["lead"]) if r["lead"] not in ("", "nan") else float("nan"), side=float(r["side"]), reason=r["reason"],
                    fall=float(r["min_z"]) < 0.7 or float(r["max_tilt"]) > math.pi / 4, tilt=math.degrees(float(r["max_tilt"])),
                    landing=(float(r["landing_x"]), float(r["landing_y"])), nets=int(r["nets"]),
                    p_err_planned=f(p, "p_error_m"), v_err_planned=f(p, "v_error_mps"), n_err_planned=f(p, "n_error_deg"),
                    p_err_contact=f(c, "p_error_m"), v_err_contact=f(c, "v_error_mps"), n_err_contact=f(c, "n_error_deg"),
                    contact_dt_ms=f(c, "dt_ms"), target_z=f(p, "target_p_z"), target_speed=math.sqrt(sum(f(p, "target_v_" + a) ** 2 for a in "xyz")) if p else float("nan")))
    return labels, shots


def cell(led):
    """The ledger row with its coverage bins: the suite's own, else from the predicted contact state
    (the main suite's edges: lateral -0.32/-0.15 m from HOME, speed 2.5/4 m/s, height 1.0/1.2 m, angle -5/5 deg)."""
    pc = led.get("predicted_contact")
    if pc is None:
        return led
    digit = lambda x, edges: sum(x >= e for e in edges)  # noqa: E731
    led = dict(led)
    led.setdefault("target_relative_y", pc[1] + 0.7625)
    led.setdefault("lateral_bin", digit(pc[1] + 0.7625, (-0.32, -0.15)))
    led.setdefault("speed_bin", digit(math.sqrt(sum(v * v for v in pc[3:6])), (2.5, 4.0)))
    led.setdefault("height_bin", digit(pc[2] + 0.76, (1.0, 1.2)))
    led.setdefault("angle_bin", digit(math.degrees(math.atan2(pc[4], -pc[3])), (-5.0, 5.0)))
    led.setdefault("section", "factorial81")
    return led


def rate(rows, key="legal"):
    return f"{sum(r[key] for r in rows):4d}/{len(rows):<4d} {100 * sum(r[key] for r in rows) / max(1, len(rows)):5.1f}%"


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    dataset = None
    if "--dataset" in argv:
        i = argv.index("--dataset")
        dataset = Path(argv[i + 1])
        del argv[i:i + 2]
    out = Path(argv[0])
    if dataset is None:
        dataset = Path(__file__).resolve().parents[2] / "logs/supergate/tree/a3_deploy/a3_deploy_example/scripts/supergate/data/main"
    labels, shots = load(out, dataset)
    report = {}
    print(f"{out}  ({dataset.name}: {len(shots[labels[0]])} launches per policy)")
    print(f"{'':28s}" + "".join(f"{label:>22s}" for label in labels))
    def line(name, select):
        print(f"{name:28s}" + "".join(f"{rate([r for r in shots[label] if select(r)]):>22s}" for label in labels))
    line("legal returns, all", lambda r: True)
    print(f"{'  valid plan':28s}" + "".join(f"{rate(shots[label], 'plan'):>22s}" for label in labels))
    print(f"{'  racket contact':28s}" + "".join(f"{rate(shots[label], 'contact'):>22s}" for label in labels))
    print(f"{'  fall-indicator trials':28s}" + "".join(f"{sum(r['fall'] for r in shots[label]):>22d}" for label in labels))
    core = lambda r: r["section"] == "factorial81"  # noqa: E731
    line("the coverage cells", core)
    line("origin-boundary", lambda r: not core(r))
    for title, key, names in (("lateral position", "lateral", LATERAL), ("incoming speed", "speed", SPEED), ("height", "height", HEIGHT), ("horizontal angle", "angle", ANGLE)):
        print(f"by {title} (core):")
        for b, name in enumerate(names):
            line("  " + name, lambda r, b=b, key=key: core(r) and r[key] == b)
    print("by ball state at contact:")
    line("  ascending", lambda r: r["ascending"])
    line("  descending", lambda r: not r["ascending"])
    print("by notice (lead at commit):")
    for lo, hi in ((0, 0.30), (0.30, 0.36), (0.36, 0.42), (0.42, 0.50), (0.50, 9)):
        line(f"  {lo:.2f}-{hi:.2f} s", lambda r, lo=lo, hi=hi: r["plan"] and lo <= r["lead"] < hi)
    print("by side, notice and height of the target (legal returns):")
    for side, name in ((1, "FH"), (-1, "BH")):
        for lo, hi in ((0, 0.30), (0.30, 0.36), (0.36, 9)):
            for zl, zh in ((0.0, 1.0), (1.0, 1.2), (1.2, 9)):
                line(f"  {name} {lo:.2f}-{min(hi, 0.42):.2f} s z {max(zl, 0.8):.1f}-{min(zh, 1.45):.2f}",
                     lambda r, side=side, lo=lo, hi=hi, zl=zl, zh=zh: r["side"] == side and lo <= r["lead"] < hi and zl <= r["target_z"] < zh)
    print("by side the Planner commanded:")
    line("  forehand", lambda r: r["side"] > 0)
    line("  backhand", lambda r: r["side"] < 0)
    line("  no command", lambda r: r["side"] == 0)
    print("where the returns were lost:")
    for name, sel in (("  no valid plan", lambda r: not r["plan"]), ("  plan but no contact", lambda r: r["plan"] and not r["contact"]),
                      ("  contact, into the net", lambda r: r["contact"] and not r["legal"] and r["nets"] > 0),
                      ("  contact, off the table", lambda r: r["contact"] and not r["legal"] and r["nets"] == 0)):
        print(f"{name:28s}" + "".join(f"{sum(sel(r) for r in shots[label]):>22d}" for label in labels))
    print("stroke at contact (shots with contact): position / velocity / face error, contact time vs plan")
    for label in labels:
        c = [r for r in shots[label] if r["contact"]]
        print(f"  {label:12s} {100 * mean([r['p_err_contact'] for r in c]):.1f} cm  {mean([r['v_err_contact'] for r in c]):.2f} m/s  "
              f"{mean([r['n_err_contact'] for r in c]):.1f} deg  {mean([r['contact_dt_ms'] for r in c]):+.0f} ms;   at the planned instant, all planned shots: "
              f"{100 * mean([r['p_err_planned'] for r in shots[label]]):.1f} cm  {mean([r['v_err_planned'] for r in shots[label]]):.2f} m/s  "
              f"{mean([r['n_err_planned'] for r in shots[label]]):.1f} deg;  commanded racket speed {mean([r['target_speed'] for r in shots[label]]):.2f} m/s, notice {mean([r['lead'] for r in shots[label] if r['plan']]):.2f} s")
    for label in labels:
        reasons = {}
        for r in shots[label]:
            if not r["plan"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(f"  {label}: reasons without a plan: {dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}")
        report[label] = shots[label]
    (out / "report.json").write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
