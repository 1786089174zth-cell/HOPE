"""Build a deterministic command replay from a completed live planner-only run."""

import argparse
import hashlib
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="scored directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--shot-limit", type=int, help="Use this completed prefix from a longer source run")
    args = parser.parse_args()
    summary = json.loads((args.source / "summary.json").read_text())
    config = json.loads((args.source / "configuration.json").read_text())
    rows = json.loads((args.source / "per-ball.json").read_text())
    if not config.get("planner_only"):
        raise ValueError("Source must be a planner-only run")
    limit = args.shot_limit or len(rows)
    if limit > len(rows):
        raise ValueError("Shot limit exceeds source scene")
    rows = rows[:limit]
    if any(not r["launched"] or not r["trajectory_complete"] for r in rows):
        raise ValueError("Selected prefix has an incomplete shot")
    if args.shot_limit is None and (summary["error"] or summary["rally_failure"]):
        raise ValueError("Full source run must have no errors")
    scene = yaml.safe_load(args.scene.read_text())
    if scene["balls"] != limit:
        raise ValueError("Target scene count differs from selected prefix")
    expected = [*scene["clean_serves"], *scene["rapid_cycle"]]
    for row, serve in zip(rows, expected):
        actual = [*row["position_table"], *row["velocity_w"]]
        if any(abs(a-b)>1e-6 for a,b in zip(actual,serve)):
            raise ValueError(f"Scene mismatch at shot {row['shot_id']}")
    commands = []
    for row in rows:
        applied = [c for c in row["commands"] if c["applied"]]
        if len(applied) > 1:
            raise ValueError(f"Multiple applied commands on shot {row['shot_id']}")
        if applied:
            if row["planned_deadline_s"] is None:
                raise ValueError(f"No simulation deadline on shot {row['shot_id']}")
            command = applied[0]
            commands.append({"shot_id":row["shot_id"], "sim_t":command["sim_t"],
                             "deadline_sim_s":row["planned_deadline_s"],
                             "values":command["values"]})
    plan = {"schema":"hope_model19000_planner_command_replay_v1",
            "source_run":args.source.parent.name,
            "scene_sha256":hashlib.sha256(args.scene.read_bytes()).hexdigest(),
            "source_scene_sha256":config["scene_sha256"],
            "source_error_after_selected_prefix":summary["error"],
            "onnx_sha256":summary["onnx_sha256"],
            "scheduled_ball_count":limit,
            "collected_commands":len(commands),
            "missing_command_shot_ids":[r["shot_id"] for r in rows if not any(c["applied"] for c in r["commands"])],
            "commands":commands}
    args.output.write_text(json.dumps(plan,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"commands":len(commands),
                      "sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
