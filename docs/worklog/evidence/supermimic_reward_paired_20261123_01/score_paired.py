"""Score each atomic training checkpoint once with the preregistered evaluation protocol."""
import json
from pathlib import Path
import subprocess
import sys
import time

from run_paired import ROOT, PKG

def main():
    plan = json.loads((ROOT / "protocol.json").read_text())
    ev = plan["evaluation"]
    for run in plan["runs"]:
        folder = ROOT / run["name"]
        for i in ev["iterations"]:
            model = folder / f"model_{i}.pt"
            out = folder / f"evaluation_{i}.json"
            if out.exists():
                completed = json.loads(out.read_text())
                if completed["iteration"] == i and completed["checkpoint"] == str(model):
                    continue
                raise ValueError(f"Unexpected evaluation at {out}")
            while not model.exists():
                time.sleep(10)
            cmd = [sys.executable, "-u", str(ROOT / "evaluate_motion.py"), str(model),
                   f"num_envs={ev['envs']}", f"seed={ev['seed']}", f"steps={ev['steps']}",
                   f"contact={ev['contact']}", f"handoff={ev['handoff']}", "device=cuda:0",
                   f"out={out.with_suffix('.tmp')}"]
            print(f"Evaluate {run['name']} iteration {i}", flush=True)
            with (folder / f"evaluation_{i}.log").open("w") as log:
                subprocess.run(cmd, cwd=PKG, stdout=log, stderr=subprocess.STDOUT, check=True)
            out.with_suffix(".tmp").replace(out)

if __name__ == "__main__":
    main()
