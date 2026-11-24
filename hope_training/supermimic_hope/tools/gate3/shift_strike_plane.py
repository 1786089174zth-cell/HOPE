"""Set the strike plane of a COPY of a Gate3 bundle by rewriting its strike box.

    /opt/drone_venv/bin/python tools/gate3/shift_strike_plane.py <bundle copy> 0.5600 0.6000     (plane 0.58)

The Runner strikes at the centre of ``hitter_pure_pos_range_per_clip`` (as exported by the Isaac
task: 0.35-0.58 m ahead of the station, centre 0.465 m, 3.5 cm short of the table edge). The two
numbers are the box's near and far x bound (the far one is optional); ``deploy.yaml``'s policy hash
is updated to match. The box is the Isaac task's exported metadata, so this is how the 0.58 m plane
was tested (docs/returns_20261119.md), not yet how a policy is officially shipped with it.
"""
import sys, hashlib, re
from pathlib import Path
import onnx
bundle, xmin = Path(sys.argv[1]), sys.argv[2]
xmax = sys.argv[3] if len(sys.argv) > 3 else None  # optional far bound; the plane is (xmin + xmax) / 2
p = bundle / "exported/policy.onnx"; old = hashlib.sha256(p.read_bytes()).hexdigest()
m = onnx.load(str(p))
for e in m.metadata_props:
    if e.key in ("hitter_pure_pos_range_per_clip", "hitter_pure_fallback_pos_range_per_clip", "hitter_pure_pos_support_per_clip"):
        e.value = ";".join(",".join([xmin, xmax or clip.split(",")[1]] + clip.split(",")[2:]) for clip in e.value.split(";")); print(e.key, "=", e.value)
onnx.save(m, str(p)); new = hashlib.sha256(p.read_bytes()).hexdigest()
y = bundle / "params/deploy.yaml"; t = y.read_text(); assert old in t; y.write_text(t.replace(old, new)); print("policy sha", old[:12], "->", new[:12])
