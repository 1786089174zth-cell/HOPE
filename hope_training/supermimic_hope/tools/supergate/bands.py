"""Make Supergate's lateral admission bands overridable (stdlib only; called by assemble.sh).

    python3 tools/supergate/bands.py <tree>/a3_deploy/a3_deploy_example/scripts/supergate/native/native_batch.cpp

The harness selects the swing side with hard-coded support bands, forehand [-0.68, -0.24] and backhand
[-0.24, 0.28] m from HOME, and commits nothing for a ball outside them (``swing_side_target_unsupported``). Main's
Planner takes the bands from the policy's own contract (ours: forehand [-0.9002, -0.275], backhand [-0.275, 0.4598]).
This edit keeps the defaults and lets ``SUPERGATE_FH_BAND=lo,hi`` / ``SUPERGATE_BH_BAND=lo,hi`` replace them, so a
suite can be run with the bands the deployed Planner would use.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
old = "{-.68,-.24},{-.24,.28}"
if "sg_band(" not in text:
    if text.count(old) != 1:
        raise SystemExit("native_batch.cpp: admission bands not found")
    helper = ('#include <cstdio>\n#include <cstdlib>\n#include <vector>\n'
              'static std::vector<double> sg_band(const char*name,double lo,double hi){if(const char*v=std::getenv(name)){double a,b;'
              'if(std::sscanf(v,"%lf,%lf",&a,&b)==2&&a<b)return{a,b};}return{lo,hi};}\n')
    first = text.index("#include")
    text = text[:first] + helper + text[first:]
    text = text.replace(old, 'sg_band("SUPERGATE_FH_BAND",-.68,-.24),sg_band("SUPERGATE_BH_BAND",-.24,.28)')
    path.write_text(text)
