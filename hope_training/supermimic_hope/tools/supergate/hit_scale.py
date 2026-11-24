"""Give Supergate the field Planner's hit-speed scale (stdlib only; called by assemble.sh).

    python3 tools/supergate/hit_scale.py <tree>/a3_deploy/a3_deploy_example/scripts/supergate/native/native_batch.cpp

Main's Planner multiplies the solved racket velocity by an operator-set scale before it publishes the command
(``planner_node.cpp``: ``command.velocity *= hit_speed_scale``; on the robot it was 0.55-0.95 on 2026-11-21). The
harness has no such scale. This edit keeps the default (1) and reads ``SUPERGATE_HIT_SPEED_SCALE``: a number for
every shot, or the path of a text file with one line per shard (six lines, comma-separated scales, one per shot),
so that a suite made from a real session (tools/real/make_suite.py) is replayed with the scale each ball had.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
old = "auto cmd=planner->plan(strike,hp::Vec3(2.055,-.7625,0),.5);r.reason=cmd.reason;if(!cmd.valid)return;"
if "sg_hit_scale(" not in text:
    if text.count(old) != 1:
        raise SystemExit("native_batch.cpp: planning call not found")
    helper = ('#include <cstdlib>\n#include <fstream>\n#include <sstream>\n#include <string>\n#include <vector>\n'
              'static double sg_hit_scale(int shard,int shot){static const std::vector<std::vector<double>> table=[]{std::vector<std::vector<double>> t;'
              'const char*v=std::getenv("SUPERGATE_HIT_SPEED_SCALE");if(!v||!*v)return t;char*end=nullptr;double x=std::strtod(v,&end);'
              'if(end&&!*end){t.push_back({x});return t;}std::ifstream f(v);std::string line;while(std::getline(f,line)){std::vector<double> row;'
              'std::istringstream s(line);std::string item;while(std::getline(s,item,\',\'))row.push_back(std::stod(item));t.push_back(row);}'
              'if(t.size()!=6)throw std::runtime_error("SUPERGATE_HIT_SPEED_SCALE: a number or a file of six lines");return t;}();'
              'if(table.empty())return 1.;if(table.size()==1)return table[0][0];return table[shard%6].at(shot);}\n')
    first = text.index("#include")
    text = text[:first] + helper + text[first:]
    text = text.replace(old, old + "cmd.velocity*=sg_hit_scale(id,current);")
    path.write_text(text)
