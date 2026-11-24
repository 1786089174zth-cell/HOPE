import json, sys, collections
import numpy as np
for O in sys.argv[1:]:
    S = json.load(open(O + "/shots.json"))
    print("==", O)
    g = collections.defaultdict(collections.Counter)
    for s in S: g[(s.get("hit_scale"), s.get("flight_s"), round(s["home"][0], 2))][s["outcome"]] += 1
    for k, v in g.items(): print("  scale, flight, home_x", k, dict(v))
    print(" n  swing lat   z_tab  tts | cmd racket v (x y z) | actual racket v     | racket err mm (x y z) | pred err: dt ms, y mm, z mm | closest mm @ms | v_in             | v_out            | net clr  land x,y | outcome scale")
    for n, s in enumerate(S):
        lat = s["target"][1] - s["home"][1]
        f3 = lambda v, sc=1, w=5, p=2: " ".join(f"{x * sc:{w}.{p}f}" for x in v) if v else " " * (3 * w + 2)
        pe = s.get("pred_err"); cl = s.get("closest")
        print(f"{n:3d} {s['swing']:+d} {lat:+.2f} {s['target'][2] - 0.76:5.2f} {s['tts']:.2f} | {f3(s['target_v'])} | {f3(s['racket_v'])} | {f3(s['racket_err'], 1000, 5, 0)} | "
              f"{(f'{pe[0]*1000:5.0f} {pe[1]*1000:5.0f} {pe[2]*1000:5.0f}' if pe else ' ' * 17)} | {(f'{cl[chr(100)]*1000:4.0f} {cl[chr(100)+chr(116)]*1000:4.0f}' if cl else ' ' * 9)} | {f3(s.get('v_in'))} | {f3(s.get('v_out'))} | "
              f"{(f'{s[chr(110)+chr(101)+chr(116)+chr(95)+chr(99)+chr(108)+chr(101)+chr(97)+chr(114)]:+.2f}' if s.get('net_clear') is not None else '     ')} {f3(s['land'][:2]) if s.get('land') else '':11s} | {s['outcome']} {s.get('hit_scale')}")
