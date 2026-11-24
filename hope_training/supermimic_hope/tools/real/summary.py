import json, sys, collections
import numpy as np
S = []
for O in sys.argv[1:]:
    S += json.load(open(O + "/shots.json"))
print(len(S), "shots", collections.Counter(s["outcome"] for s in S).most_common())
def table(name, key, order=None):
    g = collections.defaultdict(collections.Counter)
    for s in S:
        k = key(s)
        if k is not None: g[k][s["outcome"]] += 1
    print("--", name)
    for k in (order or sorted(g)):
        if k not in g: continue
        c = g[k]; n = sum(c.values()); hit = n - c["miss"] - c["no_track"] - c["lost_before_plane"]
        print(f"  {str(k):22s} n {n:3d} | contact {hit:3d} ({100*hit/n:3.0f}%) | landed {c['landed']:3d} ({100*c['landed']/n:3.0f}%) | net {c['net']:3d} long {c['long']:3d} wide {c['wide']:2d} miss {c['miss']:2d}")
lat = lambda s: s["target"][1] - s["home"][1]
zt = lambda s: s["target"][2] - 0.76
band = lambda x, e: next((f"{a:+.2f}..{b:+.2f}" for a, b in zip(e, e[1:]) if a <= x < b), None)
table("hit speed scale / flight time", lambda s: (s.get("hit_scale"), s.get("flight_s")))
table("side", lambda s: "forehand" if s["swing"] > 0 else "backhand")
table("lateral from home (m; + = robot's left)", lambda s: band(lat(s), [-1, -0.34, -0.15, 0.0, 0.15, 0.3, 1]))
table("strike height above table (m)", lambda s: band(zt(s), [0, 0.15, 0.25, 0.35, 0.45, 0.55, 1]))
table("incoming ball vx at plane (planner)", lambda s: band(-s["ball_v"][0], [0, 1.5, 1.8, 2.1, 2.6, 9]))
table("incoming ball vz at plane (planner)", lambda s: band(s["ball_v"][2], [-9, -1.5, -0.5, 0.5, 1.5, 9]))
table("notice (tts at engage, s)", lambda s: band(s["tts"], [0, 0.2, 0.3, 0.36, 1]))
C = [s for s in S if s["outcome"] not in ("miss", "no_track", "lost_before_plane")]
e = np.array([s["racket_err"] for s in S if s["tts"] > 0.3]) * 1000
print("racket FK - target at the commanded strike time, mm (tts > 0.3): mean", e.mean(0).round(0), "std", e.std(0).round(0), "| |err| median", np.median(np.linalg.norm(e, axis=1)).round(0), "90%", np.percentile(np.linalg.norm(e, axis=1), 90).round(0))
dv = np.array([np.array(s["racket_v"]) - s["target_v"] for s in S if s["tts"] > 0.3])
print("racket velocity - commanded at the strike time, m/s: mean", dv.mean(0).round(2), "std", dv.std(0).round(2))
pe = np.array([s["pred_err"] for s in S if "pred_err" in s])
print("planner prediction vs ball at the plane (n", len(pe), "): arrival ms mean", (1000*pe[:,0]).mean().round(0), "std", (1000*pe[:,0]).std().round(0), "| y mm", (1000*pe[:,1]).mean().round(0), (1000*pe[:,1]).std().round(0), "| z mm", (1000*pe[:,2]).mean().round(0), (1000*pe[:,2]).std().round(0))
vo = np.array([s["v_out"] for s in C if "v_out" in s]); 
for name in ("landed", "short", "net", "long"):
    v = np.array([s["v_out"] for s in C if s["outcome"] == name and "v_out" in s])
    if len(v): print(f"ball off the racket, {name:6s} n {len(v):3d}: vx {v[:,0].mean():.2f} +- {v[:,0].std():.2f}, vz {v[:,2].mean():+.2f} +- {v[:,2].std():.2f}, speed {np.linalg.norm(v,axis=1).mean():.2f}")
nc = np.array([s["net_clear"] for s in C if s.get("net_clear") is not None]); print("net clearance m (reaching the net): median", np.median(nc).round(2), " share within +-5 cm", (abs(nc) < 0.05).mean().round(2))
lx = np.array([s["land"][0] for s in C if "land" in s]); print("landing x (net 1.37, far edge 2.74; planner aims 2.055): median", np.median(lx).round(2), "quartiles", np.percentile(lx, [10, 25, 75, 90]).round(2))
