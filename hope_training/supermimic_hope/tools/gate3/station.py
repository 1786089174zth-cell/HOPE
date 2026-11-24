"""Where the robot stands before each ball of a Gate3 run, relative to where it entered (stdlib only).

    python3 tools/gate3/station.py <run dir>

Pelvis, feet midpoint and each foot (cm from the last sample before MOTION), feet width, pelvis and
support heading, sampled 0.2 s before each engage. The four-stroke HOLD reference has the pelvis on
the entry spot and the feet 3.8 cm ahead and 10.4 cm out on each side.
"""
import csv, math, sys, json, statistics as st
R = sys.argv[1]
def yaw(w,x,y,z): return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
trace = list(csv.DictReader(open(R+"/pp_runner_trace.csv")))
m = next(int(r["wall_time_ns"]) for r in trace if r["mode"]=="3")
# hold instants: 0.2 s before each level 0 -> nonzero transition (the robot waiting for the next ball)
starts=[]; prev="0"
for r in trace:
    if r["mode"]=="3" and prev=="0" and r["level"]!="0": starts.append(int(r["wall_time_ns"]))
    prev = r["level"] if r["mode"]=="3" else "0"
plant = csv.DictReader(open(R+"/pp_mujoco_plant.csv"))
anchor=None; out=[]; i=0; lastrow=None
for r in plant:
    t=int(r["wall_time_ns"])
    if t < m: anchor=r; continue
    while i < len(starts) and t >= starts[i]-200_000_000:
        out.append((i+1, r)); i+=1
g=lambda r,k: float(r[k])
def geom(r):
    L=(g(r,"left_foot_x"),g(r,"left_foot_y")); Rr=(g(r,"right_foot_x"),g(r,"right_foot_y"))
    mid=((L[0]+Rr[0])/2,(L[1]+Rr[1])/2); dx,dy=L[0]-Rr[0],L[1]-Rr[1]
    return dict(base=(g(r,"base_x"),g(r,"base_y")), mid=mid, width=math.hypot(dx,dy), syaw=math.degrees(math.atan2(-dx,dy)),
                byaw=yaw(*(g(r,"base_q"+a) for a in "wxyz")), stag=dx, lyaw=yaw(*(g(r,"left_foot_q"+a) for a in "wxyz")), ryaw=yaw(*(g(r,"right_foot_q"+a) for a in "wxyz")),
                L=L, R=Rr)
a=geom(anchor)
print(R.split("supermimic/")[-1])
print("  anchor (last sample before MOTION): pelvis (%.3f, %.3f) feet mid (%.3f, %.3f) width %.3f yaw %.1f"%(*a["base"],*a["mid"],a["width"],a["byaw"]))
print("  before ball | pelvis dx dy (cm) | feet-mid dx dy (cm) | L foot dx dy | R foot dx dy | width | pelvis yaw | support yaw | L/R foot yaw")
rows=[]
for n,r in out:
    q=geom(r)
    row=(n, 100*(q["base"][0]-a["base"][0]),100*(q["base"][1]-a["base"][1]),100*(q["mid"][0]-a["mid"][0]),100*(q["mid"][1]-a["mid"][1]),
         100*(q["L"][0]-a["L"][0]),100*(q["L"][1]-a["L"][1]),100*(q["R"][0]-a["R"][0]),100*(q["R"][1]-a["R"][1]),q["width"],q["byaw"]-a["byaw"],q["syaw"]-a["syaw"],q["lyaw"],q["ryaw"])
    rows.append(row)
    if n<=3 or n%10==0 or n==len(out):
        print("  %4d        | %+5.1f %+5.1f       | %+5.1f %+5.1f         | %+5.1f %+5.1f  | %+5.1f %+5.1f  | %.3f | %+5.1f      | %+5.1f       | %+5.1f / %+5.1f"%row)
col=lambda j:[r[j] for r in rows]
print("  over %d holds: pelvis dx %.1f±%.1f cm (range %+.1f..%+.1f), dy %.1f±%.1f cm (range %+.1f..%+.1f); pelvis yaw %.1f±%.1f deg (range %+.1f..%+.1f); width %.3f±%.3f m"%(
    len(rows), st.mean(col(1)),st.pstdev(col(1)),min(col(1)),max(col(1)), st.mean(col(2)),st.pstdev(col(2)),min(col(2)),max(col(2)), st.mean(col(10)),st.pstdev(col(10)),min(col(10)),max(col(10)), st.mean(col(9)),st.pstdev(col(9))))
