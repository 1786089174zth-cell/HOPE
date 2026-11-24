"""Foot tilt and trunk pitch over a Gate3 run, between balls and during strokes (stdlib only).

    python3 tools/gate3/feet.py <run dir>

Pitch is negative with the toes up. The teacher clips keep both soles flat throughout and pitch
the pelvis back by up to 4-7 degrees in the step strokes.
"""
import csv, math, sys, statistics as st, collections
R=sys.argv[1]
def rpy(w,x,y,z):
    return (math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))), math.degrees(math.asin(max(-1,min(1,2*(w*y-z*x))))))
trace=list(csv.DictReader(open(R+"/pp_runner_trace.csv")))
m=next(int(r["wall_time_ns"]) for r in trace if r["mode"]=="3"); end=max(int(r["wall_time_ns"]) for r in trace if r["mode"]=="3")
lv=[(int(r["wall_time_ns"]), r["level"]!="0", int(float(r["shot_seq"] or 0)), r["swing"]) for r in trace if r["mode"]=="3"]
import bisect
lt=[a[0] for a in lv]
rows=[]
for r in csv.DictReader(open(R+"/pp_mujoco_plant.csv")):
    t=int(r["wall_time_ns"])
    if t<m or t>end: continue
    g=lambda k: float(r[k])
    lr,lp=rpy(*(g("left_foot_q"+a) for a in "wxyz")); rr,rp=rpy(*(g("right_foot_q"+a) for a in "wxyz"))
    br,bp=rpy(*(g("base_q"+a) for a in "wxyz")); tr,tp=rpy(*(g("torso_q"+a) for a in "wxyz"))
    i=max(0,bisect.bisect_right(lt,t)-1)
    rows.append(((t-m)*1e-9, lr,lp,rr,rp,br,bp,tr,tp, g("left_foot_z"),g("right_foot_z"), g("left_foot_normal_force"),g("right_foot_normal_force"), g("base_x"), g("base_z"), lv[i][1], lv[i][2], g("left_foot_x"), g("right_foot_x"), g("base_vx")))
print(R.split("supermimic/")[-1], "MOTION %.0f s"%rows[-1][0])
col=lambda j,sel=None:[r[j] for r in rows if sel is None or sel(r)]
hold=lambda r: not r[15] and r[0]>6
swing=lambda r: r[15]
def desc(name,j,sel):
    x=col(j,sel); xs=sorted(x); n=len(xs)
    print("  %-34s mean %+6.2f  p1 %+6.2f  p50 %+6.2f  p99 %+6.2f  min %+6.2f max %+6.2f"%(name,st.mean(x),xs[n//100],xs[n//2],xs[-n//100-1],xs[0],xs[-1]))
print(" between balls (level 0):")
for name,j in (("left foot pitch (deg, + = toe down)",2),("right foot pitch",4),("left foot roll",1),("right foot roll",3),("pelvis pitch (+ = forward)",6),("torso pitch",8)):
    desc(name,j,hold)
print(" during strokes (level != 0):")
for name,j in (("left foot pitch",2),("right foot pitch",4),("left foot roll",1),("right foot roll",3),("pelvis pitch",6),("torso pitch",8)):
    desc(name,j,swing)
# foot on the ground but tilted: foot z near stance height with |pitch| or |roll| > 5 deg
fl=[r for r in rows if r[0]>6]
tilt=lambda r,side: max(abs(r[1+2*side]),abs(r[2+2*side]))
for side,name in ((0,"left"),(1,"right")):
    loaded=[r for r in fl if r[11+side]>50]
    print("  %s foot: loaded %.0f%% of the time; while loaded, tilt > 3 deg %.1f%%, > 5 deg %.1f%%, > 8 deg %.1f%%; pitch while loaded mean %+.2f (p1 %+.2f, p99 %+.2f)"%(
        name,100*len(loaded)/len(fl),100*sum(tilt(r,side)>3 for r in loaded)/len(loaded),100*sum(tilt(r,side)>5 for r in loaded)/len(loaded),100*sum(tilt(r,side)>8 for r in loaded)/len(loaded),
        st.mean(r[2+2*side] for r in loaded), sorted(r[2+2*side] for r in loaded)[len(loaded)//100], sorted(r[2+2*side] for r in loaded)[-len(loaded)//100-1]))
# backward lean events: pelvis pitch < -6 deg or torso pitch < -8
ev=[]; cur=None
for r in fl:
    back = r[6] < -5.0
    if back:
        if cur is None or r[0]-cur["end"]>0.5: cur=dict(start=r[0],end=r[0],min_pelvis=r[6],min_torso=r[8],shot=r[16],dx=r[13]); ev.append(cur)
        cur["end"]=r[0]; cur["min_pelvis"]=min(cur["min_pelvis"],r[6]); cur["min_torso"]=min(cur["min_torso"],r[8])
print("  backward pelvis pitch beyond 5 deg: %d events"%len(ev), [(round(e["start"],1), e["shot"], round(e["min_pelvis"],1), round(e["min_torso"],1)) for e in ev[:12]])
# per-shot extremes
per=collections.defaultdict(lambda:[0,0,0,0,9])
for r in fl:
    if r[16]:
        p=per[r[16]]; p[0]=min(p[0],r[6]); p[1]=max(p[1],r[6]); p[2]=min(p[2],r[2],r[4]); p[3]=max(p[3],r[2],r[4]); p[4]=min(p[4],r[14])
xs=sorted(per.items(), key=lambda kv: kv[1][0])[:6]
print("  shots with the most backward pelvis pitch:", [(k, round(v[0],1)) for k,v in xs])
xs=sorted(per.items(), key=lambda kv: kv[1][2])[:6]
print("  shots with the most toe-up foot pitch:", [(k, round(v[2],1)) for k,v in xs])
