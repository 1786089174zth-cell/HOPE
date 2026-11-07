"""120 mixed serves including outside-in diagonals and same-origin fan triplets."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import yaml
from generate_model19000_mixed_3d import NOMINAL,TEMPLATES


def generate():
    rng=np.random.default_rng(1900029999)
    rows=[dict(shot_id=i+1,category='nominal',block=0,fan_group='',x=s[0],y=s[1],z=s[2],vx=s[3],vy=s[4],vz=s[5]) for i,s in enumerate(NOMINAL)]
    def add(category,block,s,fan_group=''):
        rows.append(dict(shot_id=len(rows)+1,category=category,block=block,fan_group=fan_group,
                         **dict(zip(('x','y','z','vx','vy','vz'),[round(float(v),6) for v in s]))))
    for b in range(19):
        # World y starts beyond either sideline (+/-0.7625 m). Aim diagonally
        # across the hit plane, with both same-side and cross-body destinations.
        for side,name in [(-1,'outside_left'),(1,'outside_right')]:
            yw=side*(1.5+1.3*(b%5)/4)
            dest=(-side if b%2 else side)*(.15+.35*(b%3)/2)
            vx=-3.0-.25*(b%5)
            vy=(dest-yw)/2.32*abs(vx)
            add(name,b+1,[2.4,yw-.7625,.49+.06*(b%4),vx,vy,2.2+.08*(b%3)])
        # Exactly the same 3D origin/vx/vz for each adjacent left/center/right triplet.
        origin_y=(-.12,0.,.12)[b%3];vx=-3.0-.2*(b%4)
        height=(.40,.49,.65,.75)[b%4];vz=(2.2,2.4,2.0)[b%3]
        width=(.55,.75,.95)[b%3]
        for dest,name in [(-width,'fan_left'),(0.,'fan_center'),(width,'fan_right')]:
            add(name,b+1,[2.4,origin_y-.7625,height,vx,(dest-origin_y)/2.32*abs(vx),vz],b+1)
        category=('low','high','left','right','oblique','center')[b%6]
        s=list(TEMPLATES[category][(b//6)%4])
        add('mixed_'+category,b+1,s)
    scene={'schema_version':1,'family':'fixed_home','description':'120 directional stress serves; seed1900029999; 19 outside-left, 19 outside-right, 19 same-origin fan triplets, 19 mixed 3D plus 6 nominal',
           'balls':120,'clean_serves':NOMINAL,'rapid_cycle':[[r[k] for k in ('x','y','z','vx','vy','vz')] for r in rows[6:]],
           'timing':{'clean_flight_s':2.5,'rapid_flight_s':1.4,'final_flight_s':2.5,'clean_pause_s':1.0,'rapid_pause_s':0.0},'metrics':{}}
    return scene,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    scene,rows=generate();a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'directional-120.yaml').write_text(yaml.safe_dump(scene,sort_keys=False))
    with (a.output_dir/'directional-120-shots.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=tuple(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'balls':len(rows),'max_horizontal_angle_deg':max(math.degrees(math.atan2(abs(r['vy']),abs(r['vx']))) for r in rows)}))


if __name__=='__main__':main()
