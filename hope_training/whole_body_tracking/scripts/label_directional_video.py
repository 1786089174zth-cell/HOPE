"""Add checkpoint, serve category and mocap status to recorded-state video."""
import argparse,csv,json,subprocess
from pathlib import Path


def clock(t):
    cs=round(t*100);return f'{cs//360000}:{cs//6000%60:02}:{cs//100%60:02}.{cs%100:02}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--shots',required=True,type=Path);a=p.parse_args()
    summary=json.loads((a.output/'summary.json').read_text());rows=json.loads((a.output/'per-ball.json').read_text());states=[json.loads(l) for l in (a.output/'render-states.jsonl').open()]
    labels={int(r['shot_id']):r for r in csv.DictReader(a.shots.open())}
    subtitle=a.output/'directional.ass'
    header='''[Script Info]\nScriptType: v4.00+\nPlayResX: 800\nPlayResY: 500\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Top,DejaVu Sans,17,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,2,0,7,10,10,10,1\nStyle: Bottom,DejaVu Sans,16,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,2,0,1,10,10,10,1\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n'''
    hit_times={r['shot_id']:min((e['episode_t'] for e in r.get('events',[]) if e['kind']=='racket_contact'),default=float('inf')) for r in rows}
    with subtitle.open('w') as f:
        f.write(header)
        for i,state in enumerate(states):
            t=state['sim_t'];launched=[r for r in rows if r['launched'] and r['launched_at_s']<=t+1e-8];r=launched[-1] if launched else None
            sid=r['shot_id'] if r else 0;cat=labels[sid]['category'] if r else 'initial_stand'
            dropout=any(start<=t<end for start,end in summary['pelvis_dropout_windows_s'])
            fallen=summary['fall_time_s'] is not None and t>=summary['fall_time_s']
            top=f'MODEL {summary["iteration"]} | 120 DIRECTIONAL SERVES | RECORDED TARGETS\\NShot {sid}/120 - {cat} | t={t:.2f}s | pelvis '+('DROPOUT - LAST POSE HELD' if dropout else 'UPDATING')
            bottom=f'Racket contacts {sum(v<=t for v in hit_times.values())} | '+('FALL - RALLY STOPPED' if fallen else 'No fall so far')+' | orange incoming / cyan after contact'
            for style,text in [('Top',top),('Bottom',bottom)]:f.write(f'Dialogue: 0,{clock(i/20)},{clock((i+1)/20)},{style},,0,0,0,,{text}\n')
    video=a.output/'directional-120.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(a.output/'diagnostic.mp4'),'-vf',f"drawbox=x=0:y=0:w=iw:h=65:color=black:t=fill,subtitles=filename='{subtitle}'",'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-an',str(video)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    print('LABELED VIDEO '+str(video),flush=True)


if __name__=='__main__':main()
