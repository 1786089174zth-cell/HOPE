"""Replay recorded ROS planner commands into the GRU safety diagnostic.

This isolates actor and pelvis-observation behavior from ROS wall-clock load.
It does not evaluate how the planner responds to a new mocap outage.
"""

import argparse
import json
import math
import socket
import time
from pathlib import Path


def rebased_values(record, unix_anchor_s):
    values = list(record['values'])
    old_anchor_s = values[10] - record['deadline_sim_s']
    producer_sim_s = values[12] + values[13] * 1e-9 - old_anchor_s
    producer_s = unix_anchor_s + producer_sim_s
    sec = math.floor(producer_s)
    nsec = round((producer_s - sec) * 1e9)
    if nsec == 1_000_000_000:
        sec += 1
        nsec = 0
    values[10] = unix_anchor_s + record['deadline_sim_s']
    values[12:14] = [sec, nsec]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',required=True,type=Path)
    parser.add_argument('--socket',required=True,type=Path)
    args = parser.parse_args()
    plan = json.loads(args.replay.read_text())
    records = {r['shot_id']:r for r in plan['commands']}
    if len(records) != len(plan['commands']):
        raise ValueError('Duplicate shot ID in planner replay')
    if args.socket.exists():
        raise FileExistsError(args.socket)
    server = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        server.bind(str(args.socket))
        server.listen(1)
        conn,_ = server.accept()
        sent = set()
        with conn,conn.makefile('rwb') as channel:
            for line in channel:
                request = json.loads(line)
                if request.get('close'):
                    break
                shot_id = request['serve_id']
                record = records.get(shot_id)
                commands = []
                if record is not None and shot_id not in sent and request['sim_t'] + 1e-8 >= record['sim_t']:
                    anchor_s = request['stamp_ns'] * 1e-9 - request['sim_t']
                    commands.append({'values':rebased_values(record,anchor_s),
                                     'received_unix_ns':time.time_ns(),
                                     'replayed_from_shot_id':shot_id})
                    sent.add(shot_id)
                channel.write((json.dumps({'commands':commands,'diagnostics':[],
                                           'reply_unix_ns':time.time_ns()})+'\n').encode())
                channel.flush()
        print(json.dumps({'replayed_commands':len(sent),'expected_commands':len(records)}),flush=True)
    finally:
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == '__main__':main()
