"""Ball (and robot body) mocap samples plus a few status strings of a laptop capture -> npz / txt."""
import sys, glob, json
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

D, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ball, body, strings = [], [], {}
for f in sorted(glob.glob(D + "/laptop/capture_attempt_001/recordings/*/capture/ros_raw/*.mcap")):
    with open(f, "rb") as s:
        r = make_reader(s, decoder_factories=[DecoderFactory()])
        for _, ch, msg, ros in r.iter_decoded_messages(topics=["/ball/pose", "/UCB_P1/pose", "/hope/runner/summary", "/hope/field/status", "/hope/lifecycle/summary", "/hope/base/summary"]):
            if ch.topic in ("/ball/pose", "/UCB_P1/pose"):
                p, q, st = ros.pose.position, ros.pose.orientation, ros.header.stamp
                (ball if ch.topic == "/ball/pose" else body).append((st.sec + 1e-9 * st.nanosec, msg.log_time * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            else:
                strings.setdefault(ch.topic, []).append((msg.log_time * 1e-9, ros.data))
    print(f, len(ball), len(body), flush=True)
np.savez_compressed(out / "mocap.npz", ball=np.array(ball), body=np.array(body)[::6])
for k, v in strings.items():
    with open(out / (k.strip("/").replace("/", "_") + ".txt"), "w") as s:
        for t, d in v:
            s.write(f"{t:.3f}\t{d}\n")
