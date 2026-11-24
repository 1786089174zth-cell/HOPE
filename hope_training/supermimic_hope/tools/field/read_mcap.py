"""Mocap streams of a field recording (leaderboard ``.mcap``) as one ``.npz`` per recording.

    python -I tools/field/read_mcap.py <recording.mcap> [...] --out <directory>      (needs: mcap, mcap-ros2-support, numpy)

The recordings are VRPN mocap topics (``/vrpn_mocap/<body>/pose_id_<n>``, PoseStamped, about 300 Hz): the ball,
the two robots and other tracked bodies. For each body the output has ``<body>_t`` (seconds from the first
message, from the header stamp), ``<body>_p`` (N, 3) and ``<body>_q`` (N, 4, xyzw), plus ``log_t`` per body
(the recorder's receive time) under ``<body>_log``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def read(path: Path):
    series = {}
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, ros in reader.iter_decoded_messages():
            parts = channel.topic.split("/")
            if len(parts) != 4 or not parts[3].startswith("pose_id_"):
                continue
            p, q, stamp = ros.pose.position, ros.pose.orientation, ros.header.stamp
            series.setdefault(parts[2], []).append((stamp.sec + 1e-9 * stamp.nanosec, message.log_time * 1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    out, t0 = {}, min(rows[0][0] for rows in series.values())
    for body, rows in series.items():
        a = np.array(rows)
        out[body + "_t"], out[body + "_log"], out[body + "_p"], out[body + "_q"] = a[:, 0] - t0, a[:, 1] - a[0, 1], a[:, 2:5], a[:, 5:9]
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    i = argv.index("--out")
    out = Path(argv[i + 1])
    del argv[i:i + 2]
    out.mkdir(parents=True, exist_ok=True)
    for arg in argv:
        data = read(Path(arg))
        np.savez_compressed(out / (Path(arg).stem + ".npz"), **data)
        bodies = sorted(k[:-2] for k in data if k.endswith("_t"))
        print(f"{Path(arg).name}: " + ", ".join(f"{b} {len(data[b + '_t'])} samples over {data[b + '_t'][-1] - data[b + '_t'][0]:.1f} s" for b in bodies))


if __name__ == "__main__":
    main()
