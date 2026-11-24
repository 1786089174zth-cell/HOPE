"""Call the package training entry point; record initial state before collecting any transitions."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import train
from supermimic_hope.core.trainer import Trainer

def fingerprint(tensors):
    h = hashlib.sha256()
    for key, tensor in tensors:
        h.update(key.encode("utf-8") + b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

train_impl = Trainer.train
def train_with_receipt(self):
    path = Path(self.cfg.out)
    path.mkdir(parents=True, exist_ok=False)
    initial = {n: fingerprint(getattr(self, n).named_parameters()) for n in ("actor", "critic", "f")}
    initial["history"] = fingerprint([("obs", self.history)])
    initial["initial_sim_state"] = fingerprint([("state", self.sim.state())])
    (path / "initial.json").write_text(json.dumps(initial, indent=2))
    return train_impl(self)

Trainer.train = train_with_receipt
train.main()
