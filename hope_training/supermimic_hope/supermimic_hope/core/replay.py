"""Fixed-size replay of logged transitions (paper: 1000 steps per environment).

Time-major ring on the training device. Windows never cross an episode end, so
the exogenous task tape (references, ball commands) of a window belongs to one
episode and imagined rollouts can follow it step by step.
"""

from __future__ import annotations

import torch


class Tape:
    def __init__(self, steps: int, num_envs: int, device):
        self.T, self.N, self.device = int(steps), int(num_envs), torch.device(device)
        self.ptr = self.size = 0
        self.data: dict[str, torch.Tensor] = {}
        self._episode = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._since = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def add(self, record: dict[str, torch.Tensor], done: torch.Tensor):
        """Append one step; ``done`` marks environments whose episode ends with it."""
        record = dict(record, episode=self._episode, since=self._since)
        if not self.data:
            self.data = {
                k: torch.zeros((self.T,) + tuple(v.shape), dtype=v.dtype, device=self.device)
                for k, v in record.items()
            }
        for k, v in record.items():
            self.data[k][self.ptr].copy_(v)
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        done = done.bool()
        self._episode += done
        self._since = torch.where(done, torch.zeros_like(self._since), self._since + 1)

    def bytes(self):
        return sum(v.numel() * v.element_size() for v in self.data.values())

    def _starts(self, batch, length):
        """(slot of the first step, env) of ``batch`` windows inside one episode."""
        if self.size < length:
            raise ValueError("replay shorter than the requested window")
        m = 2 * batch
        age = torch.randint(0, self.size - length + 1, (m,), device=self.device)
        env = torch.randint(0, self.N, (m,), device=self.device)
        first = (self.ptr - self.size + age) % self.T
        last = (first + length - 1) % self.T
        episode = self.data["episode"]
        valid = episode[first, env] == episode[last, env]
        # Valid candidates first, without a host round trip (episodes are long,
        # so far more than half of the candidates are valid).
        pick = torch.argsort((~valid).int(), stable=True)[:batch]
        return first[pick], env[pick], age[pick]

    def sample(self, batch, length, keys=None, history=0, history_key="frame"):
        """Windows ``(batch, length, ...)``. With ``history`` > 0 also returns
        ``history`` (batch, history, ...): the ``history_key`` rows preceding the
        window start, oldest first, clamped at the episode start (the deployed
        runner seeds its frame history with the first frame of an episode)."""
        first, env, age = self._starts(batch, length)
        rows = (first[:, None] + torch.arange(length, device=self.device)) % self.T
        names = self.data.keys() if keys is None else keys
        out = {k: self.data[k][rows, env[:, None]] for k in names}
        out["env"] = env
        if history:
            back = torch.arange(history, 0, -1, device=self.device)[None]  # (1, history): oldest first
            limit = torch.minimum(self.data["since"][first, env], age)[:, None]
            rows_h = (first[:, None] - torch.minimum(back, limit)) % self.T
            out["history"] = self.data[history_key][rows_h, env[:, None]]
        return out
