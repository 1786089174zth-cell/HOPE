"""Batched source-time ball playback, independent of the strike planner."""

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch


HISTORY_FRAMES = 4
HISTORY_DIM = HISTORY_FRAMES * 5  # x, y, z, sample age (s), valid


def reconstruct_bank_flights(
    bank: Path, expected_sha256: str, rows_per_side: int, seed: int,
    table_offset: tuple[float, float, float], flight_step, params,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    """Reconstruct the old MuJoCo eval's final incoming segments from original bank rows."""
    with bank.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"PPQ004 bank SHA-256 mismatch: {bank}")
    with np.load(bank, allow_pickle=False) as archive:
        side = archive["clip"]
        stations = archive["station_xy"]
        xyz = archive["target_position_offset"]
        velocity = archive["incoming_velocity"]
        spin = archive["incoming_spin"]
    if rows_per_side < 1 or not np.allclose(stations, 0.0, atol=1e-6, rtol=0):
        raise ValueError("expected positive sample count and a fixed-station PPQ004 bank")

    rng = np.random.default_rng(seed)
    # Consider enough rows per side to replace short or invalid reverse paths without
    # inventing new bank rows. Selection and row IDs are reproducible for a fixed seed.
    candidates = [
        rng.permutation(np.flatnonzero(side == hand))[: rows_per_side * 4]
        for hand in (0, 1)
    ]
    if any(len(rows) < rows_per_side for rows in candidates):
        raise ValueError("PPQ004 bank does not have enough rows for each side")
    row_ids = np.concatenate(candidates)
    p, v, w = (torch.as_tensor(a[row_ids], dtype=torch.float64) for a in (xyz, velocity, spin))
    dt = 0.001
    surface_z = table_offset[2]
    floor = surface_z + params.ball_radius + 0.005
    path = [p]
    accepted = torch.isfinite(p).all(-1) & torch.isfinite(v).all(-1) & torch.isfinite(w).all(-1)
    lengths = torch.ones(len(row_ids), dtype=torch.long)
    for _ in range(600):
        pp, vv = flight_step(p, v, w, -dt, params)
        accepted &= (pp[:, 2] > floor) & (torch.linalg.vector_norm(vv, dim=-1) <= 40.0)
        lengths += accepted
        p = torch.where(accepted[:, None], pp, p)
        v = torch.where(accepted[:, None], vv, v)
        path.append(p)
    paths = torch.stack(path, dim=1).numpy()
    flights, chosen = [], []
    for hand in (0, 1):
        valid = np.flatnonzero((side[row_ids] == hand) & (lengths.numpy() >= 11))
        if len(valid) < rows_per_side:
            raise ValueError(f"PPQ004 bank has too few safe reconstructed flights for clip {hand}")
        for i in valid[:rows_per_side]:
            count = int(lengths[i])
            positions = paths[i, :count][::-1] - np.asarray(table_offset)
            flights.append((np.arange(count) * dt, positions))
            chosen.append(int(row_ids[i]))
    return flights, chosen


@dataclass
class ReplayBatch:
    times: torch.Tensor
    positions: torch.Tensor

    @classmethod
    def from_flights(cls, flights: list[tuple[np.ndarray, np.ndarray]], device) -> "ReplayBatch":
        if not flights:
            raise ValueError("at least one mocap flight is required")
        size = max(len(time) for time, _ in flights)
        times = np.full((len(flights), size), np.inf, dtype=np.float32)
        positions = np.zeros((len(flights), size, 3), dtype=np.float32)
        for i, (time, position) in enumerate(flights):
            times[i, :len(time)] = time
            positions[i, :len(time)] = position
        return cls(torch.from_numpy(times).to(device), torch.from_numpy(positions).to(device))

    def observe(
        self, flight: torch.Tensor, now: torch.Tensor, *, period_s: float, delay_s: float, max_age_s: float
    ) -> torch.Tensor:
        """Pick only samples already received at each of four observation times."""
        grid = now[:, None] - period_s * torch.arange(HISTORY_FRAMES, device=now.device)
        query = grid - delay_s
        times = self.times[flight]
        idx = torch.searchsorted(times, query.contiguous(), right=True) - 1
        safe_idx = idx.clamp_min(0)
        captured = times.gather(1, safe_idx)
        age = grid - captured
        valid = (idx >= 0) & (age <= max_age_s)
        pos = self.positions[flight[:, None], safe_idx]
        data = torch.cat((pos, age.unsqueeze(-1)), dim=-1)
        data = torch.where(valid.unsqueeze(-1), data, 0.0)
        return torch.cat((data, valid.unsqueeze(-1).to(pos.dtype)), dim=-1).flatten(1)

    def physical_sample(
        self, flight: torch.Tensor, now: torch.Tensor, *, max_gap_s: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate the simulated inbound ball between adjacent replay poses.

        This path is for contact only. The actor receives causal samples via observe().
        """
        times = self.times[flight]
        idx = torch.searchsorted(times, now[:, None].contiguous(), right=True).squeeze(1)
        hi = idx.clamp(1, times.shape[1] - 1)
        lo = hi - 1
        t0, t1 = times.gather(1, lo[:, None]).squeeze(1), times.gather(1, hi[:, None]).squeeze(1)
        interval = t1 - t0
        valid = (t0 <= now) & (now <= t1) & (interval <= max_gap_s) & (interval > 0)
        span = torch.where(valid, interval, 1.0)
        p0, p1 = self.positions[flight, lo], self.positions[flight, hi]
        velocity = (p1 - p0) / span[:, None]
        position = p0 + (now - t0).clamp(0.0, max_gap_s)[:, None] * velocity
        return torch.where(valid[:, None], position, 0.0), torch.where(valid[:, None], velocity, 0.0), valid
