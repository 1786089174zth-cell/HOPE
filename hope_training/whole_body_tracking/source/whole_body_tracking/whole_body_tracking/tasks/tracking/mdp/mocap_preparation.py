"""Training-only progress toward an incoming ball's racket preparation pose."""

import torch


def advance_preparation_progress(
    best: torch.Tensor,
    error_sq: torch.Tensor,
    lead_s: torch.Tensor,
    eligible: torch.Tensor,
    upright: torch.Tensor,
    window_s: tuple[float, float],
    distance_scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Credit new distance improvements only. A negative best marks a fresh trial."""
    earliest, latest = window_s
    in_window = eligible & (lead_s >= earliest) & (lead_s <= latest)
    score = torch.exp(-error_sq / distance_scale_m**2)
    previous = torch.where(best >= 0.0, best, score)
    progress = torch.where(in_window & upright, (score - previous).clamp_min(0.0), 0.0)
    # Even an unstable advance raises the best, so standing still while
    # straightening up cannot retroactively earn preparation progress.
    best = torch.where(in_window, torch.maximum(previous, score), best)
    return progress, best
