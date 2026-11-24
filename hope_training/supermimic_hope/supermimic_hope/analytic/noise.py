"""Sensor-noise gain penalty on the deterministic actor (CAPS-spatial).

``E_n ||W (mu(o + n) - mu(o))||^2`` with ``n`` drawn from the training
corruption model and ``W`` the affine action scales (passive head 0), i.e. the
per-tick q_des noise variance in rad^2 summed over joints that the actor mean
produces from sensor noise alone. Kept inside this package so the PPO recipe
(``algo/*.yaml``, ``projection_ppo.py``) is not modified by SuperMimic.
"""

from __future__ import annotations

import math

import torch


def parse_channel(channel):
    """``{dim, distribution: uniform|gaussian, scale: float | [dim], targets: [[start, stop,
    component, sign], ...]}``. One draw of ``dim`` values is added to every target slice
    (``obs[start:stop] += sign * n[component:component + stop - start]``), so a single mocap
    position error moves base_target_delta_xy and racket_target_rel_base together. ``scale`` is
    the half-width for uniform (Isaac ``uniform_noise`` n_max) and the std for gaussian."""
    dim = int(channel["dim"])
    distribution = str(channel.get("distribution", "gaussian"))
    if dim <= 0 or distribution not in ("uniform", "gaussian"):
        raise ValueError(f"invalid sensor noise channel {channel}")
    scale = channel["scale"]
    scale = tuple(float(x) for x in scale) if isinstance(scale, (list, tuple)) else (float(scale),) * dim
    if len(scale) != dim or any(not math.isfinite(x) or x < 0 for x in scale):
        raise ValueError(f"sensor noise channel scale must be {dim} finite non-negative values: {channel}")
    targets = []
    for start, stop, component, sign in channel["targets"]:
        start, stop, component, sign = int(start), int(stop), int(component), float(sign)
        if not (0 <= start < stop and 0 <= component and component + stop - start <= dim and sign in (-1.0, 1.0)):
            raise ValueError(f"invalid sensor noise target {[start, stop, component, sign]} in {channel}")
        targets.append((start, stop, component, sign))
    if not targets:
        raise ValueError(f"sensor noise channel has no targets: {channel}")
    return dim, distribution == "uniform", scale, tuple(targets)


class SensorNoisePenalty:
    """``coef * E||W (mu(o + n) - mu(o))||^2`` on up to ``samples`` rows of raw actor observations."""

    def __init__(self, actor, coef, channels, output_weights, samples=4096):
        self.actor = actor  # normalizer + MLP: raw 110-D observation -> 31 raw actions
        self.coef = float(coef)
        self.channels = tuple(parse_channel(dict(c)) for c in channels)
        self.output_weights = tuple(float(x) for x in output_weights)
        self.samples = int(samples)
        if not math.isfinite(self.coef) or self.coef < 0 or self.samples <= 0:
            raise ValueError("sensor noise coef must be finite and >= 0 and samples > 0")
        if self.coef > 0 and not self.channels:
            raise ValueError("sensor noise coef > 0 requires at least one channel")

    def raw(self, obs):
        clean = obs[: self.samples].detach()
        noisy = clean.clone()
        for dim, uniform, scale, targets in self.channels:
            draw = torch.rand if uniform else torch.randn
            noise = draw(clean.shape[0], dim, device=clean.device, dtype=clean.dtype)
            noise = (2.0 * noise - 1.0 if uniform else noise) * clean.new_tensor(scale)
            for start, stop, component, sign in targets:
                if stop > clean.shape[-1]:
                    raise ValueError(f"sensor noise target [{start}, {stop}) exceeds actor obs dim {clean.shape[-1]}")
                noisy[:, start:stop] += sign * noise[:, component:component + stop - start]
        mean = self.actor(torch.cat([clean, noisy]))
        delta = mean[clean.shape[0]:] - mean[: clean.shape[0]]
        if self.output_weights:
            delta = delta * mean.new_tensor(self.output_weights)
        return delta.square().sum(-1).mean()

    def __call__(self, obs):
        if self.coef <= 0:
            return obs.new_zeros(())
        return self.coef * self.raw(obs)


def from_config(actor, cfg):
    """Build from ``supermimic.sensor_noise`` (OmegaConf node or dict)."""
    from omegaconf import OmegaConf

    c = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    return SensorNoisePenalty(actor, c["coef"], c["channels"], c["output_weights"], c.get("samples", 4096))
