"""Continuous sphere contact against the replay task's moving, two-sided racket disc."""

import torch
from torch.nn.functional import normalize


def sweep_ball_against_disc(delta0, delta1, normal0, normal1, disc_radius, ball_radius):
    """Return the hit, time fraction, contact point, normal, and disc motion.

    The centers move linearly over the step. The disc normal follows normalized
    linear interpolation between the two equivalent, same-hemisphere face normals.
    Each advance uses a supporting plane of the disc and an upper bound on the
    rate at which rotation can move that plane. A step cannot skip a collision
    to reach a later sample. The search is bounded and stops within 1 µm of contact.
    """
    normal0, normal1 = normalize(normal0, dim=-1), normalize(normal1, dim=-1)
    normal1 = torch.where((normal0 * normal1).sum(dim=-1, keepdim=True) < 0, -normal1, normal1)
    dn = normal1 - normal0
    motion = delta1 - delta0
    midpoint_norm = torch.linalg.vector_norm((normal0 + normal1) * 0.5, dim=-1).clamp_min(1e-6)
    rotation_bound = disc_radius * torch.linalg.vector_norm(dn, dim=-1) / midpoint_norm
    tolerance = 1e-6

    def geometry(t):
        delta = delta0 + t[:, None] * motion
        unscaled_normal = normal0 + t[:, None] * dn
        face = normalize(unscaled_normal, dim=-1)
        tangent = delta - (delta * face).sum(dim=-1, keepdim=True) * face
        radius = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-9)
        point = tangent * (disc_radius / radius).clamp_max(1.0)
        separation = delta - point
        distance = torch.linalg.vector_norm(separation, dim=-1)
        fallback = torch.where((motion * face).sum(dim=-1, keepdim=True) > 0, -face, face)
        outward = torch.where(distance[:, None] > 1e-9, normalize(separation, dim=-1), fallback)
        return distance, point, outward

    fraction = torch.zeros_like(rotation_bound)
    distance, point, outward = geometry(fraction)
    # An interval can start at replay launch. An existing overlap counts only if
    # the sphere is entering the disc, not when a separating ball first appears.
    initial_rotation = torch.linalg.cross(normal0, dn)
    initial_relative = motion - torch.linalg.cross(initial_rotation, point)
    entering_at_start = (initial_relative * outward).sum(dim=-1) < 0
    for _ in range(8):
        for _ in range(8):
            gap = distance - ball_radius
            closing_bound = rotation_bound - (outward * motion).sum(dim=-1)
            moving = (fraction < 1.0) & (gap > tolerance) & (closing_bound > 0)
            advance = gap / closing_bound.clamp_min(1e-9)
            fraction = torch.where(moving, (fraction + advance).clamp_max(1.0), fraction)
            distance, point, outward = geometry(fraction)
        gap = distance - ball_radius
        if not bool(((fraction < 1.0) & (gap > tolerance)
                     & (rotation_bound > (outward * motion).sum(dim=-1))).any()):
            break
    hit = (distance <= ball_radius + tolerance) & ((fraction > 0) | entering_at_start)
    unscaled_normal = normal0 + fraction[:, None] * dn
    face = normalize(unscaled_normal, dim=-1)
    rotation = torch.linalg.cross(face, dn) / torch.linalg.vector_norm(
        unscaled_normal, dim=-1, keepdim=True,
    ).clamp_min(1e-6)
    return hit, fraction, point, outward, face, rotation
