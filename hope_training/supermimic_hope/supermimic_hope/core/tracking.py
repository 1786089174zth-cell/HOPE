"""Motion-tracking reward and soft termination of the paper (Table A.2, Appendix A).

BeyondMimic's Gaussian-kernel reward on world-frame bodies without the
self-collision term: the reference root is tracked in the world frame, the
other tracked bodies after re-anchoring the reference to the robot's root xy and
yaw. Everything is a differentiable function of the predicted state.
"""

from __future__ import annotations

import torch

from .rotations import quat_conj, quat_mul, quat_to_matrix, yaw_quat


def reanchor(ref_pos, ref_rot, ref_root_quat, root_pos, root_quat):
    """Reference bodies in a frame whose xy and yaw come from the robot's root and
    whose z, roll and pitch come from the reference root."""
    delta = quat_to_matrix(yaw_quat(quat_mul(root_quat, quat_conj(ref_root_quat))))[:, None]
    anchor = ref_pos[:, :1]
    origin = torch.cat((root_pos[:, :2], ref_pos[:, 0, 2:3]), -1)[:, None]
    pos = origin + (delta @ (ref_pos - anchor)[..., None])[..., 0]
    return pos, delta @ ref_rot


def rot_err_sq(Ra, Rb):
    """``2 (1 - cos theta)``: equals theta^2 to third order and has no acos singularity."""
    return (3.0 - (Ra * Rb).sum((-1, -2))).clamp_min(0.0)


def tracking_terms(s, bodies, ref, ref_bodies, track, feet=None, feet_std=0.06, height_std=0.04, lift_std=0.025,
                   trunk=None, trunk_std=0.12, feet_ori_std=None, station_std=None, weight=None):
    """The six kernel terms (each in [0, 1]) as a dict of (B,) tensors.

    ``bodies`` / ``ref_bodies`` are ``Kinematics`` outputs for the state and the
    reference state; ``track`` are the tracked body indices (root first).

    With ``feet`` (positions of the two feet inside ``track``) three tighter HOPE terms are
    added: ``feet_pos``, the feet against the re-anchored reference (the paper's 0.3 m body
    kernel averaged over 14 bodies leaves a 7 cm short step and a 3 cm missing lift almost
    free), ``feet_height`` (the lift alone, at a 2.5 cm scale) and ``root_height`` (the 0.3 m
    root kernel does not see a 2 cm crouch). With ``trunk`` (pelvis and torso inside ``track``)
    ``trunk_ori`` scores their orientation at a 0.12 rad scale: the 0.4 rad kernel averaged over
    14 bodies lets the torso lean 11 degrees for free. With ``feet_ori_std`` the two feet's
    orientation is scored as ``feet_ori``: the teacher keeps both soles flat throughout, while the
    policy brought the stepping foot back with the toes up 15 degrees, dragging the heel. With
    ``station_std = (root, feet, heading)`` three terms hold the robot on the reference's own spot in
    the world: ``root_xy``, ``feet_station`` (the feet without re-anchoring) and ``root_heading``.
    Every other position term is either 0.3 m wide or relative to the robot's root, so a robot that
    ends each stroke a millimetre nearer the table was never corrected."""
    # ``weight`` (B, tracked bodies), optional: how much each body counts in the body and trunk means (a stroke
    # asked without a pose leaves the upper body out while it swings).
    pos, rot, omega, vel = (x[:, track] for x in bodies)
    rpos, rrot, romega, rvel = (x[:, track] for x in ref_bodies)
    tpos, trot = reanchor(rpos, rrot, ref[:, 3:7], s[:, 0:3], s[:, 3:7])
    k = lambda err, std: torch.exp(-err / std ** 2)  # noqa: E731
    if weight is None:
        mean = lambda err: err.mean(-1)  # noqa: E731
        mean_trunk = mean
    else:
        mean = lambda err: (err * weight).sum(-1) / weight.sum(-1).clamp_min(1e-6)  # noqa: E731
        mean_trunk = lambda err: (err * weight[:, trunk]).sum(-1) / weight[:, trunk].sum(-1).clamp_min(1e-6)  # noqa: E731
    extra = {} if feet is None else dict(
        feet_pos=k((pos[:, feet] - tpos[:, feet]).square().sum(-1).mean(-1), feet_std),
        feet_height=k((pos[:, feet, 2] - tpos[:, feet, 2]).square().mean(-1), lift_std),
        root_height=k((pos[:, 0, 2] - rpos[:, 0, 2]).square(), height_std),
    )
    if trunk is not None:
        extra["trunk_ori"] = k(mean_trunk(rot_err_sq(rot[:, trunk], trot[:, trunk])), trunk_std)
    if feet is not None and feet_ori_std is not None:
        extra["feet_ori"] = k(rot_err_sq(rot[:, feet], trot[:, feet]).mean(-1), feet_ori_std)
    if feet is not None and station_std is not None:
        extra["root_xy"] = k((pos[:, 0, :2] - rpos[:, 0, :2]).square().sum(-1), station_std[0])
        extra["feet_station"] = k((pos[:, feet, :2] - rpos[:, feet, :2]).square().sum(-1).mean(-1), station_std[1])
        extra["root_heading"] = k(rot_err_sq(rot[:, 0], rrot[:, 0]), station_std[2])
        if len(station_std) > 3:  # the feet's own heading in the world: a pelvis can only turn back if they do
            extra["feet_heading"] = k(rot_err_sq(rot[:, feet], rrot[:, feet]).mean(-1), station_std[3])
    return extra | dict(
        root_pos=k((pos[:, 0] - rpos[:, 0]).square().sum(-1), 0.3),
        root_ori=k(rot_err_sq(rot[:, 0], rrot[:, 0]), 0.4),
        body_pos=k(mean((pos - tpos).square().sum(-1)), 0.3),
        body_ori=k(mean(rot_err_sq(rot, trot)), 0.4),
        body_lin_vel=k(mean((vel - rvel).square().sum(-1)), 1.0),
        body_ang_vel=k(mean((omega - romega).square().sum(-1)), 3.14),
    )


TRACK_WEIGHTS = dict(root_pos=0.5, root_ori=0.5, body_pos=1.0, body_ori=1.0, body_lin_vel=1.0, body_ang_vel=1.0)


def termination_error(s, ref):
    """Normalized distance to the paper's termination: root height deviation 0.25 m
    or gravity-z (tilt) deviation 0.8 from the reference root; > 1 terminates."""
    height = (ref[:, 2] - s[:, 2]).abs() / 0.25
    gz = lambda q: 2 * (q[:, 1:3] ** 2).sum(-1) - 1.0  # noqa: E731 -- gravity z in the root frame
    tilt = (gz(ref[:, 3:7]) - gz(s[:, 3:7])).abs() / 0.8
    return torch.maximum(height, tilt)


def soft_termination(s, ref, softness=0.05):
    return torch.sigmoid((termination_error(s, ref) - 1.0) / softness)
