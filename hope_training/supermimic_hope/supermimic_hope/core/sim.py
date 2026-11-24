"""Batched MuJoCo-Warp simulation of the A3 with per-world domain randomization.

One control tick is four 5 ms implicit-fast steps of the Gate3 MJCF with PD
position actuators (``a3.build_model``), replayed as one captured Warp graph.
All worlds share the origin; states are torch views of the Warp arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import dataclass, field

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .. import a3
from .dynamics import NJ, STATE


@dataclass
class SimCfg:
    """Startup domain randomization (HOPE recipe; paper Table A.3 has friction and torso COM)."""

    # Foot-floor Coulomb coefficient. Gate3's plant has 1.5 on floor and soles; the paper's range ends at 1.2.
    friction: tuple[float, float] = (0.3, 1.6)
    com_x: tuple[float, float] = (-0.025, 0.025)  # torso centre-of-mass offset (m)
    com_y: tuple[float, float] = (-0.05, 0.05)
    com_z: tuple[float, float] = (-0.05, 0.05)
    mass_scale: tuple[float, float] = (0.85, 1.15)  # whole-robot mass and inertia
    pd_scale: tuple[float, float] = (0.85, 1.15)  # per-joint kp and message-kd factors, log-uniform
    nominal_fraction: float = 0.25  # worlds with nominal mass, COM and PD gains
    # The left hand. The deployed robot carries a gripper there (main, A3PingPong-with-gripper/urdf/model.urdf:
    # 0.847 kg with its centre 0.113 m along left_wrist_yaw_Link); the Gate3 MJCF this model is built from has
    # 0.281 kg at 0.084 m. On the robot model_54000 held with the trunk 4.7 deg to the left (waist roll target
    # +2.5, actual -4.7 deg) and the left elbow 12 deg lower than in Gate3; with 0.6 kg added to the left hand
    # the simulator does the same (docs/real_hold_lean_20261120.md). 0 keeps the MJCF's hand.
    left_hand_mass: float = 0.847
    left_hand_com_x: float = 0.1128
    # Mass randomization per link. Until 2026-11-20 only the whole robot was scaled (every link by the same
    # factor: no change of balance) and the torso's centre moved; the policy corrects those (5 cm of torso COM:
    # 0.6 deg of lean) and nothing else: 0.57 kg in a hand, 1% of the robot, leaned the trunk 4.7 deg. Now
    # every link's mass and inertia vary independently by ``link_mass_scale``, the left hand is anything from bare
    # to a heavy gripper, and the right hand (racket) varies too. The nominal worlds have ``left_hand_mass``.
    # None of this is in the world model's context vector, whose size is fixed by the checkpoints.
    link_mass_scale: tuple[float, float] = (0.9, 1.1)
    left_hand_mass_range: tuple[float, float] = (0.25, 1.25)  # kg
    right_hand_mass_delta: tuple[float, float] = (-0.15, 0.30)  # kg on top of the MJCF's racket hand
    # Actuators identified on the robot. The first real sessions of ft48 137000 (2026-11-21) replayed in this model
    # (tools/real/replay_joints.py: recorded joint commands in, joint angles compared): for the same commands the
    # MJCF's right-arm joints peak at about 80% of the robot's joint speed and the waist pitch moves twice as far;
    # on the robot the forehand racket was 0.5 m/s faster than commanded and returns flew long. ``actuator_fit``
    # is the json of that tool's ``--fit``: per-joint factors on passive damping, dry friction, rotor inertia and on
    # the stiffness the motor realizes. Each world draws a position between the MJCF actuators (0) and the fitted
    # ones (1) from ``actuator_blend`` (every factor interpolated linearly; nominal worlds stay at 0, the plant of
    # Gate3 and Supergate) and that position is one more entry of the world model's context. "" = MJCF only and
    # the context of the checkpoints before this existed.
    actuator_fit: str = ""
    actuator_blend: tuple[float, float] = (-0.25, 1.25)
    # Foot-floor friction model. Gate3's plant runs MuJoCo's no-slip pass, which MuJoCo-Warp lacks;
    # under the pyramidal cone a loaded sole then creeps under a steady side force (a 0.27 m stance
    # reached the 0.47 m HOLD width in 10 s by creep alone, where Gate3 stays put). The elliptic cone
    # with this impedance ratio holds like the no-slip plant (tools/entry_replica.py).
    cone: str = "elliptic"
    impratio: float = 10.0
    nconmax: int = 48
    njmax: int = 160


class WarpSim:
    def __init__(self, num_envs: int, device: str, cfg: SimCfg | None = None, seed: int = 0, capture: bool = True):
        self.N, self.device, self.cfg = num_envs, torch.device(device), cfg or SimCfg()
        wp.config.quiet = True
        self.model, self.receipt = a3.build_model()
        self.model.opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
                               "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[self.cfg.cone]
        self.model.opt.impratio = self.cfg.impratio
        self.receipt.update(cone=self.cfg.cone, impratio=self.cfg.impratio)
        self.ix = a3.indices(self.model)
        self.qids = torch.as_tensor(self.ix["qpos"], device=self.device)
        self.vids = torch.as_tensor(self.ix["dof"], device=self.device)
        self.wp_device = str(self.device)
        d0 = mujoco.MjData(self.model)
        d0.qpos[2] = 1.0
        d0.qpos[self.ix["qpos"]] = a3.DEFAULT_Q
        d0.ctrl[:] = a3.DEFAULT_Q
        mujoco.mj_forward(self.model, d0)
        with wp.ScopedDevice(self.wp_device):
            self.m = mjw.put_model(self.model)
            # Per-world model arrays: randomization must never silently be shared.
            for key, value in vars(self.m).items():
                if isinstance(value, wp.array) and getattr(value, "_is_batched", False):
                    tensor = wp.to_torch(value)
                    tensor = tensor.repeat((self.N,) + (1,) * (tensor.ndim - 1))
                    array = wp.from_torch(tensor.contiguous(), dtype=value.dtype)
                    array._is_batched = True
                    setattr(self.m, key, array)
            self.d = mjw.put_data(self.model, d0, nworld=self.N, nconmax=self.cfg.nconmax, njmax=self.cfg.njmax)
        self.qpos, self.qvel = wp.to_torch(self.d.qpos), wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self._randomize(seed)
        self.graph = None
        with wp.ScopedDevice(self.wp_device):
            mjw.set_const(self.m, self.d)
            mjw.forward(self.m, self.d)
            self._tick()
            if capture and self.device.type == "cuda":
                with wp.ScopedCapture() as cap:
                    self._tick()
                self.graph = cap.graph

    # ------------------------------------------------------------------ randomization
    def _randomize(self, seed):
        cfg, N, dev = self.cfg, self.N, self.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        nominal = (torch.arange(N) % max(1, round(1 / cfg.nominal_fraction)) == 0) if cfg.nominal_fraction > 0 \
            else torch.zeros(N, dtype=torch.bool)

        def unit(shape=(N,), keep_nominal=True):
            u = torch.rand(shape, generator=g) * 2 - 1
            if keep_nominal:
                u = torch.where(nominal.view(-1, *([1] * (len(shape) - 1))), torch.zeros_like(u), u)
            return u

        def spread(u, rng):
            return 0.5 * (rng[0] + rng[1]) + 0.5 * (rng[1] - rng[0]) * u

        u_mu = unit(keep_nominal=False)
        u_com = unit((N, 3))
        u_mass = unit()
        u_kp, u_kd = unit((N, a3.NJ)), unit((N, a3.NJ))
        log_half = 0.5 * (np.log(cfg.pd_scale[1]) - np.log(cfg.pd_scale[0]))
        log_mid = 0.5 * (np.log(cfg.pd_scale[1]) + np.log(cfg.pd_scale[0]))
        alpha, beta = torch.exp(log_mid + log_half * u_kp).to(dev), torch.exp(log_mid + log_half * u_kd).to(dev)
        mu = spread(u_mu, cfg.friction).to(dev)
        com = torch.stack([spread(u_com[:, i], r) for i, r in enumerate((cfg.com_x, cfg.com_y, cfg.com_z))], -1).to(dev)
        scale = spread(u_mass, cfg.mass_scale).to(dev)
        # The world model's sim context: every randomized parameter in [-1, 1].
        self.ctx = torch.cat((u_mu[:, None], u_com, u_mass[:, None], u_kp, u_kd), -1).to(dev)
        model = self.model
        friction = wp.to_torch(self.m.geom_friction)
        # MuJoCo combines by max: a frictionless floor makes the foot value the contact value.
        friction[:, model.geom("floor").id, 0] = 0.0
        for name in a3.FEET:
            geoms = np.flatnonzero(model.geom_bodyid == model.body(name).id)
            friction[:, torch.as_tensor(geoms, device=dev), 0] = mu[:, None]
        wp.to_torch(self.m.body_ipos)[:, model.body("torso_Link").id] += com
        wp.to_torch(self.m.body_mass).mul_(scale[:, None])
        wp.to_torch(self.m.body_inertia).mul_(scale[:, None, None])
        mass, inertia = wp.to_torch(self.m.body_mass), wp.to_torch(self.m.body_inertia)
        left, right = model.body("left_wrist_yaw_Link").id, model.body("right_wrist_yaw_Link").id
        # drawn after everything else: the other parameters of a world are what they were before these existed
        if cfg.link_mass_scale[0] != 1.0 or cfg.link_mass_scale[1] != 1.0:
            factor = spread(unit((N, model.nbody)), cfg.link_mass_scale).to(dev)
            mass.mul_(factor)
            inertia.mul_(factor[:, :, None])
        if cfg.left_hand_mass > 0:
            bare_mass, bare_x = float(model.body_mass[left]), float(model.body_ipos[left, 0])
            drawn = spread(unit(keep_nominal=False), cfg.left_hand_mass_range)
            target = torch.where(nominal, torch.full_like(drawn, cfg.left_hand_mass), drawn).to(dev)
            # the centre moves out with the added mass: bare hand at its own, the full gripper at ``left_hand_com_x``
            x = bare_x + (cfg.left_hand_com_x - bare_x) * ((target - bare_mass) / (cfg.left_hand_mass - bare_mass)).clamp(0.0, 1.5)
            ratio = target / mass[:, left]
            mass[:, left] *= ratio
            inertia[:, left] *= (ratio * (x / bare_x) ** 2)[:, None]
            wp.to_torch(self.m.body_ipos)[:, left, 0] = x
        if cfg.right_hand_mass_delta[0] != 0.0 or cfg.right_hand_mass_delta[1] != 0.0:
            extra = torch.where(nominal, torch.zeros(N), spread(unit(keep_nominal=False), cfg.right_hand_mass_delta)).to(dev)
            factor = (mass[:, right] + extra).clamp_min(0.05) / mass[:, right]
            mass[:, right] *= factor
            inertia[:, right] *= factor[:, None]
        # Actuators between the MJCF's and the ones identified on the robot (drawn last, as above).
        self.fit_kp, self.blend = None, torch.zeros(N, device=dev)
        if cfg.actuator_fit:
            fitted = json.loads(Path(cfg.actuator_fit).read_text())["params"]["joints"]
            lo, hi = cfg.actuator_blend
            u = unit(keep_nominal=False)
            lam = torch.where(nominal, torch.zeros(N), spread(u, cfg.actuator_blend))
            self.blend = lam.to(dev)
            self.ctx = torch.cat((self.ctx, (((lam - 0.5 * (lo + hi)) / max(0.5 * (hi - lo), 1e-9))[:, None]).to(dev)), -1)
            self.fit_kp = torch.ones(a3.NJ, device=dev)
            arrays = {"damping": wp.to_torch(self.m.dof_damping), "friction": wp.to_torch(self.m.dof_frictionloss),
                      "armature": wp.to_torch(self.m.dof_armature)}
            for name, factors in fitted.items():
                j = a3.JOINTS.index(name + "_joint")
                dof = int(self.ix["dof"][j])
                for key, array in arrays.items():
                    # never exactly zero: the solver's friction rows of a joint exist for every world or for none
                    array[:, dof] *= (1.0 + self.blend * (float(factors.get(key, 1.0)) - 1.0)).clamp_min(1e-3)
                self.fit_kp[j] = float(factors.get("kp", 1.0))
            alpha = alpha * (1.0 + self.blend[:, None] * (self.fit_kp[None] - 1.0))
        # kp' = kp * alpha; kd' = kd_message * beta (the MJCF passive joint damping stays as it is, or is blended above).
        kp = torch.as_tensor(a3.KP, dtype=torch.float32, device=dev)[None] * alpha
        kd = torch.as_tensor(a3.KD, dtype=torch.float32, device=dev)[None] * beta
        self._gain, self._bias = wp.to_torch(self.m.actuator_gainprm), wp.to_torch(self.m.actuator_biasprm)
        self.kp, self.kd, self.friction = kp, kd, mu
        self.set_gains(kp, kd)

    def set_gains(self, kp, kd):
        """PD gains (N, 31) of the next ticks; ``self.kp``/``self.kd`` stay the worlds' policy gains."""
        self._gain[:, :, 0] = kp
        self._bias[:, :, 1] = -kp
        self._bias[:, :, 2] = -kd

    # ------------------------------------------------------------------ stepping
    def _tick(self):
        for _ in range(4):
            mjw.step(self.m, self.d)

    def step(self, qdes, kp=None, kd=None):
        """Advance one 20 ms control tick with PD targets ``qdes`` (N, 31); ``kp``/``kd`` replace
        the worlds' policy gains for this tick (the entry handoff)."""
        self.ctrl.copy_(qdes)
        if kp is not None:
            self.set_gains(kp, kd)
            self._custom_gains = True
        elif getattr(self, "_custom_gains", False):
            self.set_gains(self.kp, self.kd)
            self._custom_gains = False
        with wp.ScopedDevice(self.wp_device):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._tick()

    # ------------------------------------------------------------------ state access
    def state(self):
        return torch.cat((
            self.qpos[:, :7], self.qvel[:, :6], self.qpos[:, self.qids], self.qvel[:, self.vids],
        ), -1)

    def set_state(self, ids, s):
        """Write full states ``s`` (len(ids), 75) and clear warm-start accelerations."""
        self.qpos[ids, :7] = s[:, :7]
        self.qvel[ids, :6] = s[:, 7:13]
        self.qpos[ids[:, None], self.qids[None]] = s[:, 13:13 + NJ]
        self.qvel[ids[:, None], self.vids[None]] = s[:, 13 + NJ:STATE]
        wp.to_torch(self.d.qacc_warmstart)[ids] = 0.0
