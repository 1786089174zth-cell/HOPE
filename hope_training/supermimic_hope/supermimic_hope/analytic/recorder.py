"""Record physical states, controls and task context at their exact env-step instants.

Within one ``env.step`` (Isaac Lab order): physics -> terminations -> rewards
-> resets -> command update -> interval events -> observations.

* ``s_end``: captured when rewards are computed (post-physics, pre-reset);
* ``s_cmd``: captured when commands update (post-reset, pre-event), which is
  the state the command caches for the next step's racket rewards;
* ``s_start``: captured after ``env.step`` returns (what observations see).
"""

from __future__ import annotations

import torch

from . import state as S


class Recorder:
    def __init__(self, env, backend, model):
        self.env, self.backend, self.model = env, backend, model
        self.action = env.action_manager.get_term("joint_pos")
        self.s_cmd = S.from_backend(backend)
        self.s_start = self.s_cmd.clone()
        self.obs = None
        self.record = None
        self.real = None
        self._reward = env.reward_manager.compute
        self._command = env.command_manager.compute

        def reward(*args, **kwargs):
            self.record = self._capture()
            out = self._reward(*args, **kwargs)
            self.real = env.reward_manager._step_reward.detach().clone()
            return out

        def command(*args, **kwargs):
            self.s_cmd = S.from_backend(backend)
            return self._command(*args, **kwargs)

        env.reward_manager.compute = reward
        env.command_manager.compute = command

    def close(self):
        self.env.reward_manager.compute = self._reward
        self.env.command_manager.compute = self._command

    def observe(self, obs):
        """Call with the observation returned by reset/step."""
        self.s_start = S.from_backend(self.backend)
        self.obs = obs["policy"].detach().clone()

    @torch.no_grad()
    def _capture(self):
        m, a = self.model, self.action
        ctx = m.context()
        ctx.update(m.observation_context())
        action_prev = self.obs[:, m.action_slice]
        clean = m.clean_obs(self.s_start, action_prev, ctx)
        queue = getattr(a, "_qdes_delay_queue", None)
        return dict(
            s_cmd=self.s_cmd.clone(),
            s_start=self.s_start.clone(),
            s_end=S.from_backend(self.backend),
            action=a.raw_actions.detach().clone(),
            action_prev=action_prev.clone(),
            qexec=a._qdes_executed.detach().clone(),
            qexec_prev=a._previous_qdes_executed.detach().clone(),
            qcmd_prev=(queue[:, 0] if queue is not None else a._qdes_executed).detach().clone(),
            obs_delta=(self.obs - clean).clone(),
            **{"ctx/" + k: v for k, v in ctx.items()},
        )
