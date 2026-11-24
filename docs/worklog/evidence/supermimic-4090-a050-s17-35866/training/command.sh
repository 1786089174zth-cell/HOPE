#!/usr/bin/env bash
set -euo pipefail
cd /home/tianhe/ppq004-view/HOPE/hope_training/supermimic_hope
export CUDA_VISIBLE_DEVICES=0
exec /home/tianhe/ppq004-view/.supermimic-venv/bin/python -u train.py device=cuda:0 seed=17 num_envs=6144 actor_architecture=dual_body_mlp_v1 learn.body_reward_alpha=0.5 frames=10 buffer_steps=640 warmup_steps=64 wm_only_iterations=200 utd=0.4 learn.batch=1024 learn.horizon=32 learn.wm_batch=2048 learn.wm_batches=10 learn.wm_window=8 learn.critic_epochs=5 learn.critic_minibatches=4 learn.lr=0.0002 learn.gamma=0.99 learn.lam=0.95 learn.grad_norm=1.0 learn.grad_value=0.1 action_noise=0.1 ema_decay=0.99 lr_decay_iters=250000 capture=true amp=false log_interval=25 save_interval=500 max_iterations=2147483647 keep_periodic=3 run_name=supermimic-long-a050-s17-20261124-01 out=/home/tianhe/ppq004-view/supermimic-runs/supermimic-long-a050-s17-20261124-01
