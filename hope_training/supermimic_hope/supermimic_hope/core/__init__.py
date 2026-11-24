"""Standalone SuperMimic for the HOPE four-stroke task (no Isaac at training time).

Pure MuJoCo-Warp physics, a tape-driven four-stroke rally task and the training
loop of arXiv 2610.28480 / github.com/dongc1/supermimic (Apache-2.0): fit a
differentiable dynamics model on logged transitions, improve the policy by
backpropagation through 32 imagined steps with a critic bootstrap.
"""
