"""SuperMimic (arXiv 2610.28480) training for the HOPE A3 ping-pong policy.

World-model + BPTT training adapted from github.com/dongc1/supermimic_vibed
(MIT). MuJoCo-Warp supplies physics; the existing HOPE managers own the four
strokes, question bank and rewards. Kit currently bootstraps those managers.
The actor retains 110 inputs and 31 outputs; hardware qualification is separate.
"""
