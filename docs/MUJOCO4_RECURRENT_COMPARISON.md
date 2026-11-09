# MuJoCo 4.0 recurrent physical-ball comparison

This is a short, exploratory comparison of the same frozen model_19000 actor and
PPQ004 bank under three engine settings: MuJoCo 3.17.0, MuJoCo 4.0.0 with its
default contact handling, and MuJoCo 4.0.0 with speculative contacts enabled.
It does not rerun the published eight-checkpoint comparison or formal Gate3.
The original model bundle and its pinned `mujoco==3.17.0` requirements stay intact.

The optional `--speculative-contacts` switch is supported only by
`mujoco_eval_recurrent.py --physical-ball` on MuJoCo 4.0 or newer. It does not
affect the native runner or the fixed-home-100 scorer. The actor, bank, seed,
control interval, 1 ms physical-ball timestep, and contact geometry stay the
same across the three cases. Each output records the MuJoCo version and flag.

Provide two isolated Python environments. In each, install the bundle's pinned
NumPy, ONNX Runtime and PyYAML versions; install `mujoco==3.17.0` in one and
`mujoco==4.0.0` in the other. MuJoCo 4.0 splits its Python distribution into
`mujoco-core` and `mujoco-studio`; use a clean environment instead of upgrading
an existing 3.x installation in place. The script installs nothing.

From this branch's repository root:

```bash
python hope_training/whole_body_tracking/scripts/compare_mujoco4_recurrent.py \
  --python-317 /path/to/mujoco317/bin/python \
  --python-400 /path/to/mujoco400/bin/python \
  --bundle a3_deploy/a3_deploy_example/models/model_19000/policy/exported \
  --output-dir /path/outside/git/mujoco4-comparison \
  --seconds 8 --seed 0
```

The output directory must be new. It retains per-case verification and
simulation logs, the evaluator's results, and `comparison.json`. Verification
checks the frozen ONNX sequence, recurrent reset, observation and model IO before
each simulation. A failed case remains visible and makes the command exit nonzero.
The driver sets `PYTHONUTF8=1` so UTF-8 configuration files also load on Windows.

Speculative contacts may apply force **before geometric overlap**. Therefore a
positive-force contact in this mode can occur earlier than the historical
contact event, and the existing contact count and legal-return logic are only
exploratory. The PPQ004 table impulse override is not used by this three-case
driver; it uses the physical evaluator's default soft-contact mode. A better
contact count alone is not evidence of a better policy, a valid physical return,
or a formal Gate3 pass. The 4.0 plasticity feature applies to springs and flexes;
the existing rigid racket and ball do not acquire plastic deformation.

Official references: [MuJoCo 4.0 changelog](https://mujoco.readthedocs.io/en/4.0.0/changelog.html)
and [speculative contacts](https://mujoco.readthedocs.io/en/4.0.0/computation/speculative.html).

## Controlled high-speed racket-mesh sweep

`stress_mujoco4_racket_contact.py` tests the real
`right_racket_collision` mesh and physical-ball pair without the GRU, serve
bank, or scorer. The A3 begins at its nominal initial posture. A ball starts
5 cm from the racket site on either side and travels along the site's face
normal at 5, 15, 25, 40, or 60 m/s. Each trial takes one 1, 2, 5, or 16 ms
MuJoCo step. A separate 0.25 ms *non-speculative* simulation brackets the
expected first impact; all engines use the same reference windows. No ball is
teleported at a strike clock. The script retains initial and final geometric
gaps, side of the racket, solver force and contact timing in JSON.

Run each case from the branch root with the same clean environments as above:

```bash
PYTHONUTF8=1 /path/to/mujoco317/bin/python hope_training/whole_body_tracking/scripts/stress_mujoco4_racket_contact.py --output /path/outside/git/stress/mujoco317.json
PYTHONUTF8=1 /path/to/mujoco400/bin/python hope_training/whole_body_tracking/scripts/stress_mujoco4_racket_contact.py --output /path/outside/git/stress/mujoco400.json
PYTHONUTF8=1 /path/to/mujoco400/bin/python hope_training/whole_body_tracking/scripts/stress_mujoco4_racket_contact.py --speculative-contacts --output /path/outside/git/stress/mujoco400-speculative.json
```

The 2026-11-09 run found 32 combinations where the fine-step reference
bracketed impact within the coarse step. Both default engines applied zero
racket force in that *first* step; 16 of those balls ended fully beyond and
separated from the racket, a clear tunnel. With speculative contacts, all 32
had positive force, zero ended in a clear tunnel, and no force appeared in the
eight not-yet-due combinations. By timestep, the default engines had 0, 2, 6,
and 8 clear tunnels at 1, 2, 5, and 16 ms; speculative mode had zero at each.
The default engines may detect some of the other 16 overlaps on the next step;
they should not all be called tunnels.

For example, at 25 m/s and 5 ms from the front side, both default engines let
the ball finish 7.50 cm beyond the racket site with 4.14 cm geometric clearance
and no racket force. Speculative mode applied 13.47 N and left the ball 2.40 cm
on its approach side. This is evidence of collision handling in an intentionally
extreme one-step geometry test. It does not validate restitution, spin, rally
success, 100-ball scoring, or real hardware. The 60 m/s cases exceed the
physical evaluator's 50 m/s safety bound; they are isolated here on purpose.
