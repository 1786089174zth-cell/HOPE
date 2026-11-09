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
