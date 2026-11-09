# 2026-11-09 — MuJoCo 4.0 recurrent comparison

Branch `experiment/mujoco4-speculative-eval`; base `ade2073a0e120bbf0cf9963289317a3e885bbf81` (`feature/stationary-gru-mujoco-eval`). Initial tree clean. Code/docs commit `2d54f99429332ce43caf8a8a42ae42357e0b818c`. Upstream `main` was fetched and local `main` was already at `7990f18`; fork `origin/main` was 88 commits ahead and one behind upstream, so neither `main` was rewritten or pushed.

## Problem and decision

MuJoCo 4.0 adds optional speculative contacts. The published model_19000 bundle pins 3.17.0 and the formal scorer depends on actual positive-force contact timing. Add a guarded switch only to the standalone recurrent physical-ball evaluator and run three short isolated cases: 3.17.0, 4.0.0 default, and 4.0.0 speculative. Keep the published bundle, native runner, PPQ004 table-impact override, and Gate3 scorer unchanged. `docs/MUJOCO4_RECURRENT_COMPARISON.md` and its link from the existing eval guide document the command and limits. The comparison driver retains per-case logs and refuses to overwrite an output directory.

## Run and evidence

RunId `mujoco4-m19000-8s-seed0-20261109`. One plant/environment per case; seed 0; 8 simulated seconds, 400 control steps, 20 ms policy interval, 20 × 1 ms physics steps per control. Python 3.13.17 on Windows; isolated environments pin NumPy 2.5.4, ONNX Runtime 1.31.1, PyYAML 6.0.3 and the case-specific MuJoCo version. The driver sets `PYTHONUTF8=1` on all child processes. The first local attempt failed in all cases before simulation because Windows' default GBK decoder could not read a UTF-8 YAML file; it is retained under `trial-8s-seed0/`, followed by a successful provisional run and the final same-source run.

Exact final command, from this repository root:

```powershell
python hope_training/whole_body_tracking/scripts/compare_mujoco4_recurrent.py --python-317 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\venv-317\Scripts\python.exe' --python-400 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\venv-400\Scripts\python.exe' --bundle 'a3_deploy\a3_deploy_example\models\model_19000\policy\exported' --output-dir 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\run-final-8s-seed0' --seconds 8 --seed 0
```

Frozen input hashes: PT checkpoint `744526d2e7a9fdabe990070054ec3bd8126b286bc79041f94a930b0078c257c0`; ONNX actor `f3f56fe7e0c31f94bea1645fae8d2f776dca01033f2c895d850bf5c708801dda`; PPQ004 bank `02619e747333980a6f1d48aa3a9c7896e15727cf95cd3f51eb387d7687c4b419`; physics profile `df55e82102e0d5624732ea7ea03f6d98053f45d7f036367e3df3551a1c88a6f9`. All three `physical_config.json` files record the same evaluator source SHA256 `48c96cfb32d8cecf7c305ce3e75ef179920f82f9192a517f93000292b2b7f687` and matching policy/observation/ball source hashes. Original PT is not distributed or rerun; the manifest binds this published actor to its checkpoint.

Pass criterion: all three cases pass the existing 64-step ONNX/reset/plant verification, finish 8 seconds without a numerical error, and preserve the same frozen input/seed/configuration. Observed: all passed; no falls, one initial GRU reset, four shots, two racket contacts, two table contacts, three floor contacts and two simplified legal-return labels per case. Racket/table/floor contact timestamps matched across cases. MuJoCo 4.0 default and speculative `ball_events.json` and `trace.csv` SHA256 files are identical; 3.17.0 files differ at floating-point precision while contact times and summary match. Direct read-back confirmed `mjENBL_SPECULATIVE` set in the 4.0 model (`enableflags=128`). The 3.17.0 speculative request failed with the intended version guard. `py_compile` and `git diff --check` passed.

Evidence: `C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\run-final-8s-seed0\` (outside Git), especially `comparison.json` SHA256 `c570df26aa3fd57371f8d2d8ddbb20dbbcfd1a26bb508bcb8617b61b056606c14`, each case's `verification.json`, `results.json`, `physical_config.json`, `ball_events.json`, `trace.csv`, and logs. Large generated traces and the two virtual environments stay outside Git.

## Limits

This short soft-contact scenario does not establish a high-speed tunnelling benefit, equivalence of the 3.17 and 4.0 physics engines, formal Gate3 success, a 100-ball result, or hardware behavior. Speculative forces can precede geometric contact, so a later run with changed contact counts would require separate qualification before interpreting legal returns. No training was started or changed.

## Follow-up: extreme racket-mesh sweep

RunId `mujoco4-racket-sweep-20261109`; one A3/ball plant per process, deterministic reset for each of 40 combinations per engine. `stress_mujoco4_racket_contact.py` uses the same model_19000 manifest and real racket/ball collision pair, initial ball center 5 cm from the racket site on both sides, speeds 5/15/25/40/60 m/s, and one step at 1/2/5/16 ms. A separate non-speculative 0.25 ms run brackets the first contact. It does not call the GRU, bank, physical-ball flight code, PPQ004 table override, or rally scorer. This deliberately leaves the published 1 ms evaluator contract, including its 50 m/s speed bound, untouched.

The three commands were run from this repository root with `PYTHONUTF8=1`, using respectively `C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\venv-317\Scripts\python.exe`, `...\venv-400\Scripts\python.exe`, and the same 4.0 Python with `--speculative-contacts`:

```powershell
& 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\venv-317\Scripts\python.exe' hope_training/whole_body_tracking/scripts/stress_mujoco4_racket_contact.py --output 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\stress-racket-verified\mujoco317.json'
& 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\venv-400\Scripts\python.exe' hope_training/whole_body_tracking/scripts/stress_mujoco4_racket_contact.py --output 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\stress-racket-verified\mujoco400.json'
& 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\venv-400\Scripts\python.exe' hope_training/whole_body_tracking/scripts/stress_mujoco4_racket_contact.py --speculative-contacts --output 'C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\stress-racket-verified\mujoco400-speculative.json'
```

All JSON records bind the same manifest SHA256 `9d75784b8835bf4a2726bbeaa78dad864c4db8a3008ead984e4cd576f0e7f832` and script SHA256 `79e4f69370fe44125a1d424b6efde297f268da88f8ee749f67e5f48186bd006c`. The non-speculative reference windows matched across all three cases. Of 40 combinations, 32 have reference impact within the coarse step. Both 3.17 and 4.0 default produced 0/32 positive-force contacts within that *first* step and 16 clear tunnels (final ball on far side, geometrically separated, no force); 4.0 speculative produced 32/32 positive-force contacts, zero clear tunnels, and 0/8 contacts before the reference impact window. By timestep, default clear tunnels were 0/2/6/8 at 1/2/5/16 ms, versus 0/0/0/0 with speculation. The remaining default misses end in overlap and may be resolved on the following step; they are not counted as tunnels. At 25 m/s and 5 ms from the front side, the default engines let the ball end 7.50 cm beyond the site with 4.14 cm clearance, while speculative mode applied 13.47 N and left it 2.40 cm on the approach side.

Evidence is outside Git at `C:\Users\ROG\Desktop\UCB\Capstone\analysis\mujoco4-compare-20261109\stress-racket-verified\`. Output SHA256: 3.17 JSON `dcec033c685f9b7629c9f319ef1fb68741ff1c8878df54bd472d56d3a1fe53c9`, 4.0 default `863729bc9730340a76ef1b2271d4a59c91e10b9a5c89b7fb9a10efe8c610bfd5`, 4.0 speculative `fb38e4afa0b6844947bd9314055ccaa18d2b1467671cde89ea97b90edd68f507`. Earlier exploratory JSON in `stress-racket/` and `stress-racket-final/` is preserved; the verified directory is authoritative.

This proves the option prevents complete one-step passage in these constructed configurations, not that the soft contact's restitution, spin, or legal-return labels are calibrated. It is not a full rally or Gate3 result. No model weights, training, hardware process, or published eight-checkpoint evidence changed.
