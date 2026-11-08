# 2026-11-07 — four exact GRU exports, prepared; container access blocked

Repository `1786089174zth-cell/HOPE`; existing branch `feature/stationary-gru-mujoco-eval`, initial HEAD `0a1e22c0ae0cc73259671e7882eff087850910d2`. Origin fetch/push both `https://github.com/1786089174zth-cell/HOPE.git`. Initial worktree clean; no applicable AGENTS.md found. Fetch failed with `Could not resolve host: github.com`, so live remote divergence is unavailable. No remote changes were integrated.

Run `gru-formal-2b8c09938a`; training source `33c609022efbb4e85fe6a58a5a8947b0fbd46c76`, confirmed against the original formal-handoff provenance. Exact checkpoint files13000/14000/14600/20000 exist. Their non-executing pickle opcode inspection confirms saved iteration and run ID; ZIP CRC and before/after inode/size/mtime stability checks pass. Full checkpoints and their saved parameters were snapshotted outside Git at `~/workspace/hope-stationary-gru-evaluations/20261107-four-checkpoint-publication/`. No original is changed.

| Iteration | Checkpoint SHA256 |
|---|---|
|13000|`d9c0ea9c3401a40fb958c14e531a197544d3fcb2bfc57b981443adcbf240b41e`|
|14000|`91894d52862ac43085226a070148a55b3e9ca2b1551e3aa2a7a10558e2d59f1d`|
|14600|`f32bc833023e38bb2690b46865a1979bd05d8c946e54a50f162d59b1369ca43c`|
|20000|`9493981085baa94cfa4c6a592e336efb35bb609ea5ae2ae1a34c6184fd08a46b`|

No cached verified exports for these exact iterations were found. The established exporter needs the existing grasping container. Podman is blocked in this sandbox: `set sticky bit on: chmod /run/user/1001/libpod: read-only file system`. No chmod/permission workaround, container creation, dependency installation or host Torch substitution was attempted. Host validated venv has NumPy/ONNX Runtime but not Torch/ONNX. Therefore the four actual actor exports, exact initializer checks and fresh PyTorch/ONNX parity/reset checks are **pending**, not claimed passed. No new model directories/download links are fabricated.

Added an opt-in four-export publisher and focused tests. Three tests pass: rejection of relabelled actor evidence, standalone64-step/reset verification and full package/extracted-ZIP verification using explicitly labelled temporary model19000 packaging fixtures. These are publisher checks, not verification of the four pending actors. Existing model19000/29999 and all earlier results/videos remain unchanged; no simulation, behavior comparison, training, ROS or video operation is invoked.

Desktop continuation: `bash "$HOME/workspace/hope-stationary-gru-evaluations/20261107-four-checkpoint-publication/publish.sh"`. It uses the existing host venv and existing grasping only; prechecks container existence to prevent Distrobox creation fallback. It independently verifies original/snapshot/config hashes, calls the established CPU exporter and initializer verifier for each actual checkpoint, checks64 sequential ONNX outputs against its own fresh PyTorch vectors and reset repeat, and checks the extracted portable ZIP outside the repo. A failed stage stops without old-actor fallback. Completed verified exports/packages can be resumed; unrelated edits are preserved by stopping.

Only after all four pass does it add their actual portable packages under the established model layout, update shared README/model-index download/usage links, and replace this pending worklog with actual ONNX/ZIP hashes and parity errors. Runtime/normalization/joint/action/actuator/control contracts, dependency versions, source/config hashes and provenance are included; large PT/full private training YAML stay local. Readmes say `export verified; no new behavioral evaluation performed` only after checks succeed. No performance claims or videos are added.

The continuation inspects remote divergence before editing and again before push, stopping if remote changes are absent locally; no silent merge/rebase/reset. It reviews/stages only authorized package/docs paths, checks full diff/sizes/hashes, prints repository/branch/origin/destination/commits/files, uses normal local commits/push with existing hooks, and checks `ls-remote` against HEAD before claiming publication. This preparation commit contains only the publisher, its tests and this worklog; actual bundle publication is not complete.
