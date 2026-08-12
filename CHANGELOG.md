# Changelog

## Unreleased

### Phase 1 — NAO integrated into Isaac Sim

- Vendored the NAO H25 V5.0 URDF from `ros-naoqi/nao_robot` @`6747646`,
  verbatim and BSD licensed, with upstream licenses and full attribution in
  `third_party/nao/`.
- Added `scripts/fetch_nao_meshes.py`, a native-Windows bootstrap that fetches
  the CC BY-NC-ND 4.0 geometry after explicit license acceptance. Meshes and
  every USD derived from them stay untracked; the reasoning is documented
  rather than assumed.
- Added `NAO_CFG` plus the derived-URDF and USD conversion pipeline, using
  `isaacsim.asset.importer.urdf` through the Isaac Lab `UrdfConverter`.
- Added `scripts/view_nao.py`, the canonical smoke test: minimal scene,
  free-floating articulation, diagnostics read back from the simulation.
- Added `tests/test_nao_assets.py`, which validates the asset integration
  without Isaac Sim or rendering.
- Replaced the placeholder built-in-humanoid scaffolding (`view_humanoid.py`,
  `humanoids.py`, `docs/view_humanoid.md`, `configs/robots/humanoid.yaml`) with
  the NAO equivalents. The RL task scaffold is untouched and still unimplemented.
- No reinforcement learning was added.

- Reset `main` to a humanoid soccer Isaac Lab scaffold.
- Preserved the previous Franka Lagrangian MBRL work on
  `archive/franka-lagrangian-mbrl-2026-07-25`.
