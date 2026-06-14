# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Executable theory and Phase-1 handoff.**
  - One-step MSE/RMSE convention, finite-horizon control bound, beta-mixing
    on-policy assumption, and conditional (not separation) claim framing.
  - Executable Cholesky complexity counts and a mechanical LQR surrogate whose
    generated constants/results are embedded in the committed theory PDF.
  - Matched multi-budget sample-complexity sweep with empirical `kappa`, smoke
    configuration, and a Windows PowerShell launcher for the next long runs.
- **Windows 11 + CUDA deployment path.**
  - Conda environment and Isaac Sim 5.1 compatibility pins.
  - Headless CUDA/PhysX smoke test and Windows-safe Kit shutdown.
  - RSL-RL 5.0 configuration and conservative defaults for 8 GB GPUs.
- **Model-free RL track — Franka reach in Isaac Lab (PPO / RSL-RL).**
  - `rl/franka_reach_env_cfg.py`: the RL problem definition — a Franka Panda
    reach task specializing Isaac Lab's `ReachEnvCfg` (joint-position control,
    end-effector pose-tracking reward), with `num_envs` tuned for an 8 GB GPU,
    plus a small `_PLAY` variant for evaluation/video.
  - `rl/agents/rsl_rl_ppo_cfg.py`: recommended PPO agent — actor-critic MLP
    `[256, 128, 64]` (ELU) with adaptive-KL PPO hyper-parameters.
  - `rl/tasks.py`: registers `LMBRL-Franka-Reach-v0` / `-Play-v0`.
  - `scripts/train_rl.py` / `scripts/play_rl.py`: self-contained Isaac Lab
    launchers to train and visualize the PPO policy.
  - `tests/test_rl_env_cfg.py`: config-wiring tests (skip without Isaac Lab).
  - `docs/rl_quickstart.md`: how to train/play the RL baseline.
- **Phase 0 — DeLaN implementation & offline reproduction (simulator-free).**
  - `models/deep_lagrangian_network.py`: full DeLaN forward pass — Cholesky-
    parameterized PD mass matrix `M(q) = L Lᵀ + εI`, potential-energy head
    `V(q)`, Coriolis/gravity via the autograd Euler–Lagrange identity
    `c + g = (∂²L/∂q̇∂q) q̇ − ∂L/∂q` (double-backward JVP), forward/inverse
    dynamics, a symplectic one-step integrator, input normalization, and an
    inverse-/forward-dynamics training loss.
  - `models/mlp_dynamics.py`: the unstructured baseline (probabilistic head,
    normalization, NLL/MSE loss, ensemble) with `acceleration`/`delta` targets.
  - `envs/analytic_systems.py`: closed-form `Pendulum` and `TwoLinkArm`
    rigid-body systems with exact `M, c, g` — the simulator-free source of
    `(q, q̇, τ, q̈)` offline data and the unit-test oracle.
  - `eval/metrics.py`: implemented `acceleration_mse`, `rollout_mse`,
    `energy_drift`, `mass_matrix_min_eigenvalue`, and the sample-efficiency
    helpers.
  - `offline.py` + `scripts/fit_dynamics_offline.py`: fits DeLaN and the MLP on
    the same offline data and reports held-out one-step acceleration MSE — the
    Phase-0 exit criterion. DeLaN beats the MLP by ~5–11× on the 2-link arm.
  - Tests: real (non-xfail) DeLaN physics tests (PD mass matrix, forward/inverse
    consistency, energy conservation, differentiability), analytic-system and
    metric tests, and the Phase-0 DeLaN-vs-MLP comparison test.
- Initial repository scaffold: `src/` package layout, configs, tests, docs,
  theory, and CI-friendly tooling.
- `PROJECT_PLAN.md` — full theory→implementation→publication research plan.
- `README.md` — overview, motivation, method, install, usage, reproduction.
- Placeholder modules with docstrings/TODOs:
  - `models/deep_lagrangian_network.py` (energy-conserving dynamics).
  - `models/mlp_dynamics.py` (unstructured baseline + ensemble).
  - `mbrl/training_loop.py` (model-based RL loop).
  - `envs/franka_reach_wrapper.py` (Isaac Lab Franka reach task wrapper).
  - `eval/metrics.py` (sample-efficiency & physics-consistency metrics).
- `docs/`: setup guide, architecture/design doc, experiments protocol.
- MIT `LICENSE`, `.gitignore`, `pyproject.toml`, `requirements.txt`.

## [0.1.0] - 2026-06-01

### Added
- Project inception.

[Unreleased]: https://github.com/<user>/lagrangian-mbrl-franka/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/<user>/lagrangian-mbrl-franka/releases/tag/v0.1.0
