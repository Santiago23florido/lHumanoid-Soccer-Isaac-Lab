# lagrangian-mbrl-franka

**Physics-guided dynamics learning for the Franka Emika Panda arm.**

A **Physically-Informed Neural Network (PINN)** based on Deep Lagrangian
Networks (DeLaN) is trained to learn the exact equations of motion
`M(q) q̈ + c(q,q̇) + g(q) = τ` of a simulated 7-DoF Franka arm, with the
physical prior embedded directly in the network architecture.

---

## Current state — PINN training checkpoint (2026-06-30)

The project is at the **PINN-training checkpoint**: the Deep Lagrangian
dynamics model has been trained and beats the unstructured MLP baseline
on a 2-DoF arm; see §Note below for the 7-DoF Franka status.

**Key result (2-DoF two-link arm, 256 training samples, seed=42):**

| Model | Params | Test RMSE (rad/s²) |
|---|---|---|
| DeLaN (PINN) | 34,308 | **0.499** |
| MLP (unstructured) | 133,890 | 2.292 |
| **Improvement** | 3.9× smaller | **4.59×** lower RMSE |

**§Note — 7-DoF Franka:** The Franka7 planar chain has mass-matrix condition
number κ ≈ 10,000 (joint-1 ≈ 10 kg⋅m², joint-7 ≈ 0.001 kg⋅m²) and does
not yet converge with the current DeLaN settings; future work needs per-joint
conditioning or more training data/epochs.

### What has been implemented

- **Deep Lagrangian Network (DeLaN)** — the PINN.  Parameterizes the kinetic
  energy `T = ½ q̇ᵀ M(q) q̇` via a Cholesky-structured mass matrix `M(q) ≻ 0`
  and the potential energy `V(q)`, then derives `c(q,q̇)` and `g(q)` from the
  Euler–Lagrange equations using automatic differentiation.  This guarantees
  energy conservation and a positive-definite mass matrix by construction.

- **FrankaAnalytic7DoF simulator** — a planar 7-link serial arm with Franka
  Panda physical parameters (masses, link lengths from the URDF).  Provides
  exact, closed-form `M(q)`, `c(q,q̇)`, `g(q)` as ground truth for training
  and evaluation.

- **MLP baseline** — an unstructured 3-layer network that maps `(q, q̇, τ) → q̈`
  without any physics structure; serves as the comparison for the
  physics-prior hypothesis.

- **Theory** — a conditional sample-complexity bound showing the structured
  model class needs `O(N/κ)` transitions for the same model error, with `κ ≥ 1`
  the capacity reduction from the physics constraints.

### Quick start

```powershell
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate lagrangian-mbrl

# 2. Install the package
pip install -e ".[dev]"

# 3. Train the PINN on the 2-DoF arm (< 2 min on CPU)
#    Results saved to logs/pinn/pinn_results.json and figures/
python scripts/train_pinn.py --system two_link --n-train 256 --batch-size 64 --epochs 800

# 4. Run unit tests
pytest -q
```

---

## Scientific motivation

**Why a Physically-Informed Neural Network?**  Rigid-body manipulator dynamics
live in a highly constrained function class: the mass matrix must be symmetric
positive-definite, forces must be conservative, and the Euler–Lagrange equations
must hold.  An unstructured network that ignores these constraints must learn
all structure from data alone.

DeLaN embeds the Lagrangian `L = T - V` directly into the network so that every
prediction automatically satisfies:

- `M(q) ≻ 0` (Cholesky parameterization)
- `c(q,q̇) + g(q)` derived analytically from `L` (not learned separately)
- Energy conservation in unforced rollouts (symplectic Euler integration)

The working hypothesis: fewer data are needed to fit a correct model when the
function class already enforces the correct physics.

---

## Repository structure

```
lagrangian-mbrl-franka/
├── src/lagrangian_mbrl/
│   ├── models/
│   │   ├── deep_lagrangian_network.py  # DeLaN (PINN) — core contribution
│   │   └── mlp_dynamics.py             # MLP baseline
│   ├── envs/
│   │   └── analytic_systems.py         # Pendulum, TwoLinkArm, FrankaAnalytic7DoF
│   ├── eval/
│   │   └── metrics.py                  # acceleration MSE, energy drift, M eigenvalue
│   ├── theory/
│   │   ├── complexity.py               # κ complexity proxy
│   │   └── lqr_surrogate.py            # computable LQR bound anchor
│   ├── pipeline/
│   │   ├── data.py                     # dataset utilities
│   │   ├── registry.py                 # model registry
│   │   └── sample_complexity.py        # empirical κ sweep
│   ├── offline.py                      # Phase-0 offline fit helper
│   └── utils/
├── scripts/
│   ├── train_pinn.py                   # ★ PINN training — main entry point
│   ├── fit_dynamics_offline.py         # Phase-0 DeLaN vs MLP comparison
│   ├── run_sample_complexity.py        # empirical sample-complexity sweep
│   ├── generate_theory_constants.py    # compute κ, Cholesky ratios
│   └── run_lqr_surrogate.py            # LQR bound verification
├── tests/                              # pytest unit tests
├── theory/
│   └── derivations.md                  # Lagrangian derivations + checkpoint ledger
├── docs/
│   ├── experiments_protocol.md         # how to reproduce results
│   ├── architecture.md                 # design notes
│   └── setup_guide.md                  # environment setup
├── figures/                            # generated paper figures (git-ignored)
└── logs/                               # run logs (git-ignored)
```

---

## Running the PINN

```powershell
# Default: franka7 simulator, 8192 training samples, 1500 epochs (~25 min on CPU)
python scripts/train_pinn.py

# Outputs:
#   logs/pinn/pinn_results.json        — all metrics (RMSE per joint, params, ...)
#   figures/pinn_loss_curves.png       — train/test loss curves
#   figures/pinn_accel_scatter.png     — true vs. predicted q̈ scatter
#   figures/pinn_energy.png            — energy conservation check

# Fast smoke test
python scripts/train_pinn.py --epochs 30 --n-train 256 --quiet

# Simpler 2-DoF system
python scripts/train_pinn.py --system two_link --n-train 256
```

See [`docs/experiments_protocol.md`](docs/experiments_protocol.md) for the
full reproduction guide and metric definitions.

---

## Running theory experiments

```powershell
# Offline Phase-0 comparison (DeLaN vs MLP on 2-link arm)
python scripts/fit_dynamics_offline.py

# κ complexity proxy and Cholesky dimension ratios
python scripts/generate_theory_constants.py

# LQR mechanical surrogate (linear bound anchor)
python scripts/run_lqr_surrogate.py

# Empirical sample-complexity sweep (log-log MSE vs N)
python scripts/run_sample_complexity.py
```

---

## Tests

```powershell
pytest -q
```

The test suite covers the DeLaN model (PD mass matrix, inverse/forward
dynamics consistency), analytic systems (ground truth torque, 7-DoF system),
evaluation metrics, and the theory complexity proxies.

---

## Next steps (future development)

1. MBRL outer loop: online data collection → fit PINN → plan → act.
2. Model-free RL baselines (PPO, SAC) for sample-efficiency comparison.
3. Isaac Lab integration for GPU-accelerated simulation data.
4. Full benchmark matrix with ≥5 seeds and 95% confidence intervals.

---

## Citation

```bibtex
@misc{lagrangian_mbrl_franka_2026,
  title        = {Physics-Informed Dynamics Learning for Robotic Manipulation},
  author       = {Santiago},
  year         = {2026},
  note         = {Research code — PINN training checkpoint}
}
```

## License

[MIT](LICENSE).
