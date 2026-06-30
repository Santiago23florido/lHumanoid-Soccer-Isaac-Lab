# Experiments Protocol

How to run and reproduce every result at the current development checkpoint.
The project is at the **PINN-training stage**: the physics-informed dynamics
model (DeLaN) has been trained on a simulated 7-DoF Franka arm and validated
against an unstructured MLP baseline.

---

## Current checkpoint (2026-06-30)

The following experiments are implemented and reproducible:

| Experiment | Script | Output |
|---|---|---|
| Offline Phase-0 fit (2-link or pendulum) | `scripts/fit_dynamics_offline.py` | `logs/phase0/` |
| **PINN training on 7-DoF Franka simulator** | `scripts/train_pinn.py` | `logs/pinn/`, `figures/` |
| Theory constants (κ proxy, Cholesky ratio) | `scripts/generate_theory_constants.py` | `theory/` |
| LQR mechanical surrogate | `scripts/run_lqr_surrogate.py` | `theory/` |

---

## 1. PINN training on the simulated Franka arm

This is the primary result at the current checkpoint.

### Run

```powershell
# Default: 7-DoF Franka simulator, 1024 training samples, 600 epochs
python scripts/train_pinn.py

# Quick smoke test
python scripts/train_pinn.py --epochs 20 --n-train 128 --quiet

# Custom output directory
python scripts/train_pinn.py --out-dir logs/pinn_run1
```

### What it does

1. Generates `(q, q̇, τ, q̈)` transitions from `FrankaAnalytic7DoF` — a
   planar 7-link serial arm with Franka Panda physical parameters and exact
   analytic Lagrangian dynamics.
2. Trains the **Deep Lagrangian Network (DeLaN / PINN)** with inverse-dynamics
   loss (torque MSE) on the training split.
3. Trains an **unstructured MLP** baseline of comparable capacity on the same
   split.
4. Evaluates both on a held-out test set; reports one-step acceleration RMSE
   (rad/s²) per joint and overall.

### Outputs

| File | Description |
|---|---|
| `logs/pinn/pinn_results.json` | All numbers (params, RMSE per joint, history) |
| `figures/pinn_loss_curves.png` | Train loss and test RMSE vs. epoch |
| `figures/pinn_accel_scatter.png` | True vs. predicted `q̈` scatter (4 joints) |
| `figures/pinn_energy.png` | Energy conservation under unforced rollout |

### Metrics to report

- **Test acceleration RMSE** (rad/s²) — primary metric, one number per model.
- **Per-joint RMSE** — shows which joints benefit most from physical structure.
- **Energy drift** — `|E(t) − E(0)|` over 500 unforced steps; DeLaN near-zero.
- **Network size** — parameter count for DeLaN and MLP (from JSON results).
- **M(q) minimum eigenvalue** — confirms strict positive-definiteness.

---

## 2. Phase-0 offline fit (2-link arm)

Validates DeLaN on the simpler 2-DoF system used in the original Phase-0 exit
criterion.

```bash
python scripts/fit_dynamics_offline.py
python scripts/fit_dynamics_offline.py --system pendulum --n-train 64
```

---

## 3. Theory experiments

```powershell
# Complexity proxy κ and Cholesky dimension ratio
python scripts/generate_theory_constants.py

# LQR mechanical surrogate (linear system bound verification)
python scripts/run_lqr_surrogate.py
```

---

## 4. Reproducibility requirements

- Fix and record the random seed (`--seed`); default is 42 for the PINN script.
- Results in `logs/pinn/pinn_results.json` include the seed, epoch count, and
  architecture config.
- Figures regenerate deterministically from the same seed; delete `logs/pinn/`
  and rerun to get a fresh run.

---

## Future experiments (not yet implemented)

These belong to the next development phases:

- Full MBRL outer loop with online data collection.
- Model-free RL baselines (PPO, SAC) on the Franka reach task.
- Full benchmark matrix (≥5 seeds, 95% bootstrap CIs via `rliable`).
- Policy optimization inside the learned model (MPPI/CEM or Dyna-MBPO).
- Isaac Lab integration for real simulation data.
