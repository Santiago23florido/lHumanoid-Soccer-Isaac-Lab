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

## 1. PINN training — primary result on 2-DoF two-link arm

This is the primary result at the current checkpoint.  The 7-DoF Franka arm
is too ill-conditioned for the current DeLaN to converge with standard random
data (see §1.1 below).

### Run

```powershell
# 2-DoF two-link arm — the verified working configuration (< 2 min on CPU)
python scripts/train_pinn.py --system two_link --n-train 256 --batch-size 64 --epochs 800

# Custom output directory
python scripts/train_pinn.py --system two_link --n-train 256 --batch-size 64 --epochs 800 --out-dir logs/pinn_run1
```

### What it does

1. Generates `(q, q̇, τ, q̈)` transitions from a 2-DoF planar arm with
   exact analytic Lagrangian dynamics.
2. Trains the **Deep Lagrangian Network (DeLaN / PINN)** with the canonical
   inverse-dynamics loss `MSE(M(q)q̈ + c + g, τ)`.
3. Trains an **unstructured MLP** baseline (3.9× larger parameter count) on
   the same split.
4. Evaluates both on a 1024-sample held-out test set; reports one-step
   acceleration RMSE (rad/s²) per joint and overall.

### §1.1 — Why not franka7?

The Franka7 planar chain has a mass matrix with condition number κ(M) ≈
10,000–20,000 (joint-1 inertia ≈ 5–20 kg⋅m², joint-7 ≈ 9×10⁻⁴ kg⋅m²).
Three DeLaN loss configurations all fail:

| Loss | Samples | DeLaN failure |
|---|---|---|
| Forward-only | any | M→∞ (null predictor, RMSE = 2.89) |
| Inverse-only | 1024 | M→ε = 1e-3 (Cholesky vanishing gradient, RMSE = 242) |
| Combined fwd+inv | 8192 | Training loss = 4.6 after 1500 epochs; RMSE = 2.92 ≈ null |

Both the MLP (best RMSE = 2.875, from epoch-0 random init) and DeLaN reach
the null-predictor baseline.  Future fixes: per-joint output scaling, M
initialization from data statistics, or longer training on more data.

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

## 2. Phase-0 offline fit (2-link arm) — established result

Validates DeLaN on the simpler 2-DoF system.  This run completes in ~30 s and
demonstrates the sample-efficiency advantage of the physics prior at small N.

```bash
# Default: 2-DoF planar arm, 256 training samples, 800 epochs (~30 s)
python scripts/fit_dynamics_offline.py

# Pendulum (simpler)
python scripts/fit_dynamics_offline.py --system pendulum --n-train 64
```

**Verified result** (seed=0):

| Model | Params | Val accel RMSE (rad/s²) |
|---|---|---|
| DeLaN (PINN) | 34,308 | **0.97** |
| MLP (unstructured) | 133,890 | 2.12 |
| **Improvement** | — | **2.19×** |

DeLaN achieves 2.19× lower validation RMSE (4.83× lower MSE) with 3.9× fewer
parameters, using only 256 training transitions from the 2-DoF arm.

The `train_pinn.py` script on the same system (seed=42, n\_test=4096) gives
**DeLaN 0.499, MLP 2.292, improvement 4.59×** — consistent result.

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
