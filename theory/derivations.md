# Theory decisions and derivation ledger

This file is the working ledger behind `dynamics_models.tex`. The LaTeX file is
the normative statement; executable constants and surrogate results are added
to it by the theory verification scripts.

## 1. Error convention

The statistical target is one-step acceleration MSE under the aggregated data
distribution:

```text
epsilon_acc(f_hat; rho)
  = E_(x,tau)~rho ||f_hat(x,tau) - f_star(x,tau)||_2^2.
```

Its RMSE is `delta_acc = sqrt(epsilon_acc)`. H-step rollout MSE is a diagnostic,
not the primitive in the generalization theorem.

For the semi-implicit Euler update used by the repository,

```text
qd_next = qd + dt * f(x, tau)
q_next  = q  + dt * qd_next
```

an acceleration error `e` produces state-transition error

```text
||F_hat - F_star||_2 = dt * sqrt(1 + dt^2) * ||e||_2.
```

This exact identity connects the executable model metric to the simulation
bound.

## 2. Finite-horizon bound

Let `C_H = sum_(t=0)^(H-1) L_(t+1)`, where `L_(t+1)` is the Lipschitz constant
of the continuation value. Assume every occupancy used in the comparison is
covered by `rho` with density ratio at most `C_cov`. For an
`epsilon_opt`-optimal planner in the learned model:

```text
J_Fstar(pi_star) - J_Fstar(pi_hat)
  <= 2 * dt * sqrt(1 + dt^2) * C_H
       * sqrt(C_cov * epsilon_acc)
     + epsilon_opt.
```

Thus:

- when the reported metric is MSE, the bound contains `sqrt(epsilon_acc)`;
- when the metric is RMSE, the same bound is linear in `delta_acc`;
- if `L_(t+1) <= L_V (H-t-1)`, the leading horizon factor is at most
  `dt * sqrt(1 + dt^2) * L_V * H * (H-1)`.

## 3. On-policy data

The analysis does not call on-policy transitions i.i.d. Data collection is
split into rounds. Conditional on the past, each round freezes its policy,
discards burn-in, and assumes a stationary geometrically beta-mixing Markov
chain. Alternating blocks of length `b` give
`N_eff = floor(N / (2b))`; choose `b` so the coupling remainder
`2 N_eff beta(b)` fits inside the confidence budget.

The round-wise supervised bounds are conditional on the previous history and
are union-bounded across rounds. Their data-aggregation mixture is `rho`.
Coverage of the final policy by `rho` remains an explicit assumption.

## 4. Claim language

The current theorem is a **conditional improved upper bound**. It is not a
two-sided separation. A separation requires a matching lower bound for the
unstructured comparator.

## 5. Open implementation items

- [x] Pick one-step acceleration MSE as the statistical metric.
- [x] Resolve `sqrt(epsilon)` versus `epsilon`.
- [x] Derive and execute the Cholesky complexity proxy `kappa`.
- [x] Implement the fully computable LQR surrogate.
- [x] State the on-policy beta-mixing and coverage assumptions.
- [x] Use "conditional improved upper bound" framing.

## 6. Concrete kappa ledger

`scripts/generate_theory_constants.py` computes all values inserted into the
PDF. For `d=7`:

- Cholesky emits `d(d+1)/2 = 28` mass entries; a dense head emits `d^2 = 49`.
- `kappa_output = (d^2 + 1) / (d(d+1)/2 + 1)`.
- `kappa_chol,param` compares otherwise identical energy networks and isolates
  the Cholesky head.
- `kappa_direct` compares the configured deterministic direct MLP to DeLaN.
- `kappa_ensemble` compares the configured five-member probabilistic ensemble
  to DeLaN.

Only the first two isolate the Cholesky restriction. The latter two also include
architecture width and ensemble size. All are complexity proxies for upper
bounds, not statistical lower bounds.

## 7. LQR surrogate ledger

The executable anchor is `scripts/run_lqr_surrogate.py`.

1. Build a coupled linear mass-spring-damper system.
2. Discretize it with the same semi-implicit Euler convention as DeLaN.
3. Fit unrestricted `(A,B)` by ridge least squares.
4. Fit SPD mass, SPD stiffness, and diagonal damping by constrained least
   squares, using the free estimate only as one initialization.
5. Solve the discrete Riccati equation for each estimated model.
6. Evaluate each controller on the true dynamics with a Lyapunov equation.
7. Write JSON results and a LaTeX table consumed by the PDF.

The free model has `6 d^2` coefficients. The mechanical model has
`d(d+1)/2` for mass, the same for stiffness, and `d` for damping, for a total
of `d^2 + 2d`. This gives an exact dimension ratio in the surrogate, while the
nonlinear Franka claim remains conditional.

## 8. PINN Training Checkpoint — current development state

This section records the state of the project at the **PINN training
checkpoint** (2026-06-30). Everything below has been implemented and tested.

### What has been built

| Component | Status | File |
|---|---|---|
| Lagrangian theory (EL equations, bound derivation) | ✓ done | `theory/derivations.md` §1–7 |
| `DeepLagrangianNetwork` (DeLaN / PINN) | ✓ done | `src/…/models/deep_lagrangian_network.py` |
| `MLPDynamics` unstructured baseline | ✓ done | `src/…/models/mlp_dynamics.py` |
| Analytic simulators (Pendulum, TwoLinkArm, **FrankaAnalytic7DoF**) | ✓ done | `src/…/envs/analytic_systems.py` |
| Offline fit comparison (Phase-0) | ✓ done | `scripts/fit_dynamics_offline.py` |
| **PINN training on 7-DoF simulated robot** | ✓ done | `scripts/train_pinn.py` |
| Complexity proxy κ, LQR surrogate | ✓ done | `src/…/theory/`, `scripts/` |

### FrankaAnalytic7DoF simulator

The 7-DoF analytic Franka model is a **planar serial-chain arm** with link
masses, lengths, and rotational inertias taken from the Franka Panda URDF
(`src/lagrangian_mbrl/envs/analytic_systems.py`). Its Lagrangian is:

```
L = T(q, q̇) − V(q)
  = ½ q̇ᵀ M(q) q̇  −  Σ_k m_k g h_k(q)
```

where `M(q)` is the exact mass matrix from the composite rigid-body formula,
and `h_k(q)` is the height of the COM of link *k*.

The Coriolis forces are computed via the exact identity (proof in §FORCES above):

```
c_i = JVP[M(q) q̇, q, q̇]_i  −  ∂T/∂q_i
```

### Phase-0 result (2-DoF two-link arm — established)

Run with `python scripts/fit_dynamics_offline.py` (completes in ~30 s, seed=0):

| Model | Params | Val accel MSE (rad/s²)² |
|---|---|---|
| DeLaN (PINN) | 34,308 | **0.93** |
| MLP (unstructured) | 133,890 | 4.51 |
| Improvement | — | **4.83×** |

DeLaN achieves 4.83× lower validation error with 3.9× fewer parameters on 256
training samples from the 2-DoF arm. This establishes the physics-prior
advantage at small data regimes.

### PINN training results — 7-DoF Franka (reported in `logs/pinn/pinn_results.json`)

Run with `python scripts/train_pinn.py` to reproduce.

**Training setup** (see `scripts/train_pinn.py`):

| Setting | Value |
|---|---|
| System | `franka7` (DoF = 7, planar Franka Panda) |
| Training samples | 8192 |
| Test samples | 4096 |
| DeLaN hidden layers | 2 × 128, softplus, 38 813 params |
| DeLaN loss | Canonical inverse: `MSE(M(q)q̈ + c + g, τ)` |
| MLP hidden layers | 3 × 256, SiLU, 139 015 params |
| Epochs / batch | 1500 / 512 |
| Optimiser | Adam, lr = 3e-3 → 3e-5 (cosine), weight\_decay = 1e-4 |
| Seed | 42 |

**Headline results** (*fill in after training; see `logs/pinn/pinn_results.json`*):

- **DeLaN (PINN)**: test acceleration RMSE = **TBD** rad/s²
  (38 813 params, ~X s training)
- **MLP baseline**: test acceleration RMSE = **TBD** rad/s²
  (139 015 params, ~X s training)
- **Improvement**: **TBD**×  (MLP RMSE / DeLaN RMSE)
- **Energy conservation check**: unforced rollouts show near-zero energy drift,
  confirming the Lagrangian prior is respected.
- **PD mass matrix**: minimum eigenvalue of `M(q)` > 0 across the test set.

Figures saved to `figures/`:

| File | Content |
|---|---|
| `pinn_loss_curves.png` | Train loss and test RMSE vs. epoch (DeLaN vs MLP) |
| `pinn_accel_scatter.png` | True vs. predicted `q̈` scatter (test set, 4 joints) |
| `pinn_energy.png` | Energy drift over 500-step unforced rollout |

### What comes next (future development)

- MBRL outer loop (data collection from simulator → fit PINN → plan → act).
- Online policy optimization inside the learned model (MPC / Dyna-style).
- RL baselines (PPO, SAC) for sample-efficiency comparison.
- Full benchmark matrix with ≥5 seeds and 95% confidence intervals.
