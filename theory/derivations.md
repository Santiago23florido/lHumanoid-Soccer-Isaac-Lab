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
- [ ] Derive and execute the Cholesky complexity proxy `kappa`.
- [ ] Implement the fully computable LQR surrogate.
- [x] State the on-policy beta-mixing and coverage assumptions.
- [x] Use "conditional improved upper bound" framing.
