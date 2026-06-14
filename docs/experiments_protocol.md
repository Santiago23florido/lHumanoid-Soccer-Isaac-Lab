# Experiments Protocol

How to run, track, and reproduce every result in the paper. The goal is that any
figure can be regenerated from raw logs by a single command, and every reported
number is seed-averaged with a confidence interval.

## 1. The benchmark matrix

| Axis | Values |
|---|---|
| **Task** | `franka_reach` (must-have); `franka_push`, `franka_peg_insert` (stretch) |
| **Method** | `dln_mbrl` (ours) · `mlp_mbrl` (PETS/MBPO-style) · `ppo` · `sac` |
| **Seeds** | ≥ 5 per cell (more if variance is high) |
| **Data-regime sweep** | vary buffer/data budget `N` to trace the empirical sample-complexity curve |

The decisive comparison is **`dln_mbrl` vs `mlp_mbrl`**: identical loop, planner,
and hyperparameters — only `model=` changes.

## 2. Metrics (see `eval/metrics.py`)

- **Primary:** sample-efficiency curve (return vs. env steps) and
  **steps-to-threshold** (env steps to reach 90% success).
- **Model accuracy:** 1-step and H-step rollout MSE on held-out transitions.
- **Physics consistency:** energy drift over unforced rollouts; min eigenvalue
  of `M(q)` for DeLaN.
- **Compute:** wall-clock and peak VRAM (must fit 12 GB).

Report **mean ± 95% CI** via stratified bootstrap (`rliable`). Never report a
single seed.

## 3. Running experiments

```bash
# Method:
python scripts/train.py experiment=dln_mbrl env=franka_reach seed=0

# Unstructured baseline (only the dynamics model changes):
python scripts/train.py experiment=dln_mbrl model=mlp env=franka_reach seed=0

# Model-free baselines:
python scripts/train.py experiment=ppo_baseline env=franka_reach seed=0

# Sweep seeds (example with a shell loop):
for s in 0 1 2 3 4; do
  python scripts/train.py experiment=dln_mbrl seed=$s
done
```

Each run writes to `logs/<project>/<experiment>/seed_<n>/<timestamp>/` with the
resolved config, checkpoints, and metric logs.

## 4. Reproducibility requirements (do all of these)

- [ ] **Seeds:** fix and record; ≥5 per cell; report CIs.
- [ ] **Metadata per run:** git SHA + dirty flag, resolved config, seed, GPU
      name, library versions (`pip freeze` snapshot per experiment batch).
- [ ] **Determinism:** call `seed_everything(seed)`; note Isaac Sim is not
      bit-deterministic — control via seed-averaging, not bit-exact reruns.
- [ ] **Fair capacity:** match parameter counts / training budget between DeLaN
      and the MLP when claiming the gap is from *structure*, not size.
- [ ] **Held-out data:** model-accuracy metrics on transitions not used to fit.
- [ ] **Frozen results:** once Phase 4 starts, no hyperparameter changes; only
      bug fixes, documented in `CHANGELOG.md`.

## 5. From logs to figures

```bash
# (Phase 4) regenerate every paper figure from raw logs:
python scripts/evaluate.py --metrics sample_efficiency rollout_mse energy_drift \
                           --runs logs/lagrangian-mbrl-franka/** \
                           --out figures/
```

Figures must be reproducible from disk — no manual plotting steps.

## 6. Ablations checklist (see PROJECT_PLAN.md §5.3)

- [ ] Structure on/off (DeLaN vs MLP) — core.
- [ ] `M(q)` parameterization (Cholesky / diagonal / full).
- [ ] Learned dissipation & actuation model on/off.
- [ ] Ensemble size; planning horizon `H`; model-update frequency.
- [ ] Data-regime sweep → empirical `κ` vs. theory's predicted `κ`.
- [ ] Contact task degradation (probes assumption A1).

## 7. Linking experiments to theory

### Phase 1 executable handoff (Windows)

First validate the complete path:

```powershell
.\scripts\run_phase1.ps1 -Smoke
```

Then launch the predeclared long sweep:

```powershell
.\scripts\run_phase1.ps1
```

The long run tunes each model once at `N=256`, freezes the selected
hyperparameters, and retrains DeLaN and the MLP ensemble at
`N = 64, 128, 256, 512, 1024, 2048` with the same five seeds. Outputs are
written under `logs/sample_complexity/<timestamp>/`:

- `sample_complexity_results.json`: configuration, per-seed errors, and
  empirical `kappa` at the predeclared MSE thresholds.
- `sample_complexity.csv`: flat analysis table.
- `sample_complexity.png`: log-log one-step acceleration MSE versus `N`.

The theory claim is supported only if the structured curve reaches a
predeclared MSE threshold with fewer transitions across seeds. A single
favorable seed or a post-hoc threshold is not evidence.

The data-regime sweep is the bridge: plot model error (or steps-to-threshold)
vs. `N` for both model classes and compare the empirical sample-complexity ratio
to the `κ` predicted by the bound in `theory/`. Agreement (even approximate) is
the paper's strongest single figure.
