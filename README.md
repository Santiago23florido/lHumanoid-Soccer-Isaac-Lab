# lagrangian-mbrl-franka

**Physics-guided, sample-efficient model-based reinforcement learning for the
Franka Emika Panda arm in NVIDIA Isaac Lab.**

Learn the arm's dynamics with an *energy-conserving, structured* network
(**Deep Lagrangian Networks**) and use it as the predictive model inside a
model-based RL loop for reach/manipulation tasks — then show, both empirically
and with a sample-complexity bound, that the physical prior improves sample
efficiency over model-free and unstructured model-based baselines.

> Research code accompanying work targeted at **L4DC / CoRL**. See
> [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full research plan.

---

## Project overview

Model-based RL (MBRL) is sample-efficient because it reuses each real transition
many times inside a learned dynamics model — but its performance is gated by the
*accuracy* of that model. Standard practice learns dynamics with unstructured
MLP ensembles that ignore the known physics of rigid-body systems and can drift
energetically over long rollouts.

This project replaces the unstructured model with a **Deep Lagrangian Network
(DeLaN)**, which embeds the Euler–Lagrange equations
`M(q) q̈ + c(q, q̇) + g(q) = τ` directly into the network (with a guaranteed
symmetric positive-definite mass matrix `M(q)`). The hypothesis is that this
physical prior shrinks the model class, improving data efficiency and rollout
stability inside the RL loop.

## Scientific motivation

- **Why structure?** Rigid-body dynamics live in a constrained function class.
  Encoding that constraint reduces the statistical complexity of the model and,
  we argue, tightens the model-based policy-optimization error bound.
- **Why a bound?** We aim for a *conditional sample-complexity result* showing
  the structured model class needs `O(N/κ)` samples for the same model error,
  with `κ ≥ 1` the capacity reduction from the physics constraints (see
  [`PROJECT_PLAN.md` §2](PROJECT_PLAN.md) and [`theory/`](theory/)).
- **Why Franka + Isaac Lab?** A realistic 7-DoF manipulator at GPU-accelerated
  scale, with strong existing model-free baselines to compare against.

## Method summary

1. **Dynamics models** (swappable behind one interface):
   - `DeepLagrangianNetwork` — structured, energy-consistent.
   - `MLPDynamics` (ensemble) — unstructured baseline.
2. **MBRL loop:** collect transitions → fit dynamics → plan/optimize a policy
   *inside* the learned model (MPC: MPPI/CEM, or Dyna/MBPO-style) → act → repeat.
3. **Evaluation:** sample-efficiency curves (return vs. env steps,
   steps-to-threshold), model accuracy (1-step & H-step MSE), physical
   consistency (energy drift), all over ≥5 seeds.

## Repository structure

```
lagrangian-mbrl-franka/
├── README.md                 # this file
├── PROJECT_PLAN.md           # full theory→implementation→publication plan
├── CHANGELOG.md
├── LICENSE                   # MIT
├── pyproject.toml            # package + dependencies (src/ layout)
├── requirements.txt          # loosely pinned deps
├── src/lagrangian_mbrl/      # the Python package
│   ├── models/               # DeLaN + MLP dynamics models
│   ├── mbrl/                 # model-based RL training loop
│   ├── envs/                 # Isaac Lab Franka task wrappers
│   ├── eval/                 # sample-efficiency & physics metrics
│   └── utils/                # seeding, logging, config helpers
├── scripts/                  # train.py / evaluate.py entry points
├── configs/                  # Hydra/YAML experiment configs
├── tests/                    # unit tests (PD mass matrix, metrics, ...)
├── docs/                     # setup, architecture, experiment protocol
├── theory/                   # derivations, proof sketches, references.bib
├── figures/                  # generated paper figures (git-ignored content)
└── logs/                     # run logs / checkpoints (git-ignored content)
```

## Installation

> **Target platform:** Windows 11 or Linux with an NVIDIA GPU. The validated
> Windows setup uses an RTX 4070 Laptop GPU (8 GB), Conda, CUDA-enabled
> PyTorch 2.7, Isaac Sim 5.1, and Isaac Lab 2.3.2. See
> [`docs/setup_guide.md`](docs/setup_guide.md) for the detailed, step-by-step
> guide (this is a summary).

1. **Create the Conda environment:**
   ```powershell
   conda env create -f environment-windows.yml
   conda activate env_isaaclab
   ```
2. **Install Isaac Sim 5.1 + Isaac Lab 2.3.2.** Follow the exact Windows
   commands in [`docs/setup_guide.md`](docs/setup_guide.md).
3. **This package** (into the same environment Isaac Lab uses):
   ```powershell
   python -m pip install -r requirements-windows-isaacsim.txt
   python -m pip install -e ".[rl,hpo,dev]"
   ```
4. **Verify:**
   ```powershell
   python -c "import lagrangian_mbrl; print(lagrangian_mbrl.__version__)"
   python scripts/smoke_isaac_cuda.py --headless
   pytest -q
   ```

> Isaac Sim/Isaac Lab are **not** installed by `pip install -e .` — they have
> their own installer and are intentionally left out of `requirements.txt`.

## Running training

```bash
# Train the model-based method with the Lagrangian dynamics model:
python scripts/train.py experiment=dln_mbrl

# Train an unstructured MLP-ensemble MBRL baseline:
python scripts/train.py experiment=dln_mbrl model=mlp

# Train the validated model-free PPO baseline with RSL-RL:
python scripts/train_rl.py --headless --num_envs 256 --max_iterations 1000
```

Configs are composed from [`configs/`](configs/) (Hydra-style overrides). Each
run records its resolved config, git SHA, and seed.

## Running evaluation

```bash
python scripts/evaluate.py --checkpoint logs/<run_id>/checkpoints/best.pt \
                           --metrics sample_efficiency rollout_mse energy_drift
```

## Reproducing experiments

The full benchmark matrix, seeds, ablations, and figure-generation procedure are
specified in [`docs/experiments_protocol.md`](docs/experiments_protocol.md).
In short: every figure regenerates from raw logs via a single script, runs use
≥5 seeds with 95% CIs, and the resolved config + environment snapshot are saved
per run. See also [`PROJECT_PLAN.md` §5–6](PROJECT_PLAN.md).

## Citation

If you use this code, please cite (placeholder — update on submission):

```bibtex
@misc{lagrangian_mbrl_franka_2026,
  title        = {Sample-Efficient Model-Based RL with Lagrangian Dynamics
                  Priors for Robotic Manipulation},
  author       = {Santiago},
  year         = {2026},
  note         = {Preprint / under review},
  howpublished = {\url{https://github.com/<user>/lagrangian-mbrl-franka}}
}
```

## License

[MIT](LICENSE).
