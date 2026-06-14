# Model-free RL quickstart — Franka reach (PPO / RSL-RL)

This is the model-free reinforcement-learning track: a Franka Panda arm learns
to drive its end-effector to a commanded pose, trained with PPO on top of
Isaac Lab. It complements the model-based / DeLaN track described in
[`../PROJECT_PLAN.md`](../PROJECT_PLAN.md).

> Prerequisite: a working Isaac Lab install (see
> [`setup_guide.md`](setup_guide.md)) and this package installed into that same
> environment: `pip install -e ".[rl,dev]"`. All commands assume the Isaac Lab
> conda env is active (`conda activate env_isaaclab`).

## The problem (MDP)

| Element | Definition |
|---|---|
| **Robot** | Franka Emika Panda, 7-DoF (`FRANKA_PANDA_CFG`). |
| **Action** | 7-dim scaled joint-position targets around the default pose. |
| **Observation** | joint pos & vel (relative), commanded EE pose, last action. |
| **Reward** | EE position tracking (L2 + tanh) + orientation tracking − action-rate / joint-vel penalties. |
| **Termination** | episode time-out (reach-and-hold). |

Defined in [`../src/lagrangian_mbrl/rl/franka_reach_env_cfg.py`](../src/lagrangian_mbrl/rl/franka_reach_env_cfg.py).
The PPO agent (actor-critic MLP `[256, 128, 64]`, ELU) is in
[`../src/lagrangian_mbrl/rl/agents/rsl_rl_ppo_cfg.py`](../src/lagrangian_mbrl/rl/agents/rsl_rl_ppo_cfg.py).

## Train

```bash
# Headless, conservative default for an 8 GB GPU (RTX 4070 Laptop):
python scripts/train_rl.py --headless --num_envs 256 --max_iterations 1000

# Quick smoke run (verify the loop end-to-end):
python scripts/train_rl.py --headless --num_envs 64 --max_iterations 10
```

Checkpoints + TensorBoard logs land under
`logs/rsl_rl/lmbrl_franka_reach/<timestamp>/`. Watch training:

```bash
tensorboard --logdir logs/rsl_rl
```

## Play / evaluate

```bash
python scripts/play_rl.py --headless --num_envs 16 \
    --checkpoint logs/rsl_rl/lmbrl_franka_reach/<run>/model_1000.pt
```

## VRAM tips (8 GB)

- Always run training `--headless`.
- Start at `--num_envs 256`; drop to 64-128 if you hit out-of-memory. Increase
  only after checking VRAM with `nvidia-smi`.
- Reach is easy — the stock `[64, 64]` policy also solves it if you want faster
  iterations (edit `actor.hidden_dims` / `critic.hidden_dims`).

## Tasks

| ID | Purpose |
|---|---|
| `LMBRL-Franka-Reach-v0` | full training scene |
| `LMBRL-Franka-Reach-Play-v0` | small scene (16 envs, no obs noise) for eval/video |
