"""Model-free RL track for the Franka Panda *reach* task (Isaac Lab).

This subpackage frames the project as a standard reinforcement-learning problem
— a Franka 7-DoF arm learning to drive its end-effector to a commanded pose —
and provides everything needed to *train* a policy with PPO (RSL-RL) on top of
Isaac Lab:

- :mod:`lagrangian_mbrl.rl.franka_reach_env_cfg` — the **task / robot design**:
  the simulated scene (Franka articulation, table, ground, lights), the
  observation/action spaces, the reward, and the termination conditions, tuned
  for a single 8 GB GPU.
- :mod:`lagrangian_mbrl.rl.agents.rsl_rl_ppo_cfg` — the **PPO agent**: a
  recommended actor-critic MLP and PPO hyper-parameters.
- :mod:`lagrangian_mbrl.rl.tasks` — Gymnasium environment registration.

.. warning::
   Importing anything in this subpackage pulls in Isaac Lab / Isaac Sim, which
   must be installed and (for ``tasks``) launched first. It is therefore **not**
   imported by the top-level :mod:`lagrangian_mbrl` package, so the rest of the
   library (and its tests) stays importable without a simulator.

Train with::

    python scripts/train_rl.py --task LMBRL-Franka-Reach-v0 --headless --num_envs 256
"""
