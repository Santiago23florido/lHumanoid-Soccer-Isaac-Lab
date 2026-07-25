# Architecture

The project follows the Isaac Lab external project shape: a repository-level
project with `scripts/`, `source/`, and documentation at the root, plus a single
installable extension under `source/humanoid_soccer_lab`.

## Layers

- Project root: repository docs, placeholder configs, assets, and smoke tests.
- Extension: Isaac Sim extension metadata and the Python package.
- Modules: assets, task definitions, policy configuration, and utilities.
- Task: `tasks/direct/humanoid_soccer`, the first Direct RL environment target.

## Task Boundary

The first environment should stay single-agent and direct-RL focused. It should
own reset logic, observations, rewards, terminations, and action scaling for one
humanoid and one ball. Team play should be introduced later as a separate task
or as a multi-agent variant after locomotion and ball control are stable.

## Asset Boundary

Robot, ball, and field descriptions live under `humanoid_soccer_lab.assets` once
they become Isaac Lab asset configs. Raw USD files and generated assets belong
under the repository-level `assets/` directory.
