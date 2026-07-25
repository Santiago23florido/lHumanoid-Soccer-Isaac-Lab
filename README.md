# Humanoid Soccer Isaac Lab

Scaffold for an Isaac Lab external project focused on humanoid robots that
learn soccer behaviors: standing, walking, balancing under contact, dribbling,
shooting, defending, and multi-agent team play.

The previous Franka Lagrangian MBRL work is preserved on:

```text
archive/franka-lagrangian-mbrl-2026-07-25
```

## Current Scope

This repository is intentionally only a project structure reset. It defines the
directories, package boundaries, configuration placeholders, task entry points,
and documentation layout needed before implementing physics, assets, rewards,
or policies.

## Repository Layout

```text
.
|-- assets/                         # Future USD, robot, field, and ball assets
|   |-- balls/
|   |-- fields/
|   `-- robots/humanoid/
|-- configs/                        # Plain planning configs before Isaac Lab cfgs exist
|   |-- robots/
|   |-- sim/
|   |-- tasks/
|   `-- training/
|-- docs/                           # Architecture, asset pipeline, and roadmap
|-- scripts/                        # Train/play/list entry point scaffolds
|-- source/humanoid_soccer_lab/     # Isaac Lab external extension package
|   |-- config/extension.toml
|   |-- humanoid_soccer_lab/
|   |   |-- assets/
|   |   `-- tasks/direct/humanoid_soccer/
|   |-- pyproject.toml
|   `-- setup.py
`-- tests/                          # Structure checks that do not require Isaac Lab
```

## Intended Isaac Lab Workflow

1. Install Isaac Lab separately.
2. Install this extension in editable mode:

```powershell
python -m pip install -e source\humanoid_soccer_lab
```

3. List registered tasks once the task implementation exists:

```powershell
python scripts\list_tasks.py
```

4. Train or play after the environment is implemented:

```powershell
python scripts\train.py --task HumanoidSoccer-Direct-v0
python scripts\play.py --task HumanoidSoccer-Direct-v0
```

## First Implementation Targets

1. Import or author the humanoid USD/articulation and actuator model.
2. Create the soccer field, ball asset, contact sensors, and reset logic.
3. Implement a single-agent direct RL task for standing and ball approach.
4. Add reward terms for balance, gait regularity, ball control, and shooting.
5. Expand to multi-agent play only after the single-agent task is stable.
