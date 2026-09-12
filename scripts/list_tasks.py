"""List the task ids and helper scripts this project provides."""

from __future__ import annotations

TASKS = {
    "NaoStand-Direct-v0": "NAO standing balance under random pushes (training).",
    "NaoStand-Direct-Play-v0": "The same task, small and quiet, for inspection.",
    "HumanoidSoccer-Direct-v0": "Soccer task scaffold; not implemented.",
}

SCRIPTS = {
    "scripts/view_nao.py": "Load the NAO with passive joints and report the articulation.",
    "scripts/stand_nao.py": "Balance baselines: --controller joint_pd or dcm.",
    "scripts/check_nao_stand_env.py": "Validate the learning environment, score baselines.",
    "scripts/derive_gains.py": "Recompute the joint gains from the URDF.",
    "scripts/train.py": "Train a policy with RSL-RL.",
    "scripts/play.py": "Replay a trained policy.",
}


def main() -> None:
    print("Tasks")
    for name, description in TASKS.items():
        print(f"  {name:28s} {description}")
    print()
    print("Scripts")
    for name, description in SCRIPTS.items():
        print(f"  {name:32s} {description}")


if __name__ == "__main__":
    main()
