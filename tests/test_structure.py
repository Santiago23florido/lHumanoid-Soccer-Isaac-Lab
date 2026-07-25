import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "README.md",
    "CHANGELOG.md",
    "configs/robots/humanoid.yaml",
    "configs/sim/isaac_lab.yaml",
    "configs/tasks/humanoid_soccer.yaml",
    "configs/training/rsl_rl_ppo.yaml",
    "docs/architecture.md",
    "scripts/train.py",
    "scripts/play.py",
    "source/humanoid_soccer_lab/config/extension.toml",
    "source/humanoid_soccer_lab/humanoid_soccer_lab/tasks/direct/humanoid_soccer/__init__.py",
]


def test_scaffold_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    assert missing == []


def test_task_id_is_registered_in_scaffold() -> None:
    task_init = (
        ROOT
        / "source/humanoid_soccer_lab/humanoid_soccer_lab/tasks/direct/humanoid_soccer/__init__.py"
    )
    assert "HumanoidSoccer-Direct-v0" in task_init.read_text(encoding="utf-8")


def test_legacy_lagrangian_package_removed_from_main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "src/lagrangian_mbrl"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
