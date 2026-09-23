import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "README.md",
    "configs/robots/nao.yaml",
    "configs/sim/isaac_lab.yaml",
    "configs/tasks/humanoid_soccer.yaml",
    "configs/training/rsl_rl_ppo.yaml",
    "docs/architecture.md",
    "docs/asset_pipeline.md",
    "docs/task_nao_stand.md",
    "docs/view_nao.md",
    "docs/img/nao_stand.png",
    "docs/results/comparison_directional.json",
    "scripts/train.py",
    "scripts/play.py",
    "scripts/view_nao.py",
    "scripts/render_nao.py",
    "source/humanoid_soccer_lab/config/extension.toml",
    "source/humanoid_soccer_lab/humanoid_soccer_lab/soccer/tasks/humanoid_soccer/__init__.py",
]


def test_scaffold_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    assert missing == []


def test_task_id_is_registered_in_scaffold() -> None:
    task_init = (
        ROOT
        / "source/humanoid_soccer_lab/humanoid_soccer_lab/soccer/tasks/humanoid_soccer/__init__.py"
    )
    assert "HumanoidSoccer-Direct-v0" in task_init.read_text(encoding="utf-8")


def test_nao_viewer_uses_isaac_lab_launcher() -> None:
    viewer = ROOT / "scripts/view_nao.py"
    text = viewer.read_text(encoding="utf-8")
    assert "AppLauncher" in text
    assert "get_nao_cfg" in text
    assert "SimulationContext" in text


def test_placeholder_humanoid_viewer_was_replaced_by_the_nao_one() -> None:
    result = subprocess.run(
        ["git", "ls-files", "scripts/view_humanoid.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_legacy_lagrangian_package_removed_from_main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "src/lagrangian_mbrl"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
