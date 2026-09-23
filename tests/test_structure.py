import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    # Root: the three research tracks have to be visible without digging.
    "README.md",
    "ARCHITECTURE.md",
    "nao/README.md",
    "g1/README.md",
    "transfer/README.md",
    # Track 1, the target embodiment.
    "nao/assets/urdf/nao.urdf",
    "nao/assets/licenses/README.md",
    "nao/docs/task_balance.md",
    "nao/docs/asset_smoke_test.md",
    "nao/docs/asset_pipeline.md",
    "nao/img/nao_stand.png",
    "nao/results/comparison_directional.json",
    "nao/scripts/view_asset.py",
    "nao/scripts/render_pose.py",
    "nao/scripts/fetch_meshes.py",
    # Track 3, the research question.
    "transfer/docs/plan.md",
    # Cross-track entry points.
    "scripts/train.py",
    "scripts/play.py",
    "source/humanoid_transfer/config/extension.toml",
    "source/humanoid_transfer/humanoid_transfer/soccer/tasks/humanoid_soccer/__init__.py",
    "source/humanoid_transfer/humanoid_transfer/g1/assets/g1.py",
    "source/humanoid_transfer/humanoid_transfer/transfer/feasibility.py",
]


def test_scaffold_paths_exist() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    assert missing == []


def test_task_id_is_registered_in_scaffold() -> None:
    task_init = (
        ROOT
        / "source/humanoid_transfer/humanoid_transfer/soccer/tasks/humanoid_soccer/__init__.py"
    )
    assert "HumanoidSoccer-Direct-v0" in task_init.read_text(encoding="utf-8")


def test_nao_viewer_uses_isaac_lab_launcher() -> None:
    viewer = ROOT / "nao/scripts/view_asset.py"
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
