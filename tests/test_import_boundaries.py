"""What each layer of the extension is allowed to import.

``common/`` and the core of ``transfer/`` are NumPy-only so the theory can be
checked without a simulator or a GPU stack. ``transfer/latent`` adds torch but
never Isaac Lab, so the shared latent model trains and is tested on recorded
rollouts. Read from the source with ``ast``, so nothing is imported to check it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "humanoid_transfer" / "humanoid_transfer"
SIMULATOR = ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "isaacsim", "omni", "pxr")


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _modules(*parts: str, recursive: bool = False) -> list[Path]:
    root = PACKAGE.joinpath(*parts)
    found = sorted(root.rglob("*.py") if recursive else root.glob("*.py"))
    assert found, f"nothing to scan under {root}"
    return found


@pytest.mark.parametrize(
    "path",
    _modules("common") + _modules("transfer"),
    ids=lambda p: p.relative_to(PACKAGE).as_posix(),
)
def test_theory_and_transfer_core_are_numpy_only(path: Path) -> None:
    forbidden = {"torch", *SIMULATOR} & _top_level_imports(path)
    assert not forbidden, f"{path.name} imports {sorted(forbidden)}"


@pytest.mark.parametrize(
    "path",
    _modules("transfer", "latent", recursive=True),
    ids=lambda p: p.relative_to(PACKAGE).as_posix(),
)
def test_the_latent_model_never_imports_the_simulator(path: Path) -> None:
    forbidden = set(SIMULATOR) & _top_level_imports(path)
    assert not forbidden, f"{path.name} imports {sorted(forbidden)}"
