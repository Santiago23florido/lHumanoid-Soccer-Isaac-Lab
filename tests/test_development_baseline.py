"""What this branch builds on, enforced rather than remembered.

The baseline has two halves, and both are decisions that are easy to undo by
accident:

**The teacher is the trained G1.** Pinned in ``g1/policy/`` and checked by
``tests/test_teacher_policy.py``.

**The student starts from its body, not from its balance task.** The NAO's
standing work -- the ``nao_stand`` environment and the hand-written balance
controllers -- is kept, registered and documented, and is *not* an input to the
walking work. The reason is in ``ARCHITECTURE.md``: the standing task terminates
on a step, which is correct, because "recover without stepping" is the task it
poses. The walking task poses the opposite. A criterion imported from the first
into the second does not constrain the student's body, it prescribes a strategy,
and the project has already paid for that mistake twice -- once in the transfer
feasibility mask and once in the teacher's command range, where the student's
target speed was justified by the zero-step envelope.

So this file draws the line in code. The NAO's *body* is shared, because it
would be identical if the standing study had never happened:

=========================================  ============================
Shared with the walking work               Frozen out of it
=========================================  ============================
``nao.assets`` (URDF, USD, kinematics)     ``nao.tasks.nao_stand``
``common`` (LIPM, DCM, capturability)      ``nao.controllers``
=========================================  ============================

An ``importlib.import_module`` call would evade the scan below. That is accepted:
the failure being guarded is a module reaching for the standing task because it
was nearby, not someone working around the guard on purpose.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source" / "humanoid_transfer" / "humanoid_transfer"

FROZEN = (
    "humanoid_transfer.nao.tasks.nao_stand",
    "humanoid_transfer.nao.controllers",
)
"""Modules the previous NAO training lives in. Kept, not deleted, not used."""

FROZEN_RESULT_FILES = (
    "nao/results/comparison_directional.json",
    "nao/docs/task_balance.md",
    "nao/docs/baseline_joint_pd.md",
    "nao/docs/baseline_capture_point.md",
)
"""Measurements of the standing task. Readable by a human, not by the pipeline."""

DEVELOPMENT_PATH = (
    "humanoid_transfer.transfer",
    "humanoid_transfer.g1",
)
"""Packages the walking work is being built in."""

REGISTRATION_ONLY = {
    PKG / "tasks.py",
    PKG / "nao" / "tasks" / "__init__.py",
}
"""Modules whose whole job is to make gym ids exist.

These must keep importing ``nao_stand``: unregistering it would lose the standing
model, which is the opposite of the intent. They own no development logic, which
is why importing a frozen module here means nothing.
"""


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PKG.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(path: Path) -> str:
    name = _module_name(path)
    if path.name == "__init__.py":
        return name
    return name.rsplit(".", 1)[0] if "." in name else ""


def _imported_modules(path: Path) -> set[str]:
    """Absolute module names this file imports, relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _package_of(path)
            for _ in range(max(node.level - 1, 0)):
                base = base.rsplit(".", 1)[0] if "." in base else ""
            if node.level == 0:
                target = node.module or ""
            elif node.module:
                target = f"{base}.{node.module}" if base else node.module
            else:
                target = base
            if target:
                found.add(target)
                found.update(f"{target}.{alias.name}" for alias in node.names)

    return found


def _source_files() -> list[Path]:
    return sorted(p for p in PKG.rglob("*.py") if p not in REGISTRATION_ONLY)


def _on_development_path(path: Path) -> bool:
    name = _module_name(path)
    if any(name == p or name.startswith(f"{p}.") for p in DEVELOPMENT_PATH):
        return True
    # A future NAO task is development path; nao_stand itself is not.
    if name.startswith("humanoid_transfer.nao.tasks."):
        return not name.startswith("humanoid_transfer.nao.tasks.nao_stand")
    return False


# --- the frozen list points at something real --------------------------------


@pytest.mark.parametrize("frozen", FROZEN)
def test_each_frozen_module_still_exists(frozen: str) -> None:
    """A guard against modules that were deleted passes trivially and says
    nothing. The standing work is meant to be *kept*."""
    relative = frozen.removeprefix("humanoid_transfer.").replace(".", "/")
    target = PKG / relative
    assert target.is_dir() or target.with_suffix(".py").is_file()


@pytest.mark.parametrize("result_file", FROZEN_RESULT_FILES)
def test_each_frozen_result_still_exists(result_file: str) -> None:
    assert (ROOT / result_file).is_file()


def test_there_is_something_on_the_development_path_to_check() -> None:
    """If the scan finds no development-path files it cannot fail, which would
    make every assertion below vacuous."""
    assert [p for p in _source_files() if _on_development_path(p)]


# --- the boundary ------------------------------------------------------------


def test_the_walking_work_does_not_import_the_standing_work() -> None:
    offenders: list[str] = []
    for path in _source_files():
        if not _on_development_path(path):
            continue
        for imported in sorted(_imported_modules(path)):
            if any(imported == f or imported.startswith(f"{f}.") for f in FROZEN):
                offenders.append(f"{_module_name(path)} imports {imported}")

    assert offenders == [], (
        "The walking work reached into the standing work:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe NAO's body is in nao.assets and the theory is in common. "
        "If the thing needed is genuinely a property of the robot, move it "
        "there. If it is a property of the standing task, it does not belong "
        "in a walking task at all."
    )


def test_the_walking_work_does_not_read_the_standing_results() -> None:
    """Numbers measured on the standing task describe a policy trying not to
    step. Loading them into the walking pipeline makes them an assumption; the
    documents are there to be read by a person instead."""
    offenders: list[str] = []
    for path in _source_files():
        if not _on_development_path(path):
            continue
        source = path.read_text(encoding="utf-8")
        for result_file in FROZEN_RESULT_FILES:
            if result_file in source:
                offenders.append(f"{_module_name(path)} refers to {result_file}")

    assert offenders == [], "\n  ".join(["Standing results read as input:", *offenders])


def test_the_transfer_package_touches_neither_robots_task_package() -> None:
    """``transfer/`` is where the two robots meet, and it meets them at the level
    of kinematic roles and geometry. Pulling in either task package would couple
    the research to one environment's reward design."""
    for path in sorted((PKG / "transfer").rglob("*.py")):
        imported = _imported_modules(path)
        for forbidden in ("humanoid_transfer.nao.tasks", "humanoid_transfer.g1.tasks"):
            assert not any(
                name == forbidden or name.startswith(f"{forbidden}.") for name in imported
            ), f"{_module_name(path)} imports {forbidden}"


# --- the standing model survives ---------------------------------------------


def test_the_standing_task_is_still_registered() -> None:
    """ "Clean" means unused, not deleted. The standing policy is a result this
    project keeps, and losing its environment would make it unreproducible."""
    registry = ast.parse((PKG / "nao" / "tasks" / "__init__.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(registry)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
        for alias in node.names
    }
    assert "nao_stand" in imported

    task_init = (PKG / "nao" / "tasks" / "nao_stand" / "__init__.py").read_text(encoding="utf-8")
    assert 'TASK_ID = "NaoStand-Direct-v0"' in task_init


def test_the_standing_task_keeps_the_termination_that_defines_it() -> None:
    """The zero-step termination is not a leftover constraint to be cleaned up.
    It *is* the standing task: "recover without stepping" is what it poses, and
    the capturable envelope is its definition. A condition may define a task.
    """
    cfg = (PKG / "nao" / "tasks" / "nao_stand" / "nao_stand_env_cfg.py").read_text(encoding="utf-8")
    assert "termination_foot_displacement = 0.05" in cfg


def test_the_balance_controllers_are_kept_as_baselines() -> None:
    """A learned walk still has to beat something, and these are what the
    standing track measured itself against."""
    assert (PKG / "nao" / "controllers" / "dcm_balance.py").is_file()
