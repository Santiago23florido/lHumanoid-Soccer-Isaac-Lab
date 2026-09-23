"""Load the NAO H25 V5.0 into Isaac Sim and report its articulation.

Canonical Phase 1 smoke test. It builds a minimal scene — ground plane, dome
light, camera — spawns the NAO as a free-floating PhysX articulation, prints
diagnostics read back from the simulation, and then runs until the window is
closed.

There is deliberately no task, reward, observation, action or policy here. The
robot has no controller, so it is expected to collapse once physics starts;
that is what validates the articulation.

On first run the URDF is converted to USD automatically, so no manual import
through the Isaac Sim GUI is required.

Usage (Windows PowerShell)::

    cd C:\\IsaacLab
    .\\isaaclab.bat -p "C:\\...\\lagrangian-mbrl-franka\\scripts\\view_nao.py" --device cuda:0

See ``nao/assets/licenses/README.md`` for the model's provenance and licensing.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Isaac Sim terminates the process from within simulation_app.close(), which
# discards anything still sitting in a block-buffered stdout. When stdout is a
# pipe rather than a console that silently swallows every diagnostic below, so
# switch to line buffering before printing anything.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View the NAO H25 V5.0 in Isaac Sim.")
parser.add_argument(
    "--spawn-height",
    type=float,
    default=None,
    help="Root spawn height in metres. Defaults to the value in the NAO asset config.",
)
parser.add_argument(
    "--rebuild-usd",
    action="store_true",
    help="Force regeneration of the derived URDF and the USD before loading.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop after N physics steps. Use 0 to run until the window is closed.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=120,
    help="Physics steps used to verify that gravity acts on the robot.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Fail fast, before paying for an Isaac Sim launch, if the geometry is absent.
from humanoid_transfer.nao.assets.nao_paths import describe_missing_meshes, meshes_are_available

if not meshes_are_available():
    print("\n" + describe_missing_meshes() + "\n", file=sys.stderr)
    raise SystemExit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from humanoid_transfer.nao.assets.nao import (
    NAO_DEFAULT_JOINT_POS,
    NAO_EXPECTED_JOINTS,
    get_nao_cfg,
)
from humanoid_transfer.nao.assets.nao_paths import (
    DERIVED_URDF_PATH,
    NAO_URDF_PATH,
    urdf_robot_name,
)

URDF_TOTAL_MASS_KG = 5.3054
"""Sum of every ``<mass>`` in the upstream URDF, used only as a sanity reference."""

NAO_NOMINAL_HEIGHT_M = 0.58
"""Datasheet standing height of the NAO H25, used only as a scale sanity check."""

UNBOUNDED = 1.0e6
"""Above this magnitude a PhysX limit means "unlimited" (it reports FLT_MAX)."""


def _limit(value: float, unit: str = "") -> str:
    """Render a PhysX limit, collapsing the FLT_MAX sentinel to a readable word."""
    if abs(value) >= UNBOUNDED:
        return "unlimited"
    return f"{value:+.5f}{unit}" if unit else f"{value:+.5f}"


def design_scene() -> Articulation:
    """Create the minimal scene and return the NAO articulation."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Robots", "Xform")

    nao_cfg = get_nao_cfg(
        prim_path="/World/Robots/Nao",
        spawn_height=args_cli.spawn_height,
        force_conversion=args_cli.rebuild_usd,
    )
    return Articulation(cfg=nao_cfg)


def _format_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    separator = "  ".join("-" * w for w in widths)
    body = "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows
    )
    return f"{line}\n{separator}\n{body}"


def print_diagnostics(robot: Articulation) -> dict[str, object]:
    """Print articulation diagnostics read back from the simulation."""
    data = robot.data
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)

    limits = data.joint_pos_limits[0].detach().cpu()
    effort = data.joint_effort_limits[0].detach().cpu()
    velocity = data.joint_velocity_limits[0].detach().cpu()
    masses = data.default_mass[0].detach().cpu()
    total_mass = float(masses.sum())

    print("\n" + "=" * 78)
    print("NAO ARTICULATION DIAGNOSTICS")
    print("=" * 78)
    print(f"  source URDF (verbatim) : {NAO_URDF_PATH}")
    print(f"  derived URDF           : {DERIVED_URDF_PATH}")
    print(f"  USD                    : {robot.cfg.spawn.usd_path}")
    print(f"  prim path              : {robot.cfg.prim_path}")
    print(f"  robot name             : {urdf_robot_name()}")
    print(f"  instances              : {robot.num_instances}")
    print(f"  bodies / links         : {robot.num_bodies}")
    print(f"  joints                 : {robot.num_joints}")
    print(f"  degrees of freedom     : {data.joint_pos.shape[1]}")
    print(f"  fixed tendons          : {robot.num_fixed_tendons}")
    print(f"  articulation root body : {body_names[0]}")
    print(f"  root is fixed to world : {robot.is_fixed_base}")
    print(f"  total mass             : {total_mass:.4f} kg")
    print(f"  URDF reference mass    : {URDF_TOTAL_MASS_KG:.4f} kg")

    if NAO_DEFAULT_JOINT_POS:
        adjusted = ", ".join(f"{k}={v:+.5f} rad" for k, v in sorted(NAO_DEFAULT_JOINT_POS.items()))
        print(f"  zero pose overrides    : {adjusted}")
        print("                           (URDF limits exclude zero for these joints)")

    tiny = [n for i, n in enumerate(body_names) if float(masses[i]) < 1.0e-4]
    print("\n  BODIES")
    print("  " + "\n  ".join(_format_table(
        [(name, f"{float(masses[i]):.6g}") for i, name in enumerate(body_names)],
        ("body", "mass [kg]"),
    ).splitlines()))
    if tiny:
        print(
            f"\n    NOTE: {len(tiny)} finger/gripper bodies carry the upstream placeholder\n"
            "          mass of 2e-06 kg with 1.1e-09 inertia. That is what the URDF\n"
            "          declares; no value was substituted here."
        )

    print("\n  JOINTS (limits as reported by PhysX)")
    print('  "unlimited" marks a URDF <continuous> joint, which carries no limit.')
    rows = [
        (
            name,
            _limit(float(limits[i, 0])),
            _limit(float(limits[i, 1])),
            _limit(float(effort[i])),
            _limit(float(velocity[i])),
        )
        for i, name in enumerate(joint_names)
    ]
    print("  " + "\n  ".join(_format_table(
        rows, ("joint", "lower [rad]", "upper [rad]", "effort [Nm]", "velocity [rad/s]")
    ).splitlines()))

    return {
        "joint_names": joint_names,
        "body_names": body_names,
        "total_mass": total_mass,
        "limits": limits,
    }


def check_expected_joints(joint_names: list[str]) -> bool:
    """Report which of the documented H25 V5.0 joints are present."""
    present = [j for j in NAO_EXPECTED_JOINTS if j in joint_names]
    missing = [j for j in NAO_EXPECTED_JOINTS if j not in joint_names]
    extra = [j for j in joint_names if j not in NAO_EXPECTED_JOINTS]

    print("\n  EXPECTED JOINT CHECK")
    print(f"    documented joints present : {len(present)}/{len(NAO_EXPECTED_JOINTS)}")
    if missing:
        print(f"    NOT exposed as DOFs       : {', '.join(missing)}")
    if extra:
        print(f"    additional DOFs           : {len(extra)} finger/thumb joints")
        print(f"      {', '.join(extra)}")
    if not missing:
        print("    OK: every documented joint is present in the articulation.")
        print(
            "    NOTE: upstream marks RHipYawPitch and the 16 finger/thumb joints as\n"
            "          <mimic>. They are imported as PhysX mimic constraints, so they\n"
            "          still appear as DOFs but are driven by LHipYawPitch and by\n"
            "          LHand / RHand rather than being independently commandable."
        )
    return not missing


def check_geometry(robot: Articulation, total_mass: float) -> bool:
    """Programmatic scale, symmetry and orientation checks."""
    data = robot.data
    body_names = list(robot.body_names)
    positions = data.body_pos_w[0].detach().cpu()
    root_pos = data.root_pos_w[0].detach().cpu()
    local = positions - root_pos

    z_extent = float(local[:, 2].max() - local[:, 2].min())
    y_extent = float(local[:, 1].max() - local[:, 1].min())
    x_extent = float(local[:, 0].max() - local[:, 0].min())

    print("\n  GEOMETRY AND SCALE CHECK (zero pose, body origins)")
    print(
        f"    vertical extent  : {z_extent:.4f} m"
        f"  (NAO H25 nominal ~{NAO_NOMINAL_HEIGHT_M:.2f} m)"
    )
    print(f"    lateral extent   : {y_extent:.4f} m")
    print(f"    forward extent   : {x_extent:.4f} m")

    ok = True

    # Scale: a 10x or 0.1x mesh scale error would move this far outside the band.
    if not 0.30 <= z_extent <= 0.80:
        print("    FAIL: vertical extent implies a mesh scale error (check scale=0.1).")
        ok = False
    else:
        print("    OK: overall size is consistent with a ~0.58 m NAO.")

    # Mass: compare against the sum of the URDF <mass> values.
    if abs(total_mass - URDF_TOTAL_MASS_KG) > 0.05:
        print(
            f"    FAIL: total mass {total_mass:.4f} kg differs from the URDF "
            f"({URDF_TOTAL_MASS_KG:.4f} kg)."
        )
        ok = False
    else:
        print("    OK: total mass matches the URDF inertial data.")

    # Left/right: NAO's +y axis is its left side.
    def body_y(candidates: tuple[str, ...]) -> float | None:
        for name in candidates:
            if name in body_names:
                return float(local[body_names.index(name), 1])
        return None

    left_y = body_y(("l_ankle", "LAnkleRoll", "l_sole"))
    right_y = body_y(("r_ankle", "RAnkleRoll", "r_sole"))
    if left_y is None or right_y is None:
        print("    SKIP: ankle bodies not found, left/right check not possible.")
    else:
        print(f"    left ankle y     : {left_y:+.4f} m")
        print(f"    right ankle y    : {right_y:+.4f} m")
        if left_y > 0.0 > right_y:
            print("    OK: left limb is on +y and right limb on -y, not swapped.")
        else:
            print("    FAIL: left/right limbs appear swapped.")
            ok = False

    # Feet below the head: catches an upside-down import.
    head_candidates = [n for n in body_names if "Head" in n or n == "Neck"]
    if head_candidates and left_y is not None:
        head_z = max(float(local[body_names.index(n), 2]) for n in head_candidates)
        ankles = [n for n in body_names if "ankle" in n.lower()]
        foot_z = min(float(local[body_names.index(n), 2]) for n in ankles)
        print(f"    head z / foot z  : {head_z:+.4f} m / {foot_z:+.4f} m")
        if head_z > foot_z:
            print("    OK: head is above the feet, robot is upright.")
        else:
            print("    FAIL: head is below the feet, the model is inverted.")
            ok = False

    return ok


def check_gravity(sim: SimulationContext, robot: Articulation, steps: int) -> bool:
    """Step the simulation and confirm the free-floating robot falls."""
    if steps <= 0:
        return True
    sim_dt = sim.get_physics_dt()
    start_z = float(robot.data.root_pos_w[0, 2])
    lowest = start_z
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        lowest = min(lowest, float(robot.data.root_pos_w[0, 2]))
    end_z = float(robot.data.root_pos_w[0, 2])

    print(f"\n  PHYSICS CHECK ({steps} steps at dt={sim_dt:.5f} s)")
    print(f"    root height start : {start_z:.4f} m")
    print(f"    root height end   : {end_z:.4f} m")
    print(f"    lowest root height: {lowest:.4f} m")

    if end_z >= start_z - 1e-4:
        print("    FAIL: the root never descended; gravity or the root body is wrong.")
        return False
    if end_z < -0.05:
        print("    FAIL: the robot fell through the ground plane; collisions are missing.")
        return False
    print("    OK: gravity acts, the robot settles onto the ground plane and is not fixed.")
    return True


def reset(robot: Articulation) -> None:
    """Place the articulation at its configured default state."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()


def run_simulator(sim: SimulationContext, robot: Articulation, max_steps: int) -> None:
    """Run until the window closes, or until ``max_steps`` physics steps elapse."""
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        count += 1
        if max_steps > 0 and count >= max_steps:
            break


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Close viewpoint: the NAO is only ~0.58 m tall.
    sim.set_camera_view([1.1, -1.1, 0.7], [0.0, 0.0, 0.25])

    robot = design_scene()
    sim.reset()

    reset(robot)
    robot.update(sim.get_physics_dt())

    info = print_diagnostics(robot)
    joints_ok = check_expected_joints(info["joint_names"])  # type: ignore[arg-type]
    geometry_ok = check_geometry(robot, float(info["total_mass"]))  # type: ignore[arg-type]
    gravity_ok = check_gravity(sim, robot, args_cli.settle_steps)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  documented joints exposed : {'yes' if joints_ok else 'partial (see note)'}")
    print(f"  geometry and scale        : {'PASS' if geometry_ok else 'FAIL'}")
    print(f"  physics and gravity       : {'PASS' if gravity_ok else 'FAIL'}")
    print("  No RL code is involved: this is an asset validation scene only.")
    print("=" * 78 + "\n")

    reset(robot)
    print("[INFO]: Setup complete. Close the Isaac Sim window or press Ctrl+C to stop.")
    run_simulator(sim, robot, args_cli.max_steps)
    return 0 if (geometry_ok and gravity_ok) else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        # simulation_app.close(skip_cleanup=True) ends the process with an
        # immediate _exit, so an unhandled traceback would never be printed.
        # Report it here while the interpreter is still alive.
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        bounded = args_cli.max_steps > 0
        simulation_app.close(wait_for_replicator=not bounded, skip_cleanup=bounded)
    raise SystemExit(exit_code)
