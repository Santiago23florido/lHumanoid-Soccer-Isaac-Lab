"""Hold the NAO upright with a joint PD controller, and measure how well it does.

This is the balance baseline: no learning, no policy, no reward. Every joint is
commanded to hold the nominal standing posture with the gains derived in
``scripts/derive_gains.py``, and the script reports what that buys.

It exists for two reasons.

1. **Verification.** The gains are derived from the URDF's inertias on paper.
   Running them is the only way to find out whether the derivation survives
   contact, friction, solver damping and a 10 ms control period.
2. **Comparison.** A learned balance policy has to be measured against
   something. A tuned PD is the honest baseline: it is what the robot can do
   with a fixed linear feedback law and no model of its own dynamics. Anything
   the policy adds has to be visible on top of these numbers.

The quantities reported are the ones the balance theory is written in:

* the centre of mass, computed from the simulator's own body poses and masses;
* the **divergent component of motion** :math:`\\xi = x + \\dot{x}/\\omega_0`,
  which is the unstable part of the inverted-pendulum dynamics and therefore
  the thing a balance controller actually has to regulate;
* the **centre of pressure**, recovered from the ankle joint reaction wrench,
  together with how much of the foot it has left to move in;
* ankle torque against the URDF limit, which is what saturates first.

Usage (Windows PowerShell)::

    cd C:\\IsaacLab
    .\\isaaclab.bat -p "C:\\...\\scripts\\stand_nao.py" --device cuda:0 --headless

Add ``--push-velocity 0.3`` to launch a forward shove halfway through and watch
the recovery. See ``docs/stand_nao.md`` for the full option list.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Isaac Sim terminates the process from inside simulation_app.close(), which
# discards anything still sitting in a block-buffered stdout. Switch to line
# buffering before printing anything or the diagnostics vanish when piped.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_soccer_lab"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hold the NAO upright with a joint PD controller.")
parser.add_argument(
    "--duration",
    type=float,
    default=10.0,
    help="Seconds of simulated time to run. Default 10.",
)
parser.add_argument(
    "--push-velocity",
    type=float,
    default=0.0,
    help=(
        "Forward centre-of-mass velocity in m/s to impose as a push. "
        "0 disables the push. The zero-step capturable limit is printed at startup."
    ),
)
parser.add_argument(
    "--push-at",
    type=float,
    default=None,
    help="When to push, in seconds. Defaults to halfway through the run.",
)
parser.add_argument(
    "--settle-time",
    type=float,
    default=0.5,
    help="Seconds to let the robot settle onto the ground before measuring. Default 0.5.",
)
parser.add_argument(
    "--rebuild-usd",
    action="store_true",
    help="Force regeneration of the derived URDF and the USD before loading.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Fail fast, before paying for an Isaac Sim launch, if the geometry is absent.
from humanoid_soccer_lab.assets.nao_paths import describe_missing_meshes, meshes_are_available

if not meshes_are_available():
    print("\n" + describe_missing_meshes() + "\n", file=sys.stderr)
    raise SystemExit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from humanoid_soccer_lab.assets import nao_kinematics as nk
from humanoid_soccer_lab.assets.nao import (
    NAO_STAND_COM_HEIGHT,
    NAO_STAND_LIPM_OMEGA,
    get_nao_stand_cfg,
)

PHYSICS_DT = 1.0 / 200.0
"""Physics step, in seconds.

Matches the learning environment so the baseline and the policy are measured
under identical integration. 5 ms is short against the 115 ms in which an
uncorrected balance error doubles.
"""

CONTROL_DECIMATION = 2
"""Physics steps per control update, giving a 100 Hz loop.

The same rate the real NAO's DCM runs its joint controllers at, so the baseline
is not quietly given a bandwidth the hardware could not reproduce.
"""

FALL_BASE_HEIGHT = 0.20
"""Base height below which the robot counts as fallen, in metres.

The nominal standing height is 0.3207 m, so this is a drop of 12 cm: far beyond
any balance excursion, and reached well before the torso touches the ground.
"""

FALL_TILT_COSINE = -0.7
"""``projected_gravity_b[2]`` above which the robot counts as fallen.

Gravity read in the base frame is (0, 0, -1) when upright, so -0.7 is a tilt of
about 45 degrees. Past that the feet cannot generate a restoring moment.
"""


def design_scene() -> Articulation:
    """Create the ground plane, light and NAO, and return the articulation."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Robots", "Xform")

    return Articulation(
        cfg=get_nao_stand_cfg(
            prim_path="/World/Robots/Nao",
            force_conversion=args_cli.rebuild_usd,
        )
    )


def center_of_mass_world(robot: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the whole-body centre of mass position and velocity, in world frame.

    Isaac Lab exposes per-body states but no whole-body centre of mass, so it is
    assembled here as the mass-weighted mean over the bodies. ``default_mass``
    is used rather than a live read because mass is constant unless an event
    randomises it, and this script never does.
    """
    masses = robot.data.default_mass[0].to(robot.device)
    total = masses.sum()
    weights = (masses / total).view(1, -1, 1)
    position = (robot.data.body_com_pos_w * weights).sum(dim=1)
    velocity = (robot.data.body_com_lin_vel_w * weights).sum(dim=1)
    return position, velocity


def divergent_component(
    com_pos: torch.Tensor, com_vel: torch.Tensor, omega: float
) -> torch.Tensor:
    r"""Return the divergent component of motion, :math:`\xi = x + \dot{x}/\omega_0`.

    The linear inverted pendulum splits into a stable and an unstable mode. The
    unstable one is this: it obeys :math:`\dot{\xi} = \omega_0(\xi - p)`, where
    ``p`` is the centre of pressure, so it runs away unless the foot is placed
    to bring it back. The centre of mass then follows it stably. Regulating
    :math:`\xi` is therefore the whole of the balance problem; regulating the
    centre of mass alone is not, because a stationary centre of mass with the
    wrong velocity is already falling.
    """
    return com_pos[:, :2] + com_vel[:, :2] / omega


def center_of_pressure(
    robot: Articulation, foot_ids: list[int], ground_height: float = 0.0
) -> torch.Tensor | None:
    """Return the centre of pressure in world frame, or None if both feet are airborne.

    The centre of pressure is where the ground reaction can be replaced by a
    single force with no horizontal moment. It is the quantity the ankle
    strategy actually steers, and it is confined to the support polygon by
    unilateral contact: the foot can push on the ground but never pull.

    The simulator does not report it, so it is reconstructed in three steps.

    1. Take the reaction wrench the parent transmits through the ankle joint.
       Isaac Lab reports this in the child body frame, so it is rotated to
       world first. Its vertical component is negative while standing, because
       it is the leg pressing *down* on the foot.
    2. Recover the ground reaction from Newton's law on the foot alone. Under
       the quasi-static assumption the ground must balance both the ankle
       wrench and the foot's own weight::

           f_grf = -f_ankle - m_foot * g
           m_grf = -m_ankle - (c_foot - a) x (m_foot * g)

       Skipping the foot's weight would bias the normal force by about 7% and
       the pressure centre by a millimetre or two, because a NAO foot is 0.17 kg
       and its centre of mass does not sit under the ankle axis.
    3. Slide the wrench down to the contact plane and solve for the point where
       the horizontal moment vanishes::

           p_x = a_x - (m_y + (a_z - z_0) f_x) / f_z
           p_y = a_y + (m_x - (a_z - z_0) f_y) / f_z

    With both feet loaded the overall centre of pressure is the normal-force
    weighted mean of the two, which is exactly why the double-support polygon
    spans the gap between the feet rather than either foot alone.
    """
    wrench = robot.data.body_incoming_joint_wrench_b[:, foot_ids, :]
    quats = robot.data.body_quat_w[:, foot_ids, :]
    forces = math_utils.quat_apply(quats, wrench[..., :3])
    moments = math_utils.quat_apply(quats, wrench[..., 3:])

    ankles = robot.data.body_pos_w[:, foot_ids, :]
    foot_com = robot.data.body_com_pos_w[:, foot_ids, :]
    foot_mass = robot.data.default_mass[:, foot_ids].to(robot.device)

    # Gravitational force on the foot itself, and its moment about the ankle.
    weight = torch.zeros_like(forces)
    weight[..., 2] = -foot_mass * nk.GRAVITY
    lever = foot_com - ankles
    gravity_moment = torch.cross(lever, weight, dim=-1)

    forces = -forces - weight
    moments = -moments - gravity_moment

    normal = forces[..., 2]
    loaded = normal > 1.0  # newtons; below this the foot is carrying nothing
    if not bool(loaded.any()):
        return None

    safe_normal = torch.where(loaded, normal, torch.ones_like(normal))
    height = ankles[..., 2] - ground_height
    cop_x = ankles[..., 0] - (moments[..., 1] + height * forces[..., 0]) / safe_normal
    cop_y = ankles[..., 1] + (moments[..., 0] - height * forces[..., 1]) / safe_normal

    weights = torch.where(loaded, normal, torch.zeros_like(normal))
    total = weights.sum(dim=1).clamp(min=1e-6)
    return torch.stack(
        [(cop_x * weights).sum(dim=1) / total, (cop_y * weights).sum(dim=1) / total],
        dim=-1,
    )


class Recorder:
    """Accumulates the per-step metrics the summary is built from."""

    def __init__(self) -> None:
        self.com_offset: list[float] = []
        self.dcm_offset: list[float] = []
        self.cop_offset: list[float] = []
        self.base_height: list[float] = []
        self.tilt: list[float] = []
        self.ankle_torque: list[float] = []
        self.fell_at: float | None = None
        self.recovered_at: float | None = None

    @staticmethod
    def _summarise(values: list[float]) -> str:
        if not values:
            return "    (no samples)"
        peak = max(values, key=abs)
        mean = sum(values) / len(values)
        return f"mean {mean:+.4f}   peak {peak:+.4f}"


def report_startup(robot: Articulation, ankle_ids: list[int], foot_ids: list[int]) -> None:
    """Print what the robot and the theory say before any stepping happens."""
    bounds = nk.capturable_com_velocity()
    model = nk.load_model()
    efforts = {joint.name: joint.effort for joint in model.joints if joint.effort}
    ankle_effort = efforts["LAnklePitch"]
    weight = model.total_mass * nk.GRAVITY
    (x_min, x_max), (y_min, y_max) = nk.support_polygon_double_stance()

    print("\n" + "=" * 78)
    print("NAO BALANCE BASELINE -- joint PD, no learning")
    print("=" * 78)
    print(f"  bodies / joints            : {robot.num_bodies} / {robot.num_joints}")
    print(f"  total mass                 : {float(robot.data.default_mass[0].sum()):.4f} kg")
    print(f"  commanded joints           : {len(nk.ACTUATED_JOINTS)} of {robot.num_joints} DOFs")
    print(f"  control rate               : {1.0 / (PHYSICS_DT * CONTROL_DECIMATION):.0f} Hz")
    print(f"  foot bodies                : {[robot.body_names[i] for i in foot_ids]}")
    print(f"  ankle pitch joints         : {[robot.joint_names[i] for i in ankle_ids]}")

    print("\n  INVERTED PENDULUM")
    print(f"    CoM height z_c           : {NAO_STAND_COM_HEIGHT:.5f} m")
    print(f"    omega_0 = sqrt(g/z_c)    : {NAO_STAND_LIPM_OMEGA:.4f} rad/s")
    print(f"    error doubling time      : {math.log(2.0) / NAO_STAND_LIPM_OMEGA * 1e3:.1f} ms")
    print(f"    control period           : {PHYSICS_DT * CONTROL_DECIMATION * 1e3:.1f} ms")

    print("\n  SUPPORT POLYGON (double stance, relative to the sole midpoint)")
    print(f"    forward / backward       : {x_max * 1e3:+.1f} / {x_min * 1e3:+.1f} mm")
    print(f"    lateral                  : {y_max * 1e3:+.1f} / {y_min * 1e3:+.1f} mm")

    print("\n  AUTHORITY LIMITS")
    print(
        f"    ankle pitch torque       : {ankle_effort:.3f} N m each, "
        f"{2 * ankle_effort:.3f} total"
    )
    print(f"    CoP reach from torque    : {2 * ankle_effort / weight * 1e3:.1f} mm (two feet)")
    print(f"    CoP reach from geometry  : {x_max * 1e3:.1f} mm (toe edge)")
    binding = "foot geometry" if x_max < 2 * ankle_effort / weight else "ankle torque"
    print(f"    binding constraint       : {binding}")

    print("\n  ZERO-STEP CAPTURABILITY (no stepping, ankle strategy only)")
    print(f"    forward                  : {bounds['forward']:.3f} m/s")
    print(f"    backward                 : {bounds['backward']:.3f} m/s")
    print(f"    lateral                  : {bounds['lateral']:.3f} m/s")
    print(f"    forward impulse          : {bounds['forward_impulse']:.3f} N s")
    if args_cli.push_velocity > 0.0:
        ratio = args_cli.push_velocity / bounds["forward"]
        verdict = "inside" if ratio <= 1.0 else "BEYOND"
        print(
            f"    requested push           : {args_cli.push_velocity:.3f} m/s "
            f"({ratio:.0%} of the limit, {verdict})"
        )


def run(sim: SimulationContext, robot: Articulation) -> Recorder:
    """Hold the posture for the requested duration, pushing once if asked."""
    _, actuated_names = robot.find_joints(list(nk.ACTUATED_JOINTS), preserve_order=True)
    assert actuated_names == list(nk.ACTUATED_JOINTS), "joint order was not preserved"
    foot_ids, _ = robot.find_bodies(list(nk.FOOT_BODIES), preserve_order=True)
    ankle_ids, _ = robot.find_joints(["LAnklePitch", "RAnklePitch"], preserve_order=True)

    report_startup(robot, ankle_ids, foot_ids)

    targets = robot.data.default_joint_pos.clone()
    model = nk.load_model()
    ankle_effort = next(j.effort for j in model.joints if j.name == "LAnklePitch")

    push_at = args_cli.push_at
    if push_at is None:
        push_at = args_cli.settle_time + (args_cli.duration - args_cli.settle_time) * 0.5

    recorder = Recorder()
    steps = int(args_cli.duration / PHYSICS_DT)
    pushed = False

    print(f"\n  running {args_cli.duration:.1f} s ({steps} physics steps)...")

    for step in range(steps):
        elapsed = step * PHYSICS_DT

        if step % CONTROL_DECIMATION == 0:
            # The whole controller: hold every joint at its nominal angle. The
            # gains do the rest. A learned policy replaces this one line with
            # an offset on top of the same targets.
            robot.set_joint_position_target(targets)

        if args_cli.push_velocity > 0.0 and not pushed and elapsed >= push_at:
            velocity = robot.data.root_com_vel_w.clone()
            velocity[:, 0] += args_cli.push_velocity
            robot.write_root_com_velocity_to_sim(velocity)
            pushed = True
            print(f"    push at t={elapsed:.2f} s: +{args_cli.push_velocity:.2f} m/s forward")

        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYSICS_DT)

        if elapsed < args_cli.settle_time:
            continue

        com_pos, com_vel = center_of_mass_world(robot)
        feet = robot.data.body_pos_w[:, foot_ids, :]
        origin = feet.mean(dim=1)

        dcm = divergent_component(com_pos, com_vel, NAO_STAND_LIPM_OMEGA)
        recorder.com_offset.append(float(com_pos[0, 0] - origin[0, 0]))
        recorder.dcm_offset.append(float(dcm[0, 0] - origin[0, 0]))

        cop = center_of_pressure(robot, foot_ids)
        if cop is not None:
            recorder.cop_offset.append(float(cop[0, 0] - origin[0, 0]))

        height = float(robot.data.root_pos_w[0, 2])
        tilt = float(robot.data.projected_gravity_b[0, 2])
        recorder.base_height.append(height)
        recorder.tilt.append(tilt)

        torque = robot.data.applied_torque[0, ankle_ids].abs().max()
        recorder.ankle_torque.append(float(torque) / ankle_effort)

        fallen = height < FALL_BASE_HEIGHT or tilt > FALL_TILT_COSINE
        if fallen and recorder.fell_at is None:
            recorder.fell_at = elapsed
            print(f"    FELL at t={elapsed:.2f} s (height {height:.3f} m, tilt {tilt:+.3f})")
        elif not fallen and recorder.fell_at is not None and recorder.recovered_at is None:
            recorder.recovered_at = elapsed

    return recorder


def summarise(recorder: Recorder) -> bool:
    """Print the summary and return whether the baseline held."""
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)

    if not recorder.base_height:
        print("  No samples collected; --duration is shorter than --settle-time.")
        return False

    print("  Offsets are measured from the midpoint between the two feet, in metres.")
    print(f"    CoM forward offset       : {Recorder._summarise(recorder.com_offset)}")
    print(f"    DCM forward offset       : {Recorder._summarise(recorder.dcm_offset)}")
    if recorder.cop_offset:
        print(f"    CoP forward offset       : {Recorder._summarise(recorder.cop_offset)}")
    else:
        print("    CoP forward offset       : unavailable (no loaded foot)")

    (x_min, x_max), _ = nk.support_polygon_double_stance()
    peak_dcm = max(recorder.dcm_offset, key=abs) if recorder.dcm_offset else 0.0
    margin = min(x_max - peak_dcm, peak_dcm - x_min)

    print(f"\n    base height              : min {min(recorder.base_height):.4f} m")
    print(f"    tilt (gravity_b z)       : max {max(recorder.tilt):+.4f} (-1 is upright)")
    print(f"    ankle torque used        : peak {max(recorder.ankle_torque):.1%} of the limit")
    print(f"    DCM margin to polygon    : {margin * 1e3:+.1f} mm")

    upright = recorder.fell_at is None
    print()
    if upright:
        print("    OK: the robot stayed upright for the whole run.")
    elif recorder.recovered_at is not None:
        print(
            f"    RECOVERED: fell at t={recorder.fell_at:.2f} s, "
            f"back upright by t={recorder.recovered_at:.2f} s."
        )
    else:
        print(f"    FAILED: fell at t={recorder.fell_at:.2f} s and did not recover.")

    if margin < 0.0:
        print("    NOTE: the DCM left the support polygon; this push needs a step.")
    if max(recorder.ankle_torque) > 0.95:
        print("    NOTE: ankle torque saturated; the ankle strategy is exhausted.")

    print("  No learning is involved: this is a fixed linear feedback law.")
    print("=" * 78 + "\n")
    return upright


def reset(robot: Articulation) -> None:
    """Place the articulation at its configured nominal standing state."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.1, -1.1, 0.7], [0.0, 0.0, 0.25])

    robot = design_scene()
    sim.reset()

    reset(robot)
    robot.update(PHYSICS_DT)

    recorder = run(sim, robot)
    return 0 if summarise(recorder) else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close(skip_cleanup=True)
    raise SystemExit(exit_code)
