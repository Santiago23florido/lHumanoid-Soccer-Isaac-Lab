"""Recompute the NAO joint PD gains from the URDF and print the derivation.

The gains in :data:`humanoid_transfer.nao.assets.nao.NAO_STIFFNESS` are not tuned
values; they are the output of this script. Running it reproduces the whole
table, so a reviewer can check the numbers rather than trust them, and the
report can quote a derivation instead of a magic constant.

The design places each joint's closed loop at a chosen natural frequency with
critical damping::

    K_p = I * w_n^2 + M*g*l          K_d = 2 * zeta * sqrt(I * I * w_n^2)

Two corrections make this specific to a standing biped rather than generic.

1. For a leg joint the load is **not** its distal subtree. With the foot planted
   the kinematic chain inverts and the joint carries the whole body. The script
   prints both so the difference is visible; at the ankle it is a factor of 543.
2. Gravity contributes a negative stiffness ``M*g*l`` about each leg joint. A
   stiffness below it cannot be stabilised by any amount of damping, so it is a
   hard lower bound and is added back into ``K_p``.

Needs only NumPy, so it runs without Isaac Sim::

    python scripts/derive_gains.py

The scalar inverted-pendulum design is a gain-sizing approximation, not a
linearisation of the constrained multibody robot. In particular, the Euclidean
CoM lever used here is not the axis-specific gravity Hessian for every leg
joint, and double-support load sharing need not remain equal. The small held
hand/wrist gains in nao.py include engineering floors; they are not reproduced
exactly by the open-chain inertia calculation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402

LEG_NATURAL_FREQUENCY = 12.0
"""Target closed-loop frequency for the leg joints, in rad/s.

Roughly twice the open-loop unstable pole of the standing robot (5.53 rad/s) and
well inside the 100 Hz reference loop. At 200 Hz physics there are about 105
steps per 2*pi/12 oscillation period and 17 steps per 1/12 time scale.
"""

ARM_NATURAL_FREQUENCY = 20.0
"""Target closed-loop frequency for the arms, in rad/s.

Higher than the legs because the arms carry only themselves and are wanted for
fast angular-momentum corrections.
"""

HELD_NATURAL_FREQUENCY = 25.0
"""Target closed-loop frequency for the head, wrists and hands, in rad/s.

These are only ever asked to hold still, so the loop is made stiff relative to
their very small inertias.
"""

PARALLEL_LEG_JOINTS = 2
"""Left and right leg joints act on the same body inertia in double support.

Each therefore supplies half the required stiffness, so the pair together meets
the design target instead of overshooting it twofold.
"""


def critically_damped_gains(
    inertia: float, gravity_stiffness: float, natural_frequency: float, parallel: int = 1
) -> tuple[float, float]:
    """Return ``(K_p, K_d)`` for one joint of a ``parallel``-joint group.

    Args:
        inertia: Inertia the joint carries, in kg m^2.
        gravity_stiffness: Destabilising ``M*g*l`` about the axis, in N m/rad.
            Zero for an open chain that gravity does not topple.
        natural_frequency: Target closed-loop frequency, in rad/s.
        parallel: How many identical joints share the load.

    Returns:
        Stiffness in N m/rad and damping in N m s/rad.
    """
    restoring = inertia * natural_frequency**2
    stiffness = (restoring + gravity_stiffness) / parallel
    damping = 2.0 * math.sqrt(inertia * restoring) / parallel
    return stiffness, damping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zero-pose",
        action="store_true",
        help="Derive at the URDF zero pose instead of the nominal standing crouch.",
    )
    args = parser.parse_args()

    pose = None if args.zero_pose else nk.NOMINAL_STAND_JOINT_POS
    model = nk.load_model()
    mass = model.total_mass
    efforts = {joint.name: joint.effort for joint in model.joints if joint.effort}

    height = nk.com_height_above_soles(pose)
    omega = nk.lipm_omega(height)
    _, ankle_lever = nk.whole_body_inertia_about_ankle(joint_pos=pose)
    gravity_stiffness_ankle = mass * nk.GRAVITY * ankle_lever

    print("=" * 78)
    print("NAO JOINT GAIN DERIVATION")
    print("=" * 78)
    print(f"  pose                      : {'URDF zero' if args.zero_pose else 'nominal stand'}")
    print(f"  total mass M              : {mass:.4f} kg")
    print(f"  CoM height z_c            : {height:.5f} m")
    print(f"  LIPM frequency omega_0    : {omega:.4f} rad/s")
    print(f"  error doubling time       : {math.log(2.0) / omega * 1e3:.1f} ms")
    print(f"  ankle lever arm l         : {ankle_lever:.5f} m")
    print(f"  gravitational stiffness   : {gravity_stiffness_ankle:.4f} N m/rad")
    inertia_ankle, _ = nk.whole_body_inertia_about_ankle(joint_pos=pose)
    unstable_pole = math.sqrt(gravity_stiffness_ankle / inertia_ankle)
    print(f"  open-loop unstable pole   : {unstable_pole:.4f} rad/s")

    header = (
        f"\n{'joint':16s} {'distal I':>10s} {'load I':>10s} {'M g l':>8s} "
        f"{'w_n':>5s} {'K_p':>8s} {'K_d':>7s} {'tau_max':>8s} {'sat err':>8s}"
    )
    print(header)
    print("-" * len(header.strip()))

    def emit(name: str, inertia: float, gravity: float, frequency: float, parallel: int) -> None:
        stiffness, damping = critically_damped_gains(inertia, gravity, frequency, parallel)
        effort = efforts.get(name, float("nan"))
        distal = nk.composite_inertia_about_joint(name, pose)
        saturation = effort / stiffness if stiffness > 0 else float("inf")
        print(
            f"{name:16s} {distal:10.6f} {inertia:10.6f} {gravity:8.3f} "
            f"{frequency:5.1f} {stiffness:8.2f} {damping:7.2f} {effort:8.3f} {saturation:8.4f}"
        )

    print("  legs -- load is the WHOLE BODY about each axis, not the distal subtree")
    for name in nk.LEG_ACTUATED_JOINTS:
        inertia, lever = nk.whole_body_inertia_about_ankle(name, pose)
        parallel = 1 if name == "LHipYawPitch" else PARALLEL_LEG_JOINTS
        emit(name, inertia, mass * nk.GRAVITY * lever, LEG_NATURAL_FREQUENCY, parallel)

    print("\n  arms -- open chains, gravity does not topple them about these axes")
    for name in nk.ARM_ACTUATED_JOINTS:
        emit(name, nk.composite_inertia_about_joint(name, pose), 0.0, ARM_NATURAL_FREQUENCY, 1)

    print("\n  held -- never commanded, only asked to hold still")
    for name in nk.HELD_JOINTS:
        emit(name, nk.composite_inertia_about_joint(name, pose), 0.0, HELD_NATURAL_FREQUENCY, 1)

    print("\n  mimic -- driven by a PhysX constraint, drives stay at zero")
    shown = ", ".join(nk.MIMIC_DRIVEN_JOINTS[:4])
    print(f"    {shown}, ... ({len(nk.MIMIC_DRIVEN_JOINTS)} joints)")

    print("\n" + "=" * 78)
    print("NOTES")
    print("=" * 78)
    print(
        "  'distal I' is the open-chain inertia of everything below the joint.\n"
        "  'load I' is what the joint actually carries while standing. At the\n"
        "  ankle these differ by a factor of ~543: sizing gains from the distal\n"
        "  column would be two orders of magnitude too soft.\n\n"
        "  'sat err' is the joint error in radians at which the derived stiffness\n"
        "  demands more torque than the URDF allows. For the legs this is around\n"
        "  0.09-0.28 rad, comfortably outside normal balance excursions.\n\n"
        "  A leg stiffness below its 'M g l' column cannot be stabilised by any\n"
        "  amount of damping, because gravity acts as a negative spring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
