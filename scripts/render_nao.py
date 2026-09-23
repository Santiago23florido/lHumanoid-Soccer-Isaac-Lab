"""Render the NAO in its nominal standing posture to a PNG.

Produces the figure used in the repository README. The robot is held in the
nominal crouch of :data:`NOMINAL_STAND_JOINT_POS` -- the same posture every
controller and the policy are referenced to -- so the image shows the actual
configuration the numbers in the documentation describe, not a default T-pose.

The scene is deliberately minimal: ground plane, dome light, one articulation.
Physics is stepped for a moment before capture so the actuators settle the pose
against gravity rather than showing the kinematic ideal.

Usage::

    python scripts/render_nao.py --headless --output docs/img/nao_stand.png
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

parser = argparse.ArgumentParser(description="Render the NAO standing posture.")
parser.add_argument(
    "--output",
    type=Path,
    default=_REPO_ROOT / "docs" / "img" / "nao_stand.png",
    help="Where to write the PNG.",
)
parser.add_argument(
    "--width", type=int, default=1280, help="Image width in pixels. Default 1280."
)
parser.add_argument(
    "--height", type=int, default=960, help="Image height in pixels. Default 960."
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=120,
    help="Physics steps before capture, so the posture settles under gravity.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The capture needs a render product, so the renderer must stay enabled even
# when the window does not. --headless alone is fine; --kit_args disabling
# Vulkan is not, and would produce a blank image.
args_cli.enable_cameras = True

from humanoid_soccer_lab.nao.assets.nao_paths import describe_missing_meshes, meshes_are_available

if not meshes_are_available():
    print("\n" + describe_missing_meshes() + "\n", file=sys.stderr)
    raise SystemExit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext

from humanoid_soccer_lab.nao.assets.nao import NAO_STAND_CFG


def design_scene() -> Articulation:
    """Ground, light and one NAO at the origin in its standing configuration."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95))
    light_cfg.func("/World/DomeLight", light_cfg)

    sim_utils.create_prim("/World/Robots", "Xform")
    robot_cfg = NAO_STAND_CFG.replace(prim_path="/World/Robots/Nao")
    return Articulation(robot_cfg)


def main() -> int:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0)
    sim = SimulationContext(sim_cfg)

    robot = design_scene()

    camera_cfg = CameraCfg(
        prim_path="/World/RenderCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        # 20 mm is wide enough to frame a 0.58 m robot from 1.4 m without
        # clipping the head, which a 32 mm lens does at this distance.
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, clipping_range=(0.05, 20.0)
        ),
    )
    camera = Camera(camera_cfg)

    sim.reset()

    # Three-quarter view aimed at mid-torso, far enough back to frame the whole
    # robot from the soles to the head with the support polygon visible.
    # Torch tensors on the sim device: the view helper runs through
    # torch.nn.functional.normalize, which rejects NumPy arrays.
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[0.95, -0.95, 0.42]], device=sim.device),
        targets=torch.tensor([[0.0, 0.0, 0.27]], device=sim.device),
    )

    state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    robot.reset()

    for _ in range(args_cli.settle_steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt())

    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:  # drop alpha; the PNG is opaque
        rgb = rgb[..., :3]

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required to write the PNG: pip install pillow", file=sys.stderr)
        return 1

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(args_cli.output)

    height = float(robot.data.root_pos_w[0, 2])
    print(f"\n  wrote {args_cli.output}  ({args_cli.width}x{args_cli.height})")
    print(f"  base height after settling: {height:.4f} m\n")
    return 0


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
