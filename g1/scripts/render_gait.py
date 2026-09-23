"""Render the trained teacher walking, as a strip of frames.

A learning curve says the reward went up. It does not say the robot walks: a
policy that shuffles, skates or vibrates in place can track a commanded velocity
and collect the same reward. The only cheap check is to look at it.

Produces a horizontal strip sampled across roughly one gait cycle, which makes
the swing and stance phases visible in a single static image -- enough to tell
walking from the failure modes that reward alone hides.

Offscreen capture crashes in the RTX renderer under ``--headless`` on some
hybrid-graphics machines, in ``rtx.scenedb`` during Hydra engine creation. This
script therefore runs with a window, like the NAO's renderer.

Usage::

    python g1/scripts/render_gait.py \
        --checkpoint logs/rsl_rl/g1_flat/<run>/exported/policy.pt
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render the teacher gait.")
parser.add_argument("--checkpoint", type=Path, required=True, help="TorchScript policy.")
parser.add_argument(
    "--output",
    type=Path,
    default=_REPO_ROOT / "g1" / "img" / "g1_gait.png",
    help="Where to write the strip.",
)
parser.add_argument("--frames", type=int, default=5, help="Frames in the strip.")
parser.add_argument(
    "--frame-stride",
    type=int,
    default=8,
    help="Control steps between frames. 8 at 50 Hz covers a gait cycle in 5.",
)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=720)
parser.add_argument(
    "--settle-steps", type=int, default=150, help="Steps before capture, so the gait is periodic."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.sensors import CameraCfg

import humanoid_transfer.tasks  # noqa: F401  (registers the task ids)
from humanoid_transfer.g1.tasks.g1_walk import PLAY_TASK_ID


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(PLAY_TASK_ID, num_envs=1)

    # The camera has to be part of the scene configuration, not built beside it.
    # A Camera constructed standalone never has its internal buffers allocated
    # -- InteractiveScene does that -- and the first call that needs them fails
    # with a missing _ALL_INDICES rather than with anything informative.
    cfg.scene.gait_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/GaitCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0, clipping_range=(0.05, 40.0)
        ),
    )

    env = gym.make(PLAY_TASK_ID, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    robot = inner.scene["robot"]
    camera = inner.scene["gait_camera"]

    policy = torch.jit.load(str(args_cli.checkpoint), map_location=device)
    policy.eval()

    obs, _ = env.reset()

    def act(observation):
        with torch.inference_mode():
            tensor = observation["policy"] if isinstance(observation, dict) else observation
            return policy(tensor)

    for _ in range(args_cli.settle_steps):
        obs, _, _, _, _ = env.step(act(obs))

    frames = []
    # One extra pass, discarded. The render product lags a step behind the pose
    # that was just written, so the first capture comes back empty.
    for index in range(args_cli.frames + 1):
        for _ in range(args_cli.frame_stride):
            obs, _, _, _, _ = env.step(act(obs))

        # Follow the robot: it is walking, so a fixed camera loses it.
        # Place the camera in the robot's *heading* frame, not the world's.
        # Initial yaw is randomised, so a world-frame offset shows the robot
        # from an arbitrary angle -- and a world-frame position reading makes a
        # robot walking forward look like one drifting backwards, which is
        # exactly how this gait was first misread.
        base = robot.data.root_pos_w[0].detach()
        quat = robot.data.root_quat_w[0].detach()
        yaw = torch.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        cos, sin = torch.cos(yaw), torch.sin(yaw)

        # Side view: 2.6 m to the robot's left, slightly ahead and above.
        side_x, side_y, height = 0.4, -2.6, 0.35
        offset = torch.stack(
            [
                cos * side_x - sin * side_y,
                sin * side_x + cos * side_y,
                torch.full_like(cos, height),
            ]
        )
        aim = torch.tensor([0.0, 0.0, -0.15], device=device, dtype=torch.float32)
        camera.set_world_poses_from_view(
            eyes=(base + offset).unsqueeze(0),
            targets=(base + aim).unsqueeze(0),
        )
        camera.update(inner.step_dt)

        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if index > 0:
            frames.append(rgb[..., :3] if rgb.shape[-1] == 4 else rgb)
        print(f"  frame {index + 1}/{args_cli.frames}  heading {float(yaw):+.2f} rad")

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required: pip install pillow", file=sys.stderr)
        return 1

    strip = np.concatenate([f.astype(np.uint8) for f in frames], axis=1)
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(args_cli.output)

    interval = args_cli.frame_stride * inner.step_dt
    print(f"\n  wrote {args_cli.output}  ({strip.shape[1]}x{strip.shape[0]})")
    print(f"  {args_cli.frames} frames, {interval * 1e3:.0f} ms apart\n")

    env.close()
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
