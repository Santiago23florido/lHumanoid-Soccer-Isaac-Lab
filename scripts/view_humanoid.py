"""Open Isaac Lab and display the built-in humanoid articulation.

Run from an Isaac Lab Python environment. This script intentionally does not
define a task, rewards, or a policy; it only spawns the humanoid so the asset can
be inspected in Isaac Sim.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="View the built-in Isaac Lab humanoid.")
parser.add_argument("--num-humanoids", type=int, default=1, help="Number of humanoids to spawn.")
parser.add_argument(
    "--spacing",
    type=float,
    default=2.5,
    help="Spacing between humanoids in meters.",
)
parser.add_argument(
    "--reset-interval",
    type=int,
    default=0,
    help="Reset every N simulation steps. Use 0 to disable periodic resets.",
)
parser.add_argument(
    "--hold-joints",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Continuously command the default joint pose with implicit actuators.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop after N simulation steps. Use 0 to run until the app closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

try:
    from isaaclab_assets.robots.humanoid import HUMANOID_CFG
except ModuleNotFoundError:
    from isaaclab_assets import HUMANOID_CFG  # type: ignore[attr-defined]


def design_scene(
    num_humanoids: int,
    spacing: float,
) -> tuple[dict[str, Articulation], torch.Tensor]:
    """Create a simple ground plane, light, and humanoid articulation."""
    if num_humanoids < 1:
        raise ValueError("--num-humanoids must be at least 1")

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    origins = []
    sim_utils.create_prim("/World/Humanoids", "Xform")
    for index in range(num_humanoids):
        x_offset = (index - (num_humanoids - 1) / 2.0) * spacing
        origin = [x_offset, 0.0, 0.0]
        origins.append(origin)
        sim_utils.create_prim(f"/World/Humanoids/Origin{index}", "Xform", translation=origin)

    humanoid_cfg = HUMANOID_CFG.copy()
    humanoid_cfg.prim_path = "/World/Humanoids/Origin.*/Robot"
    humanoid = Articulation(cfg=humanoid_cfg)

    return {"humanoid": humanoid}, torch.tensor(origins)


def reset_humanoid(robot: Articulation, origins: torch.Tensor) -> None:
    """Reset the humanoid root and joints to the config defaults."""
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    print("[INFO]: Humanoid reset to default pose.")


def run_simulator(
    sim: SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor,
    reset_interval: int,
    hold_joints: bool,
    max_steps: int,
) -> None:
    """Run the viewer simulation loop."""
    robot = entities["humanoid"]
    sim_dt = sim.get_physics_dt()
    count = 0

    reset_humanoid(robot, origins)

    while simulation_app.is_running():
        if reset_interval > 0 and count > 0 and count % reset_interval == 0:
            reset_humanoid(robot, origins)

        if hold_joints:
            robot.set_joint_position_target(robot.data.default_joint_pos)

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        count += 1
        if max_steps > 0 and count >= max_steps:
            break


def main() -> None:
    """Launch the humanoid viewer."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, -4.0, 2.6], [0.0, 0.0, 1.1])

    scene_entities, scene_origins = design_scene(args_cli.num_humanoids, args_cli.spacing)
    scene_origins = scene_origins.to(device=sim.device)

    sim.reset()
    print("[INFO]: Setup complete. Close the Isaac Sim window or press Ctrl+C to stop.")
    run_simulator(
        sim,
        scene_entities,
        scene_origins,
        reset_interval=args_cli.reset_interval,
        hold_joints=args_cli.hold_joints,
        max_steps=args_cli.max_steps,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        fast_headless_smoke = args_cli.headless and args_cli.max_steps > 0
        simulation_app.close(wait_for_replicator=not fast_headless_smoke, skip_cleanup=fast_headless_smoke)
