"""Run the robot's runtime in MuJoCo: the deployment bundle in a second physics engine.

The exact code the robot runs (:class:`humanoid_transfer.nao.deploy.controller.Controller`:
the estimator, the precomputed plans, the NumPy actor) drives the MuJoCo NAO of
``nao/scripts/build_mjcf.py``, every 20 ms, from the sensors the robot has --
joint angles (12-bit), the IMU's roll and pitch, the gyroscope, each foot's
contact force as its force sensors read it -- with its targets reaching the
drives ``--delay-ms`` later. It starts in the crouch, as the robot
does after its hold phase: the crouch held by the drives until the robot rests on its
feet. Reports falls, the walking speed and how far the robot went, per run::

    python nao/scripts/sim2sim.py --bundle nao/bundles/<name> --plan forward --runs 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.deploy.controller import Bundle, Controller  # noqa: E402

RESOLUTION = 2 * np.pi / 4096


class MujocoNao:
    """The MuJoCo NAO with the robot's sensors and a latency on its targets."""

    def __init__(
        self,
        path: Path,
        bundle: Bundle,
        delay: float,
        friction: float | None,
        settle: float = 1.0,
        contact_time: float | None = None,
        elliptic: bool = False,
    ) -> None:
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        if friction is not None:
            self.model.geom_friction[:, 0] = friction
        if contact_time is not None:
            # Contact stiffness: MuJoCo's soft contacts settle in about this time constant.
            self.model.geom_solref[:, 0] = contact_time
        if elliptic:
            # Elliptic friction cones with a high impedance ratio: less creep under load.
            self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            self.model.opt.impratio = 10.0
        self.data = mujoco.MjData(self.model)
        self.bundle = bundle
        self.dt = float(bundle.config["control_dt"])
        self.substeps = int(round(self.dt / self.model.opt.timestep))
        self.delay_steps = int(round(delay / self.model.opt.timestep))
        names = bundle.model.joint_names
        self.qpos_index = [self.model.jnt_qposadr[self.model.joint(n).id] for n in names]
        self.actuators = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.base = self.model.body("base_link").id
        feet = ("l_ankle", "r_ankle")
        self.feet = [self.model.body(f).id for f in feet]
        self.mass = float(self.model.body_subtreemass[self.base])
        self.settle = settle
        """Seconds the drives hold the crouch before a run."""

    def reset(self, perturb: float, rng: np.random.Generator) -> None:
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        drives = self.bundle.config["drives"]
        for name, drive in drives.items():
            try:
                adr = m.jnt_qposadr[m.joint(name).id]
            except KeyError:
                continue
            d.qpos[adr] = drive["default"]
        mj.mj_forward(m, d)
        # Stand the feet's lowest vertex on the floor, at rest.
        d.qpos[2] += 0.5 - self.lowest_point()
        mj.mj_forward(m, d)
        d.qpos[2] -= self.lowest_point() - 0.0005
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        targets = np.array([drives[self.model.actuator(i).name]["default"] for i in range(m.nu)])
        self.queue = [targets.copy() for _ in range(self.delay_steps + 1)]
        self.data.ctrl[:] = targets
        # The drives hold the crouch until the robot rests on its feet, as a session's hold
        # phase does on the robot; then the push the run starts with.
        for _ in range(int(round(self.settle / m.opt.timestep))):
            mj.mj_step(m, d)
        d.qvel[:2] += rng.uniform(-perturb, perturb, size=2)

    def lowest_point(self) -> float:
        """The lowest vertex of the feet's collision meshes, world z."""
        m, d = self.model, self.data
        low = np.inf
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in self.feet or m.geom_type[g] != self.mj.mjtGeom.mjGEOM_MESH:
                continue
            mesh = m.geom_dataid[g]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = m.mesh_vert[start : start + count]
            world = d.geom_xpos[g] + vertices @ d.geom_xmat[g].reshape(3, 3).T
            low = min(low, float(world[:, 2].min()))
        return low

    def foot_forces(self) -> np.ndarray:
        mj, m, d = self.mj, self.model, self.data
        force = np.zeros(2)
        buffer = np.zeros(6)
        for i in range(d.ncon):
            contact = d.contact[i]
            bodies = (m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2])
            for k, foot in enumerate(self.feet):
                if foot in bodies:
                    mj.mj_contactForce(m, d, i, buffer)
                    force[k] += abs(buffer[0])
        return force

    def sensors(self):
        d = self.data
        q = np.array([d.qpos[a] for a in self.qpos_index])
        q = np.round(q / RESOLUTION) * RESOLUTION
        w, x, y, z = d.qpos[3:7]
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.qvel[3:6].copy()  # a free joint's angular velocity is in the body frame
        loads = self.foot_forces() / (self.bundle.model.total_mass * 9.81)
        return q, gravity, gyro, loads, rot

    def step(self, targets: dict[str, float]) -> None:
        latest = self.queue[-1].copy()
        for name, value in targets.items():
            if name in self.actuators:
                latest[self.actuators[name]] = value
        for _ in range(self.substeps):
            self.queue.append(latest)
            self.data.ctrl[:] = self.queue.pop(0)
            self.mj.mj_step(self.model, self.data)


class MujocoBackend:
    """The MuJoCo NAO behind LoLA's interface (``deploy.lola.Reading``), for whole sessions.

    Physics advances one control step at every :meth:`read`; stiffness scales the drives'
    gains as LoLA's stiffness scales the motors' current."""

    def __init__(self, nao: MujocoNao) -> None:
        self.nao = nao
        self.targets: dict[str, float] = {}
        m = nao.model
        self.gain = m.actuator_gainprm[:, 0].copy()
        self.bias = m.actuator_biasprm[:, :3].copy()
        self.started = False

    def read(self):
        from humanoid_transfer.nao.deploy import lola

        nao = self.nao
        if self.started:
            nao.step(self.targets)
        self.started = True
        q_all = {
            n: nao.data.qpos[nao.model.jnt_qposadr[nao.model.joint(n).id]] for n in lola.JOINTS
        }
        _, gravity, gyro, loads, rot = nao.sensors()
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arcsin(-np.clip(rot[2, 0], -1.0, 1.0))
        weight = nao.bundle.model.total_mass
        fsr = np.repeat(loads * weight / 4.0, 4)
        message = {
            "Position": [float(np.round(q_all[n] / RESOLUTION) * RESOLUTION) for n in lola.JOINTS],
            "Gyroscope": [float(v) for v in gyro],
            "Angles": [float(roll), float(pitch)],
            "Accelerometer": [float(v) for v in -9.81 * gravity],
            "FSR": [float(v) for v in fsr],
            "Temperature": [40.0] * 25,
            "Touch": [0.0] * 14,
        }
        return lola.Reading(message)

    def write(self, positions: dict, stiffness) -> None:
        self.targets = dict(positions)
        scale = float(stiffness) if not isinstance(stiffness, dict) else 1.0
        m = self.nao.model
        m.actuator_gainprm[:, 0] = self.gain * scale
        m.actuator_biasprm[:, 1:3] = self.bias[:, 1:3] * scale

    def close(self) -> None:
        pass


def run_session(nao: MujocoNao, bundle: Bundle, plan: str, seed: int) -> dict:
    """A whole :class:`~humanoid_transfer.nao.deploy.session.Session` (stand-up phases, safety,
    LoLA's message) on the MuJoCo NAO, starting stiff in the crouch."""
    from humanoid_transfer.nao.deploy.session import Phases, Session

    nao.reset(0.0, np.random.default_rng(seed))
    backend = MujocoBackend(nao)
    session = Session(Controller(bundle, plan), backend, phases=Phases(0.02, 0.02, 0.5, 1.0))
    start = nao.data.qpos[:2].copy()
    reason = session.run()
    moved = nao.data.qpos[:2] - start
    compute = session.log.rows.get("compute_s", [])
    return {
        "seed": seed,
        "stopped": reason,
        "policy_steps": len(compute),
        "forward_m": float(moved[0]),
        "distance_m": float(np.linalg.norm(moved)),
        "compute_ms_mean": 1000 * float(np.mean(compute)) if compute else None,
        "compute_ms_max": 1000 * float(np.max(compute)) if compute else None,
    }


def run(
    nao: MujocoNao, bundle: Bundle, plan: str, steps: int | None, perturb: float, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    controller = Controller(bundle, plan)
    nao.reset(perturb, rng)
    controller.reset()
    held = bundle.config.get("held_pos", {})
    limit = controller.steps if steps is None else min(steps, controller.steps)
    start = nao.data.qpos[:2].copy()
    heights, fell_at = [], None
    for k in range(limit):
        q, gravity, gyro, loads, rot = nao.sensors()
        targets = controller.step(q, gravity, gyro, loads)
        nao.step({**held, **dict(zip(controller.actuated, targets, strict=True))})
        height = float(nao.data.qpos[2])
        tilt = float(np.arccos(np.clip(rot[2, 2], -1, 1)))
        heights.append(height)
        if height < 0.18 or tilt > 0.8:
            fell_at = k * controller.dt
            break
    moved = nao.data.qpos[:2] - start
    seconds = len(heights) * controller.dt
    command = bundle.plan(plan)["command"]
    walking = np.abs(command[:, 0]) > 0
    return {
        "seed": seed,
        "fell_at_s": fell_at,
        "seconds": seconds,
        "distance_m": float(np.linalg.norm(moved)),
        "forward_m": float(moved[0]),
        "speed_m_s": float(np.linalg.norm(moved) / max(seconds, 1e-9)),
        "walking_s": float(walking[: len(heights)].sum() * controller.dt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--mjcf", type=Path, default=ROOT / "nao" / "assets" / "generated" / "nao_mujoco.xml"
    )
    parser.add_argument("--plan", default="forward")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--perturb", type=float, default=0.02, help="Initial base speed, m/s, ±.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--settle", type=float, default=1.0, help="Crouch held first, s.")
    parser.add_argument("--contact-time", type=float, default=None, help="Contact solref, s.")
    parser.add_argument("--elliptic", action="store_true", help="Elliptic cones, impratio 10.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--session",
        action="store_true",
        help="Run the robot's whole Session (phases, safety, LoLA's message) in real time.",
    )
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    nao = MujocoNao(
        args.mjcf,
        bundle,
        args.delay_ms / 1000.0,
        args.friction,
        args.settle,
        args.contact_time,
        args.elliptic,
    )
    if args.session:
        result = run_session(nao, bundle, args.plan, 0)
        print(json.dumps(result))
        return 0 if result["stopped"] == "end" else 1
    runs = [
        run(nao, bundle, args.plan, args.steps, args.perturb, seed) for seed in range(args.runs)
    ]
    for r in runs:
        print(json.dumps(r))
    summary = {
        "plan": args.plan,
        "delay_ms": args.delay_ms,
        "friction": args.friction,
        "falls": sum(r["fell_at_s"] is not None for r in runs),
        "runs": len(runs),
        "mean_speed_m_s": float(np.mean([r["speed_m_s"] for r in runs])),
        "mean_forward_m": float(np.mean([r["forward_m"] for r in runs])),
        "results": runs,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
