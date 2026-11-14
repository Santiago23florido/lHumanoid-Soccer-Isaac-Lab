"""The tracker as the robot runs it: sensors in, joint targets out, one control step at a time.

A deployment bundle (``nao/scripts/export_runtime.py``) holds everything the
simulator knew and the robot needs:

* ``body_model.json`` -- :class:`~.kinematics.BodyModel`'s dictionary;
* ``actor.npz`` -- :class:`~.policy.Actor`'s weights;
* ``runtime.json`` -- the control step, the actuated joints, the crouch the actions
  are offsets around, the action scale, the contract's normalisation and the
  tracker's units, and the policy's input layout;
* ``plans/<name>.npz`` -- precomputed plans (the generator never runs on the robot):
  for every control step the plan's next rows as the tracker sees them
  (``view``, in the contract's normalised units), whether a plan is held
  (``has_plan``) and where the reference will be over the view (``path``, metres
  and radians, relative to where the plans started).

:class:`Controller` builds the policy's observation exactly as the simulator's
observation manager does (``NaoTrackPolicyCfg`` with the robot's sensors,
:mod:`humanoid_transfer.nao.sim2real`), runs the actor and returns the targets.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .estimator import Estimator
from .kinematics import BodyModel
from .policy import Actor


def wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


class Bundle:
    """A deployment bundle read from its directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.config = json.loads((self.root / "runtime.json").read_text(encoding="utf-8"))
        self.model = BodyModel.load(self.root / "body_model.json")
        self.actor = Actor.load(self.root / "actor.npz")

    def plan(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.root / "plans" / f"{name}.npz", allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def plans(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "plans").glob("*.npz"))


class Controller:
    """One robot's tracker: :meth:`step` per control step."""

    def __init__(self, bundle: Bundle, plan: str) -> None:
        config = bundle.config
        self.bundle = bundle
        self.config = config
        self.dt = float(config["control_dt"])
        self.model = bundle.model
        self.actor = bundle.actor
        self.actuated = list(config["actuated"])
        self.default = np.asarray(config["default_pos"], dtype=np.float64)
        self.scale = float(config["action_scale"])
        self.mean = np.asarray(config["skill_mean"], dtype=np.float64)
        self.std = np.asarray(config["skill_std"], dtype=np.float64)
        units = config.get("units")
        self.units = (
            None if units is None else tuple(np.asarray(u, dtype=np.float64) for u in units)
        )
        self.limit = float(config["plan_limit"])
        self.length = float(config["leg_length"])
        self.layout = [(str(name), int(width)) for name, width in config["policy_terms"]]
        self.histories = {str(k): int(v) for k, v in config.get("history", {}).items()}
        index = {n: i for i, n in enumerate(self.model.joint_names)}
        self.actuated_index = np.array([index[n] for n in self.actuated])
        self.estimator = Estimator(self.model, 1, self.dt, config.get("pivot", "sole"))
        self.plan = bundle.plan(plan)
        if self.actor.inputs != sum(w for _, w in self.layout):
            raise ValueError("the actor's input width does not match the policy terms")
        self.reset()

    def reset(self) -> None:
        """Start an episode: the next step is the plans' first."""
        self.estimator.reset()
        self.t = 0
        self.last_action = np.zeros(len(self.actuated))
        self.previous_q = None
        self.anchor = None
        self.buffers = {name: deque(maxlen=n) for name, n in self.histories.items()}
        self.estimate = None

    @property
    def steps(self) -> int:
        return len(self.plan["has_plan"])

    @property
    def done(self) -> bool:
        return self.t >= self.steps

    def observe(
        self, q: np.ndarray, gravity: np.ndarray, gyro: np.ndarray, loads: np.ndarray
    ) -> np.ndarray:
        """The policy's input for this step, from the measured joint angles ``q`` (the body
        model's joint order), the IMU's gravity direction and gyroscope, and each foot's
        load in units of the robot's weight."""
        q = np.asarray(q, dtype=np.float64)
        z = self.estimator.update(q[None], gravity[None], gyro[None], loads[None])[0]
        estimate = (z - self.mean) / self.std
        self.estimate = estimate
        joints = q[self.actuated_index]
        velocity = (
            np.zeros_like(joints)
            if self.previous_q is None
            else (joints - self.previous_q) / self.dt
        )
        self.previous_q = joints
        t = min(self.t, self.steps - 1)
        position = self.estimator.position[0]
        yaw = float(self.estimator.yaw[0])
        if self.anchor is None:
            self.anchor = (position.copy(), yaw)
        terms = {
            "base_ang_vel": np.asarray(gyro, dtype=np.float64),
            "projected_gravity": np.asarray(gravity, dtype=np.float64),
            "joint_pos": joints - self.default,
            "joint_vel": velocity,
            "actions": self.last_action.copy(),
            "contract": self._contract(
                self.plan["view"][t], estimate, bool(self.plan["has_plan"][t])
            ),
            "contract_path": self._path(
                self.plan["path"][t], bool(self.plan["has_plan"][t]), position, yaw
            ),
        }
        for name, buffer in self.buffers.items():
            base = name[: -len("_history")]
            if not buffer:
                buffer.extend([terms[base]] * buffer.maxlen)
            else:
                buffer.append(terms[base])
            terms[name] = np.concatenate(list(buffer))
        return np.concatenate([terms[name] for name, _ in self.layout])

    def _contract(self, view: np.ndarray, estimate: np.ndarray, has_plan: bool) -> np.ndarray:
        relative = view - estimate[None]
        current = estimate
        if self.units is not None:
            scale, shift = self.units
            relative = relative * scale
            current = estimate * scale + shift
        if not has_plan:
            relative = np.zeros_like(relative)
        out = np.concatenate([relative.reshape(-1), current])
        return np.clip(np.nan_to_num(out, nan=0.0), -self.limit, self.limit)

    def _path(
        self, path: np.ndarray, has_plan: bool, position: np.ndarray, yaw: float
    ) -> np.ndarray:
        if not has_plan:
            return np.zeros(path.size)
        start, start_yaw = self.anchor
        c, s = np.cos(start_yaw), np.sin(start_yaw)
        ref_x = start[0] + c * path[:, 0] - s * path[:, 1]
        ref_y = start[1] + s * path[:, 0] + c * path[:, 1]
        dx, dy = ref_x - position[0], ref_y - position[1]
        cb, sb = np.cos(yaw), np.sin(yaw)
        out = np.stack(
            [(cb * dx + sb * dy) / self.length, (-sb * dx + cb * dy) / self.length,
             wrap(start_yaw + path[:, 2] - yaw)],
            axis=-1,
        )  # fmt: skip
        return np.clip(np.nan_to_num(out.reshape(-1), nan=0.0), -self.limit, self.limit)

    def act(self, observation: np.ndarray) -> np.ndarray:
        """The actuated joints' targets (``actuated`` order) for this step's observation."""
        action = self.actor(observation)
        self.last_action = action
        self.t += 1
        return self.default + self.scale * action

    def step(self, q, gravity, gyro, loads) -> np.ndarray:
        return self.act(self.observe(q, gravity, gyro, loads))


__all__ = ["Bundle", "Controller", "wrap"]
