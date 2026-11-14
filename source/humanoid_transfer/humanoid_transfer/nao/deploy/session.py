"""A run on the robot: from limp to the crouch, the policy, and back, with a safety layer.

:class:`Session` drives a :class:`~.controller.Controller` through a backend
(:mod:`~.backends`: LoLA on the robot, LoLA through a relay from a laptop, or a
simulator) in phases:

1. ``ramp`` -- stiffness from 0 to the run's over ``ramp_s``, holding the joints
   where they are;
2. ``pose`` -- the targets interpolated to the crouch the policy starts from;
3. ``hold`` -- the crouch held (until the chest button is pressed, with ``wait_button``:
   the robot is set on its feet first);
4. ``policy`` -- the tracker following the precomputed plans, one step every
   control period, the targets passed through :class:`Safety`;
5. ``stop`` -- the last targets held, then the crouch.

A fall (the torso tilted past ``fall_tilt``), the chest button, a motor past its
temperature limit, or the end of the plans stops the policy. On a fall every
joint goes soft (``fall_stiffness``): with a harness the robot hangs instead of
fighting it. Every control step is logged (:class:`Log`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .controller import Controller


@dataclass
class SafetyConfig:
    fall_tilt: float = 0.6
    """Torso roll or pitch, rad, past which the robot is falling."""
    max_target_rate: float = 12.0
    """Fastest a target may move, rad/s -- twice the leg motors' no-load speed: it
    catches a glitch, never the policy."""
    temperature_limit: float = 75.0
    """Motor temperature, deg C, at which the run stops."""
    fall_stiffness: float = 0.1
    run_stiffness: float = 1.0
    limit_margin: float = 0.0
    """How far inside the joint limits the targets are kept, rad."""


class Safety:
    """Bounds on the targets, and the conditions that end a run."""

    def __init__(self, config: SafetyConfig, lower: np.ndarray, upper: np.ndarray, dt: float):
        self.config = config
        self.lower = np.asarray(lower) + config.limit_margin
        self.upper = np.asarray(upper) - config.limit_margin
        self.dt = dt
        self.previous = None

    def reset(self, targets: np.ndarray) -> None:
        self.previous = np.asarray(targets, dtype=np.float64).copy()

    def bound(self, targets: np.ndarray) -> np.ndarray:
        out = np.clip(targets, self.lower, self.upper)
        if self.previous is not None:
            step = self.config.max_target_rate * self.dt
            out = np.clip(out, self.previous - step, self.previous + step)
        self.previous = out.copy()
        return out

    def stop_reason(
        self, gravity: np.ndarray, temperatures: np.ndarray, button: bool
    ) -> str | None:
        roll = np.arctan2(-gravity[1], -gravity[2])
        pitch = np.arcsin(np.clip(gravity[0], -1.0, 1.0))
        if max(abs(roll), abs(pitch)) > self.config.fall_tilt:
            return "fall"
        if temperatures.size and float(np.max(temperatures)) >= self.config.temperature_limit:
            return "temperature"
        if button:
            return "button"
        return None


@dataclass
class Log:
    """Every control step's readings, estimate, observation and targets."""

    rows: dict[str, list] = field(default_factory=dict)

    def add(self, **values) -> None:
        for key, value in values.items():
            self.rows.setdefault(key, []).append(np.asarray(value))

    def save(self, path: str | Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        np.savez_compressed(path, **arrays, **{k: np.asarray(v) for k, v in meta.items()})


@dataclass
class Phases:
    ramp_s: float = 1.5
    pose_s: float = 3.0
    hold_s: float = 1.0
    stop_s: float = 2.0
    wait_button: bool = False
    """After the hold, keep the crouch until the chest button is pressed and released."""


class Session:
    """One run of a controller on a backend."""

    def __init__(
        self,
        controller: Controller,
        backend,
        safety: SafetyConfig | None = None,
        phases: Phases | None = None,
        held: dict[str, float] | None = None,
    ) -> None:
        self.controller = controller
        self.backend = backend
        self.safety_config = safety or SafetyConfig()
        self.phases = phases or Phases()
        config = controller.config
        self.actuated = controller.actuated
        self.held = dict(held if held is not None else config.get("held_pos", {}))
        self.safety = Safety(
            self.safety_config,
            np.asarray(config["joint_lower"]),
            np.asarray(config["joint_upper"]),
            controller.dt,
        )
        self.log = Log()
        self.reason: str | None = None

    def _targets(self, actuated: np.ndarray) -> dict[str, float]:
        out = dict(self.held)
        out.update(dict(zip(self.actuated, (float(v) for v in actuated), strict=True)))
        return out

    def _wait(self, start: float, k: int) -> None:
        delay = start + k * self.controller.dt - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    def run(self, max_steps: int | None = None) -> str:
        """Run every phase; returns why the policy stopped."""
        c, dt, b = self.controller, self.controller.dt, self.backend
        reading = b.read()
        q0 = reading.joint(self.actuated)
        crouch = c.default
        run_stiffness = self.safety_config.run_stiffness
        start, k = time.perf_counter(), 0

        def ramp_steps(seconds: float) -> int:
            return max(1, int(round(seconds / dt)))

        # 1-3: ramp the stiffness, move to the crouch, hold.
        phases = [
            ("ramp", ramp_steps(self.phases.ramp_s)),
            ("pose", ramp_steps(self.phases.pose_s)),
            ("hold", ramp_steps(self.phases.hold_s)),
        ]
        for phase, steps in phases:
            for i in range(steps):
                s = (i + 1) / steps
                if phase == "ramp":
                    targets, stiffness = q0, run_stiffness * s
                elif phase == "pose":
                    w = 0.5 - 0.5 * np.cos(np.pi * s)
                    targets, stiffness = (1 - w) * q0 + w * crouch, run_stiffness
                else:
                    targets, stiffness = crouch, run_stiffness
                b.write(self._targets(targets), stiffness)
                reading = b.read()
                self.log.add(phase=_PHASES[phase], time=time.perf_counter() - start,
                             position=reading.position, targets=targets)  # fmt: skip
                k += 1
                self._wait(start, k)

        if self.phases.wait_button:
            # Hold the crouch until the chest button is pressed and released: the robot is
            # set on its feet (lowered from its harness) before the policy starts.
            pressed = False
            while True:
                b.write(self._targets(crouch), run_stiffness)
                reading = b.read()
                if reading.chest_button:
                    pressed = True
                elif pressed:
                    break
                k += 1
                self._wait(start, k)
            start, k = time.perf_counter(), 0

        # 4: the policy.
        c.reset()
        self.safety.reset(crouch)
        limit = c.steps if max_steps is None else min(max_steps, c.steps)
        self.reason = "end"
        last = crouch
        for _ in range(limit):
            reading = b.read()
            tic = time.perf_counter()
            gravity = reading.gravity()
            q = reading.joint(c.model.joint_names)
            loads = reading.foot_loads(c.model.total_mass)
            observation = c.observe(q, gravity, reading.gyroscope, loads)
            raw = c.act(observation)
            targets = self.safety.bound(raw)
            compute = time.perf_counter() - tic
            reason = self.safety.stop_reason(
                gravity, _legs(reading.temperature), reading.chest_button
            )
            odometry = np.append(c.estimator.position[0], c.estimator.yaw[0])
            self.log.add(
                phase=_PHASES["policy"],
                time=tic - start,
                position=reading.position,
                targets=targets,
                raw_targets=raw,
                observation=observation,
                estimate=c.estimate,
                odometry=odometry,
                gyro=reading.gyroscope,
                angles=reading.angles,
                accelerometer=reading.accelerometer,
                fsr=reading.fsr,
                temperature=reading.temperature,
                current=reading.current,
                compute_s=compute,
            )
            if reason is not None:
                self.reason = reason
                break
            b.write(self._targets(targets), run_stiffness)
            last = targets
            k += 1
            self._wait(start, k)

        # 5: stop.
        if self.reason == "fall":
            b.write(self._targets(last), self.safety_config.fall_stiffness)
            return self.reason
        steps = ramp_steps(self.phases.stop_s)
        for i in range(steps):
            w = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / steps)
            targets = (1 - w) * last + w * crouch
            b.write(self._targets(targets), run_stiffness)
            reading = b.read()
            self.log.add(phase=_PHASES["stop"], time=time.perf_counter() - start,
                         position=reading.position, targets=targets)  # fmt: skip
            k += 1
            self._wait(start, k)
        return self.reason


_PHASES = {"ramp": 0, "pose": 1, "hold": 2, "policy": 3, "stop": 4}
_LEG_INDEX = list(range(7, 18))
"""LoLA's leg joints (LHipYawPitch to RAnkleRoll)."""


def _legs(temperatures: np.ndarray) -> np.ndarray:
    return temperatures[_LEG_INDEX] if temperatures.size == 25 else temperatures


__all__ = ["Log", "Phases", "Safety", "SafetyConfig", "Session"]
