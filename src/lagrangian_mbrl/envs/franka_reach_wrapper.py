"""Isaac Lab Franka *reach* task wrapper.

Wraps an Isaac Lab (Orbit) Franka manipulation environment to expose exactly
what the MBRL loop and the dynamics models need — in particular the
**generalized coordinates** ``(q, q̇)`` and applied torques ``τ``, which the
Deep Lagrangian Network requires (the default RL observation/action spaces do
not surface these cleanly).

Why a custom wrapper
--------------------
- DeLaN learns ``M(q) q̈ + c + g = τ``; we must reliably read joint positions,
  velocities, and either applied torques or commanded targets, plus ``dt``.
- We want a single object that the generic :class:`MBRLTrainer` can drive,
  regardless of whether the underlying RL library is RSL-RL or SKRL.
- We want to log ``(q, q̇, τ, q̈)`` transitions for offline dynamics fitting in
  Phase 0 (the first deliverable).

Isaac Lab references
--------------------
- Isaac Lab docs: https://isaac-sim.github.io/IsaacLab/
- Mittal et al. "Orbit / Isaac Lab." RA-L 2023.
- Franka reach task config lives in Isaac Lab's manipulation task suite; this
  wrapper composes/overrides it rather than reimplementing the scene.

TODO (Phase 0, see PROJECT_PLAN.md):
  [ ] Construct the underlying Isaac Lab Franka-reach env (headless, GPU,
      env-count tuned for 8 GB VRAM).
  [ ] Expose obs that include (q, q̇); expose a way to read applied torques.
  [ ] Implement reset/step returning a dict with keys: q, qd, tau, reward,
      done, info, and (when available) qdd or a finite-difference estimate.
  [ ] Implement a `log_transitions(...)` helper to dump an offline dataset.
  [ ] Provide the reach reward + success threshold for steps-to-threshold.
  [ ] Later: subclass for push / peg-insertion (contact tasks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FrankaReachConfig:
    num_envs: int = 64  # tune to fit the RTX 4070 Laptop GPU (8 GB)
    episode_length_s: float = 5.0
    control_dt: float = 1.0 / 60.0
    action_mode: str = "joint_torque"  # "joint_torque" | "joint_position" | "osc"
    device: str = "cuda"
    headless: bool = True
    seed: int = 0
    success_threshold_m: float = 0.05  # end-effector distance for "reached"


class FrankaReachEnv:
    """Thin, MBRL-friendly wrapper around an Isaac Lab Franka-reach env.

    Notes
    -----
    Importing Isaac Lab pulls in Isaac Sim and must happen *after* the simulator
    app is launched. Keep those imports inside methods (lazy) so this module can
    be imported (and the package tested) without a running simulator.
    """

    def __init__(self, config: FrankaReachConfig | None = None) -> None:
        self.config = config or FrankaReachConfig()
        # TODO: lazily launch the Isaac Sim app and build the underlying env.
        raise NotImplementedError("FrankaReachEnv not yet implemented — see TODOs.")

    def reset(self) -> dict[str, Any]:
        """Reset all envs; return a dict including ``q`` and ``qd``. TODO."""
        raise NotImplementedError

    def step(self, action: Any) -> dict[str, Any]:
        """Apply ``action``; return dict with q, qd, tau, reward, done, info. TODO."""
        raise NotImplementedError

    def log_transitions(self, path: str, num_steps: int) -> None:
        """Roll out a policy and dump ``(q, q̇, τ, q̈)`` for offline dynamics
        fitting (Phase 0 deliverable). TODO."""
        raise NotImplementedError

    def close(self) -> None:
        """Shut down the simulator app. TODO."""
        raise NotImplementedError
