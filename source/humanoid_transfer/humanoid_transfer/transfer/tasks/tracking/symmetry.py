"""Training a tracker on every transition and its mirror image (rsl-rl symmetry augmentation).

The maps are :mod:`humanoid_transfer.transfer.symmetry`'s, assembled here from
a running environment: the joints from the body's measured mirror in the
articulation's order, each observation group from its active terms, and the
plan's units from the contract the environment loaded. rsl-rl's PPO doubles
each minibatch with the reflected observations and actions (Mittal et al.,
2024), so the policy is fitted to the mirror-symmetric problem the family and
the robot pose, not to whichever side its own rollouts happened to favour.

A body's tracking agent names a module-level function that binds its mirror,
since rsl-rl resolves the function by name::

    def augment(env, obs=None, actions=None):
        return mirror_augmentation(env, obs, actions, mirror=G1_MIRROR)
"""

from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

from ...symmetry import load_joint_mirror, observation_mirror
from .command import ReferenceCommand


class TrackerMirror:
    """The reflection of a tracking environment's observation groups and actions."""

    def __init__(self, env, mirror: Path | str) -> None:
        inner = env.unwrapped
        # The actuated joints, in the actions' order; the joint observations read the same.
        (action,) = inner.action_manager._terms.values()
        joints = load_joint_mirror(action._joint_names, mirror)
        if inner.action_manager.total_action_dim != joints.shape[0]:
            raise ValueError("the actions are not one per joint of the measured mirror")
        commands = inner.command_manager._terms.values()
        term = next(t for t in commands if isinstance(t, ReferenceCommand))
        norm = term.generator.skill_norm
        mean, std = norm.mean.cpu().double().numpy(), norm.std.cpu().double().numpy()
        manager = inner.observation_manager
        device = inner.device
        self.joints = torch.as_tensor(joints, dtype=torch.float32, device=device)
        self.groups: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for group, names in manager.active_terms.items():
            shapes = manager.group_obs_term_dim[group]
            layout = [(n, int(torch.tensor(s).prod())) for n, s in zip(names, shapes, strict=True)]
            try:
                affine = observation_mirror(layout, joints, mean, std)
            except KeyError:
                continue  # a group no learner reads (the teacher's labels)
            self.groups[group] = (
                torch.as_tensor(affine.linear, dtype=torch.float32, device=device),
                torch.as_tensor(affine.offset, dtype=torch.float32, device=device),
            )

    def observation(self, group: str, x: torch.Tensor) -> torch.Tensor:
        linear, offset = self.groups[group]
        return x @ linear.T + offset

    def action(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.joints.T


_MIRRORS: dict[int, TrackerMirror] = {}


@torch.no_grad()
def mirror_augmentation(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
    mirror: Path | str = "",
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Each sample followed by its mirror image: the batch doubles, originals first."""
    key = id(env)
    if key not in _MIRRORS:
        _MIRRORS[key] = TrackerMirror(env, mirror)
    reflect = _MIRRORS[key]
    augmented = None
    if obs is not None:
        n = obs.batch_size[0]
        augmented = obs.repeat(2)
        for group in obs.keys():
            augmented[group][n:] = reflect.observation(group, obs[group])
    doubled = None
    if actions is not None:
        doubled = torch.cat([actions, reflect.action(actions)], dim=0)
    return augmented, doubled


__all__ = ["TrackerMirror", "mirror_augmentation"]
