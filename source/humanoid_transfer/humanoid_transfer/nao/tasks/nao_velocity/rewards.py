"""The oracle's air-time term, Isaac Lab's, and the bound that closes what the NAO made of it.

Kept apart from the environment configuration, which needs the Kit application, so the
tests can run it on a stand-in scene.
"""

from __future__ import annotations

import torch


def feet_air_time_biped(
    env, command_name: str, threshold: float, moving: float, sensor_cfg, bounded: bool = False
) -> torch.Tensor:
    """Isaac Lab's ``feet_air_time_positive_biped`` with its standing threshold a parameter.

    The original zeroes the reward below a command of 0.1 m/s, a constant of the G1's.
    In single stance it pays the shorter of the two feet's times in their mode (the stance
    foot's contact, the swing foot's flight), clamped at ``threshold``: a stance held
    indefinitely is paid ``threshold`` at every step. The NAO's oracle settled on long
    stances: on rough both seeds stand on one leg; on flat its stride period is 5.6–11.2
    ``T`` against the teacher's 1.0, its mean stance and flight both past the threshold's
    1.55 ``T`` (``transfer/paper/README.md``, B2). That this term is the reason is the
    hypothesis ``bounded`` tests.

    ``bounded`` pays a single stance only while neither foot has held its mode longer than
    ``threshold``. On a gait whose every stance and flight is shorter, the term is the
    original's, step for step; holding a foot up, or down while the other taps, past the
    threshold is paid nothing.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact > 0.0
    in_mode = torch.where(in_contact, contact, air)
    single = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single.unsqueeze(-1), in_mode, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    if bounded:
        reward = reward * (in_mode.max(dim=1)[0] <= threshold)
    command = env.command_manager.get_command(command_name)[:, :2]
    return reward * (torch.norm(command, dim=1) > moving)
