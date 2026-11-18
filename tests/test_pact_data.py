"""PACT windows: never across an episode end, speeds in the body's own time unit."""

from __future__ import annotations

import numpy as np
from humanoid_transfer.transfer.latent.pact.data import (
    load_body,
    split_by_environment,
    windows_of_capture,
)


def _capture(path, steps=12, envs=3, joints=2, done_at=((5, 1),)):
    rng = np.random.default_rng(0)
    done = np.zeros((steps, envs), dtype=bool)
    for t, e in done_at:
        done[t, e] = True
    # joint_pos encodes (step, env) so a window's rows can be checked exactly.
    step = np.arange(steps)[:, None, None]
    env = np.arange(envs)[None, :, None]
    joint_pos = (step * 10 + env + np.zeros((1, 1, joints))).astype(np.float32)
    np.savez(
        path,
        joint_names=np.array([f"j{i}" for i in range(joints)]),
        joint_pos=joint_pos,
        joint_vel=np.ones((steps, envs, joints), dtype=np.float32),
        projected_gravity=rng.normal(size=(steps, envs, 3)).astype(np.float32),
        base_ang_vel=np.ones((steps, envs, 3), dtype=np.float32),
        skill_state=np.repeat(joint_pos[..., :1], 27, axis=-1),
        done=done,
    )
    return path


def test_windows_never_cross_an_episode_end(tmp_path):
    path = _capture(tmp_path / "c.npz")
    obs, imu, z, env = windows_of_capture(path, history=3, leg_length=1.0, joint_names=("j0", "j1"))
    steps = obs[:, 0, :, 0] // 10
    envs = obs[:, 0, :, 0] % 10
    # consecutive steps of one environment
    assert (np.diff(steps, axis=1) == 1).all() and (envs == envs[:, :1]).all()
    # env 1 ended at step 5: no window of env 1 may hold step 5 before its last row
    for row, e in zip(steps, env, strict=True):
        if e == 1:
            assert 5 not in row[:-1]
    # the target is the state at the window's last step
    np.testing.assert_array_equal(z[:, 0], obs[:, 0, -1, 0])
    assert imu.shape == (len(z), 3, 6)


def test_speeds_are_in_the_bodys_time_unit(tmp_path):
    path = _capture(tmp_path / "c.npz", done_at=())
    length = 0.25
    obs, imu, _, _ = windows_of_capture(
        path, history=2, leg_length=length, joint_names=("j0", "j1")
    )
    unit = np.sqrt(length / 9.81)
    np.testing.assert_allclose(obs[..., 1], unit, rtol=1e-6)
    np.testing.assert_allclose(imu[..., 3:], unit, rtol=1e-6)


def test_split_keeps_environments_whole(tmp_path):
    a = _capture(tmp_path / "a.npz", envs=32, done_at=())
    b = _capture(tmp_path / "b.npz", envs=32, done_at=())
    windows = load_body("x", [a, b], history=2, leg_length=1.0, joint_names=("j0", "j1"))
    test = split_by_environment(windows.group)
    assert set(windows.group[test] % 1000) == set(range(24, 32))
    assert not set(windows.group[test]) & set(windows.group[~test])
    assert {g // 1000 for g in windows.group} == {0, 1}
