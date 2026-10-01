"""Checks that the skill state is shared between bodies by construction.

The claim the transfer study makes about this state is not "it is a good
summary of a gait" -- that is for the data to say. It is that the state does
not depend on things a skill should not depend on: the size of the body, where
it stands, which way it faces, and which side is called left. Each of those is
a group of transformations, and each gets a test that applies it exactly.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from humanoid_transfer.common.capturability import dcm
from humanoid_transfer.common.frames import quat_to_matrix, yaw_matrix
from humanoid_transfer.common.scaling import froude_number as froude_number_si
from humanoid_transfer.common.similarity import UnitSystem
from humanoid_transfer.common.skill_state import (
    SKILL_LAYOUT,
    MultibodySnapshot,
    contact_state,
    dcm_offset,
    encode,
    froude_number,
    mirror,
    mirror_matrix,
)
from humanoid_transfer.nao.assets import nao_kinematics as nk

BODIES = 7
UNITS = UnitSystem(length=0.70, mass=30.0)
REFLECTION = np.diag([1.0, -1.0, 1.0])


def _random_snapshot(seed: int, batch: tuple[int, ...] = ()) -> MultibodySnapshot:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(*batch, BODIES, 3, 3))
    positions = rng.normal(size=(*batch, BODIES, 3))
    positions[..., 2] = np.abs(positions[..., 2]) + 0.2
    return MultibodySnapshot(
        masses=rng.uniform(0.5, 6.0, size=(*batch, BODIES)),
        inertia_local=a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3),
        link_rot=quat_to_matrix(rng.normal(size=(*batch, BODIES, 4))),
        com_pos=positions,
        com_lin_vel=rng.normal(size=(*batch, BODIES, 3)),
        ang_vel=rng.normal(size=(*batch, BODIES, 3)),
        foot_force=rng.normal(size=(*batch, 2, 3)) * 100.0,
        base_index=0,
        foot_index=(5, 6),
        sole_from_com=rng.normal(size=(2, 3)) * 0.05,
    )


# --- layout -----------------------------------------------------------------


def test_the_layout_has_the_27_components_it_documents() -> None:
    names = SKILL_LAYOUT.component_names()
    assert SKILL_LAYOUT.size == 27
    assert len(names) == len(set(names)) == 27
    assert "foot_position.left.y" in names
    assert "foot_load.right" in names


def test_unpacking_a_packed_state_returns_each_field_in_its_unit() -> None:
    snapshot = _random_snapshot(0)
    z = encode(snapshot, UNITS)
    fields = SKILL_LAYOUT.unpack(z)
    assert fields["foot_position"].shape == (2, 3)
    assert fields["foot_load"].shape == (2,)
    assert np.ndim(fields["height"]) == 0
    expected_load = snapshot.foot_force[..., 2] / (UNITS.mass * UNITS.gravity)
    assert fields["foot_load"] == pytest.approx(expected_load)


# --- similarity -------------------------------------------------------------


def _rescaled(snapshot: MultibodySnapshot, lam: float, mu: float) -> MultibodySnapshot:
    """The dynamically similar motion of a body ``lam`` times as long and ``mu`` as heavy.

    At fixed gravity, time scales by ``sqrt(lam)``: velocities by ``sqrt(lam)``,
    angular velocities by ``1/sqrt(lam)``, forces by ``mu``, inertias by
    ``mu lam^2``.
    """
    root = math.sqrt(lam)
    return replace(
        snapshot,
        masses=snapshot.masses * mu,
        inertia_local=snapshot.inertia_local * mu * lam**2,
        com_pos=snapshot.com_pos * lam,
        com_lin_vel=snapshot.com_lin_vel * root,
        ang_vel=snapshot.ang_vel / root,
        foot_force=snapshot.foot_force * mu,
        sole_from_com=snapshot.sole_from_com * lam,
    )


@pytest.mark.parametrize(("lam", "mu"), [(0.384, 0.165), (2.0, 8.0), (0.5, 0.5)])
def test_dynamically_similar_motions_of_different_bodies_encode_identically(
    lam: float, mu: float
) -> None:
    """The whole point. (0.384, 0.165) is roughly NAO over G1 in leg length and mass."""
    snapshot = _random_snapshot(1)
    small = UnitSystem(UNITS.length * lam, UNITS.mass * mu)
    assert encode(_rescaled(snapshot, lam, mu), small) == pytest.approx(encode(snapshot, UNITS))


def test_copying_velocities_unscaled_is_not_a_similar_motion() -> None:
    """Negative control: geometric scaling alone, speeds copied, must change the state.

    This is the "copy the teacher's speed" error the Froude section of the plan
    warns about. If the encoder were blind to it the invariance above would
    mean nothing.
    """
    lam, mu = 0.384, 0.165
    snapshot = _random_snapshot(2)
    naive = replace(
        _rescaled(snapshot, lam, mu),
        com_lin_vel=snapshot.com_lin_vel,
        ang_vel=snapshot.ang_vel,
    )
    small = UnitSystem(UNITS.length * lam, UNITS.mass * mu)
    assert not np.allclose(encode(naive, small), encode(snapshot, UNITS))


# --- gauge ------------------------------------------------------------------


@pytest.mark.parametrize("yaw", [0.3, -2.4, math.pi])
def test_where_the_robot_stands_and_which_way_it_faces_drop_out(yaw: float) -> None:
    snapshot = _random_snapshot(3)
    turn = yaw_matrix(yaw)
    shift = np.array([4.0, -7.5, 0.0])
    moved = replace(
        snapshot,
        com_pos=snapshot.com_pos @ turn.T + shift,
        com_lin_vel=snapshot.com_lin_vel @ turn.T,
        ang_vel=snapshot.ang_vel @ turn.T,
        link_rot=turn @ snapshot.link_rot,
        foot_force=snapshot.foot_force @ turn.T,
    )
    assert encode(moved, UNITS) == pytest.approx(encode(snapshot, UNITS))


# --- mirror -----------------------------------------------------------------


def _mirrored(snapshot: MultibodySnapshot) -> MultibodySnapshot:
    """Reflect the world through the sagittal plane; left foot becomes right."""
    s = REFLECTION
    left, right = snapshot.foot_index
    return replace(
        snapshot,
        com_pos=snapshot.com_pos @ s,
        com_lin_vel=snapshot.com_lin_vel @ s,
        ang_vel=-snapshot.ang_vel @ s,
        link_rot=s @ snapshot.link_rot @ s,
        inertia_local=s @ snapshot.inertia_local @ s,
        foot_force=(snapshot.foot_force @ s)[..., ::-1, :],
        foot_index=(right, left),
        sole_from_com=(snapshot.sole_from_com @ s)[::-1],
    )


def test_the_mirror_image_of_a_motion_encodes_to_the_mirrored_state() -> None:
    """Equivariance: polar vectors flip y, axial vectors flip x and z, sides swap.

    The operator is generated from the field declarations; the reflected
    snapshot is built from the physics. Agreement checks every sign rule at once.
    """
    snapshot = _random_snapshot(4)
    assert encode(_mirrored(snapshot), UNITS) == pytest.approx(mirror(encode(snapshot, UNITS)))


def test_the_mirror_is_an_involution_and_a_signed_permutation() -> None:
    m = mirror_matrix()
    assert m @ m == pytest.approx(np.eye(SKILL_LAYOUT.size))
    assert np.abs(m).sum(axis=0) == pytest.approx(np.ones(SKILL_LAYOUT.size))
    assert np.abs(m).sum(axis=1) == pytest.approx(np.ones(SKILL_LAYOUT.size))


def test_encoding_is_batched_over_time_and_environments() -> None:
    snapshot = _random_snapshot(5, batch=(4, 3))
    z = encode(snapshot, UNITS)
    single = MultibodySnapshot(
        **{
            name: getattr(snapshot, name)[2, 1]
            for name in ("masses", "inertia_local", "link_rot", "com_pos", "com_lin_vel")
        },
        ang_vel=snapshot.ang_vel[2, 1],
        foot_force=snapshot.foot_force[2, 1],
        base_index=snapshot.base_index,
        foot_index=snapshot.foot_index,
        sole_from_com=snapshot.sole_from_com,
    )
    assert z.shape == (4, 3, 27)
    assert z[2, 1] == pytest.approx(encode(single, UNITS))


# --- physics in body units --------------------------------------------------


@pytest.mark.parametrize(("length", "mass"), [(0.70, 32.0), (0.248, 5.3)])
def test_a_falling_body_accelerates_at_one_unit_whatever_its_size(
    length: float, mass: float
) -> None:
    """``dv*/dt* = sum(load) - 1``: with no contact, ``-1`` on any body.

    The dimensionless form of Newton's law is the same equation for the G1 and
    the NAO; this checks the encoder's units are the ones that make it so.
    """
    units = UnitSystem(length, mass)
    snapshot = _random_snapshot(6)
    snapshot = replace(
        snapshot,
        com_lin_vel=np.zeros_like(snapshot.com_lin_vel),
        ang_vel=np.zeros_like(snapshot.ang_vel),
        foot_force=np.zeros_like(snapshot.foot_force),
    )
    dt = 1e-3
    later = replace(
        snapshot,
        com_pos=snapshot.com_pos - np.array([0.0, 0.0, 0.5 * units.gravity * dt**2]),
        com_lin_vel=snapshot.com_lin_vel - np.array([0.0, 0.0, units.gravity * dt]),
    )
    vz = SKILL_LAYOUT.slice_of("com_velocity").start + 2
    rate = (encode(later, units)[vz] - encode(snapshot, units)[vz]) / (dt / units.time)
    loads = SKILL_LAYOUT.unpack(encode(snapshot, units))["foot_load"]
    assert rate == pytest.approx(loads.sum() - 1.0)


def test_the_froude_number_is_read_off_the_state() -> None:
    z = encode(_random_snapshot(7), UNITS)
    v = SKILL_LAYOUT.unpack(z)["com_velocity"] * UNITS.velocity
    assert froude_number(z) == pytest.approx(froude_number_si(math.hypot(v[0], v[1]), UNITS.length))


def test_the_dcm_offset_matches_the_lipm_definition_in_si_units() -> None:
    z = encode(_random_snapshot(8), UNITS)
    fields = SKILL_LAYOUT.unpack(z)
    height = float(fields["height"]) * UNITS.length
    velocity = fields["com_velocity"][:2] * UNITS.velocity
    expected = dcm(np.zeros(2), velocity, math.sqrt(UNITS.gravity / height)) / UNITS.length
    assert dcm_offset(z) == pytest.approx(expected)


def test_a_contact_threshold_in_body_weights_means_the_same_on_any_body() -> None:
    """Five percent of the weight is 15.7 N on a G1 and 2.6 N on a NAO; both count."""
    z = np.zeros(SKILL_LAYOUT.size)
    loads = SKILL_LAYOUT.slice_of("foot_load")
    z[loads] = [0.06, 0.04]
    assert contact_state(z).tolist() == [True, False]


# --- a real body ------------------------------------------------------------


def _nao_standing_snapshot() -> tuple[MultibodySnapshot, float]:
    """The NAO in its nominal pose, soles on the ground, at rest, weight shared."""
    model = nk.load_model()
    poses = nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)
    names = [name for name, link in model.links.items() if link.mass > 0.0 and name in poses]
    names = ["base_link", *[n for n in names if n != "base_link"]]
    sole_z = min(poses[frame][2, 3] for frame in nk.SOLE_FRAMES)

    def com_of(name: str) -> np.ndarray:
        return (poses[name] @ np.append(model.links[name].com, 1.0))[:3]

    masses = np.array([model.links[n].mass for n in names])
    weight = masses.sum() * nk.GRAVITY
    sole_from_com = np.stack(
        [
            (np.linalg.inv(poses[foot]) @ poses[sole])[:3, 3] - model.links[foot].com
            for foot, sole in zip(nk.FOOT_BODIES, nk.SOLE_FRAMES, strict=True)
        ]
    )
    snapshot = MultibodySnapshot(
        masses=masses,
        inertia_local=np.stack(
            [
                model.links[n].inertia_rot @ model.links[n].inertia @ model.links[n].inertia_rot.T
                for n in names
            ]
        ),
        link_rot=np.stack([poses[n][:3, :3] for n in names]),
        com_pos=np.stack([com_of(n) for n in names]) - np.array([0.0, 0.0, sole_z]),
        com_lin_vel=np.zeros((len(names), 3)),
        ang_vel=np.zeros((len(names), 3)),
        foot_force=np.array([[0.0, 0.0, weight / 2], [0.0, 0.0, weight / 2]]),
        base_index=0,
        foot_index=(names.index(nk.FOOT_BODIES[0]), names.index(nk.FOOT_BODIES[1])),
        sole_from_com=sole_from_com,
    )
    straight = nk.forward_kinematics({})
    leg = float(np.linalg.norm(straight["LThigh"][:3, 3] - straight["l_sole"][:3, 3]))
    return snapshot, leg


def test_a_standing_nao_encodes_to_its_urdf_posture() -> None:
    """Built from the URDF the standing study derived its numbers from."""
    snapshot, leg = _nao_standing_snapshot()
    units = UnitSystem(leg, float(snapshot.masses.sum()))
    fields = SKILL_LAYOUT.unpack(encode(snapshot, units))

    com_height = nk.com_height_above_soles(nk.NOMINAL_STAND_JOINT_POS)
    assert float(fields["height"]) == pytest.approx(com_height / leg)
    assert fields["foot_load"] == pytest.approx([0.5, 0.5])
    assert fields["com_velocity"] == pytest.approx(np.zeros(3))
    assert fields["angular_momentum"] == pytest.approx(np.zeros(3))
    assert fields["foot_position"][:, 2] == pytest.approx(-(com_height / leg) * np.ones(2))
    # Standing symmetric: the feet mirror each other across the sagittal plane.
    left, right = fields["foot_position"]
    assert left[1] == pytest.approx(-right[1], abs=1e-6)
    assert left[1] > 0.0
