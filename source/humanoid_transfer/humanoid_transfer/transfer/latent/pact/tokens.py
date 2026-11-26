"""The skill state ``z*`` as a set of physical entities.

``z*`` (``common/skill_state.SKILL_LAYOUT``) describes one centroid and two
contacts. The latent keeps that structure: one token per entity, each with a
linear readout anchored to that entity's fields. Splitting by entity is what lets
the same readout weights serve every contact (and, later, any number of them),
and makes the sagittal mirror a permutation of contact tokens plus a fixed
reflection of each entity's fields.
"""

from __future__ import annotations

import numpy as np
import torch

from humanoid_transfer.common.skill_state import SKILL_LAYOUT, mirror_matrix

CENTROID_FIELDS: tuple[str, ...] = (
    "height",
    "com_velocity",
    "gravity",
    "base_angular_velocity",
    "angular_momentum",
)
"""Fields that belong to the whole body, in the order a centroid token reads them."""

CONTACT_FIELDS: tuple[str, ...] = ("foot_position", "foot_velocity", "foot_load")
"""Sided fields: one block per contact, in the order a contact token reads them."""

CONTACTS = 2
"""Contacts in ``z*`` as defined today (left, right)."""


def _centroid_columns() -> list[int]:
    columns: list[int] = []
    for name in CENTROID_FIELDS:
        block = SKILL_LAYOUT.slice_of(name)
        columns.extend(range(block.start, block.stop))
    return columns


def _contact_columns() -> list[list[int]]:
    """Per contact, the columns of its sided fields in ``z*``."""
    per_contact: list[list[int]] = [[] for _ in range(CONTACTS)]
    for name in CONTACT_FIELDS:
        field = SKILL_LAYOUT.field(name)
        block = SKILL_LAYOUT.slice_of(name)
        for side in range(CONTACTS):
            start = block.start + side * field.width
            per_contact[side].extend(range(start, start + field.width))
    return per_contact


CENTROID_COLUMNS: tuple[int, ...] = tuple(_centroid_columns())
CONTACT_COLUMNS: tuple[tuple[int, ...], ...] = tuple(tuple(c) for c in _contact_columns())
CENTROID_SIZE = len(CENTROID_COLUMNS)
"""13: height 1, CoM velocity 3, gravity 3, base angular velocity 3, angular momentum 3."""
CONTACT_SIZE = len(CONTACT_COLUMNS[0])
"""7: position 3, velocity 3, load 1."""


def split(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(..., 27) -> (centroid (..., 13), contacts (..., 2, 7))``."""
    centroid = z[..., list(CENTROID_COLUMNS)]
    contacts = torch.stack([z[..., list(cols)] for cols in CONTACT_COLUMNS], dim=-2)
    return centroid, contacts


def merge(centroid: torch.Tensor, contacts: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`split`."""
    shape = (*centroid.shape[:-1], SKILL_LAYOUT.size)
    z = centroid.new_zeros(shape)
    z[..., list(CENTROID_COLUMNS)] = centroid
    for side, cols in enumerate(CONTACT_COLUMNS):
        z[..., list(cols)] = contacts[..., side, :]
    return z


def _entity_reflections() -> tuple[np.ndarray, np.ndarray]:
    """The mirror restricted to the centroid's fields and to one contact's fields.

    The full mirror maps left contact fields onto right ones with the same block; so
    the per-contact reflection is the block taking contact 0's columns to contact 1's.
    """
    full = mirror_matrix(SKILL_LAYOUT)
    centroid = full[np.ix_(CENTROID_COLUMNS, CENTROID_COLUMNS)]
    contact = full[np.ix_(CONTACT_COLUMNS[1], CONTACT_COLUMNS[0])]
    return centroid, contact


_CENTROID_MIRROR, _CONTACT_MIRROR = _entity_reflections()


def mirror_entities(
    centroid: torch.Tensor, contacts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The mirror image: reflect every entity's fields and swap the two contacts."""
    reflect_c = centroid.new_tensor(_CENTROID_MIRROR)
    reflect_k = contacts.new_tensor(_CONTACT_MIRROR)
    mirrored_centroid = centroid @ reflect_c.T
    mirrored_contacts = contacts.flip(-2) @ reflect_k.T
    return mirrored_centroid, mirrored_contacts
