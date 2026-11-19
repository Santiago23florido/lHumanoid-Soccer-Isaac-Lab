"""PACT networks: one encoder and one decoder for every body.

The pieces and the papers they come from (``references/INDEX.md``):

- **Joint tokens** (URMA, GET-Zero): one token per degree of freedom, built from
  its Pi-group descriptor (:mod:`.descriptors`) plus, for the state encoder, its
  recent observations. No per-body layer exists anywhere.
- **Graph-biased self-attention** (GET-Zero, Graphormer): each head adds a learned
  bias indexed by the tree distance between two joints, and separate biases for
  ancestor-to-descendant and descendant-to-ancestor pairs.
- **Embodiment tokens** ``E_b`` (Perceiver IO, HPT stems): a few learned queries
  read the descriptor tokens by cross-attention. They are the latent's statement
  of what the body is and can do.
- **Entity latent** (Perceiver IO with physical queries): one centroid query, one
  query per contact (tied to that foot's joint token, so its side and geometry
  come from the descriptor), and free queries, read the joint tokens, the IMU token
  and ``E_b``; then self-attention among entities. Linear heads read each entity's
  ``z*`` fields (:mod:`.tokens`): the anchor.
- **Joint decoder** (URMAv2's attention decoder): each joint's token queries the
  current entities, the plan's entities and ``E_b``, and emits that joint's
  position target.

Padded joints are masked out of every attention as keys, so a body's outputs do
not depend on what else is in the batch; and since joints form a set whose order is
carried by the descriptors, permuting a body's joints permutes its actions and
leaves its latent unchanged. Tests hold both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from humanoid_transfer.transfer.latent.pact.descriptors import (
    BODY_FEATURES,
    JOINT_FEATURES,
    BodyDescriptor,
)
from humanoid_transfer.transfer.latent.pact.tokens import CENTROID_SIZE, CONTACT_SIZE

LOG_FEATURES: tuple[str, ...] = (
    "effort_limit",
    "velocity_limit",
    "stiffness",
    "damping",
    "armature",
    "subtree_mass",
    "subtree_inertia_about_axis",
)
"""Positive Pi groups spanning several decades (inertias 1e-7 to 1e-1): read as
``log10(x + LOG_FLOOR)`` so a linear layer sees their ratios, not their sizes."""

LOG_FLOOR = 1e-6
"""Added before the logarithm; a passive joint's zero drive reads as -6."""

_LOG_COLUMNS = [JOINT_FEATURES.index(name) for name in LOG_FEATURES]


@dataclass(frozen=True)
class PactDims:
    """Sizes of the PACT networks."""

    observation: int = 3
    """Per joint and step: position from default (rad), velocity times sqrt(L/g), last action."""
    imu: int = 6
    """Per step: projected gravity (3) and base angular velocity times sqrt(L/g) (3)."""
    history: int = 4
    """Steps of joint and IMU observations the state encoder reads."""
    width: int = 128
    heads: int = 4
    ff_mult: int = 4
    body_layers: int = 2
    joint_layers: int = 2
    cross_layers: int = 2
    latent_layers: int = 2
    decoder_layers: int = 2
    body_tokens: int = 4
    free_tokens: int = 4
    contacts: int = 2
    plan_steps: int = 8
    max_distance: int = 16
    anchor_inputs: bool = False
    """Entity queries also read the analytic anchors (:mod:`.anchors`): the centroid its
    13 fields, each contact its 7, with what is not known exactly set to zero."""
    timed_plan: bool = False
    """The decoder reads a plan of any length: each step tagged by its time ahead in the
    body's own unit ``sqrt(L/g)`` instead of a learned per-index embedding (bodies see plans
    of different step counts over the same dimensionless span), each entity both as its
    value and as its difference to the current estimate, and each step's path (where the
    reference will be: CoM ahead and sideways in leg lengths, heading in radians). Steps a
    body does not have are masked."""


@dataclass
class BodyBatch:
    """Several bodies' descriptors, padded to the largest joint count."""

    joints: torch.Tensor
    """``(B, J, len(JOINT_FEATURES))``."""
    body: torch.Tensor
    """``(B, len(BODY_FEATURES))``."""
    distance: torch.Tensor
    """``(B, J, J)`` long."""
    ancestor: torch.Tensor
    """``(B, J, J)`` bool."""
    mask: torch.Tensor
    """``(B, J)`` bool, True on real joints."""
    contact_joint: torch.Tensor
    """``(B, contacts)`` long: the joint whose child link is each foot."""

    def to(self, device: torch.device | str) -> BodyBatch:
        return BodyBatch(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))

    def take(self, index: torch.Tensor) -> BodyBatch:
        """The rows of ``index`` (e.g. one body per sample from a table of bodies)."""
        return BodyBatch(*(getattr(self, f)[index] for f in self.__dataclass_fields__))


def stack_bodies(
    descriptors: list[BodyDescriptor], contact_joints: list[list[int]], pad_to: int | None = None
) -> BodyBatch:
    """Pad descriptors to a common joint count; padded joints are masked everywhere."""
    count = max(len(d.joint_names) for d in descriptors)
    count = max(count, pad_to or 0)
    size = len(descriptors)
    joints = np.zeros((size, count, len(JOINT_FEATURES)))
    distance = np.zeros((size, count, count), dtype=np.int64)
    ancestor = np.zeros((size, count, count), dtype=bool)
    mask = np.zeros((size, count), dtype=bool)
    for b, d in enumerate(descriptors):
        n = len(d.joint_names)
        joints[b, :n] = d.joints
        distance[b, :n, :n] = d.tree_distance
        ancestor[b, :n, :n] = d.ancestor
        mask[b, :n] = True
    return BodyBatch(
        joints=torch.as_tensor(joints, dtype=torch.float32),
        body=torch.as_tensor(np.stack([d.body for d in descriptors]), dtype=torch.float32),
        distance=torch.as_tensor(distance),
        ancestor=torch.as_tensor(ancestor),
        mask=torch.as_tensor(mask),
        contact_joint=torch.as_tensor(np.asarray(contact_joints), dtype=torch.long),
    )


def descriptor_inputs(joints: torch.Tensor) -> torch.Tensor:
    """Descriptor features as the networks read them: wide-range groups in log10."""
    out = joints.clone()
    out[..., _LOG_COLUMNS] = torch.log10(joints[..., _LOG_COLUMNS].clamp_min(0.0) + LOG_FLOOR)
    return out


def _mlp(inputs: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width)
    )


class GraphBias(nn.Module):
    """Per-head attention biases from the kinematic tree (GET-Zero).

    Tokens outside the tree (IMU, embodiment, entities) get one learned "global"
    bias per head in every pair that involves them.
    """

    def __init__(self, heads: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.spatial = nn.Embedding(max_distance + 2, heads)
        self.parent = nn.Embedding(max_distance + 1, heads)
        self.child = nn.Embedding(max_distance + 1, heads)

    def forward(
        self, distance: torch.Tensor, ancestor: torch.Tensor, extra: int = 0
    ) -> torch.Tensor:
        """``(B, J, J)`` tree -> ``(B, heads, J + extra, J + extra)`` bias."""
        d = distance.clamp(max=self.max_distance)
        down = d * ancestor
        up = d * ancestor.transpose(-1, -2)
        bias = self.spatial(d) + self.parent(down) + self.child(up)
        if extra:
            size = d.shape[-1] + extra
            glob = self.spatial.weight[self.max_distance + 1]
            full = glob.expand(d.shape[0], size, size, -1).clone()
            full[:, : d.shape[-1], : d.shape[-1]] = bias
            bias = full
        return bias.permute(0, 3, 1, 2)


class Block(nn.Module):
    """Pre-norm multi-head attention (self or cross) then a feed-forward layer."""

    def __init__(self, width: int, heads: int, ff_mult: int, cross: bool = False) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by {heads} heads")
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.norm_memory = nn.LayerNorm(width) if cross else None
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, ff_mult * width),
            nn.GELU(),
            nn.Linear(ff_mult * width, width),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        m = h if memory is None else self.norm_memory(memory)
        batch, queries, width = h.shape
        keys = m.shape[1]
        head = width // self.heads
        q = self.query(h).view(batch, queries, self.heads, head).transpose(1, 2)
        k, v = self.key_value(m).view(batch, keys, 2, self.heads, head).permute(2, 0, 3, 1, 4)
        mask = None
        if bias is not None or key_mask is not None:
            mask = h.new_zeros(batch, self.heads, queries, keys)
            if bias is not None:
                mask = mask + bias
            if key_mask is not None:
                mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(attended.transpose(1, 2).reshape(batch, queries, width))
        return x + self.ff(x)


class EmbodimentEncoder(nn.Module):
    """Descriptor tokens -> graph-biased attention -> ``E_b`` by learned queries."""

    def __init__(self, dims: PactDims) -> None:
        super().__init__()
        w = dims.width
        self.joint = _mlp(len(JOINT_FEATURES), w)
        self.body = _mlp(len(BODY_FEATURES), w)
        self.bias = GraphBias(dims.heads, dims.max_distance)
        self.layers = nn.ModuleList(
            Block(w, dims.heads, dims.ff_mult) for _ in range(dims.body_layers)
        )
        self.queries = nn.Parameter(torch.randn(dims.body_tokens, w) * 0.02)
        self.read = Block(w, dims.heads, dims.ff_mult, cross=True)

    def forward(self, bodies: BodyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """``-> (E_b (B, body_tokens, W), descriptor tokens (B, J, W))``."""
        descriptor = self.joint(descriptor_inputs(bodies.joints))
        body = self.body(bodies.body)[:, None, :]
        tokens = descriptor + body
        bias = self.bias(bodies.distance, bodies.ancestor)
        for layer in self.layers:
            tokens = layer(tokens, bias=bias, key_mask=bodies.mask)
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        embodiment = self.read(queries, memory=tokens, key_mask=bodies.mask)
        return embodiment, descriptor + body


def _gather(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``tokens (B, J, W)``, ``index (B, K)`` -> ``(B, K, W)``."""
    return tokens.gather(1, index[..., None].expand(-1, -1, tokens.shape[-1]))


class StateEncoder(nn.Module):
    """A body's recent observations -> entity tokens, with ``z*`` read out of them."""

    def __init__(self, dims: PactDims) -> None:
        super().__init__()
        w = dims.width
        self.dims = dims
        self.observation = _mlp(dims.history * dims.observation, w)
        self.imu = _mlp(dims.history * dims.imu, w)
        self.bias = GraphBias(dims.heads, dims.max_distance)
        self.layers = nn.ModuleList(
            Block(w, dims.heads, dims.ff_mult) for _ in range(dims.joint_layers)
        )
        self.centroid_query = nn.Parameter(torch.randn(1, w) * 0.02)
        self.contact_query = nn.Parameter(torch.randn(1, w) * 0.02)
        self.contact_from_joint = nn.Linear(w, w)
        self.free_queries = nn.Parameter(torch.randn(dims.free_tokens, w) * 0.02)
        if dims.anchor_inputs:
            self.centroid_anchor = nn.Linear(CENTROID_SIZE, w)
            self.contact_anchor = nn.Linear(CONTACT_SIZE, w)
        self.cross = nn.ModuleList(
            Block(w, dims.heads, dims.ff_mult, cross=True) for _ in range(dims.cross_layers)
        )
        self.latent = nn.ModuleList(
            Block(w, dims.heads, dims.ff_mult) for _ in range(dims.latent_layers)
        )
        self.centroid_head = nn.Linear(w, CENTROID_SIZE)
        self.contact_head = nn.Linear(w, CONTACT_SIZE)

    def forward(
        self,
        bodies: BodyBatch,
        descriptor: torch.Tensor,
        embodiment: torch.Tensor,
        joint_obs: torch.Tensor,
        imu: torch.Tensor,
        anchors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """``joint_obs (B, J, history, observation)``, ``imu (B, history, imu)``; with
        ``dims.anchor_inputs``, ``anchors = (centroid (B, 13), contacts (B, contacts, 7))``.

        Returns the entity tokens ``latent (B, 1 + contacts + free, W)`` and their
        anchored readouts ``centroid (B, 13)`` and ``contacts (B, contacts, 7)``.
        """
        batch, count = joint_obs.shape[:2]
        tokens = descriptor + self.observation(joint_obs.reshape(batch, count, -1))
        imu_token = self.imu(imu.reshape(batch, 1, -1))
        sequence = torch.cat([tokens, imu_token], dim=1)
        mask = torch.cat([bodies.mask, bodies.mask.new_ones(batch, 1)], dim=1)
        bias = self.bias(bodies.distance, bodies.ancestor, extra=1)
        for layer in self.layers:
            sequence = layer(sequence, bias=bias, key_mask=mask)

        joints = sequence[:, :count]
        contact = self.contact_query + self.contact_from_joint(
            _gather(joints, bodies.contact_joint)
        )
        centroid = self.centroid_query.expand(batch, -1, -1)
        if self.dims.anchor_inputs:
            if anchors is None:
                raise ValueError("this encoder reads the analytic anchors: pass them")
            centroid = centroid + self.centroid_anchor(anchors[0])[:, None]
            contact = contact + self.contact_anchor(anchors[1])
        queries = torch.cat(
            [
                centroid,
                contact,
                self.free_queries.expand(batch, -1, -1),
            ],
            dim=1,
        )
        memory = torch.cat([sequence, embodiment], dim=1)
        memory_mask = torch.cat([mask, mask.new_ones(batch, embodiment.shape[1])], dim=1)
        latent = queries
        for layer in self.cross:
            latent = layer(latent, memory=memory, key_mask=memory_mask)
        for layer in self.latent:
            latent = layer(latent)
        contacts = self.dims.contacts
        return {
            "latent": latent,
            "joints": joints,
            "centroid": self.centroid_head(latent[:, 0]),
            "contacts": self.contact_head(latent[:, 1 : 1 + contacts]),
        }


class JointDecoder(nn.Module):
    """Each joint queries the current entities, the plan and ``E_b`` for its target."""

    def __init__(self, dims: PactDims) -> None:
        super().__init__()
        w = dims.width
        self.dims = dims
        values = 2 if dims.timed_plan else 1
        self.plan_centroid = nn.Linear(values * CENTROID_SIZE, w)
        self.plan_contact = nn.Linear(values * CONTACT_SIZE, w)
        self.plan_contact_side = nn.Linear(w, w)
        if dims.timed_plan:
            self.plan_time = nn.Sequential(nn.Linear(1, w), nn.GELU(), nn.Linear(w, w))
            self.plan_path = nn.Linear(3, w)
        else:
            self.step = nn.Parameter(torch.randn(dims.plan_steps, w) * 0.02)
        self.bias = GraphBias(dims.heads, dims.max_distance)
        self.cross = nn.ModuleList(
            Block(w, dims.heads, dims.ff_mult, cross=True) for _ in range(dims.decoder_layers)
        )
        self.self_attention = nn.ModuleList(
            Block(w, dims.heads, dims.ff_mult) for _ in range(dims.decoder_layers)
        )
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1))

    def plan_tokens(
        self,
        joints: torch.Tensor,
        bodies: BodyBatch,
        centroid: torch.Tensor,
        contacts: torch.Tensor,
        time: torch.Tensor | None = None,
        path: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``centroid (B, H, 13)``, ``contacts (B, H, K, 7)`` -> ``(B, H * (1 + K), W)``.

        With ``dims.timed_plan`` the entities carry 26 and 14 numbers (value, then its
        difference to the estimate), ``time (B, H)`` and ``path (B, H, 3)`` are required.
        """
        batch, steps = centroid.shape[:2]
        if self.dims.timed_plan:
            if time is None or path is None:
                raise ValueError("a timed plan needs each step's time and path")
            step = self.plan_time(time[..., None]) + self.plan_path(path)
        else:
            step = self.step[:steps][None]
        side = self.plan_contact_side(_gather(joints, bodies.contact_joint))
        c = self.plan_centroid(centroid) + step
        k = self.plan_contact(contacts) + step[:, :, None] + side[:, None]
        return torch.cat([c[:, :, None], k], dim=2).reshape(batch, -1, c.shape[-1])

    def forward(
        self,
        bodies: BodyBatch,
        joints: torch.Tensor,
        latent: torch.Tensor,
        embodiment: torch.Tensor,
        plan_centroid: torch.Tensor,
        plan_contacts: torch.Tensor,
        plan_time: torch.Tensor | None = None,
        plan_path: torch.Tensor | None = None,
        plan_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``-> (B, J)`` joint position targets (padded joints are zero).

        ``plan_mask (B, H)``: True on the plan steps a body has (all when None).
        """
        plan = self.plan_tokens(joints, bodies, plan_centroid, plan_contacts, plan_time, plan_path)
        memory = torch.cat([latent, plan, embodiment], dim=1)
        memory_mask = None
        if plan_mask is not None:
            per_step = 1 + plan_contacts.shape[2]
            memory_mask = torch.cat(
                [
                    plan_mask.new_ones(latent.shape[:2]),
                    plan_mask.repeat_interleave(per_step, dim=1),
                    plan_mask.new_ones(embodiment.shape[:2]),
                ],
                dim=1,
            )
        bias = self.bias(bodies.distance, bodies.ancestor)
        x = joints
        for cross, own in zip(self.cross, self.self_attention, strict=True):
            x = cross(x, memory=memory, key_mask=memory_mask)
            x = own(x, bias=bias, key_mask=bodies.mask)
        return self.head(x)[..., 0] * bodies.mask


class Pact(nn.Module):
    """The whole model: embodiment, state encoder and joint decoder."""

    def __init__(self, dims: PactDims | None = None) -> None:
        super().__init__()
        self.dims = dims or PactDims()
        self.embodiment = EmbodimentEncoder(self.dims)
        self.encoder = StateEncoder(self.dims)
        self.decoder = JointDecoder(self.dims)

    def encode(
        self, bodies: BodyBatch, joint_obs: torch.Tensor, imu: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        embodiment, descriptor = self.embodiment(bodies)
        state = self.encoder(bodies, descriptor, embodiment, joint_obs, imu)
        state["embodiment"] = embodiment
        return state

    def forward(
        self,
        bodies: BodyBatch,
        joint_obs: torch.Tensor,
        imu: torch.Tensor,
        plan_centroid: torch.Tensor,
        plan_contacts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = self.encode(bodies, joint_obs, imu)
        state["action"] = self.decoder(
            bodies,
            state["joints"],
            state["latent"],
            state["embodiment"],
            plan_centroid,
            plan_contacts,
        )
        return state
