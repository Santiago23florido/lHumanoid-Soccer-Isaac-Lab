"""Analytic rigid-body systems with *known* Lagrangian dynamics.

Phase 0 needs an offline dataset of ``(q, q̇, τ, q̈)`` transitions to fit the
dynamics models on. The plan sources this from logged Isaac Lab Franka rollouts
(see :mod:`lagrangian_mbrl.envs.franka_reach_wrapper`), but Isaac Sim is a heavy,
GPU-only dependency. To make Phase 0 *runnable and unit-testable anywhere*, we
also provide closed-form rigid-body systems whose exact equations of motion

    M(q) q̈ + c(q, q̇) + g(q) = τ

are known analytically. These serve three purposes:

  1. Ground-truth supervision for the offline DeLaN-vs-MLP acceleration-MSE
     comparison (the Phase-0 exit criterion).
  2. A correctness oracle for the DeLaN unit tests (inverse dynamics must match
     the analytic torques on a known system).
  3. A clean A1-satisfying regime (pure Lagrangian, no contact) where the
     structural prior is *expected* to help — exactly the setting the theory in
     ``PROJECT_PLAN.md`` §2 targets.

When Isaac Lab is available, ``FrankaReachEnv.log_transitions`` dumps a dataset
with the same schema, so the offline-fit script is source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class RigidBodySystem:
    """Interface for an analytic ``M(q) q̈ + c(q,q̇) + g(q) = τ`` system."""

    dof: int

    def mass_matrix(self, q: Tensor) -> Tensor:  # (B, d, d)
        raise NotImplementedError

    def coriolis_forces(self, q: Tensor, qd: Tensor) -> Tensor:  # (B, d)
        raise NotImplementedError

    def gravity_forces(self, q: Tensor) -> Tensor:  # (B, d)
        raise NotImplementedError

    def inverse_dynamics(self, q: Tensor, qd: Tensor, qdd: Tensor) -> Tensor:
        """Analytic torques ``τ = M q̈ + c + g``."""
        M = self.mass_matrix(q)
        return (
            torch.einsum("...ij,...j->...i", M, qdd)
            + self.coriolis_forces(q, qd)
            + self.gravity_forces(q)
        )

    def forward_dynamics(self, q: Tensor, qd: Tensor, tau: Tensor) -> Tensor:
        """Analytic accelerations ``q̈ = M⁻¹ (τ − c − g)``."""
        M = self.mass_matrix(q)
        rhs = (tau - self.coriolis_forces(q, qd) - self.gravity_forces(q)).unsqueeze(-1)
        return torch.linalg.solve(M, rhs).squeeze(-1)

    def sample_dataset(
        self,
        n: int,
        *,
        q_scale: float = 3.14159,
        qd_scale: float = 2.0,
        qdd_scale: float = 5.0,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> dict[str, Tensor]:
        """Sample ``n`` i.i.d. transitions ``(q, q̇, q̈, τ)``.

        We sample ``(q, q̇, q̈)`` uniformly on a compact set (assumption A2 in
        ``PROJECT_PLAN.md``) and compute the *exact* torque ``τ`` from inverse
        dynamics — mirroring a logged robot transition, where ``(q, q̇, q̈)`` are
        observed and ``τ`` is the applied command. Sampling accelerations (rather
        than torques) keeps the regression target ``q̈`` well-scaled and avoids
        the ill-conditioning of inverting near-singular mass matrices.

        Uses double precision by default — the mass-matrix solves and second
        derivatives are better conditioned in float64 (see the risk table,
        "Cholesky / 2nd derivatives").
        """
        d = self.dof
        g = generator

        def _u(scale: float) -> Tensor:
            return (torch.rand(n, d, generator=g, dtype=dtype) * 2 - 1) * scale

        q, qd, qdd = _u(q_scale), _u(qd_scale), _u(qdd_scale)
        tau = self.inverse_dynamics(q, qd, qdd)
        return {"q": q, "qd": qd, "tau": tau, "qdd": qdd}


@dataclass
class PendulumParams:
    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81


class Pendulum(RigidBodySystem):
    """Simple 1-DoF pendulum: ``m l² θ̈ + m g l sin θ = τ`` (no Coriolis)."""

    dof = 1

    def __init__(self, params: PendulumParams | None = None) -> None:
        self.p = params or PendulumParams()

    def mass_matrix(self, q: Tensor) -> Tensor:
        ml2 = self.p.mass * self.p.length**2
        return torch.full((*q.shape[:-1], 1, 1), ml2, dtype=q.dtype, device=q.device)

    def coriolis_forces(self, q: Tensor, qd: Tensor) -> Tensor:
        return torch.zeros_like(q)

    def gravity_forces(self, q: Tensor) -> Tensor:
        p = self.p
        return p.mass * p.gravity * p.length * torch.sin(q)


@dataclass
class TwoLinkArmParams:
    """Planar 2-link arm parameters (textbook acrobot, angles from horizontal)."""

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    lc1: float = 0.5  # COM distance along link 1
    lc2: float = 0.5  # COM distance along link 2
    i1: float = 0.083  # link inertia about COM (~ m l²/12 for a rod)
    i2: float = 0.083
    gravity: float = 9.81


class TwoLinkArm(RigidBodySystem):
    """Planar 2-link manipulator with full configuration-dependent dynamics.

    Standard rigid-body model (Spong & Vidyasagar). Unlike the pendulum, ``M(q)``
    depends on ``q`` and the Coriolis/centrifugal forces are non-trivial, so it
    genuinely exercises the structured terms a DeLaN must learn.
    """

    dof = 2

    def __init__(self, params: TwoLinkArmParams | None = None) -> None:
        self.p = params or TwoLinkArmParams()

    def mass_matrix(self, q: Tensor) -> Tensor:
        p = self.p
        q2 = q[..., 1]
        c2 = torch.cos(q2)
        d1 = (
            p.m1 * p.lc1**2
            + p.m2 * (p.l1**2 + p.lc2**2 + 2 * p.l1 * p.lc2 * c2)
            + p.i1
            + p.i2
        )
        d2 = p.m2 * (p.lc2**2 + p.l1 * p.lc2 * c2) + p.i2
        d3 = torch.full_like(q2, p.m2 * p.lc2**2 + p.i2)
        M = torch.stack(
            [torch.stack([d1, d2], -1), torch.stack([d2, d3], -1)], dim=-2
        )
        return M

    def coriolis_forces(self, q: Tensor, qd: Tensor) -> Tensor:
        p = self.p
        q2 = q[..., 1]
        qd1, qd2 = qd[..., 0], qd[..., 1]
        h = p.m2 * p.l1 * p.lc2 * torch.sin(q2)
        c1 = -h * (2 * qd1 * qd2 + qd2**2)
        c2 = h * qd1**2
        return torch.stack([c1, c2], dim=-1)

    def gravity_forces(self, q: Tensor) -> Tensor:
        p = self.p
        q1, q2 = q[..., 0], q[..., 1]
        g = p.gravity
        g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * g * torch.cos(q1) + p.m2 * p.lc2 * g * torch.cos(
            q1 + q2
        )
        g2 = p.m2 * p.lc2 * g * torch.cos(q1 + q2)
        return torch.stack([g1, g2], dim=-1)


class FrankaAnalytic7DoF(RigidBodySystem):
    """7-DoF planar serial arm with approximate Franka Panda physical parameters.

    All seven revolute joints act in the same plane (a standard planar-chain
    projection of the 3-D arm used throughout teaching robotics).  The exact,
    closed-form equations of motion are

        M(q) q̈  +  c(q, q̇)  +  g(q)  =  τ

    where ``M(q)`` is computed via the composite rigid-body formula for a planar
    chain, ``g(q) = ∂V/∂q`` via autograd on the gravitational potential, and
    ``c(q,q̇)`` via the identity

        c_i  =  [JVP of  M(q)q̇  w.r.t.  q  along  q̇]_i  −  ∂T/∂q_i

    (derived in theory/derivations.md §FORCES).  This system is the primary
    "robot simulator" used for PINN training: its dynamics lie exactly in the
    Lagrangian hypothesis class (assumption A1), so the structured DeLaN model
    is expected to learn them with far fewer transitions than an unstructured MLP.

    Physical parameters are taken from the Franka Emika Panda URDF (masses,
    approximate link lengths) projected onto a planar chain for analytical
    tractability.
    """

    dof = 7

    # Approximate Franka Panda parameters (URDF, planar projection)
    _link_lengths: tuple[float, ...] = (0.333, 0.316, 0.082, 0.384, 0.088, 0.107, 0.062)
    _link_masses: tuple[float, ...] = (4.970, 0.646, 3.228, 3.587, 1.226, 1.666, 0.735)
    _com_fracs: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    _gravity: float = 9.81

    def __init__(self) -> None:
        l = list(self._link_lengths)
        m = list(self._link_masses)
        f = list(self._com_fracs)
        # rotational inertia of each link about its own COM: I = m*l²/12
        self._l_py = l
        self._m_py = m
        self._lc_py = [l_k * f_k for l_k, f_k in zip(l, f)]
        self._I_py = [m_k * l_k**2 / 12.0 for m_k, l_k in zip(m, l)]

    def _t(self, data: list[float], like: Tensor) -> Tensor:
        return torch.tensor(data, dtype=like.dtype, device=like.device)

    # ---- mass matrix ---------------------------------------------------------
    def mass_matrix(self, q: Tensor) -> Tensor:
        """Configuration-dependent mass matrix via the composite rigid-body formula.

        Uses the analytical expression for a planar serial chain:

            M(q) = Σ_k  m_k J_k^T J_k  +  Σ_k  I_k z_k z_k^T

        where ``J_k`` is the linear-velocity Jacobian of the COM of link *k*,
        and ``z_k`` is the angular-velocity Jacobian column (ones for joints
        i ≤ k, zeros otherwise).
        """
        B, d = q.shape
        l = self._t(self._l_py, q)    # (d,)
        m = self._t(self._m_py, q)
        lc = self._t(self._lc_py, q)
        I_rot = self._t(self._I_py, q)

        phi = torch.cumsum(q, dim=-1)          # (B, d)  cumulative joint angles
        cos_phi = torch.cos(phi)               # (B, d)
        sin_phi = torch.sin(phi)

        # Prefix sums: pcos[b, k] = Σ_{p=0}^{k} l_p cos φ_p
        pcos = torch.cumsum(l * cos_phi, dim=-1)   # (B, d)
        psin = torch.cumsum(l * sin_phi, dim=-1)

        # Pad with a leading zero so pcos_pad[:, k] = Σ_{p=0}^{k-1} l_p cos φ_p
        zero = torch.zeros(B, 1, dtype=q.dtype, device=q.device)
        pcos_pad = torch.cat([zero, pcos], dim=1)  # (B, d+1)
        psin_pad = torch.cat([zero, psin], dim=1)

        # (B, d_k, d_i) Jacobians via broadcasting.
        # For link k, joint i (i ≤ k):
        #   Jx[b,k,i] = −(Σ_{p=i}^{k-1} l_p sin φ_p) − lc_k sin φ_k
        #              = −(psin_pad[b,k] − psin_pad[b,i]) − lc_k sin φ_k
        #   Jy[b,k,i] = +(Σ_{p=i}^{k-1} l_p cos φ_p) + lc_k cos φ_k
        psin_k = psin_pad[:, :d].unsqueeze(2)   # (B, d_k, 1)
        psin_i = psin_pad[:, :d].unsqueeze(1)   # (B, 1, d_i)
        pcos_k = pcos_pad[:, :d].unsqueeze(2)
        pcos_i = pcos_pad[:, :d].unsqueeze(1)

        lck_sin = (lc * sin_phi).unsqueeze(2)   # (B, d_k, 1)
        lck_cos = (lc * cos_phi).unsqueeze(2)

        Jx = -(psin_k - psin_i) - lck_sin       # (B, d_k, d_i)
        Jy = (pcos_k - pcos_i) + lck_cos

        # Zero out entries where i > k (those joints don't affect link k).
        i_idx = torch.arange(d, device=q.device)
        k_idx = torch.arange(d, device=q.device)
        mask = (i_idx.unsqueeze(0) <= k_idx.unsqueeze(1)).to(q.dtype)  # (d_k, d_i)
        Jx = Jx * mask.unsqueeze(0)
        Jy = Jy * mask.unsqueeze(0)

        # Linear-inertia contribution: Σ_k m_k J_k^T J_k
        m_bcast = m.view(1, d, 1)
        M_lin = (torch.einsum("bki,bkj->bij", m_bcast * Jx, Jx) +
                 torch.einsum("bki,bkj->bij", m_bcast * Jy, Jy))

        # Rotational-inertia contribution: Σ_k I_k (k ≥ i) (k ≥ j)
        I_mat = torch.einsum("k,ki,kj->ij", I_rot, mask, mask)  # (d, d)
        M_rot = I_mat.unsqueeze(0).expand(B, -1, -1)

        eps = 1e-6 * torch.eye(d, dtype=q.dtype, device=q.device)
        return M_lin + M_rot + eps.unsqueeze(0)

    # ---- potential energy ----------------------------------------------------
    def potential_energy(self, q: Tensor) -> Tensor:
        """V(q) = Σ_k m_k g h_k(q) where h_k is the height of COM_k."""
        l = self._t(self._l_py, q)
        m = self._t(self._m_py, q)
        lc = self._t(self._lc_py, q)

        phi = torch.cumsum(q, dim=-1)         # (B, d)
        sin_phi = torch.sin(phi)

        # psin_pad[:, k] = Σ_{p=0}^{k-1} l_p sin φ_p  (height from full links)
        zero = torch.zeros(q.shape[0], 1, dtype=q.dtype, device=q.device)
        psin_pad = torch.cat([zero, torch.cumsum(l * sin_phi, dim=-1)], dim=1)  # (B, d+1)

        # height of COM_k = sum of link heights before k + partial link k
        h = psin_pad[:, :self.dof] + lc * sin_phi   # (B, d)
        return (m * h * self._gravity).sum(dim=-1)   # (B,)

    # ---- gravity forces (∂V/∂q) -----------------------------------------------
    def gravity_forces(self, q: Tensor) -> Tensor:
        with torch.enable_grad():
            qg = q.clone().requires_grad_(True)
            V = self.potential_energy(qg).sum()
            (g,) = torch.autograd.grad(V, qg)
        return g.detach()

    # ---- Coriolis forces -----------------------------------------------------
    def coriolis_forces(self, q: Tensor, qd: Tensor) -> Tensor:
        """Coriolis / centrifugal forces via the JVP identity.

        From the Euler–Lagrange equations one can show:

            c_i  =  JVP[M(q)q̇, q, q̇]_i  −  ∂T/∂q_i

        where T = ½ q̇ᵀ M(q) q̇.  See theory/derivations.md for the proof.
        """
        with torch.enable_grad():
            qg = q.clone().requires_grad_(True)

            def _momentum(q_: Tensor) -> Tensor:
                return torch.einsum("bij,bj->bi", self.mass_matrix(q_), qd)

            # JVP: directional derivative of momentum w.r.t. q along q̇.
            p = _momentum(qg)
            (dp_dq,) = torch.autograd.grad(p.sum(), qg, create_graph=True)
            jvp = (dp_dq * qd).sum(dim=-1, keepdim=True)

            # Compute term by term for each output dimension.
            # Faster: use the full Jacobian via repeated scalar grad.
            # Re-derive JVP properly: Σ_k (∂p_i/∂q_k) qd_k
            jvp_vec = torch.zeros_like(q)
            for i in range(self.dof):
                pi = _momentum(qg)[:, i]
                (grad_pi,) = torch.autograd.grad(pi.sum(), qg, retain_graph=True)
                jvp_vec[:, i] = (grad_pi * qd).sum(dim=-1)

            # ∂T/∂q with qd fixed.
            M = self.mass_matrix(qg)
            T = 0.5 * torch.einsum("bi,bij,bj->b", qd, M, qd)
            (dT_dq,) = torch.autograd.grad(T.sum(), qg)

        return (jvp_vec - dT_dq).detach()


def make_system(name: str) -> RigidBodySystem:
    """Factory: ``pendulum`` | ``two_link`` | ``franka7``."""
    systems: dict[str, type[RigidBodySystem]] = {
        "pendulum": Pendulum,
        "two_link": TwoLinkArm,
        "franka7": FrankaAnalytic7DoF,
    }
    try:
        return systems[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown system {name!r}; choose from {sorted(systems)}") from exc
