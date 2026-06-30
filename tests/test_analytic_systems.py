"""Tests for the analytic rigid-body systems used as the Phase-0 oracle."""

import torch

from lagrangian_mbrl.envs.analytic_systems import FrankaAnalytic7DoF, Pendulum, TwoLinkArm, make_system

torch.set_default_dtype(torch.float64)


def test_two_link_mass_matrix_is_symmetric_pd():
    arm = TwoLinkArm()
    q = torch.randn(64, 2)
    M = arm.mass_matrix(q)
    assert M.shape == (64, 2, 2)
    assert torch.allclose(M, M.transpose(-1, -2))
    assert (torch.linalg.eigvalsh(M) > 0).all()


def test_forward_inverse_dynamics_roundtrip():
    for system in (Pendulum(), TwoLinkArm()):
        d = system.dof
        q, qd, qdd = torch.randn(32, d), torch.randn(32, d), torch.randn(32, d)
        tau = system.inverse_dynamics(q, qd, qdd)
        qdd_rec = system.forward_dynamics(q, qd, tau)
        assert torch.allclose(qdd, qdd_rec, atol=1e-9)


def test_sample_dataset_schema_and_consistency():
    arm = make_system("two_link")
    gen = torch.Generator().manual_seed(0)
    data = arm.sample_dataset(128, generator=gen)
    assert set(data) == {"q", "qd", "tau", "qdd"}
    for v in data.values():
        assert v.shape == (128, 2)
        assert torch.isfinite(v).all()
    # qdd must be the exact forward dynamics of (q, qd, tau)
    qdd = arm.forward_dynamics(data["q"], data["qd"], data["tau"])
    assert torch.allclose(qdd, data["qdd"], atol=1e-9)


# ── FrankaAnalytic7DoF tests ──────────────────────────────────────────────────

def test_franka7_mass_matrix_symmetric_pd():
    arm = FrankaAnalytic7DoF()
    q = torch.randn(16, 7) * 0.5
    M = arm.mass_matrix(q)
    assert M.shape == (16, 7, 7)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-10)
    eigs = torch.linalg.eigvalsh(M)
    assert (eigs > 0).all(), f"M not PD; min eig = {eigs.min():.3e}"


def test_franka7_forward_inverse_roundtrip():
    """M(q)^{-1} * (M(q)*qdd + c + g - c - g) == qdd within tolerance."""
    arm = FrankaAnalytic7DoF()
    gen = torch.Generator().manual_seed(7)
    q = torch.rand(8, 7, generator=gen) * 1.0
    qd = torch.rand(8, 7, generator=gen) * 0.5
    qdd = torch.rand(8, 7, generator=gen) * 2.0
    tau = arm.inverse_dynamics(q, qd, qdd)
    qdd_rec = arm.forward_dynamics(q, qd, tau)
    assert torch.allclose(qdd, qdd_rec, atol=1e-6), \
        f"Forward/inverse roundtrip error: {(qdd - qdd_rec).abs().max():.3e}"


def test_franka7_sample_dataset_finite():
    arm = FrankaAnalytic7DoF()
    gen = torch.Generator().manual_seed(42)
    data = arm.sample_dataset(32, generator=gen)
    assert set(data) == {"q", "qd", "tau", "qdd"}
    for k, v in data.items():
        assert v.shape == (32, 7), f"wrong shape for {k}"
        assert torch.isfinite(v).all(), f"non-finite values in {k}"


def test_franka7_make_system_factory():
    arm = make_system("franka7")
    assert isinstance(arm, FrankaAnalytic7DoF)
    assert arm.dof == 7
