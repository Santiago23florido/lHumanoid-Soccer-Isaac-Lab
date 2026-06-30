"""Tests for the executable complexity constants used by the LaTeX note."""

from lagrangian_mbrl.models import (
    DeepLagrangianNetwork,
    DeLaNConfig,
    MLPDynamics,
    MLPDynamicsConfig,
    MLPDynamicsEnsemble,
    MLPEnsembleConfig,
)
from lagrangian_mbrl.theory import complexity_report


def _parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_default_complexity_report_matches_models():
    report = complexity_report()
    delan = DeepLagrangianNetwork(DeLaNConfig())
    direct = MLPDynamics(MLPDynamicsConfig(probabilistic=False))
    member = MLPDynamicsConfig(probabilistic=True)
    ensemble = MLPDynamicsEnsemble(MLPEnsembleConfig(member=member, size=5))

    assert report.delan_parameters == _parameters(delan)
    assert report.direct_mlp_parameters == _parameters(direct)
    assert report.probabilistic_ensemble_parameters == _parameters(ensemble)


def test_franka_cholesky_kappas_are_explicit():
    report = complexity_report(dof=7)
    assert report.cholesky_entries == 28
    assert report.full_mass_entries == 49
    assert report.kappa_output > 1.0
    assert report.kappa_cholesky_parameters > 1.0
    assert report.kappa_direct_parameters > report.kappa_cholesky_parameters
