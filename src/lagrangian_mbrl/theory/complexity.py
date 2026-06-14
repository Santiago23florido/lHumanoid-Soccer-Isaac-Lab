"""Concrete complexity proxies for the structured and unstructured models.

These counts support a conditional upper-bound comparison. They are not lower
bounds and therefore do not establish a two-sided statistical separation.
"""

from __future__ import annotations

from dataclasses import dataclass


def mlp_parameter_count(
    input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int
) -> int:
    """Count weights and biases in a fully connected MLP."""
    widths = (input_dim, *hidden_sizes, output_dim)
    return sum(
        (left + 1) * right for left, right in zip(widths, widths[1:], strict=False)
    )


@dataclass(frozen=True)
class ComplexityReport:
    """Architecture-specific counts used in the theory document."""

    dof: int
    cholesky_entries: int
    full_mass_entries: int
    delan_parameters: int
    full_mass_energy_parameters: int
    direct_mlp_parameters: int
    probabilistic_ensemble_parameters: int

    @property
    def kappa_output(self) -> float:
        """Free output-coordinate ratio: full mass plus V over Cholesky plus V."""
        return (self.full_mass_entries + 1) / (self.cholesky_entries + 1)

    @property
    def kappa_cholesky_parameters(self) -> float:
        """Parameter ratio isolating the full-mass versus Cholesky head."""
        return self.full_mass_energy_parameters / self.delan_parameters

    @property
    def kappa_direct_parameters(self) -> float:
        """Current deterministic direct-MLP parameter ratio over DeLaN."""
        return self.direct_mlp_parameters / self.delan_parameters

    @property
    def kappa_ensemble_parameters(self) -> float:
        """Current probabilistic ensemble parameter ratio over DeLaN."""
        return self.probabilistic_ensemble_parameters / self.delan_parameters


def complexity_report(
    dof: int = 7,
    delan_hidden: tuple[int, ...] = (128, 128),
    mlp_hidden: tuple[int, ...] = (256, 256, 256),
    ensemble_size: int = 5,
) -> ComplexityReport:
    """Return exact counts for the repository's default model architectures."""
    n_tril = dof * (dof + 1) // 2
    n_full = dof * dof

    potential = mlp_parameter_count(dof, delan_hidden, 1)
    delan = mlp_parameter_count(dof, delan_hidden, n_tril) + potential
    full_mass_energy = mlp_parameter_count(dof, delan_hidden, n_full) + potential

    direct_mlp = mlp_parameter_count(3 * dof, mlp_hidden, dof)
    probabilistic_member = mlp_parameter_count(3 * dof, mlp_hidden, 2 * dof)
    probabilistic_member += 2 * dof  # learned max/min log-variance bounds

    return ComplexityReport(
        dof=dof,
        cholesky_entries=n_tril,
        full_mass_entries=n_full,
        delan_parameters=delan,
        full_mass_energy_parameters=full_mass_energy,
        direct_mlp_parameters=direct_mlp,
        probabilistic_ensemble_parameters=ensemble_size * probabilistic_member,
    )
