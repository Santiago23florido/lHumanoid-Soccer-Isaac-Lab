"""Tests for the fully computable mechanical LQR surrogate."""

import numpy as np

from lagrangian_mbrl.theory.lqr_surrogate import (
    fit_structured_mechanics,
    fit_unstructured,
    lqr_cost,
    lqr_gain,
    make_mechanical_system,
    project_to_mechanics,
    sample_transitions,
    structured_parameter_count,
    unstructured_parameter_count,
)


def test_exact_mechanical_projection_is_identity():
    system = make_mechanical_system(dof=3)
    a_hat, b_hat, mass, stiffness, damping = project_to_mechanics(
        system.a, system.b, system.dof, system.dt
    )
    assert np.allclose(a_hat, system.a)
    assert np.allclose(b_hat, system.b)
    assert np.allclose(mass, system.mass)
    assert np.allclose(stiffness, system.stiffness)
    assert np.allclose(damping, system.damping)


def test_noiseless_identification_recovers_optimal_controller():
    system = make_mechanical_system(dof=2)
    state, action, next_state = sample_transitions(
        system, 256, np.random.default_rng(0), noise_std=0.0
    )
    a_free, b_free = fit_unstructured(state, action, next_state)
    a_struct, b_struct, _, _, _ = fit_structured_mechanics(
        state,
        action,
        next_state,
        dof=system.dof,
        dt=system.dt,
        a_initial=a_free,
        b_initial=b_free,
    )
    optimal_gain = lqr_gain(system.a, system.b, system.q_cost, system.r_cost)
    structured_gain = lqr_gain(a_struct, b_struct, system.q_cost, system.r_cost)
    optimal_cost = lqr_cost(
        system.a, system.b, optimal_gain, system.q_cost, system.r_cost
    )
    structured_cost = lqr_cost(
        system.a, system.b, structured_gain, system.q_cost, system.r_cost
    )
    assert np.allclose(a_free, system.a, atol=1e-8)
    assert np.allclose(b_free, system.b, atol=1e-8)
    assert abs(structured_cost - optimal_cost) < 1e-8


def test_structured_dimension_is_smaller():
    assert structured_parameter_count(7) == 63
    assert unstructured_parameter_count(7) == 294
