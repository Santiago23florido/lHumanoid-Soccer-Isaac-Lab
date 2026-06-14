"""Tests for the matched data-budget sweep."""

from lagrangian_mbrl.pipeline.sample_complexity import (
    SampleComplexityConfig,
    empirical_kappas,
    first_budget_to_threshold,
    run_sample_complexity,
)


def test_threshold_and_empirical_kappa():
    structured = [
        {"n_train": 32, "accel_mse_mean": 0.2},
        {"n_train": 64, "accel_mse_mean": 0.04},
    ]
    comparator = [
        {"n_train": 32, "accel_mse_mean": 0.3},
        {"n_train": 64, "accel_mse_mean": 0.1},
        {"n_train": 128, "accel_mse_mean": 0.03},
    ]
    assert first_budget_to_threshold(structured, 0.05) == 64
    result = empirical_kappas(
        {"delan": structured, "mlp": comparator}, (0.05,), "delan", "mlp"
    )
    assert result[0]["kappa_empirical"] == 2.0


def test_sample_complexity_smoke(tmp_path):
    config = SampleComplexityConfig(
        models=("mlp",),
        budgets=(24, 32),
        n_val=32,
        tuning_n_train=24,
        seeds=(0,),
        hpo_trials=1,
        hpo_epochs=1,
        train_epochs=1,
        n_bootstrap=20,
        target_mses=(10.0,),
        structured_model="mlp",
        comparator_model="mlp",
    )
    results = run_sample_complexity(config, out_dir=tmp_path)
    assert len(results["models"]["mlp"]["rows"]) == 2
    assert (tmp_path / "sample_complexity_results.json").exists()
    assert (tmp_path / "sample_complexity.csv").exists()
