import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "folded_metric_optimizer_gate.py"
)
SPEC = importlib.util.spec_from_file_location("folded_metric_optimizer_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_folded_update_matches_explicit_redundant_chart():
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(9, 7))
    a = MODULE.orthonormal_columns(rng, 9, 3)
    b = rng.normal(scale=0.1, size=(3, 7))
    gradient = rng.normal(size=(9, 7))
    merged = weight + a @ b
    folded = MODULE.chart_sgd_update(merged, a, b, gradient, 0.01)
    redundant = MODULE.redundant_parameterization_sgd_update(
        weight, a, b, gradient, 0.01
    )
    np.testing.assert_allclose(folded[0], redundant[0], atol=1e-14)
    np.testing.assert_allclose(folded[1], redundant[2], atol=1e-14)
    np.testing.assert_allclose(folded[2], redundant[3], atol=1e-14)


def test_zero_B_first_step_has_closed_form():
    rng = np.random.default_rng(2)
    weight = rng.normal(size=(11, 8))
    a = MODULE.orthonormal_columns(rng, 11, 4)
    b = np.zeros((4, 8))
    gradient = rng.normal(size=(11, 8))
    next_merged, _, _ = MODULE.chart_sgd_update(
        weight, a, b, gradient, 0.02, scale=0.7
    )
    expected = MODULE.first_step_closed_form(gradient, a, 0.02, scale=0.7)
    np.testing.assert_allclose(next_merged - weight, expected, atol=1e-14)


def test_learning_A_is_not_fixed_metric_after_B_moves():
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(10, 6))
    a = MODULE.orthonormal_columns(rng, 10, 3)
    b = rng.normal(scale=0.2, size=(3, 6))
    gradient = rng.normal(size=(10, 6))
    learned = MODULE.chart_sgd_update(weight, a, b, gradient, 0.01, learn_a=True)[0]
    fixed = MODULE.chart_sgd_update(weight, a, b, gradient, 0.01, learn_a=False)[0]
    assert np.linalg.norm(learned - fixed) > 0.0


def test_resource_ledger_has_no_model_or_serving_overhead():
    ledger = MODULE.resource_ledger()
    assert ledger["targeted_served_weight_scalars"] == 18_874_368
    assert ledger["extra_optimizer_factor_scalars"] == 1_302_528
    assert ledger["extra_model_state_scalars"] == 0
    assert ledger["extra_served_parameters"] == 0
    assert ledger["extra_served_macs"] == 0
    assert ledger["fixed_a_extra_macs_per_optimizer_update"] == 3 * ledger["single_factor_product_macs"]
    assert ledger["learned_ab_extra_macs_per_optimizer_update"] == 4 * ledger["single_factor_product_macs"]


def test_all_frozen_gates_pass():
    assert MODULE.run_gate(seed=331, trials=16)["all_gates_pass"]
