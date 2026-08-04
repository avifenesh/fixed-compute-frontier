import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "two_sided_shadow_factor_gate.py"
)
SPEC = importlib.util.spec_from_file_location("two_sided_shadow_factor_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_zero_branches_are_exact_dense_endpoint():
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(9, 7))
    factors = MODULE.initialize_chart(rng, 9, 7, 3)
    assert np.array_equal(MODULE.merged_weight(weight, *factors), weight)


def test_first_step_is_exact_two_sided_preconditioner():
    rng = np.random.default_rng(2)
    weight = rng.normal(size=(11, 8))
    gradient = rng.normal(size=(11, 8))
    factors = MODULE.initialize_chart(rng, 11, 8, 4)
    initial = MODULE.merged_weight(weight, *factors)
    stepped = MODULE.one_sgd_step(
        weight, *factors, gradient, learning_rate=0.01, scale=0.7
    )
    actual = MODULE.merged_weight(*stepped, scale=0.7) - initial
    expected = MODULE.predicted_first_effective_update(
        factors[0], factors[3], gradient, learning_rate=0.01, scale=0.7
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_projectors_have_frozen_rank_and_are_idempotent():
    rng = np.random.default_rng(3)
    a_left, _, _, b_right = MODULE.initialize_chart(rng, 13, 10, 4)
    left = a_left @ a_left.T
    right = b_right.T @ b_right
    assert np.linalg.matrix_rank(left, tol=1e-10) == 4
    assert np.linalg.matrix_rank(right, tol=1e-10) == 4
    np.testing.assert_allclose(left @ left, left, atol=1e-12)
    np.testing.assert_allclose(right @ right, right, atol=1e-12)


def test_resource_ledger_moves_cost_only_to_training():
    ledger = MODULE.resource_ledger()
    assert ledger["served_ffn_weight_parameters"] == 14_155_776
    assert ledger["candidate_served_ffn_weight_parameters"] == 14_155_776
    assert ledger["candidate_extra_served_parameters"] == 0
    assert ledger["candidate_extra_served_dense_macs_per_token"] == 0
    assert ledger["extra_training_parameters"] == 1_622_016


def test_all_frozen_gates_pass():
    assert MODULE.run_gate(seed=307, trials=16)["all_gates_pass"]
