import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "block_algebra_swiglu_gate.py"
)
SPEC = importlib.util.spec_from_file_location("block_algebra_swiglu_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_core_and_gauss_match_direct_complex_multiplication():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(11, 7, 2))
    u = rng.normal(size=(11, 7, 2))
    expected = MODULE.complex_direct(a, u)
    np.testing.assert_allclose(
        MODULE.apply_core(a, u, MODULE.complex_core()), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        MODULE.complex_gauss_three_multiply(a, u), expected, atol=1e-12
    )


def test_determinant_pencils_separate_real_tensor_orbits():
    assert MODULE.determinant_pencil_coefficients(MODULE.diagonal_core()) == (0.0, 1.0, 0.0)
    assert MODULE.determinant_pencil_coefficients(MODULE.complex_core()) == (-1.0, 0.0, -1.0)
    assert MODULE.pencil_discriminant(MODULE.diagonal_core()) == 1.0
    assert MODULE.pencil_discriminant(MODULE.complex_core()) == -4.0


def test_tiny_zero_init_blend_does_not_reach_rank_three_region():
    boundary = MODULE.rank_three_boundary()
    assert 0.431 < boundary < 0.433
    assert MODULE.interpolation_discriminant(0.01) > 0.0
    assert MODULE.interpolation_discriminant(boundary - 1e-6) > 0.0
    assert MODULE.interpolation_discriminant(boundary + 1e-6) < 0.0
    for alpha in (0.0, 0.1, 0.25, 0.5, 1.0):
        np.testing.assert_allclose(
            MODULE.pencil_discriminant(MODULE.interpolated_core(alpha)),
            MODULE.interpolation_discriminant(alpha),
            atol=1e-12,
        )


def test_target_scale_resource_ledger_keeps_dense_budget():
    ledger = MODULE.resource_ledger()
    assert ledger["dense_projection_macs_per_token"] == 176_160_768
    assert ledger["candidate_dense_weight_parameters"] == ledger["dense_weight_parameters"]
    assert ledger["candidate_extra_parameters_fixed_core"] == 0
    assert ledger["complex_gauss_products_per_token"] == 21_504
    assert ledger["gauss_extra_product_fraction_of_dense_macs"] < 1e-4


def test_all_frozen_algebra_gates_pass():
    assert MODULE.run_gate(seed=211, trials=16)["all_gates_pass"]
