import json

import numpy as np

from experiments.gauge_embedded_curved_keys import (
    atlas_and_polar_gate,
    curvature_coefficients,
    functional_rank_gate,
    gauge_containment_gate,
    gauge_fix_pair,
    polar_pivot,
    choose_pivot,
    resource_ledger,
    strict_extension_witness,
)


def test_pair_gauge_sets_aliased_coefficient_to_zero():
    query = np.arange(24, dtype=np.float64).reshape(4, 2, 3) / 11.0
    key = np.array([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    fixed_query, fixed_key, gauge = gauge_fix_pair(query, key, pivot=1, tau=0.25)
    assert abs(fixed_key[0, 1]) < 1e-12
    assert abs(fixed_key[1, 1] - 0.25) < 1e-12
    np.testing.assert_allclose(
        gauge.T @ gauge,
        np.eye(2) * gauge[0].dot(gauge[0]),
        atol=1e-12,
    )
    assert fixed_query.shape == query.shape
    assert choose_pivot(key, tau=np.linalg.norm(key[:, 1])) == 1


def test_geck_contains_gqa_rope_baseline_and_is_strict_extension():
    containment = gauge_containment_gate()
    assert containment["pass"]
    assert all(containment["checks"].values())
    extension = strict_extension_witness()
    assert extension["second_difference"] > 1.0
    assert extension["ordinary_bilinear_second_difference"] == 0.0
    assert extension["strict_extension"]


def test_atlas_edge_cases_and_centered_polar_chart():
    result = atlas_and_polar_gate()
    json.dumps(result)
    assert result["pass"]
    assert all(result["checks"].values())
    tau = 0.125
    pivot = polar_pivot(phi=0.3, log_scale=-0.4, tau=tau)
    coefficients = curvature_coefficients(float(pivot[0]), float(pivot[1]), tau)
    np.testing.assert_allclose(
        coefficients, (np.sin(0.3), np.tanh(-0.4)), atol=1e-12
    )


def test_geck_recovers_both_rope_gauge_directions_as_functional_rank():
    result = functional_rank_gate()
    assert result["pass"]
    assert result["ordinary_rank"] == 10
    assert result["curved_baseline_slice_rank"] == 12
    assert result["generic_curved_rank"] == 12
    assert len(result["worlds"]) == 5


def test_geck_ledger_adds_no_weights_or_cache_width():
    ledger = resource_ledger(hidden_width=640, head_dim=64, kv_heads=2)
    assert ledger["dense_key_weights_both"] == 81_920
    assert ledger["dense_key_macs_per_token_both"] == 81_920
    assert ledger["key_cache_scalars_per_token_both"] == 128
    assert ledger["extra_learned_scalars"] == 0
    assert ledger["extra_scalar_multiplications_per_token"] == 256
    assert ledger["extra_scalar_additions_per_token"] == 128
    assert ledger["existing_weight_coefficients_read_by_epilogue"] == 128
    assert ledger["compile_time_pivot_indices"] == 64
    assert ledger["compile_time_pivot_index_bits"] == 640
    assert ledger["weight_only_coefficient_square_roots"] == 64
    assert ledger["weight_only_coefficient_divisions"] == 128
