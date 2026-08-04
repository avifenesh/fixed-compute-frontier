import numpy as np

from experiments.t32r_rank32_scan_stage0 import (
    HIDDEN,
    RANK,
    RECORD_TOKENS,
    concentration_world,
    depthwise_context,
    multiplication_ledger,
    parameter_ledger,
    position_code,
)


def test_position_code_shape_and_finiteness():
    code = position_code(RECORD_TOKENS)
    assert code.shape == (RECORD_TOKENS, RANK)
    assert np.isfinite(code).all()
    assert np.array_equal(code[0, 0::2], np.zeros(RANK // 2, dtype=np.float32))
    assert np.array_equal(code[0, 1::2], np.ones(RANK // 2, dtype=np.float32))


def test_depthwise_filter_has_zero_boundary_padding():
    values = np.ones((RECORD_TOKENS, RANK), dtype=np.float32)
    taps = np.ones((3, RANK), dtype=np.float32)
    result = depthwise_context(values, taps)
    assert np.array_equal(result[0], np.full(RANK, 2.0, dtype=np.float32))
    assert np.array_equal(result[1], np.full(RANK, 3.0, dtype=np.float32))
    assert np.array_equal(result[-1], np.full(RANK, 2.0, dtype=np.float32))


def test_concentration_world_matches_closed_form_and_bound():
    for delta in (2, 4, 8, 12, 16):
        row = concentration_world(delta)
        assert abs(row["target_weight"] - row["exact_target_weight"]) <= 1e-12
        assert row["selected_value_error"] <= row["theorem_bound"] + 1e-6
        assert np.isfinite(row["target_weight_derivative"])


def test_resource_ledgers_are_exact():
    parameters = parameter_ledger()
    multiplies = multiplication_ledger()
    assert HIDDEN == 384 and RANK == 32
    assert parameters["total_allocated_entries"] == 943_538
    assert parameters["matched_slack_entries"] == 24_142
    assert multiplies["total_multiplies"] == 3_230_144
    assert multiplies["total_multiplies"] < multiplies["frozen_envelope"]
