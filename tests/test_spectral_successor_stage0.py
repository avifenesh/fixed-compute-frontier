import numpy as np

from experiments.spectral_successor_stage0 import (
    direct_targets,
    fixed_codes,
    recurrence_targets,
    spectral_points,
    vandermonde_recovery,
)


def test_reverse_recurrence_matches_direct_sum() -> None:
    rng = np.random.default_rng(11)
    codes = fixed_codes(31, 4, 12)
    points = spectral_points(8)
    tokens = rng.integers(0, 31, size=29)
    direct = direct_targets(tokens, codes, points)
    recurrent = recurrence_targets(tokens, codes, points, np.float64)
    np.testing.assert_allclose(recurrent, direct, atol=1e-12, rtol=1e-12)


def test_order_changes_spectral_target_but_not_bag() -> None:
    codes = fixed_codes(16, 4, 13)
    points = spectral_points(8)
    left = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    right = left[::-1]
    np.testing.assert_array_equal(codes[left].sum(axis=0), codes[right].sum(axis=0))
    left_target = recurrence_targets(
        np.concatenate(([0], left)), codes, points, np.float64
    )[0]
    right_target = recurrence_targets(
        np.concatenate(([0], right)), codes, points, np.float64
    )[0]
    assert not np.allclose(left_target, right_target)


def test_vandermonde_summary_recovers_short_horizon() -> None:
    measurement = vandermonde_recovery(8, 4, 14)
    assert measurement["maximum_reconstruction_error"] <= 1e-10
    assert measurement["condition_number"] <= 2.0
