from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.dense_exception_scaling_law import (
    feature_exception_cost_ratio,
    isotropic_error,
    isotropic_required_rank,
    near_linear_fallback_rate,
    random_orthogonal_witness,
    required_rank_from_spectrum,
    spectrum,
)


class DenseExceptionScalingLawTests(unittest.TestCase):
    def test_isotropic_rank_formula_meets_frozen_tolerances(self) -> None:
        width = 4096
        expected_fractions = {0.10: 0.90, 0.05: 0.975, 0.01: 0.999}
        for tolerance, expected in expected_fractions.items():
            rank = isotropic_required_rank(width, tolerance)
            self.assertGreaterEqual(rank / width, expected - 1 / width)
            self.assertLessEqual(isotropic_error(width, rank), tolerance + 1e-12)
            if rank > 0:
                self.assertGreater(isotropic_error(width, rank - 1), tolerance)

    def test_flat_spectrum_matches_closed_form(self) -> None:
        for width in (64, 257):
            values = spectrum(width, 0.0)
            for tolerance in (0.10, 0.05, 0.01):
                self.assertEqual(
                    required_rank_from_spectrum(values, tolerance),
                    isotropic_required_rank(width, tolerance),
                )

    def test_steep_spectrum_needs_smaller_fraction_as_width_grows(self) -> None:
        ranks = [
            required_rank_from_spectrum(spectrum(width, 1.0), 0.05)
            for width in (256, 1024, 4096)
        ]
        fractions = [rank / width for rank, width in zip(ranks, (256, 1024, 4096))]
        self.assertGreater(fractions[0], fractions[1])
        self.assertGreater(fractions[1], fractions[2])

    def test_rank_correction_loses_compute_edge_above_half_width(self) -> None:
        ratio = feature_exception_cost_ratio(4096, 2048)["total_to_dense_ratio"]
        self.assertGreater(ratio, 1.0)

    def test_near_linear_dense_fallback_must_vanish_with_width(self) -> None:
        for width in (256, 1024, 4096):
            self.assertEqual(
                near_linear_fallback_rate(width), math.log2(width) / width
            )
        self.assertGreater(
            near_linear_fallback_rate(256), near_linear_fallback_rate(4096)
        )

    def test_random_orthogonal_exception_has_flat_spectrum(self) -> None:
        witness = random_orthogonal_witness(width=32)
        self.assertTrue(witness["all_singular_values_equal"])
        self.assertAlmostEqual(witness["exception_relative_errors"]["0"], 1.0)
        self.assertAlmostEqual(witness["exception_relative_errors"]["32"], 0.0)
        self.assertAlmostEqual(
            witness["exception_relative_errors"]["16"], np.sqrt(0.5)
        )


if __name__ == "__main__":
    unittest.main()
