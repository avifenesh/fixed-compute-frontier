from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.gauge_curvature_permutation_mask import (
    cyclic_forbidden_sources,
    factor_scalar_quadratic_coefficients,
    mask_covers_all_quadratic_monomials,
    permutation_mask_ledger,
    permutation_mask_value,
    run_stage0_gate,
    scalar_quadratic,
)


class GaugeCurvaturePermutationMaskTests(unittest.TestCase):
    def test_cyclic_mask_covers_squares_and_cross_terms(self) -> None:
        for width in (3, 4, 8, 128):
            self.assertTrue(
                mask_covers_all_quadratic_monomials(
                    cyclic_forbidden_sources(width)
                )
            )
        self.assertFalse(mask_covers_all_quadratic_monomials((1, 0)))
        self.assertFalse(mask_covers_all_quadratic_monomials((0, 2, 1)))

    def test_zero_gate_recovers_identity_value_map(self) -> None:
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        actual = permutation_mask_value(
            (2.0, -3.0, 0.5),
            identity,
            (0, 1, 2),
            ((0.0, 0.0, 0.0),) * 3,
            cyclic_forbidden_sources(3),
        )
        self.assertEqual(actual, [2.0, -3.0, 0.5])

    def test_mask_factors_arbitrary_scalar_quadratic(self) -> None:
        coefficients = (
            (2.0, -1.0, 0.25, 3.0),
            (0.0, -0.5, 1.5, 0.75),
            (0.0, 0.0, 4.0, -2.0),
            (0.0, 0.0, 0.0, 1.25),
        )
        readout = (1.0, 2.0, -1.0, 0.5)
        forbidden = cyclic_forbidden_sources(4)
        gate = factor_scalar_quadratic_coefficients(
            coefficients, forbidden, readout
        )
        identity = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(4))
            for row in range(4)
        )
        for x in ((1.0, 2.0, 3.0, 4.0), (-0.5, 1.25, 2.0, -3.0)):
            value = permutation_mask_value(x, identity, (0, 1, 2, 3), gate, forbidden)
            quadratic = sum(
                readout[channel] * (value[channel] - x[channel])
                for channel in range(4)
            )
            self.assertAlmostEqual(quadratic, scalar_quadratic(x, coefficients), places=11)

    def test_ledger_exactly_matches_dense_value_projection(self) -> None:
        for model_width, value_width in ((3, 3), (16, 4), (4_096, 128)):
            ledger = permutation_mask_ledger(model_width, value_width)
            self.assertEqual(
                ledger["dense_scalar_multiplications"],
                ledger["candidate_scalar_multiplications"],
            )
            self.assertEqual(
                ledger["dense_scalar_additions"],
                ledger["candidate_scalar_additions"],
            )
            self.assertEqual(
                ledger["covered_unique_quadratic_monomials"],
                ledger["total_unique_quadratic_monomials"],
            )

    def test_forbidden_gate_entry_is_rejected(self) -> None:
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            permutation_mask_value(
                (1.0, 2.0, 3.0),
                identity,
                (0, 1, 2),
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                cyclic_forbidden_sources(3),
            )

    def test_persisted_stage0_report(self) -> None:
        report = run_stage0_gate()
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "gauge-curvature-permutation-mask-stage0.json"
        )
        self.assertEqual(json.loads(result_path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
