from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.gauge_funded_quadratic_values import (
    gauge_fix_value_output,
    key_coupled_kv_ledger,
    key_coupled_quadratic_value,
    matmul,
    mixed_finite_difference,
    row_times_matrix,
    run_stage0_gate,
    structured_quadratic_value,
    value_projection_ledger,
)


class GaugeFundedQuadraticValueTests(unittest.TestCase):
    def test_value_output_gauge_preserves_product_and_sets_identity_minor(self) -> None:
        value = [
            [2.0, 1.0],
            [1.0, 3.0],
            [4.0, -2.0],
            [0.5, 1.5],
        ]
        output = [[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]]
        fixed_value, fixed_output = gauge_fix_value_output(value, output, (0, 1))
        self.assertEqual(
            [[round(entry, 12) for entry in fixed_value[row]] for row in (0, 1)],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        original_product = matmul(value, output)
        fixed_product = matmul(fixed_value, fixed_output)
        for original_row, fixed_row in zip(
            original_product, fixed_product, strict=True
        ):
            for original, fixed in zip(original_row, fixed_row, strict=True):
                self.assertAlmostEqual(original, fixed, places=11)

    def test_zero_gate_exactly_recovers_gauge_fixed_linear_values(self) -> None:
        fixed_value = [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, -1.0],
            [0.5, 3.0],
        ]
        x = [0.25, -2.0, 1.5, 0.75]
        zero_gate = [[0.0, 0.0], [0.0, 0.0]]
        expected = row_times_matrix(x, fixed_value)
        actual = structured_quadratic_value(x, fixed_value, (0, 1), zero_gate)
        for expected_value, actual_value in zip(expected, actual, strict=True):
            self.assertAlmostEqual(expected_value, actual_value, places=12)

    def test_quadratic_value_has_nonzero_mixed_finite_difference(self) -> None:
        fixed_value = [[1.0, 0.0], [0.0, 1.0]]
        gate = [[0.0, 0.0], [1.0, 0.0]]

        def candidate(x: tuple[float, float]) -> float:
            return structured_quadratic_value(x, fixed_value, (0, 1), gate)[0]

        witness = mixed_finite_difference(
            candidate,
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
        )
        self.assertEqual(witness, 1.0)

    def test_scalar_arithmetic_ledger_matches_dense_projection(self) -> None:
        for model_width, value_width in ((8, 2), (16, 4), (4_096, 128)):
            ledger = value_projection_ledger(model_width, value_width)
            self.assertEqual(
                ledger["dense_scalar_multiplications"],
                ledger["candidate_scalar_multiplications"],
            )
            self.assertEqual(
                ledger["dense_scalar_additions"],
                ledger["candidate_scalar_additions"],
            )
            self.assertEqual(ledger["candidate_weight_saving"], value_width)

    def test_key_coupled_gate_contains_linear_values_and_is_quadratic(self) -> None:
        fixed_value = [[1.0, 0.0], [0.0, 1.0]]
        x = (2.0, -3.0)
        self.assertEqual(
            key_coupled_quadratic_value(
                x, fixed_value, (0, 1), key=x, gate_scale=(0.0, 0.0)
            ),
            [2.0, -3.0],
        )

        def candidate(first: float) -> float:
            point = (first, 0.0)
            return key_coupled_quadratic_value(
                point,
                fixed_value,
                (0, 1),
                key=point,
                gate_scale=(1.0, 0.0),
            )[0]

        self.assertEqual(candidate(1.0) - 2 * candidate(0.0) + candidate(-1.0), 2.0)

    def test_key_coupled_variant_reduces_logical_kv_work(self) -> None:
        ledger = key_coupled_kv_ledger(4_096, 128)
        self.assertEqual(ledger["candidate_weight_saving"], 128 * 127)
        self.assertEqual(ledger["candidate_multiplication_saving"], 128 * 126)
        self.assertEqual(ledger["candidate_addition_saving"], 128 * 126)

    def test_invalid_gate_diagonal_and_singular_pivot_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            structured_quadratic_value(
                (1.0, 2.0),
                ((1.0, 0.0), (0.0, 1.0)),
                (0, 1),
                ((1.0, 0.0), (0.0, 0.0)),
            )
        with self.assertRaises(ValueError):
            gauge_fix_value_output(
                ((1.0, 2.0), (2.0, 4.0), (0.0, 1.0)),
                ((1.0,), (2.0,)),
                (0, 1),
            )

    def test_persisted_stage0_report(self) -> None:
        report = run_stage0_gate()
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "gauge-funded-quadratic-values-stage0.json"
        )
        self.assertEqual(json.loads(result_path.read_text()), report)
        self.assertEqual(
            report["decision"],
            "retain_as_pre_candidate_gpu_kernel_and_lm_quality_unproven",
        )
        self.assertFalse(report["gpu_required_now"])


if __name__ == "__main__":
    unittest.main()
