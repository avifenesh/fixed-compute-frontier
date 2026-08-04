from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.epsilon_fingerprint_register import (
    FIELD_PRIME_64,
    FINISHED_PHASE,
    FingerprintRegister,
    bounded_streaming_gate,
    equal_with_fingerprint,
    exhaustive_root_bound_gate,
    fingerprint,
    is_prime_64,
    run_stage0_gate,
    state_ledger,
    streaming_equal_with_fingerprint,
)


class EpsilonFingerprintRegisterTests(unittest.TestCase):
    def test_frozen_moduli_are_prime(self) -> None:
        self.assertTrue(is_prime_64(257))
        self.assertTrue(is_prime_64(FIELD_PRIME_64))
        for composite in (0, 1, 4, 9, 341, 561, 1_105):
            self.assertFalse(is_prime_64(composite))

    def test_equal_streams_never_reject(self) -> None:
        stream = (0, 1, 255, 8, 13, 21)
        for point in (0, 1, 2, 42, FIELD_PRIME_64 - 1):
            self.assertTrue(
                equal_with_fingerprint(stream, stream, point, FIELD_PRIME_64)
            )

    def test_length_is_part_of_the_decision(self) -> None:
        self.assertEqual(fingerprint((1,), 0, 257), fingerprint((0, 1), 0, 257))
        self.assertFalse(equal_with_fingerprint((1,), (0, 1), 0, 257))

    def test_bounded_streaming_register_and_empty_segments(self) -> None:
        self.assertTrue(
            streaming_equal_with_fingerprint(
                (), (), evaluation_point=7, prime=257, maximum_length=0
            )
        )
        self.assertTrue(
            streaming_equal_with_fingerprint(
                (1, 2, 3),
                (1, 2, 3),
                evaluation_point=11,
                prime=257,
                maximum_length=3,
            )
        )
        self.assertFalse(
            streaming_equal_with_fingerprint(
                (1, 2, 3),
                (1, 2),
                evaluation_point=11,
                prime=257,
                maximum_length=3,
            )
        )

    def test_invalid_transitions_and_overflow_do_not_mutate(self) -> None:
        with self.assertRaises(ValueError):
            FingerprintRegister(evaluation_point=1, prime=15, maximum_length=1)
        register = FingerprintRegister(
            evaluation_point=11, prime=257, maximum_length=1
        )
        with self.assertRaises(ValueError):
            register.finish()
        register.push(7)
        snapshot = register.snapshot()
        with self.assertRaises(OverflowError):
            register.push(8)
        self.assertEqual(register.snapshot(), snapshot)
        with self.assertRaises(ValueError):
            register.push(256)
        self.assertEqual(register.snapshot(), snapshot)
        register.separator()
        with self.assertRaises(ValueError):
            register.separator()
        register.push(7)
        self.assertTrue(register.finish())
        self.assertEqual(register.phase, FINISHED_PHASE)
        finished_snapshot = register.snapshot()
        with self.assertRaises(ValueError):
            register.push(7)
        with self.assertRaises(ValueError):
            register.finish()
        self.assertEqual(register.snapshot(), finished_snapshot)

    def test_million_token_boundary_gate(self) -> None:
        result = bounded_streaming_gate()
        self.assertTrue(result["accepted_equal_segments_at_boundary"])
        self.assertTrue(result["overflow_rejected_without_mutation"])
        self.assertEqual(result["maximum_length_per_segment"], 1_000_000)
        self.assertNotIn("tokens", result["stored_fields"])

    def test_exhaustive_polynomial_root_bound(self) -> None:
        result = exhaustive_root_bound_gate()
        self.assertEqual(result["unequal_pairs"], 5_334)
        self.assertEqual(result["field_evaluations"], 1_370_838)
        self.assertLessEqual(
            result["maximum_collision_roots"], result["maximum_length"] - 1
        )

    def test_million_byte_ledger(self) -> None:
        ledger = state_ledger(
            maximum_length=1_000_000,
            alphabet_size=256,
            prime=FIELD_PRIME_64,
        )
        self.assertEqual(ledger["logical_register_bits"], 234)
        self.assertEqual(ledger["logical_register_bytes_ceiling"], 30)
        self.assertEqual(
            ledger["deterministic_exact_one_pass_bytes_lower_bound"], 1_000_000
        )
        self.assertLess(
            ledger["worst_case_false_positive_probability_bound"], 5.5e-14
        )

    def test_persisted_stage0_report(self) -> None:
        report = run_stage0_gate()
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "epsilon-fingerprint-register-stage0.json"
        )
        self.assertEqual(json.loads(result_path.read_text()), report)
        self.assertEqual(
            report["decision"],
            "retain_as_classical_epsilon_trade_not_architecture_candidate",
        )
        self.assertEqual(report["schema_version"], 2)
        self.assertFalse(report["gpu_required"])


if __name__ == "__main__":
    unittest.main()
