from __future__ import annotations

import itertools
import json
import unittest
from pathlib import Path

from experiments.constraint_basin_no_go import (
    HAMMING_7_4_CHECKS,
    attention_read,
    basin_step,
    hamming_7_4_codeword,
    hamming_syndrome,
    message_passing_step,
    numerical_parity_step,
    numerical_unit_energy_step,
    run_no_go_gate,
)


class ConstraintBasinNoGoTests(unittest.TestCase):
    def test_prototype_energy_step_is_attention_read(self) -> None:
        patterns = ((1.0, 0.0), (0.0, 1.0), (-0.5, -0.25))
        state = (0.2, -0.3)
        expected = attention_read(patterns, state, beta=1.4)
        actual = numerical_unit_energy_step(patterns, state, beta=1.4)
        for left, right in zip(expected, actual, strict=True):
            self.assertAlmostEqual(left, right, delta=2e-8)

    def test_basin_step_is_message_passing_step(self) -> None:
        state = (0.8, -0.6, 0.2, -0.4, 0.5, -0.7, 0.3)
        cue = (0.7, -0.2, 0.1, -0.9, 0.6, -0.1, 0.4)
        direct = basin_step(
            state, cue, HAMMING_7_4_CHECKS, penalty=0.9, step_size=0.2
        )
        messages = message_passing_step(
            state, cue, HAMMING_7_4_CHECKS, penalty=0.9, step_size=0.2
        )
        for left, right in zip(direct, messages, strict=True):
            self.assertAlmostEqual(left, right, delta=2e-15)
        numerical = numerical_parity_step(
            state, cue, HAMMING_7_4_CHECKS, penalty=0.9, step_size=0.2
        )
        for left, right in zip(direct, numerical, strict=True):
            self.assertAlmostEqual(left, right, delta=2e-9)

    def test_hamming_codebook_is_valid(self) -> None:
        codebook = {
            hamming_7_4_codeword(message)
            for message in itertools.product((0, 1), repeat=4)
        }
        self.assertEqual(len(codebook), 16)
        self.assertTrue(all(hamming_syndrome(word) == 0 for word in codebook))

    def test_full_gate_closes_before_training(self) -> None:
        report = run_no_go_gate()
        self.assertEqual(report["status"], "closed_before_training")
        self.assertIsNone(report["candidate_number"])
        self.assertFalse(report["gpu_required"])
        self.assertFalse(report["resource_dominance_claimed"])
        self.assertEqual(
            report["decision"], "no_distinct_capability_algebra_admitted"
        )
        ceiling = report["direct_control_reaches_map_ceiling"]
        self.assertEqual(ceiling["clean_plus_single_flip_episodes"], 128)
        self.assertEqual(ceiling["syndrome_accuracy"], 1.0)
        self.assertEqual(ceiling["exact_map_accuracy"], 1.0)
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "constraint-basin-no-go.json"
        )
        self.assertEqual(json.loads(result_path.read_text()), report)


if __name__ == "__main__":
    unittest.main()
