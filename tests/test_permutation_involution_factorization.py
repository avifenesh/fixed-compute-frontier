from __future__ import annotations

import itertools
import random
import unittest

from experiments.permutation_involution_factorization import (
    apply_factored_permutation_in_place,
    apply_permutation_out_of_place,
    compose,
    factor_two_involutions,
    is_involution,
    permutation_value_traffic,
    transport_ledger,
    validate_permutation,
)


class TwoInvolutionFactorizationTests(unittest.TestCase):
    def assert_factorization(self, permutation: tuple[int, ...]) -> None:
        first, second = factor_two_involutions(permutation)
        self.assertTrue(is_involution(first))
        self.assertTrue(is_involution(second))
        self.assertEqual(compose(first, second), permutation)
        values = list(range(len(permutation)))
        expected = apply_permutation_out_of_place(values, permutation)
        apply_factored_permutation_in_place(values, permutation)
        self.assertEqual(values, expected)
        traffic = permutation_value_traffic(
            permutation, feature_width=1, value_bytes=1
        )
        moved_factor_rows = sum(
            target != source
            for factor in (first, second)
            for source, target in enumerate(factor)
        )
        self.assertEqual(
            traffic.canonical_two_involution_bytes,
            2 * moved_factor_rows,
        )

    def test_every_permutation_through_n7(self) -> None:
        for size in range(8):
            for permutation in itertools.permutations(range(size)):
                self.assert_factorization(permutation)

    def test_random_large_permutations(self) -> None:
        rng = random.Random(20260726)
        for size in (16, 64, 257):
            for _ in range(20):
                permutation = list(range(size))
                rng.shuffle(permutation)
                self.assert_factorization(tuple(permutation))

    def test_invalid_permutation_is_rejected(self) -> None:
        for invalid in ((0, 0), (1,), (-1,), (True,)):
            with self.assertRaises(ValueError):
                validate_permutation(invalid)

    def test_transport_ledger_exposes_memory_traffic_trade(self) -> None:
        ledger = transport_ledger(64, 16)
        self.assertEqual(ledger.value_payload_bytes, 2048)
        self.assertEqual(ledger.permutation_index_bytes, 128)
        self.assertEqual(ledger.packed_visit_bitset_bytes, 8)
        self.assertEqual(ledger.direct_out_of_place_peak_bytes, 4224)
        self.assertEqual(ledger.ideal_cycle_peak_optimistic_bytes, 2216)
        self.assertEqual(ledger.factored_execution_peak_bytes, 2304)
        self.assertEqual(ledger.factored_end_to_end_optimistic_bytes, 2312)
        self.assertEqual(ledger.factored_optimistic_saving_vs_direct_bytes, 1912)
        self.assertEqual(
            ledger.one_dense_stage_value_traffic_assumption_bytes, 4096
        )
        self.assertEqual(
            ledger.two_dense_stage_value_traffic_assumption_bytes, 8192
        )
        self.assertEqual(
            ledger.extra_dense_stage_value_traffic_assumption_bytes, 4096
        )
        self.assertEqual(ledger.one_index_stream_traffic_assumption_bytes, 128)
        self.assertEqual(ledger.two_index_stream_traffic_assumption_bytes, 256)
        self.assertEqual(
            ledger.one_dense_stage_total_traffic_assumption_bytes, 4224
        )
        self.assertEqual(
            ledger.two_dense_stage_total_traffic_assumption_bytes, 8448
        )

    def test_transport_ledger_checks_unsigned_index_capacity(self) -> None:
        transport_ledger(256, 1, index_bytes=1)
        with self.assertRaises(ValueError):
            transport_ledger(257, 1, index_bytes=1)

    def test_exact_value_traffic_depends_on_cycle_structure(self) -> None:
        identity = permutation_value_traffic((0, 1, 2, 3), 1)
        self.assertEqual(identity.direct_out_of_place_bytes, 16)
        self.assertEqual(identity.ideal_cycle_bytes, 0)
        self.assertEqual(identity.canonical_two_involution_bytes, 0)

        transpositions = permutation_value_traffic((1, 0, 3, 2), 1)
        self.assertEqual(transpositions.ideal_cycle_bytes, 16)
        self.assertEqual(transpositions.canonical_two_involution_bytes, 16)
        self.assertEqual(transpositions.canonical_extra_vs_cycle_bytes, 0)

        long_cycle = permutation_value_traffic((1, 2, 3, 0), 1)
        self.assertEqual(long_cycle.cycle_count, 1)
        self.assertEqual(long_cycle.fixed_point_count, 0)
        self.assertEqual(long_cycle.ideal_cycle_bytes, 16)
        self.assertEqual(long_cycle.canonical_two_involution_bytes, 24)
        self.assertEqual(long_cycle.canonical_extra_vs_cycle_bytes, 8)


if __name__ == "__main__":
    unittest.main()
