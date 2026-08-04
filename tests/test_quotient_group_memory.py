from __future__ import annotations

import itertools
import json
import unittest
from pathlib import Path

from experiments.quotient_group_memory import (
    IDENTITY_S3,
    QuotientGroupMemory,
    compose,
    crosscheck_trace,
    inverse,
    labeled_partition_state_count,
    logical_state_ledger,
    run_stage0_gate,
    s3_elements,
)


class QuotientGroupMemoryTests(unittest.TestCase):
    def test_s3_group_operations(self) -> None:
        group = s3_elements()
        self.assertEqual(len(group), 6)
        for value in group:
            self.assertEqual(compose(value, inverse(value)), IDENTITY_S3)
            self.assertEqual(compose(inverse(value), value), IDENTITY_S3)
        for left, middle, right in itertools.product(group, repeat=3):
            self.assertEqual(
                compose(compose(left, middle), right),
                compose(left, compose(middle, right)),
            )

    def test_non_commutative_relations_and_contradiction(self) -> None:
        swap_01 = (1, 0, 2)
        swap_12 = (0, 2, 1)
        self.assertNotEqual(compose(swap_01, swap_12), compose(swap_12, swap_01))

        memory = QuotientGroupMemory(3, IDENTITY_S3)
        self.assertEqual(memory.link(0, 1, swap_01), "merged")
        self.assertEqual(memory.link(1, 2, swap_12), "merged")
        composed = compose(swap_12, swap_01)
        self.assertEqual(memory.relation(0, 2), composed)
        self.assertEqual(memory.relation(2, 0), inverse(composed))
        self.assertEqual(memory.link(0, 2, composed), "consistent")
        wrong = next(value for value in s3_elements() if value != composed)
        self.assertEqual(memory.link(0, 2, wrong), "contradiction")
        self.assertEqual(memory.relation(0, 2), composed)

    def test_state_count_reduces_to_bell_numbers_for_trivial_group(self) -> None:
        self.assertEqual(
            [labeled_partition_state_count(size, 1) for size in range(1, 6)],
            [1, 2, 5, 15, 52],
        )

    def test_state_count_matches_brute_force_for_nontrivial_groups(self) -> None:
        def restricted_growth_partitions(size: int) -> list[tuple[int, ...]]:
            partitions: list[tuple[int, ...]] = []
            for labels in itertools.product(range(size), repeat=size):
                if labels[0] != 0:
                    continue
                if all(
                    labels[index] <= 1 + max(labels[:index])
                    for index in range(1, size)
                ):
                    partitions.append(labels)
            return partitions

        def brute_force_count(size: int, group_order: int) -> int:
            count = 0
            for partition in restricted_growth_partitions(size):
                roots = {
                    partition.index(block) for block in set(partition)
                }
                count += sum(
                    all(potentials[root] == 0 for root in roots)
                    for potentials in itertools.product(range(group_order), repeat=size)
                )
            return count

        for size in range(1, 5):
            for group_order in (2, 3):
                self.assertEqual(
                    labeled_partition_state_count(size, group_order),
                    brute_force_count(size, group_order),
                )

    def test_state_ledger_boundary_and_frozen_values(self) -> None:
        with self.assertRaises(ValueError):
            logical_state_ledger(1, 6)
        ledger = logical_state_ledger(64, 6)
        self.assertEqual(ledger["minimum_distinguishing_bits"], 339)
        self.assertEqual(ledger["simple_packed_representation_bits"], 768)
        self.assertEqual(ledger["simple_bit_packed_array_payload_bytes"], 96)
        self.assertEqual(
            ledger["hypothetical_packed_uint32_uint8_uint8_array_payload_bytes"],
            384,
        )

    def test_independent_graph_reference(self) -> None:
        report = crosscheck_trace(seed=17, size=24, operations=512)
        self.assertEqual(report["operations"], 512)
        self.assertEqual(report["relation_checks"], 2_048)
        self.assertEqual(
            report["merged"] + report["consistent"] + report["contradiction"],
            512,
        )

    def test_persisted_stage0_report(self) -> None:
        report = run_stage0_gate()
        result_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "quotient-group-memory-stage0.json"
        )
        self.assertEqual(json.loads(result_path.read_text()), report)
        self.assertEqual(
            report["decision"],
            "retain_as_classical_typed_ram_primitive_not_architecture_candidate",
        )
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(
            report["resource_model"]["machine"],
            "online_word_RAM_or_cell_probe",
        )
        self.assertIn(
            "language_or_entity_name_to_dense_slot_mapping",
            report["excluded_accounting"],
        )
        self.assertFalse(report["gpu_required"])


if __name__ == "__main__":
    unittest.main()
