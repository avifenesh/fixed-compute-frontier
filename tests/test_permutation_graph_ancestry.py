from __future__ import annotations

import unittest

from experiments.permutation_graph_ancestry import (
    CUT_PARENT,
    ROOT,
    SET_PARENT,
    AncestorBloom64Rebuild,
    ExactIntervalRebuild,
    MaterializedClosureBitset,
    ParentMutation,
    ParentRecompute,
    allocation_table,
    batch_one_allocated_bytes,
    forest_entropy_bits,
    generate_mutation_trace,
    generate_valid_forest,
    packed_direct_sentinel_parent_bytes,
    splitmix64,
    validate_forest,
)


EXACT_TYPES = (ParentRecompute, ExactIntervalRebuild, MaterializedClosureBitset)


def reference_strict_ancestor(
    parents: tuple[int, ...], ancestor: int, node: int
) -> bool:
    if ancestor == node:
        return False
    current = parents[node]
    while current != ROOT:
        if current == ancestor:
            return True
        current = parents[current]
    return False


class AncestryRepresentationTests(unittest.TestCase):
    def assert_exact_modes_agree(self, modes: list[object]) -> None:
        parents = modes[0].parents
        for ancestor in range(len(parents)):
            for node in range(len(parents)):
                expected = reference_strict_ancestor(parents, ancestor, node)
                for mode in modes:
                    self.assertEqual(
                        mode.is_ancestor(ancestor, node),
                        expected,
                        (type(mode).__name__, ancestor, node),
                    )

    def assert_bloom_has_no_false_negatives(
        self, exact: ParentRecompute, bloom: AncestorBloom64Rebuild
    ) -> None:
        for ancestor in range(exact.node_count):
            self.assertFalse(bloom.is_ancestor(ancestor, ancestor))
            for node in range(exact.node_count):
                if exact.is_ancestor(ancestor, node):
                    self.assertTrue(
                        bloom.is_ancestor(ancestor, node),
                        (ancestor, node),
                    )

    def test_exact_modes_match_independent_parent_chase(self) -> None:
        for seed in (0, 17, 20260725):
            parents = generate_valid_forest(32, seed)
            modes = [mode_type(parents) for mode_type in EXACT_TYPES]
            self.assert_exact_modes_agree(modes)

    def test_strict_ancestry_excludes_self(self) -> None:
        parents = (ROOT, 0, 0, 1, 3)
        modes = [mode_type(parents) for mode_type in EXACT_TYPES]
        bloom = AncestorBloom64Rebuild(parents)
        for node in range(len(parents)):
            for mode in [*modes, bloom]:
                self.assertFalse(mode.is_ancestor(node, node))
        self.assertTrue(modes[0].is_ancestor(0, 4))
        self.assertTrue(modes[0].is_ancestor(1, 4))
        self.assertFalse(modes[0].is_ancestor(2, 4))

    def test_bloom_has_no_false_negatives(self) -> None:
        for seed in (3, 29, 43):
            parents = generate_valid_forest(64, seed)
            exact = ParentRecompute(parents)
            bloom = AncestorBloom64Rebuild(parents)
            self.assert_bloom_has_no_false_negatives(exact, bloom)

    def test_reparent_and_cut_rebuild_every_cached_mode(self) -> None:
        parents = (ROOT, 0, 0, 1, 1, 2, 5)
        exact_modes = [mode_type(parents) for mode_type in EXACT_TYPES]
        bloom = AncestorBloom64Rebuild(parents)
        all_modes = [*exact_modes, bloom]

        for mode in all_modes:
            self.assertTrue(mode.set_parent(3, 2))
        self.assert_exact_modes_agree(exact_modes)
        self.assert_bloom_has_no_false_negatives(exact_modes[0], bloom)
        self.assertTrue(exact_modes[0].is_ancestor(2, 3))
        self.assertFalse(exact_modes[0].is_ancestor(1, 3))

        for mode in all_modes:
            self.assertTrue(mode.cut_parent(2))
        self.assert_exact_modes_agree(exact_modes)
        self.assert_bloom_has_no_false_negatives(exact_modes[0], bloom)
        self.assertFalse(exact_modes[0].is_ancestor(0, 3))
        self.assertTrue(exact_modes[0].is_ancestor(2, 3))

    def test_parent_mutation_acceptance_and_rejection_semantics(self) -> None:
        parents = (ROOT, 0, 1, ROOT)
        for mode_type in (*EXACT_TYPES, AncestorBloom64Rebuild):
            mode = mode_type(parents)

            before = mode.parents
            self.assertTrue(mode.set_parent(2, 1))  # valid idempotent
            self.assertEqual(mode.parents, before)

            self.assertTrue(mode.cut_parent(3))  # valid root idempotent
            self.assertEqual(mode.parents, before)

            for child, parent in ((1, 1), (0, 2), (-1, 0), (1, 99)):
                self.assertFalse(mode.set_parent(child, parent))
                self.assertEqual(mode.parents, before)
            self.assertFalse(mode.cut_parent(99))
            self.assertEqual(mode.parents, before)

            self.assertFalse(mode.apply(ParentMutation(SET_PARENT, 1)))
            self.assertFalse(mode.apply(ParentMutation(CUT_PARENT, 1, 0)))
            self.assertFalse(mode.apply(ParentMutation("UNKNOWN", 1)))
            self.assertEqual(mode.parents, before)

    def test_deterministic_trace_stays_valid_and_keeps_modes_aligned(self) -> None:
        parents = generate_valid_forest(24, 1234)
        trace = generate_mutation_trace(parents, 128, 5678)
        self.assertEqual(trace, generate_mutation_trace(parents, 128, 5678))

        exact_modes = [mode_type(parents) for mode_type in EXACT_TYPES]
        bloom = AncestorBloom64Rebuild(parents)
        all_modes = [*exact_modes, bloom]
        for mutation in trace:
            for mode in all_modes:
                self.assertTrue(mode.apply(mutation))
            validate_forest(exact_modes[0].parents)
            self.assertEqual(
                {mode.parents for mode in all_modes},
                {exact_modes[0].parents},
            )

        self.assert_exact_modes_agree(exact_modes)
        self.assert_bloom_has_no_false_negatives(exact_modes[0], bloom)

    def test_invalid_forests_are_rejected(self) -> None:
        for parents in ((ROOT, 2), (0,), (1, 0), (ROOT, -1)):
            with self.assertRaises(ValueError):
                validate_forest(parents)

    def test_splitmix64_frozen_vectors(self) -> None:
        self.assertEqual(splitmix64(0), 0xE220A8397B1DCDAF)
        self.assertEqual(splitmix64(1), 0x910A2DEC89025CC1)


class AncestryAccountingTests(unittest.TestCase):
    def test_frozen_allocation_table(self) -> None:
        self.assertEqual(
            allocation_table(),
            {
                64: {
                    "parent_recompute": 128,
                    "exact_interval_rebuild": 384,
                    "materialized_closure_bitset": 640,
                    "ancestor_bloom64_rebuild": 672,
                    "continuous_ancestor_bf16x16": 2176,
                },
                256: {
                    "parent_recompute": 512,
                    "exact_interval_rebuild": 1536,
                    "materialized_closure_bitset": 8704,
                    "ancestor_bloom64_rebuild": 2592,
                    "continuous_ancestor_bf16x16": 8704,
                },
                1024: {
                    "parent_recompute": 2048,
                    "exact_interval_rebuild": 6144,
                    "materialized_closure_bitset": 133120,
                    "ancestor_bloom64_rebuild": 10272,
                    "continuous_ancestor_bf16x16": 34816,
                },
            },
        )

    def test_allocation_formulas_and_payload_accounting(self) -> None:
        for node_count in (64, 256, 1024):
            allocated = batch_one_allocated_bytes(node_count)
            self.assertEqual(allocated["parent_recompute"], 2 * node_count)
            self.assertEqual(allocated["exact_interval_rebuild"], 6 * node_count)
            self.assertEqual(
                allocated["materialized_closure_bitset"],
                2 * node_count + 8 * node_count * ((node_count + 63) // 64),
            )
            self.assertEqual(
                allocated["ancestor_bloom64_rebuild"], 10 * node_count + 32
            )
            self.assertEqual(
                allocated["continuous_ancestor_bf16x16"], 34 * node_count
            )

        self.assertAlmostEqual(forest_entropy_bits(64) / 8, 47.4261465)
        self.assertEqual(packed_direct_sentinel_parent_bytes(64), 56)
        self.assertEqual(packed_direct_sentinel_parent_bytes(256), 288)
        self.assertEqual(packed_direct_sentinel_parent_bytes(1024), 1408)


if __name__ == "__main__":
    unittest.main()
