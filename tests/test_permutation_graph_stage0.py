from __future__ import annotations

import random
import unittest

from experiments.permutation_graph_stage0 import (
    ARENA_BYTES,
    Command,
    alternating_word,
    apply_command,
    hamiltonian_matchings,
    initial_state,
    path_aggregate,
    strict_ancestor,
    to_arena,
    traverse,
    validate_arena_layout,
    validate_state,
)


class PermutationGraphStage0Tests(unittest.TestCase):
    def test_frozen_arena_is_exact_and_padding_is_zero(self) -> None:
        state = initial_state(64, 17)
        validate_arena_layout()
        arena = to_arena(state)
        self.assertEqual(len(arena), ARENA_BYTES)
        self.assertEqual(arena[2626:], bytes(1470))

    def test_hamiltonian_matchings_give_effective_32_hops(self) -> None:
        relations = hamiltonian_matchings(64, random.Random(29))
        state = initial_state(64, 29)
        state.relations = relations
        word = alternating_word(32)
        visited = [0]
        node = 0
        for relation_index in word:
            node = state.relations[relation_index][node]
            visited.append(node)
        self.assertEqual(len(set(visited)), 33)
        self.assertEqual(traverse(state, 0, word), visited[-1])

    def test_rejected_command_changes_only_status(self) -> None:
        state = initial_state(8, 43)
        before = state.task_snapshot()
        self.assertFalse(apply_command(state, Command("SET", (99, 1))))
        after = state.task_snapshot()
        self.assertEqual(before[:-1], after[:-1])
        self.assertEqual(after[-1], 1)
        self.assertTrue(apply_command(state, Command("TOGGLE", (0,))))
        self.assertEqual(state.status, 0)

    def test_two_switch_preserves_involution_and_is_invertible(self) -> None:
        state = initial_state(8, 17)
        relation = state.relations[0]
        edges = [(i, target) for i, target in enumerate(relation) if i < target]
        (a, b), (c, d) = edges[:2]
        before = state.task_snapshot()
        self.assertTrue(apply_command(state, Command("TWO_SWITCH", (0, a, b, c, d))))
        validate_state(state)
        self.assertTrue(apply_command(state, Command("TWO_SWITCH", (0, a, c, b, d))))
        self.assertEqual(state.task_snapshot()[:-1], before[:-1])

    def test_parent_cycle_is_rejected_and_queries_are_strict(self) -> None:
        state = initial_state(4, 1)
        state.parents = [65535, 0, 1, 2]
        self.assertTrue(strict_ancestor(state, 0, 3))
        self.assertFalse(strict_ancestor(state, 3, 3))
        before = state.task_snapshot()
        self.assertFalse(apply_command(state, Command("SET_PARENT", (0, 3))))
        self.assertEqual(state.task_snapshot()[:-1], before[:-1])

    def test_path_aggregate_includes_start_and_each_visit(self) -> None:
        state = initial_state(4, 3)
        state.values = [1, 2, 3, 4]
        state.occupied = [True] * 4
        state.relations = ([1, 0, 3, 2], [3, 2, 1, 0])
        word = (0, 1)
        self.assertEqual(path_aggregate(state, 0, word), 1 + 2 + 3)


if __name__ == "__main__":
    unittest.main()
