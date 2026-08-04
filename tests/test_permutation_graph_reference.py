from __future__ import annotations

import itertools
import unittest
from dataclasses import FrozenInstanceError

from experiments.permutation_graph_reference import (
    Command,
    ReferenceState,
    apply_command,
    hop_traversal,
    is_involution,
    is_parent_forest,
    is_reduced_relation_word,
    path_aggregate_mod257,
    path_nodes,
    run_trajectory,
    state_snapshot,
    strict_ancestor,
)


def all_involutions(size: int) -> list[tuple[int, ...]]:
    """Generate involutions independently of the reference implementation."""

    result: list[tuple[int, ...]] = []
    mapping = [-1] * size

    def visit(remaining: tuple[int, ...]) -> None:
        if not remaining:
            result.append(tuple(mapping))
            return
        first = remaining[0]
        mapping[first] = first
        visit(remaining[1:])
        for offset, partner in enumerate(remaining[1:], start=1):
            mapping[first] = partner
            mapping[partner] = first
            visit(remaining[1:offset] + remaining[offset + 1 :])
            mapping[partner] = -1
        mapping[first] = -1

    visit(tuple(range(size)))
    return result


def independently_is_forest(parents: tuple[int | None, ...]) -> bool:
    size = len(parents)
    for start in range(size):
        cursor: int | None = start
        for _ in range(size + 1):
            if cursor is None:
                break
            cursor = parents[cursor]
        else:
            return False
    return True


def rename_state(state: ReferenceState, old_to_new: tuple[int, ...]) -> ReferenceState:
    size = state.node_count
    values = [0] * size
    occupied = [False] * size
    conditions = [False] * size
    relations = [[0] * size for _ in range(2)]
    parents: list[int | None] = [None] * size
    for old in range(size):
        new = old_to_new[old]
        values[new] = state.values[old]
        occupied[new] = state.occupied[old]
        conditions[new] = state.conditions[old]
        for relation_index in range(2):
            relations[relation_index][new] = old_to_new[state.relations[relation_index][old]]
        parent = state.parents[old]
        parents[new] = None if parent is None else old_to_new[parent]
    return ReferenceState(
        tuple(values),
        tuple(occupied),
        tuple(conditions),
        (tuple(relations[0]), tuple(relations[1])),
        tuple(parents),
        state.status,
    )


def rename_command(command: Command, old_to_new: tuple[int, ...]) -> Command:
    opcode = command.opcode
    args = command.args
    if opcode == "SET":
        return Command(opcode, (old_to_new[args[0]], args[1]))
    if opcode in ("SWAP_VALUE", "MOVE_VALUE", "SET_PARENT"):
        return Command(opcode, tuple(old_to_new[node] for node in args))
    if opcode in ("TOGGLE", "CUT_PARENT"):
        return Command(opcode, (old_to_new[args[0]],))
    if opcode == "TWO_SWITCH":
        return Command(opcode, (args[0], *(old_to_new[node] for node in args[1:])))
    return command


class ReferenceInterpreterTests(unittest.TestCase):
    def test_value_and_condition_commands_have_frozen_semantics(self) -> None:
        initial = ReferenceState.empty(3)
        set_zero = apply_command(initial, Command("SET", (0, 17)))
        set_one = apply_command(set_zero, Command("SET", (1, 99)))
        toggled = apply_command(set_one, Command("TOGGLE", (1,)))

        swapped = apply_command(toggled, Command("SWAP_VALUE", (0, 1)))
        self.assertEqual(swapped.values, (99, 17, 0))
        self.assertEqual(swapped.occupied, (True, True, False))
        self.assertEqual(swapped.conditions, (False, True, False))

        moved = apply_command(swapped, Command("MOVE_VALUE", (0, 1)))
        self.assertEqual(moved.values, (0, 99, 0))
        self.assertEqual(moved.occupied, (False, True, False))
        self.assertEqual(moved.conditions, swapped.conditions)

        for opcode in ("SWAP_VALUE", "MOVE_VALUE"):
            same_node = apply_command(moved, Command(opcode, (1, 1)))
            self.assertEqual(state_snapshot(same_node)[:-1], state_snapshot(moved)[:-1])
            self.assertEqual(same_node.status, 0)

        self.assertEqual(initial.values, (0, 0, 0))
        with self.assertRaises(FrozenInstanceError):
            initial.status = 1  # type: ignore[misc]

    def test_rejection_changes_only_status_and_valid_command_resets_it(self) -> None:
        state = apply_command(ReferenceState.empty(3), Command("SET", (0, 255)))
        invalid_commands = (
            Command("SET", (0, 256)),
            Command("SET", (True, 1)),
            Command("SWAP_VALUE", (0, 3)),
            Command("MOVE_VALUE", (-1, 0)),
            Command("TOGGLE", ()),
            Command("UNKNOWN", ()),
        )
        for command in invalid_commands:
            rejected = apply_command(state, command)
            self.assertEqual(state_snapshot(rejected)[:-1], state_snapshot(state)[:-1])
            self.assertEqual(rejected.status, 1)

        rejected = apply_command(state, invalid_commands[0])
        accepted = apply_command(rejected, Command("CUT_PARENT", (0,)))
        self.assertEqual(accepted.status, 0)
        self.assertEqual(state_snapshot(accepted)[:-1], state_snapshot(state)[:-1])

    def test_two_switch_is_checked_and_invertible(self) -> None:
        relation = (1, 0, 3, 2, 4)
        state = ReferenceState.empty(5, (relation, tuple(range(5))))
        switched = apply_command(state, Command("TWO_SWITCH", (0, 0, 1, 2, 3)))
        self.assertEqual(switched.relations[0], (2, 3, 0, 1, 4))
        self.assertTrue(is_involution(switched.relations[0]))

        restored = apply_command(switched, Command("TWO_SWITCH", (0, 0, 2, 1, 3)))
        self.assertEqual(restored.relations, state.relations)
        self.assertEqual(restored.status, 0)

        for command in (
            Command("TWO_SWITCH", (2, 0, 1, 2, 3)),
            Command("TWO_SWITCH", (0, 0, 1, 2, 2)),
            Command("TWO_SWITCH", (0, 0, 2, 1, 3)),
            Command("TWO_SWITCH", (0, 0, 1, 2, 4)),
        ):
            rejected = apply_command(state, command)
            self.assertEqual(state_snapshot(rejected)[:-1], state_snapshot(state)[:-1])
            self.assertEqual(rejected.status, 1)

    def test_parent_commands_preserve_a_strict_forest(self) -> None:
        state = ReferenceState.empty(4)
        state = apply_command(state, Command("SET_PARENT", (1, 0)))
        state = apply_command(state, Command("SET_PARENT", (2, 1)))
        state = apply_command(state, Command("SET_PARENT", (3, 2)))
        self.assertEqual(state.parents, (None, 0, 1, 2))
        self.assertTrue(strict_ancestor(state, 0, 3))
        self.assertTrue(strict_ancestor(state, 1, 3))
        self.assertFalse(strict_ancestor(state, 3, 3))

        idempotent = apply_command(state, Command("SET_PARENT", (2, 1)))
        self.assertEqual(idempotent.parents, state.parents)
        self.assertEqual(idempotent.status, 0)

        for command in (
            Command("SET_PARENT", (0, 3)),
            Command("SET_PARENT", (2, 2)),
            Command("SET_PARENT", (4, 0)),
        ):
            rejected = apply_command(state, command)
            self.assertEqual(rejected.parents, state.parents)
            self.assertEqual(rejected.status, 1)

        cut = apply_command(state, Command("CUT_PARENT", (2,)))
        self.assertEqual(cut.parents, (None, 0, None, 2))
        self.assertFalse(strict_ancestor(cut, 0, 3))
        cut_again = apply_command(cut, Command("CUT_PARENT", (2,)))
        self.assertEqual(cut_again.parents, cut.parents)
        self.assertEqual(cut_again.status, 0)

    def test_reduced_hops_and_path_aggregate_include_every_visit(self) -> None:
        state = ReferenceState(
            values=(255, 2, 3, 4, 5, 6),
            occupied=(True, True, True, False, True, True),
            conditions=(False,) * 6,
            relations=((1, 0, 3, 2, 5, 4), (2, 3, 0, 1, 5, 4)),
            parents=(None, 0, 1, None, 3, 4),
        )
        word = (0, 1, 0, 1)
        self.assertEqual(path_nodes(state, 0, word), (0, 1, 3, 2, 0))
        self.assertEqual(hop_traversal(state, 0, word), 0)
        self.assertEqual(path_aggregate_mod257(state, 0, word), (255 + 2 + 3 + 255) % 257)
        self.assertTrue(is_reduced_relation_word(word))
        self.assertFalse(is_reduced_relation_word((0, 0)))

        for bad_word in ((0, 0), (0, 2), (True,)):
            with self.assertRaises(ValueError):
                hop_traversal(state, 0, bad_word)
        with self.assertRaises(ValueError):
            path_aggregate_mod257(state, 6, ())

    def test_exact_trajectory_contains_initial_and_every_post_state(self) -> None:
        initial = ReferenceState.empty(2)
        commands = (
            Command("SET", (0, 1)),
            Command("SET", (1, 255)),
            Command("SWAP_VALUE", (0, 1)),
        )
        trajectory = run_trajectory(initial, commands)
        self.assertEqual(len(trajectory), 4)
        self.assertIs(trajectory[0], initial)
        self.assertEqual(trajectory[-1].values, (255, 1))
        self.assertEqual(
            run_trajectory(initial, commands, include_initial=False), trajectory[1:]
        )

    def test_all_involutions_through_n6_and_every_available_two_switch(self) -> None:
        for size in range(1, 7):
            identity = tuple(range(size))
            for relation in all_involutions(size):
                self.assertTrue(is_involution(relation))
                self.assertEqual(tuple(relation[relation[node]] for node in range(size)), identity)
                state = ReferenceState.empty(size, (relation, identity))
                edges = [(node, target) for node, target in enumerate(relation) if node < target]
                for (a, b), (c, d) in itertools.combinations(edges, 2):
                    switched = apply_command(
                        state, Command("TWO_SWITCH", (0, a, b, c, d))
                    )
                    self.assertEqual(switched.status, 0)
                    self.assertTrue(is_involution(switched.relations[0]))
                    restored = apply_command(
                        switched, Command("TWO_SWITCH", (0, a, c, b, d))
                    )
                    self.assertEqual(restored.relations[0], relation)

    def test_all_parent_assignments_through_n4_are_classified(self) -> None:
        for size in range(1, 5):
            identity = tuple(range(size))
            choices: tuple[int | None, ...] = (None, *range(size))
            for parents in itertools.product(choices, repeat=size):
                expected = independently_is_forest(parents)
                self.assertEqual(is_parent_forest(parents), expected)
                fields = dict(
                    values=(0,) * size,
                    occupied=(False,) * size,
                    conditions=(False,) * size,
                    relations=(identity, identity),
                    parents=parents,
                )
                if expected:
                    self.assertEqual(ReferenceState(**fields).parents, parents)
                else:
                    with self.assertRaises(ValueError):
                        ReferenceState(**fields)

    def test_node_renaming_equivariance_for_commands_and_queries(self) -> None:
        base = ReferenceState(
            values=(10, 20, 30, 40),
            occupied=(True, False, True, True),
            conditions=(False, True, False, True),
            relations=((1, 0, 3, 2), (2, 3, 0, 1)),
            parents=(None, 0, None, 2),
        )
        commands = (
            Command("SET", (1, 255)),
            Command("SWAP_VALUE", (0, 3)),
            Command("MOVE_VALUE", (2, 0)),
            Command("TOGGLE", (3,)),
            Command("TWO_SWITCH", (0, 0, 1, 2, 3)),
            Command("CUT_PARENT", (1,)),
            Command("SET_PARENT", (1, 3)),
            Command("SET_PARENT", (2, 1)),  # rejected cycle
        )
        word = (0, 1, 0)

        for renaming in itertools.permutations(range(base.node_count)):
            original = base
            renamed = rename_state(base, renaming)
            for command in commands:
                original = apply_command(original, command)
                renamed = apply_command(renamed, rename_command(command, renaming))
                self.assertEqual(renamed, rename_state(original, renaming))

            self.assertEqual(
                hop_traversal(renamed, renaming[0], word),
                renaming[hop_traversal(original, 0, word)],
            )
            self.assertEqual(
                path_aggregate_mod257(renamed, renaming[0], word),
                path_aggregate_mod257(original, 0, word),
            )
            self.assertEqual(
                strict_ancestor(renamed, renaming[2], renaming[1]),
                strict_ancestor(original, 2, 1),
            )


if __name__ == "__main__":
    unittest.main()
