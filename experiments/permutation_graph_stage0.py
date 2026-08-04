"""Stage 0 tested executor for the permutation-graph experiment.

This module intentionally does not import the independent gold interpreter.
Cross-implementation agreement belongs in the integration harness.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


ROOT = 65_535
ARENA_BYTES = 4_096
ARENA_LAYOUT = {
    "X": (0, 2_048),
    "Z": (2_048, 128),
    "pi_0": (2_176, 128),
    "pi_1": (2_304, 128),
    "parent": (2_432, 128),
    "controller": (2_560, 64),
    "status": (2_624, 2),
    "masked_padding": (2_626, 1_470),
}


@dataclass(frozen=True)
class Command:
    opcode: str
    args: tuple[int, ...] = ()


@dataclass
class TypedOps:
    integer_arithmetic: int = 0
    comparison: int = 0
    gather: int = 0
    scatter: int = 0
    branch: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "integer_arithmetic": self.integer_arithmetic,
            "comparison": self.comparison,
            "gather": self.gather,
            "scatter": self.scatter,
            "branch": self.branch,
        }


@dataclass
class GraphState:
    values: list[int]
    occupied: list[bool]
    conditions: list[bool]
    relations: tuple[list[int], list[int]]
    parents: list[int]
    status: int = 0
    ops: TypedOps = field(default_factory=TypedOps, compare=False)

    @property
    def n(self) -> int:
        return len(self.values)

    def clone(self) -> "GraphState":
        return GraphState(
            self.values.copy(),
            self.occupied.copy(),
            self.conditions.copy(),
            (self.relations[0].copy(), self.relations[1].copy()),
            self.parents.copy(),
            self.status,
            TypedOps(**self.ops.as_dict()),
        )

    def task_snapshot(self) -> tuple[object, ...]:
        z = tuple(
            value | (int(occupied) << 8) | (int(condition) << 9)
            for value, occupied, condition in zip(
                self.values, self.occupied, self.conditions, strict=True
            )
        )
        return (
            z,
            tuple(self.relations[0]),
            tuple(self.relations[1]),
            tuple(self.parents),
            self.status,
        )


def validate_arena_layout() -> None:
    cursor = 0
    for offset, size in ARENA_LAYOUT.values():
        if offset != cursor or size < 0:
            raise ValueError("arena views must be contiguous and nonnegative")
        cursor += size
    if cursor != ARENA_BYTES:
        raise ValueError(f"arena ends at {cursor}, expected {ARENA_BYTES}")


def _valid_node(state: GraphState, node: object) -> bool:
    state.ops.comparison += 2
    return isinstance(node, int) and not isinstance(node, bool) and 0 <= node < state.n


def _reject(state: GraphState) -> bool:
    state.status = 1
    state.ops.scatter += 1
    return False


def _valid_involution(relation: Sequence[int]) -> bool:
    n = len(relation)
    return all(0 <= target < n and relation[target] == node for node, target in enumerate(relation))


def _valid_forest(parents: Sequence[int]) -> bool:
    n = len(parents)
    for start in range(n):
        seen: set[int] = set()
        node = start
        while node != ROOT:
            if not 0 <= node < n or node in seen:
                return False
            seen.add(node)
            node = parents[node]
    return True


def validate_state(state: GraphState) -> None:
    n = state.n
    if not (
        len(state.occupied)
        == len(state.conditions)
        == len(state.parents)
        == len(state.relations[0])
        == len(state.relations[1])
        == n
    ):
        raise ValueError("all state arrays must have length N")
    if n < 1 or n > 64:
        raise ValueError("tested executor supports 1 <= N <= 64")
    if any(not 0 <= value <= 255 for value in state.values):
        raise ValueError("values must be bytes")
    if state.status not in (0, 1):
        raise ValueError("status must be one bit")
    if not all(_valid_involution(relation) for relation in state.relations):
        raise ValueError("relations must be involutions")
    if not _valid_forest(state.parents):
        raise ValueError("parents must form a rooted forest")


def _would_create_cycle(state: GraphState, child: int, parent: int) -> bool:
    node = parent
    while node != ROOT:
        state.ops.comparison += 2
        state.ops.branch += 1
        if node == child:
            return True
        state.ops.gather += 1
        node = state.parents[node]
    return False


def apply_command(state: GraphState, command: Command) -> bool:
    """Apply one command; return whether it was accepted.

    Status is the last-command rejection bit. It resets before validation, and
    a rejected command changes no other task state.
    """

    state.status = 0
    state.ops.scatter += 1
    op = command.opcode
    args = command.args

    if op == "SET" and len(args) == 2:
        node, value = args
        if not _valid_node(state, node) or not isinstance(value, int) or isinstance(value, bool):
            return _reject(state)
        state.ops.comparison += 2
        if not 0 <= value <= 255:
            return _reject(state)
        state.values[node] = value
        state.occupied[node] = True
        state.ops.scatter += 2
        return True

    if op == "SWAP_VALUE" and len(args) == 2:
        left, right = args
        if not _valid_node(state, left) or not _valid_node(state, right):
            return _reject(state)
        state.ops.gather += 4
        state.values[left], state.values[right] = state.values[right], state.values[left]
        state.occupied[left], state.occupied[right] = (
            state.occupied[right],
            state.occupied[left],
        )
        state.ops.scatter += 4
        return True

    if op == "MOVE_VALUE" and len(args) == 2:
        source, destination = args
        if not _valid_node(state, source) or not _valid_node(state, destination):
            return _reject(state)
        if source == destination:
            state.ops.comparison += 1
            return True
        state.ops.gather += 2
        state.values[destination] = state.values[source]
        state.occupied[destination] = state.occupied[source]
        state.values[source] = 0
        state.occupied[source] = False
        state.ops.scatter += 4
        return True

    if op == "TOGGLE" and len(args) == 1:
        (node,) = args
        if not _valid_node(state, node):
            return _reject(state)
        state.ops.gather += 1
        state.conditions[node] = not state.conditions[node]
        state.ops.scatter += 1
        return True

    if op == "TWO_SWITCH" and len(args) == 5:
        relation_index, a, b, c, d = args
        if relation_index not in (0, 1):
            state.ops.comparison += 1
            return _reject(state)
        nodes = (a, b, c, d)
        if not all(_valid_node(state, node) for node in nodes) or len(set(nodes)) != 4:
            state.ops.comparison += 1
            return _reject(state)
        relation = state.relations[relation_index]
        state.ops.gather += 2
        if relation[a] != b or relation[c] != d:
            state.ops.comparison += 2
            return _reject(state)
        relation[a], relation[c] = c, a
        relation[b], relation[d] = d, b
        state.ops.scatter += 4
        return True

    if op == "SET_PARENT" and len(args) == 2:
        child, parent = args
        if not _valid_node(state, child) or not _valid_node(state, parent) or child == parent:
            state.ops.comparison += 1
            return _reject(state)
        if _would_create_cycle(state, child, parent):
            return _reject(state)
        state.parents[child] = parent
        state.ops.scatter += 1
        return True

    if op == "CUT_PARENT" and len(args) == 1:
        (child,) = args
        if not _valid_node(state, child):
            return _reject(state)
        state.parents[child] = ROOT
        state.ops.scatter += 1
        return True

    return _reject(state)


def strict_ancestor(state: GraphState, ancestor: int, node: int) -> bool:
    if not _valid_node(state, ancestor) or not _valid_node(state, node):
        raise ValueError("query node out of range")
    cursor = state.parents[node]
    state.ops.gather += 1
    while cursor != ROOT:
        state.ops.comparison += 2
        if cursor == ancestor:
            return True
        cursor = state.parents[cursor]
        state.ops.gather += 1
    return False


def traverse(state: GraphState, start: int, relation_word: Sequence[int]) -> int:
    if not _valid_node(state, start):
        raise ValueError("query node out of range")
    node = start
    for relation_index in relation_word:
        if relation_index not in (0, 1):
            raise ValueError("relation labels must be 0 or 1")
        node = state.relations[relation_index][node]
        state.ops.gather += 1
    return node


def path_aggregate(state: GraphState, start: int, relation_word: Sequence[int]) -> int:
    if not _valid_node(state, start):
        raise ValueError("query node out of range")
    node = start
    total = state.values[node] if state.occupied[node] else 0
    state.ops.gather += 2
    for relation_index in relation_word:
        if relation_index not in (0, 1):
            raise ValueError("relation labels must be 0 or 1")
        node = state.relations[relation_index][node]
        total = (total + (state.values[node] if state.occupied[node] else 0)) % 257
        state.ops.gather += 3
        state.ops.integer_arithmetic += 2
    return total


def to_arena(state: GraphState) -> bytes:
    if state.n != 64:
        raise ValueError("the frozen arena layout requires N=64")
    validate_state(state)
    validate_arena_layout()
    arena = bytearray(ARENA_BYTES)
    for node, (value, occupied, condition) in enumerate(
        zip(state.values, state.occupied, state.conditions, strict=True)
    ):
        encoded = value | (int(occupied) << 8) | (int(condition) << 9)
        offset = ARENA_LAYOUT["Z"][0] + 2 * node
        arena[offset : offset + 2] = encoded.to_bytes(2, "little")
    for name, relation in zip(("pi_0", "pi_1"), state.relations, strict=True):
        base = ARENA_LAYOUT[name][0]
        for node, target in enumerate(relation):
            arena[base + 2 * node : base + 2 * node + 2] = target.to_bytes(2, "little")
    parent_base = ARENA_LAYOUT["parent"][0]
    for node, parent in enumerate(state.parents):
        arena[parent_base + 2 * node : parent_base + 2 * node + 2] = parent.to_bytes(2, "little")
    status_base = ARENA_LAYOUT["status"][0]
    arena[status_base : status_base + 2] = state.status.to_bytes(2, "little")
    return bytes(arena)


def hamiltonian_matchings(n: int, rng: random.Random) -> tuple[list[int], list[int]]:
    if n < 2 or n % 2:
        raise ValueError("Hamiltonian matching pair requires positive even N")
    order = list(range(n))
    rng.shuffle(order)
    first = list(range(n))
    second = list(range(n))
    for index in range(0, n, 2):
        a, b = order[index], order[index + 1]
        first[a], first[b] = b, a
    for index in range(1, n, 2):
        a, b = order[index], order[(index + 1) % n]
        second[a], second[b] = b, a
    return first, second


def random_forest(n: int, rng: random.Random) -> list[int]:
    order = list(range(n))
    rng.shuffle(order)
    parents = [ROOT] * n
    for index, node in enumerate(order[1:], start=1):
        parents[node] = ROOT if rng.random() < 0.2 else rng.choice(order[:index])
    return parents


def initial_state(n: int, seed: int) -> GraphState:
    rng = random.Random(seed)
    if n % 2:
        relations = (list(range(n)), list(range(n)))
        nodes = list(range(n - 1))
        rng.shuffle(nodes)
        for relation in relations:
            for index in range(0, len(nodes), 2):
                a, b = nodes[index], nodes[index + 1]
                relation[a], relation[b] = b, a
    else:
        relations = hamiltonian_matchings(n, rng)
    state = GraphState(
        values=[rng.randrange(256) for _ in range(n)],
        occupied=[bool(rng.randrange(2)) for _ in range(n)],
        conditions=[bool(rng.randrange(2)) for _ in range(n)],
        relations=relations,
        parents=random_forest(n, rng),
    )
    validate_state(state)
    return state


def command_trace(n: int, length: int, seed: int) -> list[Command]:
    """Generate commands without consulting either transition implementation."""

    rng = random.Random(seed)
    opcodes = (
        "SET",
        "SWAP_VALUE",
        "MOVE_VALUE",
        "TOGGLE",
        "TWO_SWITCH",
        "SET_PARENT",
        "CUT_PARENT",
    )
    trace: list[Command] = []
    for step in range(length):
        opcode = opcodes[step % len(opcodes)]
        invalid = step % 11 == 10
        node = n + 1 if invalid else rng.randrange(n)
        if opcode == "SET":
            trace.append(Command(opcode, (node, 256 if invalid else rng.randrange(256))))
        elif opcode in ("SWAP_VALUE", "MOVE_VALUE", "SET_PARENT"):
            trace.append(Command(opcode, (node, rng.randrange(n))))
        elif opcode in ("TOGGLE", "CUT_PARENT"):
            trace.append(Command(opcode, (node,)))
        else:
            trace.append(
                Command(
                    opcode,
                    (rng.randrange(2), node, rng.randrange(n), rng.randrange(n), rng.randrange(n)),
                )
            )
    return trace


def alternating_word(length: int, first_relation: int = 0) -> tuple[int, ...]:
    if length < 0 or first_relation not in (0, 1):
        raise ValueError("invalid reduced relation word")
    return tuple((first_relation + offset) % 2 for offset in range(length))


def run_trace(state: GraphState, commands: Iterable[Command]) -> list[tuple[object, ...]]:
    trajectory: list[tuple[object, ...]] = []
    for command in commands:
        apply_command(state, command)
        validate_state(state)
        trajectory.append(state.task_snapshot())
    return trajectory


def write_stage0_executor_report(path: Path, *, seed: int = 17) -> dict[str, object]:
    state = initial_state(64, seed)
    commands = command_trace(64, 1_024, seed + 1)
    trajectory = run_trace(state, commands)
    arena = to_arena(state)
    report = {
        "schema_version": 1,
        "seed": seed,
        "node_count": 64,
        "commands": len(commands),
        "trajectory_steps": len(trajectory),
        "final_snapshot_sha256": hashlib.sha256(repr(trajectory[-1]).encode()).hexdigest(),
        "final_arena_sha256": hashlib.sha256(arena).hexdigest(),
        "arena_bytes": len(arena),
        "typed_operations": state.ops.as_dict(),
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    write_stage0_executor_report(
        Path(__file__).resolve().parents[1] / "results" / "permutation-graph-stage0-executor.json"
    )
