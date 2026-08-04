"""Independent immutable gold interpreter for the permutation-graph workload.

This module deliberately contains its own state validation and transition
logic.  It must not import transition code from the tested Stage 0 executor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence


VALUE_MODULUS = 256
PATH_AGGREGATE_MODULUS = 257


def _is_plain_int(value: object) -> bool:
    return type(value) is int


def is_involution(relation: Sequence[int]) -> bool:
    """Return whether ``relation`` is an in-range involution."""

    size = len(relation)
    for node, target in enumerate(relation):
        if not _is_plain_int(target) or not 0 <= target < size:
            return False
        if relation[target] != node:
            return False
    return True


def is_parent_forest(parents: Sequence[int | None]) -> bool:
    """Return whether parent pointers describe an acyclic rooted forest."""

    size = len(parents)
    for parent in parents:
        if parent is not None and (
            not _is_plain_int(parent) or not 0 <= parent < size
        ):
            return False

    for start in range(size):
        seen: set[int] = set()
        cursor: int | None = start
        while cursor is not None:
            if cursor in seen:
                return False
            seen.add(cursor)
            cursor = parents[cursor]
    return True


@dataclass(frozen=True)
class Command:
    opcode: str
    args: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))


@dataclass(frozen=True)
class ReferenceState:
    """Complete logical state of one graph world.

    ``None`` is the logical root marker in ``parents``.  All containers are
    normalized to tuples so a command can only produce a new state.
    """

    values: tuple[int, ...]
    occupied: tuple[bool, ...]
    conditions: tuple[bool, ...]
    relations: tuple[tuple[int, ...], tuple[int, ...]]
    parents: tuple[int | None, ...]
    status: int = 0

    def __post_init__(self) -> None:
        values = tuple(self.values)
        occupied = tuple(self.occupied)
        conditions = tuple(self.conditions)
        relations = tuple(tuple(relation) for relation in self.relations)
        parents = tuple(self.parents)

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "parents", parents)

        size = len(values)
        if size < 1:
            raise ValueError("state must contain at least one node")
        if not (
            len(occupied) == len(conditions) == len(parents) == size
            and len(relations) == 2
            and all(len(relation) == size for relation in relations)
        ):
            raise ValueError("all node arrays must have equal length and two relations")
        if any(not _is_plain_int(value) or not 0 <= value < VALUE_MODULUS for value in values):
            raise ValueError("values must be integers in 0..255")
        if any(type(flag) is not bool for flag in occupied + conditions):
            raise ValueError("occupancy and condition entries must be booleans")
        if not all(is_involution(relation) for relation in relations):
            raise ValueError("each relation must be an involution")
        if not is_parent_forest(parents):
            raise ValueError("parents must form a rooted forest")
        if not _is_plain_int(self.status) or self.status not in (0, 1):
            raise ValueError("status must be 0 or 1")

    @classmethod
    def empty(
        cls,
        node_count: int,
        relations: tuple[Sequence[int], Sequence[int]] | None = None,
    ) -> "ReferenceState":
        if not _is_plain_int(node_count) or node_count < 1:
            raise ValueError("node_count must be a positive integer")
        if relations is None:
            identity = tuple(range(node_count))
            frozen_relations = (identity, identity)
        else:
            frozen_relations = (tuple(relations[0]), tuple(relations[1]))
        return cls(
            values=(0,) * node_count,
            occupied=(False,) * node_count,
            conditions=(False,) * node_count,
            relations=frozen_relations,
            parents=(None,) * node_count,
        )

    @property
    def node_count(self) -> int:
        return len(self.values)

    @property
    def n(self) -> int:
        return self.node_count

    def snapshot(self) -> tuple[object, ...]:
        return state_snapshot(self)


# A neutral alias makes the reference convenient in small standalone probes.
GraphState = ReferenceState


def state_snapshot(state: ReferenceState) -> tuple[object, ...]:
    """Return a canonical exact snapshot, including the rejection bit."""

    return (
        state.values,
        state.occupied,
        state.conditions,
        state.relations[0],
        state.relations[1],
        state.parents,
        state.status,
    )


def _valid_node(state: ReferenceState, node: object) -> bool:
    return _is_plain_int(node) and 0 <= node < state.node_count


def _reject(state: ReferenceState) -> ReferenceState:
    return replace(state, status=1)


def _accept_without_task_change(state: ReferenceState) -> ReferenceState:
    return replace(state, status=0)


def _tuple_with(items: Sequence[object], index: int, value: object) -> tuple[object, ...]:
    changed = list(items)
    changed[index] = value
    return tuple(changed)


def apply_command(state: ReferenceState, command: Command) -> ReferenceState:
    """Apply one command and return a new complete state.

    A rejected command sets ``status`` to one and changes no other field.  A
    valid command, including an idempotent one, resets ``status`` to zero.
    """

    if not isinstance(command, Command):
        return _reject(state)

    opcode = command.opcode
    args = command.args

    if opcode == "SET" and len(args) == 2:
        node, value = args
        if not _valid_node(state, node) or not _is_plain_int(value) or not 0 <= value < VALUE_MODULUS:
            return _reject(state)
        values = _tuple_with(state.values, node, value)
        occupied = _tuple_with(state.occupied, node, True)
        return replace(state, values=values, occupied=occupied, status=0)

    if opcode == "SWAP_VALUE" and len(args) == 2:
        left, right = args
        if not _valid_node(state, left) or not _valid_node(state, right):
            return _reject(state)
        if left == right:
            return _accept_without_task_change(state)
        values = list(state.values)
        occupied = list(state.occupied)
        values[left], values[right] = values[right], values[left]
        occupied[left], occupied[right] = occupied[right], occupied[left]
        return replace(
            state, values=tuple(values), occupied=tuple(occupied), status=0
        )

    if opcode == "MOVE_VALUE" and len(args) == 2:
        source, destination = args
        if not _valid_node(state, source) or not _valid_node(state, destination):
            return _reject(state)
        if source == destination:
            return _accept_without_task_change(state)
        values = list(state.values)
        occupied = list(state.occupied)
        values[destination] = values[source]
        occupied[destination] = occupied[source]
        values[source] = 0
        occupied[source] = False
        return replace(
            state, values=tuple(values), occupied=tuple(occupied), status=0
        )

    if opcode == "TOGGLE" and len(args) == 1:
        (node,) = args
        if not _valid_node(state, node):
            return _reject(state)
        conditions = _tuple_with(state.conditions, node, not state.conditions[node])
        return replace(state, conditions=conditions, status=0)

    if opcode == "TWO_SWITCH" and len(args) == 5:
        relation_index, a, b, c, d = args
        nodes = (a, b, c, d)
        if (
            not _is_plain_int(relation_index)
            or relation_index not in (0, 1)
            or not all(_valid_node(state, node) for node in nodes)
            or len(set(nodes)) != 4
        ):
            return _reject(state)
        relation = state.relations[relation_index]
        if not (
            relation[a] == b
            and relation[b] == a
            and relation[c] == d
            and relation[d] == c
        ):
            return _reject(state)
        switched = list(relation)
        switched[a], switched[c] = c, a
        switched[b], switched[d] = d, b
        relations = list(state.relations)
        relations[relation_index] = tuple(switched)
        return replace(state, relations=tuple(relations), status=0)

    if opcode == "SET_PARENT" and len(args) == 2:
        child, parent = args
        if (
            not _valid_node(state, child)
            or not _valid_node(state, parent)
            or child == parent
        ):
            return _reject(state)
        if state.parents[child] == parent:
            return _accept_without_task_change(state)
        cursor: int | None = parent
        while cursor is not None:
            if cursor == child:
                return _reject(state)
            cursor = state.parents[cursor]
        parents = _tuple_with(state.parents, child, parent)
        return replace(state, parents=parents, status=0)

    if opcode == "CUT_PARENT" and len(args) == 1:
        (child,) = args
        if not _valid_node(state, child):
            return _reject(state)
        parents = _tuple_with(state.parents, child, None)
        return replace(state, parents=parents, status=0)

    return _reject(state)


def run_trajectory(
    initial: ReferenceState,
    commands: Iterable[Command],
    *,
    include_initial: bool = True,
) -> tuple[ReferenceState, ...]:
    """Return the exact immutable state trajectory for ``commands``."""

    current = initial
    trajectory = [current] if include_initial else []
    for command in commands:
        current = apply_command(current, command)
        trajectory.append(current)
    return tuple(trajectory)


def _validated_reduced_word(relation_word: Iterable[int]) -> tuple[int, ...]:
    try:
        word = tuple(relation_word)
    except TypeError as error:
        raise ValueError("relation word must be iterable") from error
    previous: int | None = None
    for label in word:
        if not _is_plain_int(label) or label not in (0, 1):
            raise ValueError("relation labels must be 0 or 1")
        if previous == label:
            raise ValueError("relation word must be reduced")
        previous = label
    return word


def is_reduced_relation_word(relation_word: Iterable[int]) -> bool:
    try:
        _validated_reduced_word(relation_word)
    except ValueError:
        return False
    return True


def path_nodes(
    state: ReferenceState, start: int, relation_word: Iterable[int]
) -> tuple[int, ...]:
    """Return the start node followed by every node visited by a reduced word."""

    if not _valid_node(state, start):
        raise ValueError("query node out of range")
    word = _validated_reduced_word(relation_word)
    cursor = start
    visited = [cursor]
    for relation_index in word:
        cursor = state.relations[relation_index][cursor]
        visited.append(cursor)
    return tuple(visited)


def hop_traversal(
    state: ReferenceState, start: int, relation_word: Iterable[int]
) -> int:
    return path_nodes(state, start, relation_word)[-1]


def strict_ancestor(state: ReferenceState, ancestor: int, node: int) -> bool:
    if not _valid_node(state, ancestor) or not _valid_node(state, node):
        raise ValueError("query node out of range")
    cursor = state.parents[node]
    while cursor is not None:
        if cursor == ancestor:
            return True
        cursor = state.parents[cursor]
    return False


def path_aggregate_mod257(
    state: ReferenceState, start: int, relation_word: Iterable[int]
) -> int:
    total = sum(
        state.values[node] for node in path_nodes(state, start, relation_word) if state.occupied[node]
    )
    return total % PATH_AGGREGATE_MODULUS


# Familiar query aliases; both retain the reference interpreter's reduced-word
# validation rather than calling into the tested executor.
traverse = hop_traversal
path_aggregate = path_aggregate_mod257

