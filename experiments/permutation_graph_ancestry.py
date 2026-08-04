"""Stage 0 ancestry representations for permutation-graph refinement.

The implementations follow the frozen preregistration: ancestry is strict,
cached modes retain a uint16 parent array, and every accepted state-changing
parent mutation rebuilds its derived representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from random import Random
from typing import Final, Sequence


ROOT: Final = 0xFFFF
WORD_BITS: Final = 64
WORD_MASK: Final = (1 << WORD_BITS) - 1
FROZEN_NODE_COUNTS: Final = (64, 256, 1024)
BLOOM_SEEDS: Final = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
)

SET_PARENT: Final = "SET_PARENT"
CUT_PARENT: Final = "CUT_PARENT"


def _require_node_count(node_count: int) -> None:
    if type(node_count) is not int or not 0 < node_count <= ROOT:
        raise ValueError(f"node_count must be in [1, {ROOT}]")


def _is_node(node: object, node_count: int) -> bool:
    return type(node) is int and 0 <= node < node_count


def validate_forest(parents: Sequence[int]) -> None:
    """Raise ``ValueError`` unless ``parents`` encodes a valid rooted forest."""
    node_count = len(parents)
    _require_node_count(node_count)

    for child, parent in enumerate(parents):
        if parent == ROOT:
            continue
        if not _is_node(parent, node_count):
            raise ValueError(f"parent[{child}]={parent!r} is out of range")
        if parent == child:
            raise ValueError(f"parent[{child}] is a self-link")

    state = [0] * node_count  # 0 unseen, 1 on current path, 2 complete
    for start in range(node_count):
        node = start
        while node != ROOT and state[node] == 0:
            state[node] = 1
            node = parents[node]
        if node != ROOT and state[node] == 1:
            raise ValueError("parent relation contains a cycle")

        node = start
        while node != ROOT and state[node] == 1:
            state[node] = 2
            node = parents[node]


def _normalized_forest(parents: Sequence[int]) -> list[int]:
    normalized = list(parents)
    validate_forest(normalized)
    return normalized


def _require_query_nodes(node_count: int, ancestor: int, node: int) -> None:
    if not _is_node(ancestor, node_count) or not _is_node(node, node_count):
        raise IndexError("ancestor and node must be valid node IDs")


def _strict_ancestor_unchecked(
    parents: Sequence[int], ancestor: int, node: int
) -> bool:
    if ancestor == node:
        return False
    current = parents[node]
    while current != ROOT:
        if current == ancestor:
            return True
        current = parents[current]
    return False


def strict_parent_chase_membership(
    parents: Sequence[int], ancestor: int, node: int
) -> bool:
    """Return whether ``ancestor`` is a strict ancestor of ``node``."""
    normalized = _normalized_forest(parents)
    _require_query_nodes(len(normalized), ancestor, node)
    return _strict_ancestor_unchecked(normalized, ancestor, node)


def splitmix64(value: int) -> int:
    """Return the frozen unsigned SplitMix64 permutation of ``value``."""
    value = (value + 0x9E3779B97F4A7C15) & WORD_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & WORD_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & WORD_MASK
    return (value ^ (value >> 31)) & WORD_MASK


def _bloom_mask(node_id: int) -> int:
    mask = 0
    for seed in BLOOM_SEEDS:
        mask |= 1 << (splitmix64(node_id ^ seed) & 63)
    return mask


def forest_entropy_bits(node_count: int) -> float:
    """Information lower bound for labeled rooted forests on ``node_count`` nodes."""
    _require_node_count(node_count)
    return (node_count - 1) * log2(node_count + 1)


def packed_direct_sentinel_parent_bytes(node_count: int) -> int:
    """Bytes for fixed-width direct IDs plus one distinct root sentinel."""
    _require_node_count(node_count)
    bits_per_parent = ceil(log2(node_count + 1))
    return ceil(node_count * bits_per_parent / 8)


def batch_one_allocated_bytes(node_count: int) -> dict[str, int]:
    """Return frozen tensor-storage allocation formulas for one forest."""
    _require_node_count(node_count)
    parent_bytes = 2 * node_count
    closure_bytes = 8 * node_count * ceil(node_count / WORD_BITS)
    return {
        "parent_recompute": parent_bytes,
        "exact_interval_rebuild": 6 * node_count,
        "materialized_closure_bitset": parent_bytes + closure_bytes,
        "ancestor_bloom64_rebuild": 10 * node_count + 32,
        "continuous_ancestor_bf16x16": 34 * node_count,
    }


def allocation_table(
    node_counts: Sequence[int] = FROZEN_NODE_COUNTS,
) -> dict[int, dict[str, int]]:
    """Return the batch-one allocation table for the requested node counts."""
    return {
        node_count: batch_one_allocated_bytes(node_count)
        for node_count in node_counts
    }


FROZEN_ALLOCATION_TABLE: Final = allocation_table()


@dataclass(frozen=True, slots=True)
class ParentMutation:
    opcode: str
    child: int
    parent: int | None = None


def _can_set_parent(parents: Sequence[int], child: int, parent: int) -> bool:
    node_count = len(parents)
    if not _is_node(child, node_count) or not _is_node(parent, node_count):
        return False
    if child == parent:
        return False
    if parents[child] == parent:
        return True
    return not _strict_ancestor_unchecked(parents, child, parent)


def _set_parent_in_place(
    parents: list[int], child: int, parent: int
) -> tuple[bool, bool]:
    if not _can_set_parent(parents, child, parent):
        return False, False
    if parents[child] == parent:
        return True, False
    parents[child] = parent
    return True, True


def _cut_parent_in_place(parents: list[int], child: int) -> tuple[bool, bool]:
    if not _is_node(child, len(parents)):
        return False, False
    if parents[child] == ROOT:
        return True, False
    parents[child] = ROOT
    return True, True


class _MutableAncestry:
    mode_name = ""

    def __init__(self, parents: Sequence[int]) -> None:
        self._parents = _normalized_forest(parents)

    @property
    def parents(self) -> tuple[int, ...]:
        return tuple(self._parents)

    @property
    def node_count(self) -> int:
        return len(self._parents)

    @property
    def allocated_bytes(self) -> int:
        return batch_one_allocated_bytes(self.node_count)[self.mode_name]

    def _rebuild(self) -> None:
        pass

    def set_parent(self, child: int, parent: int) -> bool:
        accepted, changed = _set_parent_in_place(self._parents, child, parent)
        if changed:
            self._rebuild()
        return accepted

    def cut_parent(self, child: int) -> bool:
        accepted, changed = _cut_parent_in_place(self._parents, child)
        if changed:
            self._rebuild()
        return accepted

    def apply(self, mutation: ParentMutation) -> bool:
        if mutation.opcode == SET_PARENT:
            if mutation.parent is None:
                return False
            return self.set_parent(mutation.child, mutation.parent)
        if mutation.opcode == CUT_PARENT:
            if mutation.parent is not None:
                return False
            return self.cut_parent(mutation.child)
        return False

    def is_ancestor(self, ancestor: int, node: int) -> bool:
        raise NotImplementedError


class ParentRecompute(_MutableAncestry):
    """Exact strict ancestry by chasing the frozen uint16 parent array."""

    mode_name = "parent_recompute"

    def is_ancestor(self, ancestor: int, node: int) -> bool:
        _require_query_nodes(self.node_count, ancestor, node)
        return _strict_ancestor_unchecked(self._parents, ancestor, node)


class ExactIntervalRebuild(_MutableAncestry):
    """Exact strict ancestry using rebuilt preorder subtree intervals."""

    mode_name = "exact_interval_rebuild"

    def __init__(self, parents: Sequence[int]) -> None:
        super().__init__(parents)
        self.entry: tuple[int, ...] = ()
        self.exit: tuple[int, ...] = ()
        self._rebuild()

    def _rebuild(self) -> None:
        children = [[] for _ in self._parents]
        roots: list[int] = []
        for child, parent in enumerate(self._parents):
            if parent == ROOT:
                roots.append(child)
            else:
                children[parent].append(child)

        entry = [0] * self.node_count
        exit_ = [0] * self.node_count
        clock = 0
        for root in roots:
            stack = [(root, False)]
            while stack:
                node, leaving = stack.pop()
                if leaving:
                    exit_[node] = clock
                    continue
                entry[node] = clock
                clock += 1
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(children[node]))

        if clock != self.node_count:
            raise AssertionError("validated forest traversal did not visit every node")
        self.entry = tuple(entry)
        self.exit = tuple(exit_)

    def is_ancestor(self, ancestor: int, node: int) -> bool:
        _require_query_nodes(self.node_count, ancestor, node)
        return (
            ancestor != node
            and self.entry[ancestor] <= self.entry[node] < self.exit[ancestor]
        )


class MaterializedClosureBitset(_MutableAncestry):
    """Exact strict ancestry stored as packed uint64 closure rows."""

    mode_name = "materialized_closure_bitset"

    def __init__(self, parents: Sequence[int]) -> None:
        super().__init__(parents)
        self.words_per_row = ceil(self.node_count / WORD_BITS)
        self.rows: tuple[tuple[int, ...], ...] = ()
        self._rebuild()

    def _rebuild(self) -> None:
        rows: list[tuple[int, ...]] = []
        for node in range(self.node_count):
            words = [0] * self.words_per_row
            current = self._parents[node]
            while current != ROOT:
                words[current // WORD_BITS] |= 1 << (current % WORD_BITS)
                current = self._parents[current]
            rows.append(tuple(words))
        self.rows = tuple(rows)

    def is_ancestor(self, ancestor: int, node: int) -> bool:
        _require_query_nodes(self.node_count, ancestor, node)
        if ancestor == node:
            return False
        word = self.rows[node][ancestor // WORD_BITS]
        return bool(word & (1 << (ancestor % WORD_BITS)))


class AncestorBloom64Rebuild(_MutableAncestry):
    """Frozen four-hash, 64-bit approximate strict-ancestry cache."""

    mode_name = "ancestor_bloom64_rebuild"

    def __init__(self, parents: Sequence[int]) -> None:
        super().__init__(parents)
        self.rows: tuple[int, ...] = ()
        self._rebuild()

    def _rebuild(self) -> None:
        rows: list[int] = []
        for node in range(self.node_count):
            signature = 0
            current = self._parents[node]
            while current != ROOT:
                signature |= _bloom_mask(current)
                current = self._parents[current]
            rows.append(signature)
        self.rows = tuple(rows)

    def is_ancestor(self, ancestor: int, node: int) -> bool:
        _require_query_nodes(self.node_count, ancestor, node)
        if ancestor == node:
            return False
        mask = _bloom_mask(ancestor)
        return self.rows[node] & mask == mask


def generate_valid_forest(node_count: int, seed: int) -> tuple[int, ...]:
    """Generate a deterministic randomized forest without rejection sampling."""
    _require_node_count(node_count)
    rng = Random(seed)
    order = list(range(node_count))
    rng.shuffle(order)
    parents = [ROOT] * node_count
    for position, node in enumerate(order[1:], start=1):
        if rng.randrange(4) != 0:
            parents[node] = order[rng.randrange(position)]
    return tuple(parents)


def _choose_reparent(
    parents: Sequence[int], rng: Random
) -> tuple[int, int] | None:
    node_count = len(parents)
    if node_count < 2:
        return None
    for _ in range(16 + 8 * node_count):
        child = rng.randrange(node_count)
        parent = rng.randrange(node_count)
        if parents[child] != parent and _can_set_parent(parents, child, parent):
            return child, parent
    for child in range(node_count):
        for parent in range(node_count):
            if parents[child] != parent and _can_set_parent(parents, child, parent):
                return child, parent
    for child, parent in enumerate(parents):
        if parent != ROOT:
            return child, parent
    return None


def _choose_leaf_attachment(
    parents: Sequence[int], rng: Random
) -> tuple[int, int] | None:
    if len(parents) < 2:
        return None
    child_counts = [0] * len(parents)
    for parent in parents:
        if parent != ROOT:
            child_counts[parent] += 1
    candidates = [
        node
        for node, parent in enumerate(parents)
        if parent == ROOT and child_counts[node] == 0
    ]
    if not candidates:
        return None
    child = candidates[rng.randrange(len(candidates))]
    parent = rng.randrange(len(parents) - 1)
    if parent >= child:
        parent += 1
    return child, parent


def generate_mutation_trace(
    initial_parents: Sequence[int], length: int, seed: int
) -> tuple[ParentMutation, ...]:
    """Generate a deterministic accepted trace while preserving a valid forest."""
    if type(length) is not int or length < 0:
        raise ValueError("length must be a nonnegative integer")
    parents = _normalized_forest(initial_parents)
    rng = Random(seed)
    trace: list[ParentMutation] = []

    for index in range(length):
        lane = index % 4
        if lane == 0:
            mutation = ParentMutation(CUT_PARENT, rng.randrange(len(parents)))
        else:
            pair = (
                _choose_leaf_attachment(parents, rng)
                if lane == 3
                else _choose_reparent(parents, rng)
            )
            if pair is None:
                mutation = ParentMutation(CUT_PARENT, rng.randrange(len(parents)))
            else:
                mutation = ParentMutation(SET_PARENT, pair[0], pair[1])

        if mutation.opcode == CUT_PARENT:
            accepted, _ = _cut_parent_in_place(parents, mutation.child)
        else:
            assert mutation.parent is not None
            accepted, _ = _set_parent_in_place(
                parents, mutation.child, mutation.parent
            )
        if not accepted:
            raise AssertionError("trace generator emitted a rejected mutation")
        trace.append(mutation)

    validate_forest(parents)
    return tuple(trace)
