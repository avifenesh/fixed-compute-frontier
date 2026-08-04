#!/usr/bin/env python3
"""Exact CPU reference and frozen data builder for CPKV-TOPK-001.

This file intentionally contains no optimizer or training loop.  It implements
only the pre-training gates authorized in the preregistration:

* two exact finite-beam stack representations;
* causal action-history coverage for S32 -> K3 compilation;
* exhaustive binary-prefix comparison through length six;
* the analytic X4 sensitivity witness; and
* deterministic generation and hashing of fixed data splits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np


N_CONTROL = 3
N_SYMBOL = 3
N_ACTION = 15
S_BEAM = 32
K_BEAM = 3
PAYLOAD_WIDTH = 64

StackItem = tuple[int, int]  # (symbol, payload-source id)
StackSig = tuple[StackItem, ...]
TupleKey = tuple[int, StackSig]
Vector = tuple[float, ...]


@dataclass(frozen=True)
class Action:
    next_q: int
    kind: str
    symbol: int | None = None


@dataclass(frozen=True)
class Node:
    parent: int
    payload_source: int
    symbol: int


class PersistentArena:
    """Canonical persistent nodes; node zero is the root."""

    def __init__(self) -> None:
        self.nodes = [Node(parent=0, payload_source=-1, symbol=-1)]
        self._intern: dict[tuple[int, int, int], int] = {}

    def push(self, parent: int, symbol: int, payload_source: int) -> int:
        key = (parent, symbol, payload_source)
        existing = self._intern.get(key)
        if existing is not None:
            return existing
        index = len(self.nodes)
        self.nodes.append(
            Node(parent=parent, payload_source=payload_source, symbol=symbol)
        )
        self._intern[key] = index
        return index

    def signature(self, node_index: int) -> StackSig:
        items: list[StackItem] = []
        while node_index != 0:
            node = self.nodes[node_index]
            items.append((node.symbol, node.payload_source))
            node_index = node.parent
        items.reverse()
        return tuple(items)

    def materialize(self, signature: StackSig) -> int:
        """Materialize only nodes on a retained stack signature."""

        node_index = 0
        for symbol, payload_source in signature:
            node_index = self.push(node_index, symbol, payload_source)
        return node_index


def action_space() -> tuple[Action, ...]:
    actions: list[Action] = []
    for next_q in range(N_CONTROL):
        for symbol in range(N_SYMBOL):
            actions.append(Action(next_q=next_q, kind="push", symbol=symbol))
        actions.append(Action(next_q=next_q, kind="pop"))
        actions.append(Action(next_q=next_q, kind="noop"))
    assert len(actions) == N_ACTION
    return tuple(actions)


ACTIONS = action_space()


def _softmax(scores: Sequence[float]) -> tuple[float, ...]:
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(exponentials)
    return tuple(value / denominator for value in exponentials)


def action_probabilities(
    *, time_index: int, token: int, q: int, top_symbol: int
) -> tuple[tuple[Action, float], ...]:
    """Deterministic non-degenerate oracle probabilities.

    Root pops are absent before normalization, exactly matching the frozen
    learned operator.  The formula is arbitrary but fixed and deliberately
    separates token, controller, top symbol, time, and action effects.
    """

    valid_actions: list[Action] = []
    scores: list[float] = []
    for action_index, action in enumerate(ACTIONS):
        if top_symbol < 0 and action.kind == "pop":
            continue
        kind_code = {"push": 1, "pop": 2, "noop": 3}[action.kind]
        symbol_code = -1 if action.symbol is None else action.symbol
        interaction = (
            (action_index + 1)
            * (q + 2)
            * (token + 3)
            * (top_symbol + 5)
        ) % 17
        score = (
            0.071 * (action_index + 1)
            + 0.113 * q
            - 0.047 * top_symbol
            + 0.137 * token
            + 0.029 * time_index
            + 0.019 * kind_code
            + 0.013 * symbol_code
            + 0.007 * interaction
        )
        valid_actions.append(action)
        scores.append(score)
    probabilities = _softmax(scores)
    return tuple(zip(valid_actions, probabilities, strict=True))


def payload(payload_source: int) -> Vector:
    if payload_source < 0:
        return (0.0, 0.0, 0.0, 0.0)
    raw = (
        math.sin(0.31 * (payload_source + 1)),
        math.cos(0.17 * (payload_source + 2)),
        math.sin(0.23 * (payload_source + 3)),
        math.cos(0.11 * (payload_source + 5)),
    )
    norm = math.sqrt(math.fsum(value * value for value in raw))
    scale = max(1.0, norm)
    return tuple(value / scale for value in raw)


def vector_distance(left: Vector, right: Vector) -> float:
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))


def weighted_read_tuple(weights: Mapping[TupleKey, float]) -> Vector:
    result = [0.0, 0.0, 0.0, 0.0]
    for (_, stack), weight in weights.items():
        value = payload(stack[-1][1]) if stack else payload(-1)
        for index, component in enumerate(value):
            result[index] += weight * component
    return tuple(result)


def _select_tuple(
    candidates: Mapping[TupleKey, float], beam_size: int
) -> dict[TupleKey, float]:
    ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
    kept = ranked[:beam_size]
    denominator = math.fsum(weight for _, weight in kept)
    if denominator <= 0.0:
        raise AssertionError("beam normalization denominator is not positive")
    return {key: weight / denominator for key, weight in kept}


def expand_tuple(
    weights: Mapping[TupleKey, float], *, time_index: int, token: int
) -> dict[TupleKey, float]:
    candidates: dict[TupleKey, float] = {}
    payload_source = 2 * time_index + token
    for (q, stack), parent_weight in weights.items():
        top_symbol = stack[-1][0] if stack else -1
        for action, probability in action_probabilities(
            time_index=time_index, token=token, q=q, top_symbol=top_symbol
        ):
            if action.kind == "push":
                assert action.symbol is not None
                next_stack = stack + ((action.symbol, payload_source),)
            elif action.kind == "pop":
                if not stack:
                    raise AssertionError("root pop escaped the action mask")
                next_stack = stack[:-1]
            else:
                next_stack = stack
            key = (action.next_q, next_stack)
            candidates[key] = candidates.get(key, 0.0) + (
                parent_weight * probability
            )
    return candidates


def step_tuple(
    weights: Mapping[TupleKey, float],
    *,
    time_index: int,
    token: int,
    beam_size: int,
) -> dict[TupleKey, float]:
    return _select_tuple(
        expand_tuple(weights, time_index=time_index, token=token), beam_size
    )


ArenaKey = tuple[int, int]  # (controller, node index)


def arena_to_tuple(
    arena: PersistentArena, weights: Mapping[ArenaKey, float]
) -> dict[TupleKey, float]:
    converted: dict[TupleKey, float] = {}
    for (q, node_index), weight in weights.items():
        key = (q, arena.signature(node_index))
        converted[key] = converted.get(key, 0.0) + weight
    return converted


def expand_arena_virtual(
    arena: PersistentArena,
    weights: Mapping[ArenaKey, float],
    *,
    time_index: int,
    token: int,
) -> dict[TupleKey, float]:
    """Expand through pointers without allocating rejected push candidates."""

    candidates: dict[TupleKey, float] = {}
    payload_source = 2 * time_index + token
    for (q, node_index), parent_weight in weights.items():
        top_symbol = arena.nodes[node_index].symbol
        stack = arena.signature(node_index)
        for action, probability in action_probabilities(
            time_index=time_index, token=token, q=q, top_symbol=top_symbol
        ):
            if action.kind == "push":
                assert action.symbol is not None
                next_stack = stack + ((action.symbol, payload_source),)
            elif action.kind == "pop":
                if node_index == 0:
                    raise AssertionError("root pop escaped the action mask")
                next_stack = arena.signature(arena.nodes[node_index].parent)
            else:
                next_stack = stack
            key = (action.next_q, next_stack)
            candidates[key] = candidates.get(key, 0.0) + (
                parent_weight * probability
            )
    return candidates


def materialize_arena_distribution(
    arena: PersistentArena,
    weights: Mapping[TupleKey, float],
) -> dict[ArenaKey, float]:
    materialized: dict[ArenaKey, float] = {}
    for (q, signature), weight in weights.items():
        key = (q, arena.materialize(signature))
        materialized[key] = materialized.get(key, 0.0) + weight
    return materialized


def step_arena(
    arena: PersistentArena,
    weights: Mapping[ArenaKey, float],
    *,
    time_index: int,
    token: int,
    beam_size: int,
) -> dict[ArenaKey, float]:
    candidates = expand_arena_virtual(
        arena, weights, time_index=time_index, token=token
    )
    selected = _select_tuple(candidates, beam_size)
    return materialize_arena_distribution(arena, selected)


@dataclass
class TrackedTupleState:
    total: dict[TupleKey, float]
    covered: dict[TupleKey, float]
    k_state: dict[TupleKey, float]


@dataclass
class TrackedArenaState:
    total: dict[ArenaKey, float]
    covered: dict[ArenaKey, float]
    k_state: dict[ArenaKey, float]


def step_tracked_tuple(
    state: TrackedTupleState, *, time_index: int, token: int
) -> TrackedTupleState:
    k_candidates = expand_tuple(state.k_state, time_index=time_index, token=token)
    next_k = _select_tuple(k_candidates, K_BEAM)
    selected_keys = set(next_k)

    total_candidates = expand_tuple(
        state.total, time_index=time_index, token=token
    )
    covered_expanded = expand_tuple(
        state.covered, time_index=time_index, token=token
    )
    covered_candidates = {
        key: weight
        for key, weight in covered_expanded.items()
        if key in selected_keys
    }

    next_total, next_covered = select_total_and_covered(
        total_candidates,
        covered_candidates,
        beam_size=S_BEAM,
    )
    return TrackedTupleState(
        total=next_total, covered=next_covered, k_state=next_k
    )


def select_total_and_covered(
    total_candidates: Mapping[TupleKey, float],
    covered_candidates: Mapping[TupleKey, float],
    *,
    beam_size: int,
) -> tuple[dict[TupleKey, float], dict[TupleKey, float]]:
    """The actual S-beam selection and causal-history provenance path."""

    ranked = sorted(
        total_candidates.items(), key=lambda item: (-item[1], item[0])
    )[:beam_size]
    denominator = math.fsum(weight for _, weight in ranked)
    if denominator <= 0.0:
        raise AssertionError("tracked denominator is not positive")
    total = {key: weight / denominator for key, weight in ranked}
    covered = {
        key: covered_candidates.get(key, 0.0) / denominator
        for key in total
        if covered_candidates.get(key, 0.0) > 0.0
    }
    return total, covered


def step_tracked_arena(
    s_arena: PersistentArena,
    k_arena: PersistentArena,
    state: TrackedArenaState,
    *,
    time_index: int,
    token: int,
) -> TrackedArenaState:
    k_candidates = expand_arena_virtual(
        k_arena, state.k_state, time_index=time_index, token=token
    )
    next_k_tuple = _select_tuple(k_candidates, K_BEAM)
    selected_keys = set(next_k_tuple)

    total_candidates = expand_arena_virtual(
        s_arena, state.total, time_index=time_index, token=token
    )
    covered_expanded = expand_arena_virtual(
        s_arena, state.covered, time_index=time_index, token=token
    )
    covered_candidates = {
        key: weight
        for key, weight in covered_expanded.items()
        if key in selected_keys
    }
    next_total_tuple, next_covered_tuple = select_total_and_covered(
        total_candidates,
        covered_candidates,
        beam_size=S_BEAM,
    )
    return TrackedArenaState(
        total=materialize_arena_distribution(s_arena, next_total_tuple),
        covered=materialize_arena_distribution(s_arena, next_covered_tuple),
        k_state=materialize_arena_distribution(k_arena, next_k_tuple),
    )


def _assert_distributions_close(
    left: Mapping[TupleKey, float],
    right: Mapping[TupleKey, float],
    *,
    atol: float = 2e-13,
) -> None:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:3]
        missing_right = sorted(set(left) - set(right))[:3]
        raise AssertionError(
            f"key mismatch left_missing={missing_left} right_missing={missing_right}"
        )
    for key in left:
        if not math.isclose(left[key], right[key], abs_tol=atol, rel_tol=atol):
            raise AssertionError(
                f"weight mismatch key={key} left={left[key]} right={right[key]}"
            )


def _binary_sequences(max_length: int) -> Iterable[tuple[int, ...]]:
    yield ()
    for length in range(1, max_length + 1):
        for bits in range(1 << length):
            yield tuple((bits >> shift) & 1 for shift in reversed(range(length)))


def run_oracle(max_length: int = 6) -> dict[str, object]:
    checked_sequences = 0
    checked_prefixes = 0
    minimum_bound_slack = math.inf
    minimum_covered_mass = 1.0

    root_actions = action_probabilities(time_index=0, token=0, q=0, top_symbol=-1)
    if any(action.kind == "pop" for action, _ in root_actions):
        raise AssertionError("root pop was not masked")
    if not math.isclose(
        math.fsum(probability for _, probability in root_actions),
        1.0,
        abs_tol=1e-15,
    ):
        raise AssertionError("root transition probabilities do not sum to one")

    for sequence in _binary_sequences(max_length):
        checked_sequences += 1
        tuple_state: dict[TupleKey, float] = {(0, ()): 1.0}
        arena = PersistentArena()
        arena_state: dict[ArenaKey, float] = {(0, 0): 1.0}
        tracked = TrackedTupleState(
            total={(0, ()): 1.0},
            covered={(0, ()): 1.0},
            k_state={(0, ()): 1.0},
        )
        tracked_s_arena = PersistentArena()
        tracked_k_arena = PersistentArena()
        tracked_arena = TrackedArenaState(
            total={(0, 0): 1.0},
            covered={(0, 0): 1.0},
            k_state={(0, 0): 1.0},
        )

        for time_index, token in enumerate(sequence):
            checked_prefixes += 1
            tuple_state = step_tuple(
                tuple_state,
                time_index=time_index,
                token=token,
                beam_size=S_BEAM,
            )
            arena_state = step_arena(
                arena,
                arena_state,
                time_index=time_index,
                token=token,
                beam_size=S_BEAM,
            )
            _assert_distributions_close(tuple_state, arena_to_tuple(arena, arena_state))

            tracked = step_tracked_tuple(
                tracked, time_index=time_index, token=token
            )
            tracked_arena = step_tracked_arena(
                tracked_s_arena,
                tracked_k_arena,
                tracked_arena,
                time_index=time_index,
                token=token,
            )
            _assert_distributions_close(tuple_state, tracked.total)
            _assert_distributions_close(
                tracked.total,
                arena_to_tuple(tracked_s_arena, tracked_arena.total),
            )
            _assert_distributions_close(
                tracked.covered,
                arena_to_tuple(tracked_s_arena, tracked_arena.covered),
            )
            _assert_distributions_close(
                tracked.k_state,
                arena_to_tuple(tracked_k_arena, tracked_arena.k_state),
            )

            covered_mass = math.fsum(tracked.covered.values())
            if not 0.0 < covered_mass <= 1.0 + 1e-12:
                raise AssertionError(f"invalid covered mass {covered_mass}")
            minimum_covered_mass = min(minimum_covered_mass, covered_mass)
            covered_distribution = {
                key: weight / covered_mass
                for key, weight in tracked.covered.items()
            }
            full_read = weighted_read_tuple(tracked.total)
            covered_read = weighted_read_tuple(covered_distribution)
            error = vector_distance(full_read, covered_read)
            bound = 2.0 * (1.0 - covered_mass)
            slack = bound - error
            minimum_bound_slack = min(minimum_bound_slack, slack)
            if error > bound + 1e-12:
                raise AssertionError(
                    f"2delta bound failed sequence={sequence} t={time_index} "
                    f"error={error} bound={bound}"
                )

    x4_keys: tuple[TupleKey, ...] = tuple(
        (index % N_CONTROL, ((index % N_SYMBOL, 10_000 + index),))
        for index in range(4)
    )
    x4_candidates = {key: 0.25 for key in x4_keys}
    x4_selected = _select_tuple(x4_candidates, K_BEAM)
    x4_covered_candidates = {
        key: weight for key, weight in x4_candidates.items() if key in x4_selected
    }
    x4_total, x4_covered = select_total_and_covered(
        x4_candidates,
        x4_covered_candidates,
        beam_size=S_BEAM,
    )
    x4_mass = math.fsum(x4_covered.values())
    x4_values = {
        key: tuple(1.0 if axis == index else 0.0 for axis in range(4))
        for index, key in enumerate(x4_keys)
    }

    def x4_read(weights: Mapping[TupleKey, float]) -> Vector:
        return tuple(
            math.fsum(weight * x4_values[key][axis] for key, weight in weights.items())
            for axis in range(4)
        )

    x4_full = x4_read(x4_total)
    x4_top3 = x4_read(
        {key: weight / x4_mass for key, weight in x4_covered.items()}
    )
    x4_error = vector_distance(x4_full, x4_top3)
    x4_expected_error = 1.0 / math.sqrt(12.0)
    if not math.isclose(x4_mass, 0.75, abs_tol=1e-15):
        raise AssertionError(f"X4 actual selector mass mismatch: {x4_mass}")
    if not math.isclose(x4_error, x4_expected_error, abs_tol=1e-15):
        raise AssertionError("X4 analytic read error mismatch")

    return {
        "status": "pass",
        "lineage": "CPKV-TOPK-001",
        "max_length": max_length,
        "checked_sequences": checked_sequences,
        "checked_prefixes": checked_prefixes,
        "minimum_k3_history_mass": minimum_covered_mass,
        "minimum_bound_slack": minimum_bound_slack,
        "x4": {
            "top3_history_mass": x4_mass,
            "discarded_mass": 1.0 - x4_mass,
            "read_error": x4_error,
            "bound": 0.5,
        },
    }


TOKEN_MAPS = {
    "D1": ["BOS", "EOS", "open0", "open1", "close0", "close1"],
    "A1": ["BOS", "EOS", "0", "1"],
    "A2": ["BOS", "EOS", "a", "b", "c"],
}

FAMILY_BASE_SEEDS = {"D1": 9101, "A1": 9201, "A2": 9301}
MODEL_SEEDS = tuple(range(1701, 1711))
TRAINED_ARMS = ("S32", "DIRECT3", "HARD3", "MLP", "RNN32", "LINK32")
TRAIN_SPEC = {"low": 1, "high": 8, "count": 20_000}
DEVELOPMENT_SPEC = {"low": 9, "high": 12, "count": 5_000, "offset": 0}
EVALUATION_SPECS = {
    "E1": {"low": 13, "high": 20, "count": 10_000, "offset": 1},
    "E2": {"low": 21, "high": 32, "count": 10_000, "offset": 2},
}


def _sample_sequence(
    family: str, low: int, high: int, generator: np.random.Generator
) -> list[str]:
    n = int(generator.integers(low, high + 1))
    if family == "D1":
        bits = generator.integers(0, 2, size=n).tolist()
        opens = [f"open{bit}" for bit in bits]
        closes = [f"close{bit}" for bit in reversed(bits)]
        body = opens + closes
    elif family == "A1":
        bits = [str(bit) for bit in generator.integers(0, 2, size=n).tolist()]
        odd = bool(generator.integers(0, 2))
        center = [str(int(generator.integers(0, 2)))] if odd else []
        body = bits + center + list(reversed(bits))
    elif family == "A2":
        branch = int(generator.integers(0, 2))
        free_count = int(generator.integers(low, high + 1))
        if branch == 0:
            i, j, k = n, n, free_count
        else:
            i, j, k = free_count, n, n
        body = ["a"] * i + ["b"] * j + ["c"] * k
    else:
        raise ValueError(f"unknown family {family}")
    return ["BOS", *body, "EOS"]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verify_preflight_receipt(
    *,
    project_root: Path,
    receipt_path: Path,
    expected_sources: Mapping[str, Path],
) -> Mapping[str, object]:
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "cpkv-topk-001-preflight-v1"
        or value.get("lineage") != "CPKV-TOPK-001"
        or value.get("status") != "pass"
        or value.get("seed") != 1701
        or value.get("base_sample_count") != 512
        or value.get("projection_limit_hours") != 5.7
    ):
        raise ValueError("passing preflight receipt is absent or invalid")
    families = value.get("families")
    if not isinstance(families, dict) or set(families) != {"D1", "A1", "A2"}:
        raise ValueError("preflight family ledger is incomplete")
    for family, record in families.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid preflight family: {family}")
        hours = record.get("projected_all_arm_cpu_hours")
        if (
            record.get("status") != "pass"
            or isinstance(hours, bool)
            or not isinstance(hours, (int, float))
            or not math.isfinite(float(hours))
            or float(hours) > 5.7
            or record.get("projection_limit_hours") != 5.7
            or record.get("actual_freeze_limit_hours") != 6.0
        ):
            raise ValueError(f"preflight resource gate failed: {family}")
    hashes = value.get("source_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(expected_sources):
        raise ValueError("preflight source seal is incomplete")
    for label, path in expected_sources.items():
        record = hashes.get(label)
        if (
            not isinstance(record, dict)
            or record.get("path") != path.relative_to(project_root).as_posix()
            or record.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"preflight source mismatch: {label}")
    return value


def _build_split(
    *,
    family: str,
    low: int,
    high: int,
    count: int,
    seed: int,
    path: Path | None,
) -> str:
    """Build a split or only its commitment hash.

    Evaluation uses the hash-only path until a checkpoint-freeze receipt exists.
    """

    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    digest = hashlib.sha256()
    temporary: Path | None = None
    handle = None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, delete=False
        )
        temporary = Path(handle.name)
    try:
        for _ in range(count):
            tokens = _sample_sequence(family, low, high, generator)
            line = (
                json.dumps({"tokens": tokens}, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            digest.update(line)
            if handle is not None:
                handle.write(line)
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, path)
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


def prepare_fixed_data(project_root: Path) -> dict[str, object]:
    data_root = project_root / "data" / "topk-pushdown-cpkv001"
    split_records: dict[str, object] = {}
    for family, base_seed in FAMILY_BASE_SEEDS.items():
        development_path = data_root / f"{family.lower()}-development.jsonl"
        development_hash = _build_split(
            family=family,
            low=int(DEVELOPMENT_SPEC["low"]),
            high=int(DEVELOPMENT_SPEC["high"]),
            count=int(DEVELOPMENT_SPEC["count"]),
            seed=base_seed,
            path=development_path,
        )
        split_records[f"{family}/development"] = {
            "path": development_path.relative_to(project_root).as_posix(),
            "sha256": development_hash,
            "sequences": int(DEVELOPMENT_SPEC["count"]),
            "seed": base_seed,
            "range": [
                int(DEVELOPMENT_SPEC["low"]),
                int(DEVELOPMENT_SPEC["high"]),
            ],
            "materialized": True,
        }

        for model_seed in MODEL_SEEDS:
            train_path = (
                data_root / "train" / f"{family.lower()}-{model_seed}.jsonl"
            )
            train_hash = _build_split(
                family=family,
                low=int(TRAIN_SPEC["low"]),
                high=int(TRAIN_SPEC["high"]),
                count=int(TRAIN_SPEC["count"]),
                seed=model_seed,
                path=train_path,
            )
            split_records[f"{family}/train/{model_seed}"] = {
                "path": train_path.relative_to(project_root).as_posix(),
                "sha256": train_hash,
                "sequences": int(TRAIN_SPEC["count"]),
                "seed": model_seed,
                "range": [int(TRAIN_SPEC["low"]), int(TRAIN_SPEC["high"])],
                "paired_across_arms": True,
                "materialized": True,
            }

        for split_name, spec in EVALUATION_SPECS.items():
            split_seed = base_seed + int(spec["offset"])
            path = data_root / f"{family.lower()}-{split_name.lower()}.jsonl"
            expected_hash = _build_split(
                family=family,
                low=int(spec["low"]),
                high=int(spec["high"]),
                count=int(spec["count"]),
                seed=split_seed,
                path=None,
            )
            split_records[f"{family}/{split_name}"] = {
                "post_freeze_path": path.relative_to(project_root).as_posix(),
                "sha256_commitment": expected_hash,
                "sequences": int(spec["count"]),
                "seed": split_seed,
                "range": [int(spec["low"]), int(spec["high"])],
                "materialized": False,
                "requires_checkpoint_freeze_receipt": True,
            }

    source_paths = {
        "reference": project_root
        / "experiments"
        / "topk_compiled_pushdown_reference.py",
        "preregistration": project_root
        / "results"
        / "topk-compiled-pushdown-acquisition-preregistration.md",
        "candidate": project_root
        / "results"
        / "compiled-pushdown-kv-substitution-paper-candidate.md",
        "evaluator": project_root
        / "experiments"
        / "topk_compiled_pushdown_evaluator.py",
        "training": project_root
        / "experiments"
        / "topk_compiled_pushdown_train.py",
        "reference_tests": project_root
        / "tests"
        / "test_topk_compiled_pushdown_reference.py",
        "evaluator_tests": project_root
        / "tests"
        / "test_topk_compiled_pushdown_evaluator.py",
        "locked_evaluation": project_root
        / "experiments"
        / "topk_compiled_pushdown_locked_eval.py",
        "locked_evaluation_tests": project_root
        / "tests"
        / "test_topk_compiled_pushdown_locked_eval.py",
        "training_tests": project_root
        / "tests"
        / "test_topk_compiled_pushdown_train.py",
        "preflight": project_root
        / "experiments"
        / "topk_compiled_pushdown_preflight.py",
        "preflight_tests": project_root
        / "tests"
        / "test_topk_compiled_pushdown_preflight.py",
    }
    source_hashes = {
        label: {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": _sha256(path),
        }
        for label, path in source_paths.items()
    }
    parameter_counts = {
        family: 193 * len(tokens) + 17_588
        for family, tokens in TOKEN_MAPS.items()
    }
    oracle_path = project_root / "results" / "topk-compiled-pushdown-oracle.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    x4 = oracle.get("x4") if isinstance(oracle, dict) else None
    expected_x4 = {
        "top3_history_mass": 0.75,
        "discarded_mass": 0.25,
        "read_error": 1.0 / math.sqrt(12.0),
        "bound": 0.5,
    }
    x4_matches = isinstance(x4, dict) and all(
        not isinstance(x4.get(name), bool)
        and isinstance(x4.get(name), (int, float))
        and math.isclose(
            float(x4[name]), expected, rel_tol=0.0, abs_tol=1e-15
        )
        for name, expected in expected_x4.items()
    )
    if (
        not isinstance(oracle, dict)
        or oracle.get("lineage") != "CPKV-TOPK-001"
        or oracle.get("status") != "pass"
        or oracle.get("max_length") != 6
        or oracle.get("checked_sequences") != 127
        or not x4_matches
    ):
        raise ValueError("length-six oracle receipt is absent or invalid")
    preflight_path = project_root / "results" / "topk-pushdown-preflight.json"
    preflight_sources = {
        label: source_paths[label]
        for label in (
            "preflight",
            "preflight_tests",
            "training",
            "locked_evaluation",
        )
    }
    preflight = _verify_preflight_receipt(
        project_root=project_root,
        receipt_path=preflight_path,
        expected_sources=preflight_sources,
    )
    manifest: dict[str, object] = {
        "schema": "cpkv-topk-001-manifest-v1",
        "lineage": "CPKV-TOPK-001",
        "status": "oracle-data-and-training-protocol-frozen",
        "token_maps": TOKEN_MAPS,
        "fixed_splits": split_records,
        "source_hashes": source_hashes,
        "oracle_receipt": {
            "path": oracle_path.relative_to(project_root).as_posix(),
            "sha256": _sha256(oracle_path),
            "max_length": 6,
            "checked_sequences": 127,
            "x4_top3_history_mass": 0.75,
        },
        "preflight_receipt": {
            "path": preflight_path.relative_to(project_root).as_posix(),
            "sha256": _sha256(preflight_path),
            "projection_limit_hours": 5.7,
            "projected_all_arm_cpu_hours": {
                family: preflight["families"][family][
                    "projected_all_arm_cpu_hours"
                ]
                for family in ("D1", "A1", "A2")
            },
        },
        "model": {
            "hidden_width": 64,
            "payload_width": PAYLOAD_WIDTH,
            "training_beam": S_BEAM,
            "served_beams": [1, 2, 3],
            "control_states": N_CONTROL,
            "stack_symbols": N_SYMBOL,
            "actions": N_ACTION,
            "parameter_formula": "193*V + 17588",
            "parameter_counts": parameter_counts,
            "dense_macs_per_token_formula": "12288 + 960 + 4096 + 128*V",
            "stack_module_parameters": 5_236,
        },
        "served_resource_ledger": {
            "removed_gqa_group": {
                "shape": {"g": 4, "h": 128, "kv_dtype_bytes": 2},
                "projection_macs_and_coefficients": "1280*d",
                "cache_bytes_per_token": 512,
            },
            "candidate": {
                "minimum_d": 512,
                "dense_macs_per_token": "1271*d",
                "sparse_branch_scalar_equivalents_upper_bound": 3_700,
                "reserved_scalar_equivalents": "9*d",
                "persistent_bytes_per_token_k3": 176,
                "bounded_per_sequence_beam_bytes": 52,
                "swi_glu_width": 376,
            },
        },
        "precision": {
            "learned_tensors": "float32",
            "evaluation_probabilities_and_kl": "float64",
            "persistent_index": "uint32",
            "served_payload": "bfloat16",
        },
        "training_protocol": {
            "model_seeds": list(MODEL_SEEDS),
            "trained_arms": list(TRAINED_ARMS),
            "paired_training_file_across_arms": True,
            "initialization": {
                "embedding": "normal(mean=0,std=0.02)",
                "dense_matrices": "xavier_uniform(gain=1)",
                "transition_table": "normal(mean=0,std=0.02)",
                "biases": "zero",
                "control_gains": "one",
                "rng": (
                    "PCG64DXSM(little_endian_uint64(" 
                    "SHA256(f'{model_seed}:{parameter_name}')[:8]))"
                ),
                "xavier_bounds": "+-sqrt(6/(fan_in+fan_out))",
            },
            "minibatch_order": (
                "PCG64DXSM(model_seed+100000+epoch), one fixed permutation "
                "per epoch shared across arms"
            ),
            "batching": (
                "64 sequences, no padding; sum per-sequence target losses and "
                "divide by total non-BOS targets including EOS"
            ),
            "adamw_decay": (
                "decay only 2D learned matrices including embeddings; exclude "
                "biases, transition table, and gains"
            ),
            "determinism": (
                "CPU only; torch.use_deterministic_algorithms(True); one torch "
                "intra-op thread; no mixed precision"
            ),
            "runtime_requirement": "CPython 3.14, torch==2.13.0 CPU wheel",
            "hard3": "straight-through argmax without sampling noise",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch_installed": _package_version("torch"),
            "torch_required": "2.13.0+cpu",
            "platform": platform.platform(),
        },
        "commands": {
            "oracle": (
                "python experiments/topk_compiled_pushdown_reference.py "
                "--oracle --oracle-output results/topk-compiled-pushdown-oracle.json"
            ),
            "prepare": (
                "python experiments/topk_compiled_pushdown_reference.py "
                "--prepare --project-root ."
            ),
            "training": (
                "python experiments/topk_compiled_pushdown_train.py "
                "--manifest manifests/topk-pushdown-cpkv001.json "
                "--family {D1,A1,A2} --arm {S32,DIRECT3,HARD3,MLP,RNN32,LINK32} "
                "--seed {1701..1710}"
            ),
            "preflight": (
                "python experiments/topk_compiled_pushdown_preflight.py "
                "--manifest manifests/topk-pushdown-cpkv001.json "
                "--output results/topk-pushdown-preflight.json"
            ),
            "post_freeze_evaluation": (
                "python experiments/topk_compiled_pushdown_evaluator.py "
                "--manifest manifests/topk-pushdown-cpkv001.json "
                "--checkpoint-freeze-receipt PATH"
            ),
            "build_checkpoint_freeze_receipt": (
                "python experiments/topk_compiled_pushdown_evaluator.py "
                "--manifest manifests/topk-pushdown-cpkv001.json "
                "--build-checkpoint-freeze-receipt "
                "results/topk-pushdown-checkpoints-frozen.json"
            ),
            "locked_evaluation_shard": (
                "python experiments/topk_compiled_pushdown_locked_eval.py "
                "--manifest manifests/topk-pushdown-cpkv001.json "
                "--checkpoint-freeze-receipt "
                "results/topk-pushdown-checkpoints-frozen.json "
                "--shard-index {0..59} "
                "--output results/topk-pushdown-shards/shard-{00..59}.json"
            ),
            "locked_evaluation_merge": (
                "python experiments/topk_compiled_pushdown_locked_eval.py "
                "--manifest manifests/topk-pushdown-cpkv001.json "
                "--checkpoint-freeze-receipt "
                "results/topk-pushdown-checkpoints-frozen.json "
                "--merge-shards results/topk-pushdown-shards "
                "--output results/topk-pushdown-locked-evaluation.json"
            ),
        },
    }
    manifest_path = project_root / "manifests" / "topk-pushdown-cpkv001.json"
    _atomic_write_text(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest["manifest_path"] = manifest_path.relative_to(project_root).as_posix()
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--max-length", type=int, default=6)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--oracle-output",
        type=Path,
        default=Path("results/topk-compiled-pushdown-oracle.json"),
    )
    args = parser.parse_args()
    if not args.oracle and not args.prepare:
        parser.error("at least one of --oracle or --prepare is required")

    project_root = args.project_root.resolve()
    output: MutableMapping[str, object] = {}
    if args.oracle:
        oracle = run_oracle(max_length=args.max_length)
        oracle_path = args.oracle_output
        if not oracle_path.is_absolute():
            oracle_path = project_root / oracle_path
        _write_json(oracle_path, oracle)
        output["oracle"] = oracle
        output["oracle_path"] = oracle_path.relative_to(project_root).as_posix()
    if args.prepare:
        output["manifest"] = prepare_fixed_data(project_root)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
