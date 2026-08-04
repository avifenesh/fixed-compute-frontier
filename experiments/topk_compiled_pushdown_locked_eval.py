#!/usr/bin/env python3
"""Locked teacher-forced evaluator for CPKV-TOPK-001.

The module intentionally keeps structural probabilities in FP64 while learned
matrix operations remain FP32.  It evaluates the frozen arms, compiled beams,
causal K3 history coverage, and the two preregistered interventions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

try:
    from experiments.topk_compiled_pushdown_evaluator import (
        _read_json,
        _resolve_inside,
        _sha256,
        materialize,
    )
    from experiments.topk_compiled_pushdown_reference import (
        ACTIONS,
        N_SYMBOL,
    )
    from experiments.topk_compiled_pushdown_train import (
        HIDDEN,
        AcquisitionModel,
    )
except ModuleNotFoundError:  # Direct `python experiments/...py` execution.
    from topk_compiled_pushdown_evaluator import (  # type: ignore[no-redef]
        _read_json,
        _resolve_inside,
        _sha256,
        materialize,
    )
    from topk_compiled_pushdown_reference import (  # type: ignore[no-redef]
        ACTIONS,
        N_SYMBOL,
    )
    from topk_compiled_pushdown_train import (  # type: ignore[no-redef]
        HIDDEN,
        AcquisitionModel,
    )


LINEAGE = "CPKV-TOPK-001"
BASE_ARMS = ("S32", "DIRECT3", "HARD3", "MLP", "RNN32", "LINK32")
COMPILED_ARMS = ("K1", "K2", "K3")
EVALUATED_ARMS = ("S32", *COMPILED_ARMS, *BASE_ARMS[1:])
CONTROLS = ("DIRECT3", "HARD3", "MLP", "RNN32", "LINK32")
INTERVENTIONS = ("ZERO-READ", "CHRONO-LINK")
EVALUATION_CELL_KEYS = tuple(
    sorted(
        f"{family}/{split}/{seed}"
        for family in ("D1", "A1", "A2")
        for split in ("E1", "E2")
        for seed in range(1701, 1711)
    )
)
SHARD_SCHEMA = "cpkv-topk-001-locked-evaluation-shard-v1"
FINAL_SCHEMA = "cpkv-topk-001-locked-evaluation-v1"
MAXIMUM_STATS = {
    "maximum_retained_configurations",
    "maximum_stack_depth",
    "logical_peak_auxiliary_bytes",
}
PAYLOAD_VECTOR_BYTES_FP32 = 64 * 4
PERSISTENT_NODE_BYTES_LOGICAL = 4 + 4 + 1
CONFIGURATION_BYTES_LOGICAL = 4 + 4 + 8
RNN32_STATE_BYTES_FP32 = 32 * 4


def _write_once_text(path: Path, value: str) -> None:
    """Publish a complete file atomically, failing if its name already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

StackItem = tuple[int, int]
StackSig = tuple[StackItem, ...]
ConfigKey = tuple[int, StackSig]


@dataclass
class Trace:
    logits: list[Tensor]
    stats: dict[str, int]


@dataclass
class MetricAccumulator:
    token_loss: float = 0.0
    tokens: int = 0
    sequence_nll_sum: float = 0.0
    sequences: int = 0
    delayed_loss: float = 0.0
    delayed_tokens: int = 0
    delayed_correct: int = 0
    early_loss: float = 0.0
    early_tokens: int = 0
    early_correct: int = 0
    conditional_suffix_loss: float = 0.0
    conditional_suffix_tokens: int = 0
    stats: dict[str, int] = field(default_factory=dict)
    position_types: dict[str, dict[str, float]] = field(default_factory=dict)
    early_by_length: dict[int, dict[str, float]] = field(default_factory=dict)

    def add_stats(self, values: Mapping[str, int]) -> None:
        for name, value in values.items():
            if name in MAXIMUM_STATS:
                self.stats[name] = max(self.stats.get(name, 0), int(value))
            else:
                self.stats[name] = self.stats.get(name, 0) + int(value)

    def result(self) -> dict[str, object]:
        if self.tokens <= 0 or self.sequences <= 0:
            raise ValueError("empty metric accumulator")
        return {
            "token_nll": self.token_loss / self.tokens,
            "sequence_weighted_nll": self.sequence_nll_sum / self.sequences,
            "tokens": self.tokens,
            "sequences": self.sequences,
            "delayed_nll": (
                self.delayed_loss / self.delayed_tokens
                if self.delayed_tokens
                else None
            ),
            "delayed_accuracy": (
                self.delayed_correct / self.delayed_tokens
                if self.delayed_tokens
                else None
            ),
            "delayed_tokens": self.delayed_tokens,
            "early_nll": (
                self.early_loss / self.early_tokens if self.early_tokens else None
            ),
            "early_accuracy": (
                self.early_correct / self.early_tokens
                if self.early_tokens
                else None
            ),
            "early_tokens": self.early_tokens,
            "conditional_suffix_nll": self.conditional_suffix_loss / self.sequences,
            "conditional_suffix_tokens": self.conditional_suffix_tokens,
            "position_types": {
                name: {
                    "nll": values["loss"] / values["count"],
                    "accuracy": values["correct"] / values["count"],
                    "tokens": int(values["count"]),
                }
                for name, values in sorted(self.position_types.items())
            },
            "early_length_slices": {
                str(length): {
                    "nll": values["loss"] / values["count"],
                    "accuracy": values["correct"] / values["count"],
                    "tokens": int(values["count"]),
                }
                for length, values in sorted(self.early_by_length.items())
            },
            "executed_state": self.stats,
        }


def _transition_probabilities(
    model: AcquisitionModel, base_logits: Tensor, q: int, top_symbol: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    top_index = N_SYMBOL if top_symbol < 0 else top_symbol
    logits = (base_logits + model.T[q, top_index]).to(torch.float64)
    valid = tuple(
        index
        for index, action in enumerate(ACTIONS)
        if not (top_symbol < 0 and action.kind == "pop")
    )
    probabilities = torch.softmax(logits[list(valid)], dim=0)
    return valid, tuple(float(value) for value in probabilities)


def _select(
    candidates: Mapping[ConfigKey, float], beam_size: int
) -> dict[ConfigKey, float]:
    ranked = sorted(
        candidates.items(), key=lambda item: (-item[1], item[0])
    )[:beam_size]
    denominator = math.fsum(weight for _, weight in ranked)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("nonpositive beam denominator")
    return {key: weight / denominator for key, weight in ranked}


def _expand(
    model: AcquisitionModel,
    state: Mapping[ConfigKey, float],
    *,
    base_logits: Tensor,
    time_index: int,
    chronological_parent: StackSig,
    use_chronological_parent: bool,
) -> tuple[dict[ConfigKey, float], set[StackSig], dict[str, int]]:
    buckets: dict[ConfigKey, list[float]] = {}
    pushed: set[StackSig] = set()
    executed_branches = 0
    push_candidates = 0
    cache: dict[tuple[int, int], tuple[tuple[int, ...], tuple[float, ...]]] = {}
    for (q, stack), parent_weight in state.items():
        top_symbol = stack[-1][0] if stack else -1
        cache_key = (q, top_symbol)
        transition = cache.get(cache_key)
        if transition is None:
            transition = _transition_probabilities(
                model, base_logits, q, top_symbol
            )
            cache[cache_key] = transition
        valid, probabilities = transition
        executed_branches += len(valid)
        for action_index, probability in zip(valid, probabilities, strict=True):
            action = ACTIONS[action_index]
            if action.kind == "push":
                push_candidates += 1
                assert action.symbol is not None
                parent = chronological_parent if use_chronological_parent else stack
                next_stack = parent + ((action.symbol, time_index),)
                pushed.add(next_stack)
            elif action.kind == "pop":
                if not stack:
                    raise AssertionError("root pop escaped evaluation mask")
                next_stack = stack[:-1]
            else:
                next_stack = stack
            key = (action.next_q, next_stack)
            buckets.setdefault(key, []).append(parent_weight * probability)
    return (
        {key: math.fsum(values) for key, values in buckets.items()},
        pushed,
        {
            "branches": executed_branches,
            "softmaxes": len(cache),
            "push_candidates": push_candidates,
        },
    )


def _next_chronological_parent(
    state: Mapping[ConfigKey, float],
    pushed: set[StackSig],
    current: StackSig,
) -> StackSig:
    allocated: set[StackSig] = set()
    for _, stack in sorted(key for key in state if key[1] in pushed):
        if stack not in allocated:
            current = stack
            allocated.add(stack)
    return current


def _weighted_read(
    state: Mapping[ConfigKey, float],
    payloads: Mapping[int, Tensor],
    zero: Tensor,
) -> Tensor:
    weights = torch.tensor(list(state.values()), dtype=torch.float64)
    values = torch.stack(
        [
            payloads[stack[-1][1]].to(torch.float64) if stack else zero.to(torch.float64)
            for _, stack in state
        ]
    )
    return (weights.unsqueeze(1) * values).sum(dim=0)


def _logical_auxiliary_bytes(
    *, payloads: int, persistent_nodes: int, configurations: int
) -> int:
    """Implementation-independent bytes for the evaluator's logical state."""
    return (
        payloads * PAYLOAD_VECTOR_BYTES_FP32
        + persistent_nodes * PERSISTENT_NODE_BYTES_LOGICAL
        + configurations * CONFIGURATION_BYTES_LOGICAL
    )


@torch.no_grad()
def weighted_stack_trace(
    model: AcquisitionModel,
    tokens: Sequence[int],
    *,
    beam_size: int,
    intervention: str | None = None,
) -> Trace:
    """Independent whole-sequence reference for decoder differential tests."""
    if intervention not in {None, *INTERVENTIONS}:
        raise ValueError(f"unknown intervention {intervention}")
    zero = model.E.new_zeros(HIDDEN)
    h = zero
    r = zero
    state: dict[ConfigKey, float] = {(0, ()): 1.0}
    payloads: dict[int, Tensor] = {}
    chronological_parent: StackSig = ()
    logits: list[Tensor] = []
    stats = {
        "tokens": 0,
        "configurations_expanded": 0,
        "valid_transition_branches": 0,
        "merged_candidates": 0,
        "retained_configurations": 0,
        "maximum_retained_configurations": 0,
        "maximum_stack_depth": 0,
        "action_projections": 0,
        "softmaxes": 0,
        "push_candidates": 0,
        "persistent_nodes_written": 0,
        "payload_vectors_written": 0,
        "payload_reads": 0,
        "dense_macs": 0,
        "logical_peak_auxiliary_bytes": 0,
    }
    use_chronological = model.arm == "LINK32" or intervention == "CHRONO-LINK"
    for time_index, token in enumerate(tokens[:-1]):
        h = model.common_hidden(token, h, r)
        raw_payload = model.W_v @ h
        payloads[time_index] = raw_payload / torch.clamp(
            torch.linalg.vector_norm(raw_payload), min=1.0
        )
        stats["tokens"] += 1
        stats["payload_vectors_written"] += 1
        stats["dense_macs"] += 12_288 + 960 + 4_096 + 128 * model.vocabulary_size
        stats["configurations_expanded"] += len(state)
        base_logits = model.W_a @ h
        stats["action_projections"] += 1
        candidates, pushed, expansion = _expand(
            model,
            state,
            base_logits=base_logits,
            time_index=time_index,
            chronological_parent=chronological_parent,
            use_chronological_parent=use_chronological,
        )
        stats["valid_transition_branches"] += expansion["branches"]
        stats["softmaxes"] += expansion["softmaxes"]
        stats["push_candidates"] += expansion["push_candidates"]
        stats["merged_candidates"] += len(candidates)
        state = _select(candidates, beam_size)
        chronological_parent = _next_chronological_parent(
            state, pushed, chronological_parent
        )
        stats["retained_configurations"] += len(state)
        retained_pushes = {stack for _, stack in state if stack in pushed}
        stats["persistent_nodes_written"] += len(retained_pushes)
        stats["payload_reads"] += sum(bool(stack) for _, stack in state)
        stats["maximum_retained_configurations"] = max(
            stats["maximum_retained_configurations"], len(state)
        )
        stats["maximum_stack_depth"] = max(
            stats["maximum_stack_depth"],
            max((len(stack) for _, stack in state), default=0),
        )
        stats["logical_peak_auxiliary_bytes"] = max(
            stats["logical_peak_auxiliary_bytes"],
            _logical_auxiliary_bytes(
                payloads=len(payloads),
                persistent_nodes=stats["persistent_nodes_written"],
                configurations=len(state),
            ),
        )
        read64 = _weighted_read(state, payloads, zero)
        r = read64.to(torch.float32)
        if intervention == "ZERO-READ":
            r = zero
        logits.append(model.output_logits(h, r).to(torch.float64))
    return Trace(logits=logits, stats=stats)


@torch.no_grad()
def hard3_trace(model: AcquisitionModel, tokens: Sequence[int]) -> Trace:
    """Independent HARD3 reference for decoder differential tests."""
    zero = model.E.new_zeros(HIDDEN)
    h = zero
    r = zero
    lanes: list[tuple[int, StackSig, float]] = [
        (q, (), 1.0 / 3.0) for q in range(3)
    ]
    payloads: dict[int, Tensor] = {}
    logits: list[Tensor] = []
    stats = {
        "tokens": 0,
        "configurations_expanded": 0,
        "valid_transition_branches": 0,
        "merged_candidates": 0,
        "retained_configurations": 0,
        "maximum_retained_configurations": 3,
        "maximum_stack_depth": 0,
        "action_projections": 0,
        "softmaxes": 0,
        "push_candidates": 0,
        "persistent_nodes_written": 0,
        "payload_vectors_written": 0,
        "payload_reads": 0,
        "dense_macs": 0,
        "logical_peak_auxiliary_bytes": 0,
    }
    for time_index, token in enumerate(tokens[:-1]):
        h = model.common_hidden(token, h, r)
        raw_payload = model.W_v @ h
        payloads[time_index] = raw_payload / torch.clamp(
            torch.linalg.vector_norm(raw_payload), min=1.0
        )
        next_lanes: list[tuple[int, StackSig, float]] = []
        unnormalized: list[float] = []
        stats["tokens"] += 1
        stats["payload_vectors_written"] += 1
        stats["dense_macs"] += 12_288 + 960 + 4_096 + 128 * model.vocabulary_size
        stats["configurations_expanded"] += len(lanes)
        base_logits = model.W_a @ h
        stats["action_projections"] += 1
        for q, stack, lane_weight in lanes:
            top_symbol = stack[-1][0] if stack else -1
            valid, probabilities = _transition_probabilities(
                model, base_logits, q, top_symbol
            )
            stats["valid_transition_branches"] += len(valid)
            stats["softmaxes"] += 1
            stats["push_candidates"] += sum(
                ACTIONS[index].kind == "push" for index in valid
            )
            hard_local = max(range(len(valid)), key=probabilities.__getitem__)
            action = ACTIONS[valid[hard_local]]
            if action.kind == "push":
                assert action.symbol is not None
                next_stack = stack + ((action.symbol, time_index),)
                stats["persistent_nodes_written"] += 1
            elif action.kind == "pop":
                next_stack = stack[:-1]
            else:
                next_stack = stack
            weight = lane_weight * probabilities[hard_local]
            next_lanes.append((action.next_q, next_stack, weight))
            unnormalized.append(weight)
        denominator = math.fsum(unnormalized)
        lanes = [
            (q, stack, weight / denominator)
            for q, stack, weight in next_lanes
        ]
        stats["merged_candidates"] += len(lanes)
        stats["retained_configurations"] += len(lanes)
        stats["maximum_stack_depth"] = max(
            stats["maximum_stack_depth"],
            max((len(stack) for _, stack, _ in lanes), default=0),
        )
        stats["payload_reads"] += sum(bool(stack) for _, stack, _ in lanes)
        stats["logical_peak_auxiliary_bytes"] = max(
            stats["logical_peak_auxiliary_bytes"],
            _logical_auxiliary_bytes(
                payloads=len(payloads),
                persistent_nodes=stats["persistent_nodes_written"],
                configurations=len(lanes),
            ),
        )
        weights = torch.tensor(
            [weight for _, _, weight in lanes], dtype=torch.float64
        )
        reads = torch.stack(
            [
                payloads[stack[-1][1]].to(torch.float64)
                if stack
                else zero.to(torch.float64)
                for _, stack, _ in lanes
            ]
        )
        r = (weights.unsqueeze(1) * reads).sum(dim=0).to(torch.float32)
        logits.append(model.output_logits(h, r).to(torch.float64))
    return Trace(logits=logits, stats=stats)


@torch.no_grad()
def control_trace(model: AcquisitionModel, tokens: Sequence[int]) -> Trace:
    """Independent dense-control reference for decoder differential tests."""
    zero = model.E.new_zeros(HIDDEN)
    h = zero
    r = zero
    s = model.E.new_zeros(32) if model.arm == "RNN32" else None
    logits: list[Tensor] = []
    for token in tokens[:-1]:
        h = model.common_hidden(token, h, r)
        if model.arm == "MLP":
            r = model.W2 @ torch.tanh(model.W1 @ h + model.b1) + model.b2
            r = torch.cat((r[:12] * model.gains, r[12:]))
        elif model.arm == "RNN32":
            assert s is not None
            s = torch.tanh(model.A @ h + model.B @ s + model.b_s)
            r = model.C @ s + model.b_r
            r = torch.cat((r[:20] * model.gains, r[20:]))
        else:
            raise ValueError(f"not a dense control: {model.arm}")
        logits.append(model.output_logits(h, r).to(torch.float64))
    targets = len(tokens) - 1
    dense_module_macs = 5_120
    return Trace(
        logits=logits,
        stats={
            "tokens": targets,
            "configurations_expanded": 0,
            "valid_transition_branches": 0,
            "merged_candidates": 0,
            "retained_configurations": 0,
            "maximum_retained_configurations": 0,
            "maximum_stack_depth": 0,
            "action_projections": 0,
            "softmaxes": 0,
            "push_candidates": 0,
            "persistent_nodes_written": 0,
            "payload_vectors_written": 0,
            "payload_reads": 0,
            "dense_macs": targets
            * (12_288 + dense_module_macs + 128 * model.vocabulary_size),
            "logical_peak_auxiliary_bytes": (
                RNN32_STATE_BYTES_FP32 if model.arm == "RNN32" else 0
            ),
        },
    )


class IncrementalDecoder:
    """One locked inference state machine shared by scoring and generation."""

    def __init__(
        self,
        model: AcquisitionModel,
        *,
        evaluated_arm: str,
        intervention: str | None = None,
    ) -> None:
        if intervention is not None and evaluated_arm not in {"S32", "K3"}:
            raise ValueError("interventions apply only to S32 and compiled K3")
        if intervention not in {None, *INTERVENTIONS}:
            raise ValueError(f"unknown intervention {intervention}")
        self.model = model
        self.evaluated_arm = evaluated_arm
        self.intervention = intervention
        self.zero = model.E.new_zeros(HIDDEN)
        self.h = self.zero
        self.r = self.zero
        self.time_index = 0
        self.payloads: dict[int, Tensor] = {}
        self.state: dict[ConfigKey, float] = {(0, ()): 1.0}
        self.chronological_parent: StackSig = ()
        self.lanes: list[tuple[int, StackSig, float]] = [
            (q, (), 1.0 / 3.0) for q in range(3)
        ]
        self.s = self.zero.new_zeros(32) if evaluated_arm == "RNN32" else None
        self.stats = {
            "tokens": 0,
            "configurations_expanded": 0,
            "valid_transition_branches": 0,
            "merged_candidates": 0,
            "retained_configurations": 0,
            "maximum_retained_configurations": (
                3 if evaluated_arm == "HARD3" else 0
            ),
            "maximum_stack_depth": 0,
            "action_projections": 0,
            "softmaxes": 0,
            "push_candidates": 0,
            "persistent_nodes_written": 0,
            "payload_vectors_written": 0,
            "payload_reads": 0,
            "dense_macs": 0,
            "logical_peak_auxiliary_bytes": (
                RNN32_STATE_BYTES_FP32 if evaluated_arm == "RNN32" else 0
            ),
        }

    def _weighted_step(self, token: int) -> Tensor:
        model = self.model
        self.h = model.common_hidden(token, self.h, self.r)
        raw_payload = model.W_v @ self.h
        self.payloads[self.time_index] = raw_payload / torch.clamp(
            torch.linalg.vector_norm(raw_payload), min=1.0
        )
        self.stats["payload_vectors_written"] += 1
        self.stats["dense_macs"] += (
            12_288 + 960 + 4_096 + 128 * model.vocabulary_size
        )
        self.stats["configurations_expanded"] += len(self.state)
        base_logits = model.W_a @ self.h
        self.stats["action_projections"] += 1
        use_chronological = (
            model.arm == "LINK32" or self.intervention == "CHRONO-LINK"
        )
        candidates, pushed, expansion = _expand(
            model,
            self.state,
            base_logits=base_logits,
            time_index=self.time_index,
            chronological_parent=self.chronological_parent,
            use_chronological_parent=use_chronological,
        )
        self.stats["valid_transition_branches"] += expansion["branches"]
        self.stats["softmaxes"] += expansion["softmaxes"]
        self.stats["push_candidates"] += expansion["push_candidates"]
        self.stats["merged_candidates"] += len(candidates)
        beam_size = (
            int(self.evaluated_arm[1:])
            if self.evaluated_arm in COMPILED_ARMS
            else 3
            if self.evaluated_arm == "DIRECT3"
            else 32
        )
        self.state = _select(candidates, beam_size)
        self.chronological_parent = _next_chronological_parent(
            self.state, pushed, self.chronological_parent
        )
        self.stats["retained_configurations"] += len(self.state)
        retained_pushes = {
            stack for _, stack in self.state if stack in pushed
        }
        self.stats["persistent_nodes_written"] += len(retained_pushes)
        self.stats["payload_reads"] += sum(bool(stack) for _, stack in self.state)
        self.stats["maximum_retained_configurations"] = max(
            self.stats["maximum_retained_configurations"], len(self.state)
        )
        self.stats["maximum_stack_depth"] = max(
            self.stats["maximum_stack_depth"],
            max((len(stack) for _, stack in self.state), default=0),
        )
        self.stats["logical_peak_auxiliary_bytes"] = max(
            self.stats["logical_peak_auxiliary_bytes"],
            _logical_auxiliary_bytes(
                payloads=len(self.payloads),
                persistent_nodes=self.stats["persistent_nodes_written"],
                configurations=len(self.state),
            ),
        )
        self.r = _weighted_read(
            self.state, self.payloads, self.zero
        ).to(torch.float32)
        if self.intervention == "ZERO-READ":
            self.r = self.zero
        return model.output_logits(self.h, self.r).to(torch.float64)

    def _hard_step(self, token: int) -> Tensor:
        model = self.model
        self.h = model.common_hidden(token, self.h, self.r)
        raw_payload = model.W_v @ self.h
        self.payloads[self.time_index] = raw_payload / torch.clamp(
            torch.linalg.vector_norm(raw_payload), min=1.0
        )
        next_lanes: list[tuple[int, StackSig, float]] = []
        weights: list[float] = []
        self.stats["payload_vectors_written"] += 1
        self.stats["dense_macs"] += (
            12_288 + 960 + 4_096 + 128 * model.vocabulary_size
        )
        self.stats["configurations_expanded"] += len(self.lanes)
        base_logits = model.W_a @ self.h
        self.stats["action_projections"] += 1
        for q, stack, lane_weight in self.lanes:
            top_symbol = stack[-1][0] if stack else -1
            valid, probabilities = _transition_probabilities(
                model, base_logits, q, top_symbol
            )
            self.stats["valid_transition_branches"] += len(valid)
            self.stats["softmaxes"] += 1
            self.stats["push_candidates"] += sum(
                ACTIONS[index].kind == "push" for index in valid
            )
            chosen = max(range(len(valid)), key=probabilities.__getitem__)
            action = ACTIONS[valid[chosen]]
            if action.kind == "push":
                assert action.symbol is not None
                next_stack = stack + ((action.symbol, self.time_index),)
                self.stats["persistent_nodes_written"] += 1
            elif action.kind == "pop":
                next_stack = stack[:-1]
            else:
                next_stack = stack
            weight = lane_weight * probabilities[chosen]
            next_lanes.append((action.next_q, next_stack, weight))
            weights.append(weight)
        denominator = math.fsum(weights)
        self.lanes = [
            (q, stack, weight / denominator)
            for q, stack, weight in next_lanes
        ]
        self.stats["merged_candidates"] += len(self.lanes)
        self.stats["retained_configurations"] += len(self.lanes)
        self.stats["maximum_stack_depth"] = max(
            self.stats["maximum_stack_depth"],
            max((len(stack) for _, stack, _ in self.lanes), default=0),
        )
        self.stats["payload_reads"] += sum(
            bool(stack) for _, stack, _ in self.lanes
        )
        self.stats["logical_peak_auxiliary_bytes"] = max(
            self.stats["logical_peak_auxiliary_bytes"],
            _logical_auxiliary_bytes(
                payloads=len(self.payloads),
                persistent_nodes=self.stats["persistent_nodes_written"],
                configurations=len(self.lanes),
            ),
        )
        lane_weights = torch.tensor(
            [weight for _, _, weight in self.lanes], dtype=torch.float64
        )
        reads = torch.stack(
            [
                self.payloads[stack[-1][1]].to(torch.float64)
                if stack
                else self.zero.to(torch.float64)
                for _, stack, _ in self.lanes
            ]
        )
        self.r = (lane_weights.unsqueeze(1) * reads).sum(dim=0).to(torch.float32)
        return model.output_logits(self.h, self.r).to(torch.float64)

    def _control_step(self, token: int) -> Tensor:
        model = self.model
        self.h = model.common_hidden(token, self.h, self.r)
        if self.evaluated_arm == "MLP":
            self.r = model.W2 @ torch.tanh(model.W1 @ self.h + model.b1) + model.b2
            self.r = torch.cat((self.r[:12] * model.gains, self.r[12:]))
        elif self.evaluated_arm == "RNN32":
            assert self.s is not None
            self.s = torch.tanh(model.A @ self.h + model.B @ self.s + model.b_s)
            self.r = model.C @ self.s + model.b_r
            self.r = torch.cat((self.r[:20] * model.gains, self.r[20:]))
        else:
            raise ValueError(f"not a dense control: {self.evaluated_arm}")
        self.stats["dense_macs"] += (
            12_288 + 5_120 + 128 * model.vocabulary_size
        )
        return model.output_logits(self.h, self.r).to(torch.float64)

    @torch.no_grad()
    def step(self, token: int) -> Tensor:
        self.stats["tokens"] += 1
        if self.evaluated_arm in {"MLP", "RNN32"}:
            logits = self._control_step(token)
        elif self.evaluated_arm == "HARD3":
            logits = self._hard_step(token)
        elif self.evaluated_arm in {
            "S32",
            "K1",
            "K2",
            "K3",
            "DIRECT3",
            "LINK32",
        }:
            logits = self._weighted_step(token)
        else:
            raise ValueError(f"unknown evaluated arm {self.evaluated_arm}")
        self.time_index += 1
        return logits

    def trace(self, tokens: Sequence[int]) -> Trace:
        return Trace(
            logits=[self.step(token) for token in tokens[:-1]],
            stats=dict(self.stats),
        )


def teacher_forced_trace(
    model: AcquisitionModel,
    tokens: Sequence[int],
    *,
    evaluated_arm: str,
    intervention: str | None = None,
) -> Trace:
    return IncrementalDecoder(
        model,
        evaluated_arm=evaluated_arm,
        intervention=intervention,
    ).trace(tokens)


def delayed_positions(
    family: str,
    tokens: Sequence[int],
    token_names: Sequence[str],
) -> dict[int, int]:
    names = [token_names[token] for token in tokens]
    positions: dict[int, int] = {}
    if family == "D1":
        close_ids = [token_names.index("close0"), token_names.index("close1")]
        for target_position, name in enumerate(names[1:], start=1):
            if name.startswith("close"):
                correct = tokens[target_position]
                positions[target_position - 1] = (
                    close_ids[1] if correct == close_ids[0] else close_ids[0]
                )
    elif family == "A1":
        bit_ids = [token_names.index("0"), token_names.index("1")]
        content_length = len(tokens) - 2
        suffix_start = 1 + math.ceil(content_length / 2)
        for target_position in range(suffix_start, len(tokens) - 1):
            correct = tokens[target_position]
            positions[target_position - 1] = (
                bit_ids[1] if correct == bit_ids[0] else bit_ids[0]
            )
    elif family == "A2":
        b_id = token_names.index("b")
        c_id = token_names.index("c")
        eos_id = token_names.index("EOS")
        first_c = names.index("c")
        positions[first_c - 1] = b_id
        positions[len(tokens) - 2] = c_id
        if tokens[-1] != eos_id:
            raise AssertionError("A2 sequence does not terminate in EOS")
    else:
        raise ValueError(f"unknown family {family}")
    return positions


def delayed_position_types(
    family: str,
    tokens: Sequence[int],
    token_names: Sequence[str],
) -> dict[int, tuple[int, str]]:
    positions = delayed_positions(family, tokens, token_names)
    if family == "D1":
        return {
            position: (distractor, "typed_closer")
            for position, distractor in positions.items()
        }
    if family == "A1":
        return {
            position: (distractor, "mirrored_suffix")
            for position, distractor in positions.items()
        }
    ordered = sorted(positions)
    if len(ordered) != 2:
        raise AssertionError("A2 must expose two delayed boundaries")
    return {
        ordered[0]: (positions[ordered[0]], "b_to_c_boundary"),
        ordered[1]: (positions[ordered[1]], "c_to_eos_boundary"),
    }


def accumulate_trace(
    accumulator: MetricAccumulator,
    trace: Trace,
    tokens: Sequence[int],
    *,
    family: str,
    token_names: Sequence[str],
) -> None:
    if len(trace.logits) != len(tokens) - 1:
        raise AssertionError("trace target count mismatch")
    logits = torch.stack(trace.logits)
    targets = torch.tensor(tokens[1:], dtype=torch.long)
    log_probabilities = torch.log_softmax(logits, dim=1)
    losses = -log_probabilities[torch.arange(len(targets)), targets]
    delayed = delayed_position_types(family, tokens, token_names)
    first_delayed = min(delayed) if delayed else len(targets)
    accumulator.token_loss += math.fsum(float(value) for value in losses)
    accumulator.tokens += len(targets)
    accumulator.sequence_nll_sum += float(losses.mean())
    accumulator.sequences += 1
    # Score the complete continuation conditioned on the frozen prompt.
    accumulator.conditional_suffix_loss += math.fsum(
        float(value) for value in losses[first_delayed:]
    )
    accumulator.conditional_suffix_tokens += len(losses) - first_delayed
    for position, loss in enumerate(losses):
        if position in delayed:
            accumulator.delayed_loss += float(loss)
            accumulator.delayed_tokens += 1
            correct = tokens[position + 1]
            distractor, position_type = delayed[position]
            # Forced-choice ties are conservatively counted as incorrect.
            correct_choice = int(
                float(logits[position, correct])
                > float(logits[position, distractor])
            )
            accumulator.delayed_correct += correct_choice
            values = accumulator.position_types.setdefault(
                position_type, {"loss": 0.0, "count": 0.0, "correct": 0.0}
            )
            values["loss"] += float(loss)
            values["count"] += 1.0
            values["correct"] += correct_choice
        elif position < first_delayed:
            accumulator.early_loss += float(loss)
            accumulator.early_tokens += 1
            accumulator.early_correct += int(
                int(torch.argmax(logits[position])) == int(targets[position])
            )
            content_length = len(tokens) - 2
            values = accumulator.early_by_length.setdefault(
                content_length, {"loss": 0.0, "count": 0.0, "correct": 0.0}
            )
            values["loss"] += float(loss)
            values["count"] += 1.0
            values["correct"] += int(
                int(torch.argmax(logits[position])) == int(targets[position])
            )
    accumulator.add_stats(trace.stats)


@torch.no_grad()
def causal_history_metrics(
    model: AcquisitionModel,
    tokens: Sequence[int],
    *,
    k: int = 3,
) -> dict[str, object]:
    zero = model.E.new_zeros(HIDDEN)
    h = zero
    r = zero
    total: dict[ConfigKey, float] = {(0, ()): 1.0}
    covered: dict[ConfigKey, float] = {(0, ()): 1.0}
    k_state: dict[ConfigKey, float] = {(0, ()): 1.0}
    payloads: dict[int, Tensor] = {}
    masses: list[float] = []
    read_errors: list[float] = []
    bounds: list[float] = []
    for time_index, token in enumerate(tokens[:-1]):
        h = model.common_hidden(token, h, r)
        raw_payload = model.W_v @ h
        payloads[time_index] = raw_payload / torch.clamp(
            torch.linalg.vector_norm(raw_payload), min=1.0
        )
        base_logits = model.W_a @ h
        k_candidates, _, _ = _expand(
            model,
            k_state,
            base_logits=base_logits,
            time_index=time_index,
            chronological_parent=(),
            use_chronological_parent=False,
        )
        next_k = _select(k_candidates, k)
        selected_k_keys = set(next_k)
        total_candidates, _, _ = _expand(
            model,
            total,
            base_logits=base_logits,
            time_index=time_index,
            chronological_parent=(),
            use_chronological_parent=False,
        )
        covered_expanded, _, _ = _expand(
            model,
            covered,
            base_logits=base_logits,
            time_index=time_index,
            chronological_parent=(),
            use_chronological_parent=False,
        )
        covered_candidates = {
            key: weight
            for key, weight in covered_expanded.items()
            if key in selected_k_keys
        }
        ranked = sorted(
            total_candidates.items(), key=lambda item: (-item[1], item[0])
        )[:32]
        denominator = math.fsum(weight for _, weight in ranked)
        total = {key: weight / denominator for key, weight in ranked}
        covered = {
            key: covered_candidates.get(key, 0.0) / denominator
            for key in total
            if covered_candidates.get(key, 0.0) > 0.0
        }
        k_state = next_k
        mass = math.fsum(covered.values())
        if not 0.0 <= mass <= 1.0 + 1e-12:
            raise AssertionError(f"invalid causal history mass {mass}")
        full_read = _weighted_read(total, payloads, zero)
        if mass > 0.0:
            covered_distribution = {
                key: weight / mass for key, weight in covered.items()
            }
            covered_read = _weighted_read(covered_distribution, payloads, zero)
        else:
            covered_read = zero.to(torch.float64)
        error = float(torch.linalg.vector_norm(full_read - covered_read))
        bound = 2.0 * (1.0 - mass)
        if error > bound + 1e-6:
            raise AssertionError(
                f"2delta violation error={error} bound={bound} t={time_index}"
            )
        masses.append(mass)
        read_errors.append(error)
        bounds.append(bound)
        r = full_read.to(torch.float32)
    return {
        "prefixes": len(masses),
        "masses": masses,
        "read_errors": read_errors,
        "bounds": bounds,
    }


def _load_model(
    *,
    project_root: Path,
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
    family: str,
    arm: str,
    seed: int,
) -> AcquisitionModel:
    token_maps = manifest.get("token_maps")
    checkpoint_hashes = receipt.get("checkpoint_hashes")
    if not isinstance(token_maps, dict) or not isinstance(checkpoint_hashes, dict):
        raise ValueError("manifest or checkpoint receipt incomplete")
    source_arm = "S32" if arm in COMPILED_ARMS else arm
    record = checkpoint_hashes.get(f"{family}/{source_arm}/{seed}")
    if not isinstance(record, dict):
        raise ValueError(f"checkpoint missing: {family}/{source_arm}/{seed}")
    run_path = _resolve_inside(
        project_root, str(record["path"]), label="locked evaluation run"
    )
    run = _read_json(run_path)
    weights_path = _resolve_inside(
        project_root, str(run["weights_path"]), label="locked evaluation weights"
    )
    model = AcquisitionModel(
        vocabulary_size=len(token_maps[family]), arm=source_arm, model_seed=seed
    )
    with np.load(weights_path, allow_pickle=False) as arrays:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(torch.from_numpy(arrays[name]))
    model.eval()
    return model


def _load_evaluation_sequences(
    *, project_root: Path, manifest: Mapping[str, object], family: str, split: str
) -> list[list[int]]:
    token_maps = manifest.get("token_maps")
    fixed_splits = manifest.get("fixed_splits")
    if not isinstance(token_maps, dict) or not isinstance(fixed_splits, dict):
        raise ValueError("manifest split data incomplete")
    record = fixed_splits.get(f"{family}/{split}")
    if not isinstance(record, dict):
        raise ValueError(f"evaluation split missing: {family}/{split}")
    relative = record.get("post_freeze_path")
    expected_hash = record.get("sha256_commitment")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError("sealed evaluation split record incomplete")
    path = _resolve_inside(project_root, relative, label="locked evaluation split")
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError(f"sealed evaluation split hash mismatch: {family}/{split}")
    token_to_id = {
        str(token): index for index, token in enumerate(token_maps[family])
    }
    sequences: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            sequences.append([token_to_id[str(token)] for token in raw["tokens"]])
    if len(sequences) != int(record["sequences"]):
        raise ValueError("sealed evaluation split count mismatch")
    return sequences


def _kl_s32_k3(left: Trace, right: Trace) -> tuple[float, int]:
    if len(left.logits) != len(right.logits):
        raise AssertionError("KL trace length mismatch")
    total = 0.0
    for left_logits, right_logits in zip(
        left.logits, right.logits, strict=True
    ):
        left_log = torch.log_softmax(left_logits, dim=0)
        right_log = torch.log_softmax(right_logits, dim=0)
        left_probability = torch.exp(left_log)
        total += float(torch.sum(left_probability * (left_log - right_log)))
    return total, len(left.logits)


def evaluate_teacher_forced(
    *,
    project_root: Path,
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
    requested_cells: set[str] | None = None,
) -> dict[str, object]:
    token_maps = manifest.get("token_maps")
    protocol = manifest.get("training_protocol")
    if not isinstance(token_maps, dict) or not isinstance(protocol, dict):
        raise ValueError("locked evaluation manifest incomplete")
    seeds = protocol.get("model_seeds")
    if not isinstance(seeds, list) or seeds != list(range(1701, 1711)):
        raise ValueError("locked evaluation seed contract mismatch")
    if requested_cells is not None and not requested_cells.issubset(
        EVALUATION_CELL_KEYS
    ):
        raise ValueError("unknown requested teacher-forced cell")
    cells: dict[str, object] = {}
    for family in ("D1", "A1", "A2"):
        token_names = [str(value) for value in token_maps[family]]
        for split in ("E1", "E2"):
            selected_seeds = [
                int(seed)
                for seed in seeds
                if requested_cells is None
                or f"{family}/{split}/{int(seed)}" in requested_cells
            ]
            if not selected_seeds:
                continue
            sequences = _load_evaluation_sequences(
                project_root=project_root,
                manifest=manifest,
                family=family,
                split=split,
            )
            for seed in selected_seeds:
                models = {
                    arm: _load_model(
                        project_root=project_root,
                        manifest=manifest,
                        receipt=receipt,
                        family=family,
                        arm=arm,
                        seed=seed,
                    )
                    for arm in BASE_ARMS
                }
                accumulators = {
                    arm: MetricAccumulator() for arm in EVALUATED_ARMS
                }
                intervention_accumulators = {
                    f"{arm}/{intervention}": MetricAccumulator()
                    for arm in ("S32", "K3")
                    for intervention in INTERVENTIONS
                }
                kl_sum = 0.0
                kl_tokens = 0
                mass_prefixes = 0
                mass_delta_sum = 0.0
                mass_delta_le_001 = 0
                mass_max_delta = 0.0
                mass_read_error_sum = 0.0
                mass_max_bound_excess = -math.inf
                for tokens in sequences:
                    traces: dict[str, Trace] = {}
                    for arm in EVALUATED_ARMS:
                        model = models["S32"] if arm in COMPILED_ARMS else models[arm]
                        trace = teacher_forced_trace(
                            model, tokens, evaluated_arm=arm
                        )
                        traces[arm] = trace
                        accumulate_trace(
                            accumulators[arm],
                            trace,
                            tokens,
                            family=family,
                            token_names=token_names,
                        )
                    sequence_kl, sequence_kl_tokens = _kl_s32_k3(
                        traces["S32"], traces["K3"]
                    )
                    kl_sum += sequence_kl
                    kl_tokens += sequence_kl_tokens
                    mass = causal_history_metrics(models["S32"], tokens, k=3)
                    for value, error, bound in zip(
                        mass["masses"],
                        mass["read_errors"],
                        mass["bounds"],
                        strict=True,
                    ):
                        delta = 1.0 - float(value)
                        mass_prefixes += 1
                        mass_delta_sum += delta
                        mass_delta_le_001 += int(delta <= 0.01)
                        mass_max_delta = max(mass_max_delta, delta)
                        mass_read_error_sum += float(error)
                        mass_max_bound_excess = max(
                            mass_max_bound_excess, float(error) - float(bound)
                        )
                    for arm in ("S32", "K3"):
                        for intervention in INTERVENTIONS:
                            trace = teacher_forced_trace(
                                models["S32"],
                                tokens,
                                evaluated_arm=arm,
                                intervention=intervention,
                            )
                            accumulate_trace(
                                intervention_accumulators[
                                    f"{arm}/{intervention}"
                                ],
                                trace,
                                tokens,
                                family=family,
                                token_names=token_names,
                            )
                if mass_prefixes <= 0 or kl_tokens <= 0:
                    raise AssertionError("locked evaluation cell is empty")
                key = f"{family}/{split}/{seed}"
                cells[key] = {
                    "family": family,
                    "split": split,
                    "seed": seed,
                    "arms": {
                        arm: accumulator.result()
                        for arm, accumulator in accumulators.items()
                    },
                    "kl_s32_to_k3": {
                        "nat_per_token": kl_sum / kl_tokens,
                        "tokens": kl_tokens,
                    },
                    "causal_k3_history": {
                        "prefixes": mass_prefixes,
                        "mean_delta": mass_delta_sum / mass_prefixes,
                        "fraction_delta_le_0_01": (
                            mass_delta_le_001 / mass_prefixes
                        ),
                        "maximum_delta": mass_max_delta,
                        "mean_read_error": mass_read_error_sum / mass_prefixes,
                        "maximum_bound_excess": mass_max_bound_excess,
                    },
                    "interventions": {
                        name: accumulator.result()
                        for name, accumulator in intervention_accumulators.items()
                    },
                }
    return {
        "schema": "cpkv-topk-001-teacher-forced-v1",
        "logical_state_byte_accounting": {
            "payload_vector_fp32": PAYLOAD_VECTOR_BYTES_FP32,
            "persistent_node_parent_source_symbol": PERSISTENT_NODE_BYTES_LOGICAL,
            "configuration_q_pointer_weight_fp64": CONFIGURATION_BYTES_LOGICAL,
            "rnn32_state_fp32": RNN32_STATE_BYTES_FP32,
            "scope": "peak logical auxiliary state per sequence; Python object overhead excluded",
        },
        "cells": cells,
    }


def _median(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite or empty gate values")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _at_least(value: float, threshold: float) -> bool:
    return value + 1e-12 >= threshold


def _at_most(value: float, threshold: float) -> bool:
    return value <= threshold + 1e-12


def _cell(
    teacher_forced: Mapping[str, object], family: str, split: str, seed: int
) -> Mapping[str, object]:
    cells = teacher_forced.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("teacher-forced cell mapping missing")
    value = cells.get(f"{family}/{split}/{seed}")
    if not isinstance(value, dict):
        raise ValueError(f"teacher-forced cell missing: {family}/{split}/{seed}")
    return value


def _finite_metric(record: Mapping[str, object], name: str) -> float:
    value = record.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"metric missing or nonfinite: {name}")
    return float(value)


def teacher_forced_decision(
    teacher_forced: Mapping[str, object],
) -> dict[str, object]:
    gate_cells: dict[str, object] = {}
    overall_pass = True
    for family in ("D1", "A1", "A2"):
        for split in ("E1", "E2"):
            acquisition_checks: dict[str, object] = {}
            for control in CONTROLS:
                nll_gains: list[float] = []
                accuracy_gains: list[float] = []
                early_regressions: list[float] = []
                length_regressions: list[float] = []
                for seed in range(1701, 1711):
                    cell = _cell(teacher_forced, family, split, seed)
                    arms = cell["arms"]
                    stack = arms["S32"]
                    baseline = arms[control]
                    nll_gains.append(
                        _finite_metric(baseline, "token_nll")
                        - _finite_metric(stack, "token_nll")
                    )
                    accuracy_gains.append(
                        _finite_metric(stack, "delayed_accuracy")
                        - _finite_metric(baseline, "delayed_accuracy")
                    )
                    early_regressions.append(
                        _finite_metric(stack, "early_nll")
                        - _finite_metric(baseline, "early_nll")
                    )
                    stack_lengths = stack["early_length_slices"]
                    control_lengths = baseline["early_length_slices"]
                    if set(stack_lengths) != set(control_lengths):
                        raise ValueError("early length-slice mismatch")
                    length_regressions.extend(
                        _finite_metric(stack_lengths[length], "nll")
                        - _finite_metric(control_lengths[length], "nll")
                        for length in stack_lengths
                    )
                passed = (
                    min(nll_gains) > 0.0
                    and _at_least(_median(nll_gains), 0.05)
                    and min(accuracy_gains) > 0.0
                    and _at_least(_median(accuracy_gains), 0.05)
                    and _at_most(max(early_regressions), 0.01)
                    and _at_most(max(length_regressions), 0.01)
                )
                overall_pass = overall_pass and passed
                acquisition_checks[control] = {
                    "pass": passed,
                    "nll_gain_min": min(nll_gains),
                    "nll_gain_median": _median(nll_gains),
                    "delayed_accuracy_gain_min": min(accuracy_gains),
                    "delayed_accuracy_gain_median": _median(accuracy_gains),
                    "maximum_early_nll_regression": max(early_regressions),
                    "maximum_length_slice_nll_regression": max(
                        length_regressions
                    ),
                }

            compression_arm = "K1" if family == "D1" else "K3"
            nll_retentions: list[float] = []
            accuracy_retentions: list[float] = []
            kl_values: list[float] = []
            mass_means: list[float] = []
            mass_fractions: list[float] = []
            bound_excesses: list[float] = []
            intervention_values: dict[str, list[float]] = {
                f"{arm}/{intervention}": []
                for arm in ("S32", "K3")
                for intervention in INTERVENTIONS
            }
            intervention_early_changes: dict[str, list[float]] = {
                key: [] for key in intervention_values
            }
            compression_positive = True
            for seed in range(1701, 1711):
                cell = _cell(teacher_forced, family, split, seed)
                arms = cell["arms"]
                control = min(CONTROLS, key=lambda arm: arms[arm]["token_nll"])
                control_nll = _finite_metric(arms[control], "token_nll")
                control_accuracy = _finite_metric(
                    arms[control], "delayed_accuracy"
                )
                s_nll_gain = control_nll - _finite_metric(arms["S32"], "token_nll")
                k_nll_gain = control_nll - _finite_metric(
                    arms[compression_arm], "token_nll"
                )
                s_accuracy_gain = _finite_metric(
                    arms["S32"], "delayed_accuracy"
                ) - control_accuracy
                k_accuracy_gain = _finite_metric(
                    arms[compression_arm], "delayed_accuracy"
                ) - control_accuracy
                if min(s_nll_gain, k_nll_gain, s_accuracy_gain, k_accuracy_gain) <= 0:
                    compression_positive = False
                    nll_retentions.append(math.nan)
                    accuracy_retentions.append(math.nan)
                else:
                    nll_retentions.append(k_nll_gain / s_nll_gain)
                    accuracy_retentions.append(k_accuracy_gain / s_accuracy_gain)
                kl_values.append(_finite_metric(cell["kl_s32_to_k3"], "nat_per_token"))
                mass = cell["causal_k3_history"]
                mass_means.append(_finite_metric(mass, "mean_delta"))
                mass_fractions.append(
                    _finite_metric(mass, "fraction_delta_le_0_01")
                )
                bound_excesses.append(
                    _finite_metric(mass, "maximum_bound_excess")
                )
                for key in intervention_values:
                    clean_arm, _ = key.split("/")
                    clean = arms[clean_arm]
                    intervened = cell["interventions"][key]
                    denominator = (
                        _finite_metric(clean, "delayed_accuracy")
                        - control_accuracy
                    )
                    if denominator <= 0.0:
                        intervention_values[key].append(math.nan)
                    else:
                        intervention_values[key].append(
                            (
                                _finite_metric(clean, "delayed_accuracy")
                                - _finite_metric(intervened, "delayed_accuracy")
                            )
                            / denominator
                        )
                    intervention_early_changes[key].append(
                        abs(
                            _finite_metric(clean, "early_accuracy")
                            - _finite_metric(intervened, "early_accuracy")
                        )
                    )
            compression_pass = (
                compression_positive
                and all(math.isfinite(value) for value in nll_retentions)
                and all(math.isfinite(value) for value in accuracy_retentions)
                and _at_least(min(nll_retentions), 0.80)
                and _at_least(_median(nll_retentions), 0.90)
                and _at_least(min(accuracy_retentions), 0.80)
                and _at_least(_median(accuracy_retentions), 0.90)
            )
            mass_pass = (
                family == "D1"
                or (
                    _at_most(max(mass_means), 0.01)
                    and _at_least(min(mass_fractions), 0.99)
                    and _at_most(max(bound_excesses), 1e-6)
                )
            )
            kl_mean = math.fsum(kl_values) / len(kl_values)
            kl_pass = _at_most(kl_mean, 0.01) and _at_most(
                max(kl_values), 0.02
            )
            intervention_checks: dict[str, object] = {}
            for key, values in intervention_values.items():
                finite_values = [value for value in values if math.isfinite(value)]
                passed = (
                    all(math.isfinite(value) and value > 0.0 for value in values)
                    and _at_least(min(values), 0.60)
                    and _at_least(_median(values), 0.80)
                    and _at_most(max(intervention_early_changes[key]), 0.01)
                )
                intervention_checks[key] = {
                    "pass": passed,
                    "disappearance_min": (
                        min(finite_values) if finite_values else None
                    ),
                    "disappearance_median": (
                        _median(finite_values) if finite_values else None
                    ),
                    "maximum_early_accuracy_change": max(
                        intervention_early_changes[key]
                    ),
                }
            interventions_pass = all(
                value["pass"] for value in intervention_checks.values()
            )
            cell_pass = (
                all(value["pass"] for value in acquisition_checks.values())
                and compression_pass
                and mass_pass
                and kl_pass
                and interventions_pass
            )
            overall_pass = overall_pass and cell_pass
            gate_cells[f"{family}/{split}"] = {
                "pass": cell_pass,
                "acquisition": acquisition_checks,
                "compression": {
                    "arm": compression_arm,
                    "pass": compression_pass,
                    "nll_retention_min": (
                        min(value for value in nll_retentions if math.isfinite(value))
                        if any(math.isfinite(value) for value in nll_retentions)
                        else None
                    ),
                    "nll_retention_median": (
                        _median(
                            [value for value in nll_retentions if math.isfinite(value)]
                        )
                        if any(math.isfinite(value) for value in nll_retentions)
                        else None
                    ),
                    "accuracy_retention_min": (
                        min(
                            value
                            for value in accuracy_retentions
                            if math.isfinite(value)
                        )
                        if any(math.isfinite(value) for value in accuracy_retentions)
                        else None
                    ),
                    "accuracy_retention_median": (
                        _median(
                            [
                                value
                                for value in accuracy_retentions
                                if math.isfinite(value)
                            ]
                        )
                        if any(
                            math.isfinite(value) for value in accuracy_retentions
                        )
                        else None
                    ),
                },
                "causal_history": {
                    "pass": mass_pass,
                    "maximum_mean_delta": max(mass_means),
                    "minimum_fraction_delta_le_0_01": min(mass_fractions),
                    "maximum_bound_excess": max(bound_excesses),
                },
                "recurrent_kl": {
                    "pass": kl_pass,
                    "mean_nat_per_token": kl_mean,
                    "median_nat_per_token": _median(kl_values),
                    "maximum_nat_per_token": max(kl_values),
                },
                "interventions": intervention_checks,
            }
    return {
        "schema": "cpkv-topk-001-teacher-forced-decision-v1",
        "pass": overall_pass,
        "cells": gate_cells,
    }


def _variable_mapping(
    family: str, n: int, parity: int, position: int
) -> tuple[int, str]:
    if family == "D1":
        if position < n:
            return position, "open"
        return 2 * n - 1 - position, "close"
    if family == "A1":
        if position < n:
            return position, "bit"
        if parity and position == n:
            return n, "bit"
        return 2 * n + parity - 1 - position, "bit"
    raise ValueError(f"variable mapping unsupported for {family}")


def _symbol_bit(family: str, role: str, symbol: str) -> int | None:
    if family == "D1":
        mapping = {
            ("open", "open0"): 0,
            ("open", "open1"): 1,
            ("close", "close0"): 0,
            ("close", "close1"): 1,
        }
        return mapping.get((role, symbol))
    if family == "A1" and symbol in {"0", "1"}:
        return int(symbol)
    return None


def _variable_symbol(family: str, role: str, bit: int) -> str:
    if family == "D1":
        return f"{role}{bit}"
    if family == "A1":
        return str(bit)
    raise ValueError(f"variable symbol unsupported for {family}")


def analytic_states(
    family: str,
    prefix_content: Sequence[str],
    *,
    low: int,
    high: int,
) -> list[tuple[float, str, dict[str, float]]]:
    """Return unnormalized latent mass, outcome, and next-token law."""

    states: list[tuple[float, str, dict[str, float]]] = []
    if family in {"D1", "A1"}:
        parities = (0,) if family == "D1" else (0, 1)
        for n in range(low, high + 1):
            for parity in parities:
                full_length = 2 * n + parity
                if len(prefix_content) > full_length:
                    continue
                assigned: dict[int, int] = {}
                consistent = True
                for position, symbol in enumerate(prefix_content):
                    variable, role = _variable_mapping(
                        family, n, parity, position
                    )
                    bit = _symbol_bit(family, role, symbol)
                    if bit is None or (
                        variable in assigned and assigned[variable] != bit
                    ):
                        consistent = False
                        break
                    assigned[variable] = bit
                if not consistent:
                    continue
                mass = 2.0 ** (-len(assigned))
                outcome = f"length:{full_length}"
                if len(prefix_content) == full_length:
                    next_law = {"EOS": 1.0}
                else:
                    variable, role = _variable_mapping(
                        family, n, parity, len(prefix_content)
                    )
                    if variable in assigned:
                        next_law = {
                            _variable_symbol(
                                family, role, assigned[variable]
                            ): 1.0
                        }
                    else:
                        next_law = {
                            _variable_symbol(family, role, 0): 0.5,
                            _variable_symbol(family, role, 1): 0.5,
                        }
                states.append((mass, outcome, next_law))
        return states

    if family == "A2":
        for branch in (0, 1):
            for n in range(low, high + 1):
                for m in range(low, high + 1):
                    counts = (n, n, m) if branch == 0 else (m, n, n)
                    content = (
                        ["a"] * counts[0]
                        + ["b"] * counts[1]
                        + ["c"] * counts[2]
                    )
                    if list(prefix_content) != content[: len(prefix_content)]:
                        continue
                    next_symbol = (
                        "EOS"
                        if len(prefix_content) == len(content)
                        else content[len(prefix_content)]
                    )
                    outcome = f"a:{counts[0]},b:{counts[1]},c:{counts[2]}"
                    states.append((1.0, outcome, {next_symbol: 1.0}))
        return states
    raise ValueError(f"unknown analytic family {family}")


def analytic_next_distribution(
    family: str,
    prefix_content: Sequence[str],
    *,
    low: int,
    high: int,
) -> dict[str, float]:
    states = analytic_states(
        family, prefix_content, low=low, high=high
    )
    denominator = math.fsum(mass for mass, _, _ in states)
    if denominator <= 0.0:
        return {}
    buckets: dict[str, list[float]] = {}
    for mass, _, law in states:
        for symbol, probability in law.items():
            buckets.setdefault(symbol, []).append(mass * probability)
    return {
        symbol: math.fsum(values) / denominator
        for symbol, values in buckets.items()
    }


def analytic_outcome_distribution(
    family: str,
    prefix_content: Sequence[str],
    *,
    low: int,
    high: int,
) -> dict[str, float]:
    states = analytic_states(
        family, prefix_content, low=low, high=high
    )
    buckets: dict[str, list[float]] = {}
    for mass, outcome, _ in states:
        buckets.setdefault(outcome, []).append(mass)
    denominator = math.fsum(
        math.fsum(values) for values in buckets.values()
    )
    if denominator <= 0.0:
        raise ValueError("prompt has no analytic continuation")
    return {
        outcome: math.fsum(values) / denominator
        for outcome, values in buckets.items()
    }


def analytic_map_continuation(
    family: str,
    prefix_content: Sequence[str],
    *,
    low: int,
    high: int,
    horizon: int,
) -> list[str]:
    candidates: dict[tuple[str, ...], float] = {}
    if family in {"D1", "A1"}:
        parities = (0,) if family == "D1" else (0, 1)
        for n in range(low, high + 1):
            for parity in parities:
                full_length = 2 * n + parity
                if len(prefix_content) > full_length:
                    continue
                assigned: dict[int, int] = {}
                consistent = True
                for position, symbol in enumerate(prefix_content):
                    variable, role = _variable_mapping(
                        family, n, parity, position
                    )
                    bit = _symbol_bit(family, role, symbol)
                    if bit is None or (
                        variable in assigned and assigned[variable] != bit
                    ):
                        consistent = False
                        break
                    assigned[variable] = bit
                if not consistent:
                    continue
                full: list[str] = []
                for position in range(full_length):
                    variable, role = _variable_mapping(
                        family, n, parity, position
                    )
                    full.append(
                        _variable_symbol(
                            family, role, assigned.get(variable, 0)
                        )
                    )
                continuation = tuple(full[len(prefix_content) :] + ["EOS"])
                latent_width = n if family == "D1" else n + parity
                candidates[continuation] = 2.0 ** (-latent_width)
    elif family == "A2":
        multiplicities: dict[tuple[str, ...], float] = {}
        for branch in (0, 1):
            for n in range(low, high + 1):
                for m in range(low, high + 1):
                    counts = (n, n, m) if branch == 0 else (m, n, n)
                    full = tuple(
                        ["a"] * counts[0]
                        + ["b"] * counts[1]
                        + ["c"] * counts[2]
                    )
                    if tuple(prefix_content) != full[: len(prefix_content)]:
                        continue
                    continuation = full[len(prefix_content) :] + ("EOS",)
                    multiplicities[continuation] = (
                        multiplicities.get(continuation, 0.0) + 1.0
                    )
        candidates = multiplicities
    else:
        raise ValueError(f"unknown analytic family {family}")
    if not candidates:
        return []
    maximum = max(candidates.values())
    result = list(min(key for key, value in candidates.items() if value == maximum))
    if len(result) > horizon:
        raise AssertionError("analytic MAP exceeds frozen generation horizon")
    return result


def _prompt(tokens: Sequence[int], family: str, token_names: Sequence[str]) -> list[int]:
    positions = delayed_positions(family, tokens, token_names)
    if not positions:
        raise ValueError("sequence has no delayed position")
    return list(tokens[: min(positions) + 1])


def model_continuation(
    model: AcquisitionModel,
    evaluated_arm: str,
    prompt: Sequence[int],
    *,
    eos_id: int,
    token_names: Sequence[str],
    uniforms: Sequence[float] | None,
    horizon: int,
) -> list[int]:
    decoder = IncrementalDecoder(model, evaluated_arm=evaluated_arm)
    logits: Tensor | None = None
    for token in prompt:
        logits = decoder.step(token)
    if logits is None:
        raise ValueError("empty continuation prompt")
    continuation: list[int] = []
    for step in range(horizon):
        if uniforms is None:
            maximum = float(torch.max(logits))
            tied = [
                index
                for index, value in enumerate(logits)
                if float(value) == maximum
            ]
            token = min(tied, key=lambda index: token_names[index])
        else:
            probabilities = torch.softmax(logits, dim=0).numpy()
            cumulative = np.cumsum(probabilities, dtype=np.float64)
            token = int(np.searchsorted(cumulative, uniforms[step], side="right"))
            token = min(token, len(token_names) - 1)
        continuation.append(token)
        if token == eos_id:
            return continuation
        logits = decoder.step(token)
    return continuation


def completion_outcome(
    family: str,
    full_tokens: Sequence[str],
    *,
    low: int,
    high: int,
) -> tuple[bool, str | None, int | None]:
    if not full_tokens or full_tokens[-1] != "EOS":
        return False, None, None
    content = list(full_tokens[:-1])
    if family == "D1":
        if len(content) % 2:
            return False, None, len(content)
        n = len(content) // 2
        valid = (
            low <= n <= high
            and all(value in {"open0", "open1"} for value in content[:n])
            and content[n:]
            == [
                "close0" if value == "open0" else "close1"
                for value in reversed(content[:n])
            ]
        )
        return valid, f"length:{len(content)}" if valid else None, len(content)
    if family == "A1":
        n = len(content) // 2
        valid = (
            low <= n <= high
            and len(content) in {2 * n, 2 * n + 1}
            and all(value in {"0", "1"} for value in content)
            and content == list(reversed(content))
        )
        return valid, f"length:{len(content)}" if valid else None, len(content)
    if family == "A2":
        a_count = 0
        while a_count < len(content) and content[a_count] == "a":
            a_count += 1
        b_end = a_count
        while b_end < len(content) and content[b_end] == "b":
            b_end += 1
        c_end = b_end
        while c_end < len(content) and content[c_end] == "c":
            c_end += 1
        b_count = b_end - a_count
        c_count = c_end - b_end
        valid = (
            c_end == len(content)
            and min(a_count, b_count, c_count) >= low
            and max(a_count, b_count, c_count) <= high
            and (a_count == b_count or b_count == c_count)
        )
        outcome = f"a:{a_count},b:{b_count},c:{c_count}" if valid else None
        return valid, outcome, len(content)
    raise ValueError(f"unknown completion family {family}")


def _continuation_horizon(family: str, high: int) -> int:
    maximum_content = 2 * high if family == "D1" else 2 * high + 1
    if family == "A2":
        maximum_content = 3 * high
    return maximum_content + 1


def _total_variation(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    keys = set(left) | set(right)
    return 0.5 * math.fsum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def evaluate_continuations(
    *,
    project_root: Path,
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
    teacher_forced: Mapping[str, object],
    requested_cells: set[str] | None = None,
) -> dict[str, object]:
    token_maps = manifest["token_maps"]
    protocol = manifest["training_protocol"]
    fixed_splits = manifest["fixed_splits"]
    if requested_cells is not None and not requested_cells.issubset(
        EVALUATION_CELL_KEYS
    ):
        raise ValueError("unknown requested continuation cell")
    cells: dict[str, object] = {}
    for family in ("D1", "A1", "A2"):
        token_names = [str(value) for value in token_maps[family]]
        eos_id = token_names.index("EOS")
        for split in ("E1", "E2"):
            selected_seeds = [
                int(seed)
                for seed in protocol["model_seeds"]
                if requested_cells is None
                or f"{family}/{split}/{int(seed)}" in requested_cells
            ]
            if not selected_seeds:
                continue
            split_record = fixed_splits[f"{family}/{split}"]
            low, high = [int(value) for value in split_record["range"]]
            horizon = _continuation_horizon(family, high)
            sequences = _load_evaluation_sequences(
                project_root=project_root,
                manifest=manifest,
                family=family,
                split=split,
            )
            prompts = [_prompt(tokens, family, token_names) for tokens in sequences]
            generator = np.random.Generator(np.random.PCG64DXSM(9401))
            paired_uniforms = generator.random((len(prompts), horizon))
            analytic_records = []
            for prompt in prompts:
                content = [token_names[token] for token in prompt[1:]]
                outcomes = analytic_outcome_distribution(
                    family, content, low=low, high=high
                )
                legal_lengths = [
                    int(outcome.split(":", 1)[1])
                    if family != "A2"
                    else sum(
                        int(part.split(":", 1)[1])
                        for part in outcome.split(",")
                    )
                    for outcome in outcomes
                ]
                analytic_records.append(
                    {
                        "outcomes": outcomes,
                        "minimum_length": min(legal_lengths),
                        "maximum_length": max(legal_lengths),
                        "map": analytic_map_continuation(
                            family,
                            content,
                            low=low,
                            high=high,
                            horizon=horizon,
                        ),
                    }
                )
            for seed in selected_seeds:
                teacher_cell = _cell(teacher_forced, family, split, seed)
                teacher_arms = teacher_cell["arms"]
                models = {
                    arm: _load_model(
                        project_root=project_root,
                        manifest=manifest,
                        receipt=receipt,
                        family=family,
                        arm=arm,
                        seed=seed,
                    )
                    for arm in BASE_ARMS
                }
                arm_results: dict[str, object] = {}
                for arm in EVALUATED_ARMS:
                    model = models["S32"] if arm in COMPILED_ARMS else models[arm]
                    valid = premature = late = no_eos = 0
                    greedy_valid = greedy_map = 0
                    strata_model: dict[str, dict[str, int]] = {}
                    strata_analytic: dict[str, dict[str, float]] = {}
                    strata_counts: dict[str, int] = {}
                    for index, (prompt, analytic) in enumerate(
                        zip(prompts, analytic_records, strict=True)
                    ):
                        stratum = (
                            f"prefix:{len(prompt)-1}/last:{token_names[prompt[-1]]}"
                        )
                        strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
                        expected = strata_analytic.setdefault(stratum, {})
                        for outcome, probability in analytic["outcomes"].items():
                            expected[outcome] = expected.get(outcome, 0.0) + probability
                        sampled = model_continuation(
                            model,
                            arm,
                            prompt,
                            eos_id=eos_id,
                            token_names=token_names,
                            uniforms=paired_uniforms[index],
                            horizon=horizon,
                        )
                        sampled_names = [token_names[token] for token in sampled]
                        full_names = [token_names[token] for token in prompt[1:]] + sampled_names
                        is_valid, outcome, length = completion_outcome(
                            family, full_names, low=low, high=high
                        )
                        valid += int(is_valid)
                        no_eos += int(not sampled_names or sampled_names[-1] != "EOS")
                        if length is None:
                            late += int(not sampled_names or sampled_names[-1] != "EOS")
                        else:
                            premature += int(length < analytic["minimum_length"])
                            late += int(length > analytic["maximum_length"])
                        observed_outcome = outcome if outcome is not None else "__INVALID__"
                        observed = strata_model.setdefault(stratum, {})
                        observed[observed_outcome] = observed.get(observed_outcome, 0) + 1

                        greedy = model_continuation(
                            model,
                            arm,
                            prompt,
                            eos_id=eos_id,
                            token_names=token_names,
                            uniforms=None,
                            horizon=horizon,
                        )
                        greedy_names = [token_names[token] for token in greedy]
                        greedy_full = [token_names[token] for token in prompt[1:]] + greedy_names
                        greedy_is_valid, _, _ = completion_outcome(
                            family, greedy_full, low=low, high=high
                        )
                        greedy_valid += int(greedy_is_valid)
                        greedy_map += int(greedy_names == analytic["map"])
                    televisions: list[float] = []
                    for stratum, count in strata_counts.items():
                        observed = {
                            outcome: value / count
                            for outcome, value in strata_model[stratum].items()
                        }
                        expected = {
                            outcome: value / count
                            for outcome, value in strata_analytic[stratum].items()
                        }
                        televisions.append(_total_variation(observed, expected))
                    total = len(prompts)
                    arm_results[arm] = {
                        "paired_samples": total,
                        "grammar_valid_rate": valid / total,
                        "premature_eos_rate": premature / total,
                        "late_eos_rate": late / total,
                        "no_eos_rate": no_eos / total,
                        "stratified_outcome_tv": math.fsum(televisions)
                        / len(televisions),
                        "greedy_grammar_valid_rate": greedy_valid / total,
                        "greedy_analytic_map_rate": greedy_map / total,
                        "conditional_suffix_nll": _finite_metric(
                            teacher_arms[arm], "conditional_suffix_nll"
                        ),
                        "strata": len(televisions),
                    }
                cells[f"{family}/{split}/{seed}"] = {
                    "family": family,
                    "split": split,
                    "seed": seed,
                    "arms": arm_results,
                }
    return {
        "schema": "cpkv-topk-001-continuations-v1",
        "sampling_seed": 9401,
        "paired_samples_per_cell": 10_000,
        "cells": cells,
    }


def continuation_decision(continuations: Mapping[str, object]) -> dict[str, object]:
    cells = continuations.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("continuation cells missing")
    metric_directions = {
        "grammar_valid_rate": 1.0,
        "premature_eos_rate": -1.0,
        "late_eos_rate": -1.0,
        "no_eos_rate": -1.0,
        "stratified_outcome_tv": -1.0,
        "greedy_grammar_valid_rate": 1.0,
        "greedy_analytic_map_rate": 1.0,
        "conditional_suffix_nll": -1.0,
    }
    decision_cells: dict[str, object] = {}
    overall_pass = True
    for family in ("D1", "A1", "A2"):
        for split in ("E1", "E2"):
            metric_records: dict[str, object] = {}
            for metric, direction in metric_directions.items():
                retentions: list[float] = []
                for seed in range(1701, 1711):
                    cell = cells[f"{family}/{split}/{seed}"]
                    arms = cell["arms"]
                    control_value = max(
                        direction * _finite_metric(arms[control], metric)
                        for control in CONTROLS
                    )
                    s_value = direction * _finite_metric(arms["S32"], metric)
                    k_value = direction * _finite_metric(arms["K3"], metric)
                    denominator = s_value - control_value
                    numerator = k_value - control_value
                    retentions.append(
                        numerator / denominator
                        if denominator > 0.0 and numerator > 0.0
                        else math.nan
                    )
                passed = (
                    all(math.isfinite(value) for value in retentions)
                    and _at_least(min(retentions), 0.80)
                    and _at_least(_median(retentions), 0.90)
                )
                overall_pass = overall_pass and passed
                finite = [value for value in retentions if math.isfinite(value)]
                metric_records[metric] = {
                    "pass": passed,
                    "minimum_retention": min(finite) if finite else None,
                    "median_retention": _median(finite) if finite else None,
                }
            decision_cells[f"{family}/{split}"] = {
                "pass": all(value["pass"] for value in metric_records.values()),
                "metrics": metric_records,
            }
    return {
        "schema": "cpkv-topk-001-continuation-decision-v1",
        "pass": overall_pass,
        "cells": decision_cells,
    }


def build_evaluation_shard(
    *,
    project_root: Path,
    manifest_path: Path,
    receipt_path: Path,
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
    shard_index: int,
) -> dict[str, object]:
    if not 0 <= shard_index < len(EVALUATION_CELL_KEYS):
        raise ValueError("shard index must be in [0, 59]")
    cell_key = EVALUATION_CELL_KEYS[shard_index]
    requested = {cell_key}
    teacher_forced = evaluate_teacher_forced(
        project_root=project_root,
        manifest=manifest,
        receipt=receipt,
        requested_cells=requested,
    )
    continuations = evaluate_continuations(
        project_root=project_root,
        manifest=manifest,
        receipt=receipt,
        teacher_forced=teacher_forced,
        requested_cells=requested,
    )
    if set(teacher_forced["cells"]) != requested or set(
        continuations["cells"]
    ) != requested:
        raise AssertionError("evaluation shard did not produce exactly one cell")
    return {
        "schema": SHARD_SCHEMA,
        "lineage": LINEAGE,
        "status": "complete",
        "shard_index": shard_index,
        "cell_key": cell_key,
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_freeze_receipt_sha256": _sha256(receipt_path),
        "locked_evaluator_sha256": _sha256(Path(__file__).resolve()),
        "teacher_forced": teacher_forced,
        "continuations": continuations,
    }


def merge_evaluation_shard_records(
    records: Sequence[Mapping[str, object]],
    *,
    manifest_sha256: str,
    receipt_sha256: str,
    locked_evaluator_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if len(records) != len(EVALUATION_CELL_KEYS):
        raise ValueError("locked evaluation requires exactly 60 shards")
    teacher_cells: dict[str, object] = {}
    continuation_cells: dict[str, object] = {}
    seen_indices: set[int] = set()
    for record in records:
        index = record.get("shard_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("invalid shard index")
        if not 0 <= index < len(EVALUATION_CELL_KEYS) or index in seen_indices:
            raise ValueError("duplicate or out-of-range shard index")
        seen_indices.add(index)
        cell_key = EVALUATION_CELL_KEYS[index]
        if (
            record.get("schema") != SHARD_SCHEMA
            or record.get("lineage") != LINEAGE
            or record.get("status") != "complete"
            or record.get("cell_key") != cell_key
            or record.get("manifest_sha256") != manifest_sha256
            or record.get("checkpoint_freeze_receipt_sha256") != receipt_sha256
            or record.get("locked_evaluator_sha256") != locked_evaluator_sha256
        ):
            raise ValueError(f"shard binding mismatch: {index}")
        teacher = record.get("teacher_forced")
        continuation = record.get("continuations")
        if not isinstance(teacher, dict) or not isinstance(continuation, dict):
            raise ValueError(f"shard payload missing: {index}")
        raw_teacher_cells = teacher.get("cells")
        raw_continuation_cells = continuation.get("cells")
        if (
            not isinstance(raw_teacher_cells, dict)
            or set(raw_teacher_cells) != {cell_key}
            or not isinstance(raw_continuation_cells, dict)
            or set(raw_continuation_cells) != {cell_key}
        ):
            raise ValueError(f"shard cell payload mismatch: {index}")
        teacher_cells[cell_key] = raw_teacher_cells[cell_key]
        continuation_cells[cell_key] = raw_continuation_cells[cell_key]
    if seen_indices != set(range(len(EVALUATION_CELL_KEYS))):
        raise ValueError("missing locked-evaluation shard")
    ordered_teacher = {
        key: teacher_cells[key] for key in EVALUATION_CELL_KEYS
    }
    ordered_continuations = {
        key: continuation_cells[key] for key in EVALUATION_CELL_KEYS
    }
    return (
        {
            "schema": "cpkv-topk-001-teacher-forced-v1",
            "logical_state_byte_accounting": {
                "payload_vector_fp32": PAYLOAD_VECTOR_BYTES_FP32,
                "persistent_node_parent_source_symbol": PERSISTENT_NODE_BYTES_LOGICAL,
                "configuration_q_pointer_weight_fp64": CONFIGURATION_BYTES_LOGICAL,
                "rnn32_state_fp32": RNN32_STATE_BYTES_FP32,
                "scope": "peak logical auxiliary state per sequence; Python object overhead excluded",
            },
            "cells": ordered_teacher,
        },
        {
            "schema": "cpkv-topk-001-continuations-v1",
            "sampling_seed": 9401,
            "paired_samples_per_cell": 10_000,
            "cells": ordered_continuations,
        },
    )


def merge_evaluation_shards(
    *,
    project_root: Path,
    shard_directory: Path,
    manifest_path: Path,
    receipt_path: Path,
    materialization: Mapping[str, object],
) -> dict[str, object]:
    root = project_root.resolve()
    directory = shard_directory.resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError("shard directory is absent or escapes project root")
    expected_names = {
        f"shard-{index:02d}.json" for index in range(len(EVALUATION_CELL_KEYS))
    }
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)[:5]
        extra = sorted(actual_names - expected_names)[:5]
        raise ValueError(f"shard filename set mismatch missing={missing} extra={extra}")
    records = [
        _read_json(directory / f"shard-{index:02d}.json")
        for index in range(len(EVALUATION_CELL_KEYS))
    ]
    manifest_hash = _sha256(manifest_path)
    receipt_hash = _sha256(receipt_path)
    evaluator_hash = _sha256(Path(__file__).resolve())
    teacher_forced, continuations = merge_evaluation_shard_records(
        records,
        manifest_sha256=manifest_hash,
        receipt_sha256=receipt_hash,
        locked_evaluator_sha256=evaluator_hash,
    )
    teacher_decision = teacher_forced_decision(teacher_forced)
    continuation_gate = continuation_decision(continuations)
    passed = bool(teacher_decision["pass"] and continuation_gate["pass"])
    return {
        "schema": FINAL_SCHEMA,
        "lineage": LINEAGE,
        "status": "pass" if passed else "fail",
        "manifest_sha256": manifest_hash,
        "checkpoint_freeze_receipt_sha256": receipt_hash,
        "locked_evaluator_sha256": evaluator_hash,
        "materialization": materialization,
        "shard_hashes": {
            f"shard-{index:02d}.json": _sha256(
                directory / f"shard-{index:02d}.json"
            )
            for index in range(len(EVALUATION_CELL_KEYS))
        },
        "teacher_forced": teacher_forced,
        "teacher_forced_decision": teacher_decision,
        "continuations": continuations,
        "continuation_decision": continuation_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-freeze-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shard-index", type=int)
    mode.add_argument("--merge-shards", type=Path)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    project_root = args.project_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    receipt_path = args.checkpoint_freeze_receipt
    if not receipt_path.is_absolute():
        receipt_path = project_root / receipt_path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path = output_path.resolve()
    if not output_path.is_relative_to(project_root):
        raise ValueError("locked evaluation output escapes project root")
    if output_path.exists():
        raise FileExistsError("locked evaluation output already exists")
    materialization = materialize(
        project_root=project_root,
        manifest_path=manifest_path,
        freeze_receipt_path=receipt_path,
    )
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    if args.shard_index is not None:
        result = build_evaluation_shard(
            project_root=project_root,
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            manifest=manifest,
            receipt=receipt,
            shard_index=args.shard_index,
        )
    else:
        assert args.merge_shards is not None
        shard_directory = args.merge_shards
        if not shard_directory.is_absolute():
            shard_directory = project_root / shard_directory
        result = merge_evaluation_shards(
            project_root=project_root,
            shard_directory=shard_directory,
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            materialization=materialization,
        )
    _write_once_text(
        output_path, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "lineage": LINEAGE,
                "status": result["status"],
                "output": output_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
