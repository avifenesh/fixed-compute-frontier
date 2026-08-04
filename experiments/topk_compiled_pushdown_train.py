#!/usr/bin/env python3
"""Frozen CPU trainer for the CPKV-TOPK-001 acquisition experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from experiments.topk_compiled_pushdown_reference import ACTIONS, _atomic_write_text
except ModuleNotFoundError:  # Direct `python experiments/...py` execution.
    from topk_compiled_pushdown_reference import (  # type: ignore[no-redef]
        ACTIONS,
        _atomic_write_text,
    )


LINEAGE = "CPKV-TOPK-001"
ARMS = ("S32", "DIRECT3", "HARD3", "MLP", "RNN32", "LINK32")
STACK_ARMS = {"S32", "DIRECT3", "HARD3", "LINK32"}
BEAM_BY_ARM = {"S32": 32, "DIRECT3": 3, "LINK32": 32}
HIDDEN = 64
N_ACTION = 15
N_CONTROL = 3
N_SYMBOL = 3
EPOCHS = 20
BATCH_SIZE = 64
TRAIN_EXAMPLES = 20_000
STEPS_PER_EPOCH = math.ceil(TRAIN_EXAMPLES / BATCH_SIZE)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = int(0.05 * TOTAL_STEPS)

StackItem = tuple[int, int]
StackSig = tuple[StackItem, ...]
ConfigKey = tuple[int, StackSig]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(path: Path) -> str:
    return sha256_file(path)


def parameter_seed(model_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{model_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def initialized_array(
    *, model_seed: int, name: str, shape: tuple[int, ...], kind: str
) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64DXSM(parameter_seed(model_seed, name)))
    if kind == "normal":
        array = generator.normal(0.0, 0.02, size=shape)
    elif kind == "xavier":
        if len(shape) != 2:
            raise ValueError(f"Xavier parameter is not a matrix: {name}/{shape}")
        bound = math.sqrt(6.0 / (shape[0] + shape[1]))
        array = generator.uniform(-bound, bound, size=shape)
    elif kind == "zero":
        array = np.zeros(shape)
    elif kind == "one":
        array = np.ones(shape)
    else:
        raise ValueError(f"unknown initializer {kind}")
    return np.asarray(array, dtype=np.float32)


def make_parameter(
    *, model_seed: int, name: str, shape: tuple[int, ...], kind: str
) -> nn.Parameter:
    return nn.Parameter(torch.from_numpy(initialized_array(
        model_seed=model_seed, name=name, shape=shape, kind=kind
    )).clone())


def parameter_shapes(vocabulary_size: int, arm: str) -> dict[str, tuple[int, ...]]:
    common = {
        "E": (vocabulary_size, HIDDEN),
        "W_e": (HIDDEN, HIDDEN),
        "W_h": (HIDDEN, HIDDEN),
        "W_r": (HIDDEN, HIDDEN),
        "b_h": (HIDDEN,),
        "W_y": (vocabulary_size, HIDDEN),
        "U_y": (vocabulary_size, HIDDEN),
        "b_y": (vocabulary_size,),
    }
    if arm in STACK_ARMS:
        module = {
            "W_a": (N_ACTION, HIDDEN),
            "T": (N_CONTROL, N_SYMBOL + 1, N_ACTION),
            "W_v": (HIDDEN, HIDDEN),
        }
    elif arm == "MLP":
        module = {
            "W1": (40, HIDDEN),
            "W2": (HIDDEN, 40),
            "b1": (40,),
            "b2": (HIDDEN,),
            "gains": (12,),
        }
    elif arm == "RNN32":
        module = {
            "A": (32, HIDDEN),
            "B": (32, 32),
            "C": (HIDDEN, 32),
            "b_s": (32,),
            "b_r": (HIDDEN,),
            "gains": (20,),
        }
    else:
        raise ValueError(f"unknown arm {arm}")
    return {**common, **module}


class AcquisitionModel(nn.Module):
    def __init__(self, *, vocabulary_size: int, arm: str, model_seed: int) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.vocabulary_size = vocabulary_size
        self.arm = arm
        self.model_seed = model_seed
        shapes = parameter_shapes(vocabulary_size, arm)
        for name, shape in shapes.items():
            if name in {"E", "T"}:
                kind = "normal"
            elif name == "gains":
                kind = "one"
            elif len(shape) == 1:
                kind = "zero"
            else:
                kind = "xavier"
            setattr(
                self,
                name,
                make_parameter(
                    model_seed=model_seed,
                    name=name,
                    shape=shape,
                    kind=kind,
                ),
            )
        expected = 193 * vocabulary_size + 17_588
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != expected:
            raise AssertionError(f"parameter count {actual} != {expected}")

    def common_hidden(self, token: int, h: Tensor, r: Tensor) -> Tensor:
        embedding = self.E[token]
        return torch.tanh(
            self.W_e @ embedding + self.W_h @ h + self.W_r @ r + self.b_h
        )

    def output_logits(self, h: Tensor, r: Tensor) -> Tensor:
        return self.W_y @ h + self.U_y @ r + self.b_y

    def transition_probabilities(
        self, base_logits: Tensor, q: int, top_symbol: int
    ) -> tuple[tuple[int, ...], Tensor]:
        top_index = N_SYMBOL if top_symbol < 0 else top_symbol
        logits = base_logits + self.T[q, top_index]
        valid = tuple(
            index
            for index, action in enumerate(ACTIONS)
            if not (top_symbol < 0 and action.kind == "pop")
        )
        index_tensor = torch.tensor(valid, dtype=torch.long)
        return valid, torch.softmax(logits[index_tensor], dim=0)

    @staticmethod
    def select_beam(
        candidates: Mapping[ConfigKey, Tensor], beam_size: int
    ) -> dict[ConfigKey, Tensor]:
        keys = list(candidates)
        structural_order = sorted(range(len(keys)), key=keys.__getitem__)
        detached = torch.stack(
            [candidates[key].detach() for key in keys]
        ).numpy()
        ranked_positions = np.argsort(
            -detached[structural_order], kind="stable"
        )[:beam_size]
        selected_indices = [structural_order[int(index)] for index in ranked_positions]
        selected = torch.stack([candidates[keys[index]] for index in selected_indices])
        selected = selected / selected.sum()
        return {
            keys[candidate_index]: selected[result_index]
            for result_index, candidate_index in enumerate(selected_indices)
        }

    def _beam_expansion(
        self,
        *,
        state: Mapping[ConfigKey, Tensor],
        h: Tensor,
        time_index: int,
        chronological_parent: StackSig | None,
    ) -> tuple[list[ConfigKey], list[int], Tensor, set[StackSig]]:
        state_items = list(state.items())
        candidate_keys: list[ConfigKey] = []
        pushed: set[StackSig] = set()
        base_logits = self.W_a @ h
        combo_positions: dict[tuple[int, int], int] = {}
        combos: list[tuple[int, int]] = []
        state_combo_indices: list[int] = []
        for (q, stack), _ in state_items:
            top_symbol = stack[-1][0] if stack else -1
            combo = (q, top_symbol)
            combo_index = combo_positions.get(combo)
            if combo_index is None:
                combo_index = len(combos)
                combo_positions[combo] = combo_index
                combos.append(combo)
            state_combo_indices.append(combo_index)
        combo_q = torch.tensor([q for q, _ in combos], dtype=torch.long)
        combo_top = torch.tensor(
            [N_SYMBOL if top < 0 else top for _, top in combos], dtype=torch.long
        )
        combo_logits = base_logits.unsqueeze(0) + self.T[combo_q, combo_top]
        combo_valid = torch.ones(
            (len(combos), N_ACTION), dtype=torch.bool
        )
        pop_indices = [
            index for index, action in enumerate(ACTIONS) if action.kind == "pop"
        ]
        for combo_index, (_, top_symbol) in enumerate(combos):
            if top_symbol < 0:
                combo_valid[combo_index, pop_indices] = False
        combo_probabilities = torch.softmax(
            combo_logits.masked_fill(~combo_valid, -torch.inf), dim=1
        )
        state_combo = torch.tensor(state_combo_indices, dtype=torch.long)
        state_probabilities = combo_probabilities[state_combo]
        state_valid = combo_valid[state_combo]
        parent_weights = torch.stack([weight for _, weight in state_items])
        contributions = (parent_weights.unsqueeze(1) * state_probabilities)[
            state_valid
        ]
        for (q, stack), _ in state_items:
            top_symbol = stack[-1][0] if stack else -1
            for action_index, action in enumerate(ACTIONS):
                if top_symbol < 0 and action.kind == "pop":
                    continue
                if action.kind == "push":
                    assert action.symbol is not None
                    parent = chronological_parent if self.arm == "LINK32" else stack
                    assert parent is not None
                    next_stack = parent + ((action.symbol, time_index),)
                    pushed.add(next_stack)
                elif action.kind == "pop":
                    if not stack:
                        raise AssertionError("root pop escaped mask")
                    next_stack = stack[:-1]
                else:
                    next_stack = stack
                candidate_keys.append((action.next_q, next_stack))
        merge_positions: dict[ConfigKey, int] = {}
        merged_keys: list[ConfigKey] = []
        merge_indices: list[int] = []
        for key in candidate_keys:
            position = merge_positions.get(key)
            if position is None:
                position = len(merged_keys)
                merge_positions[key] = position
                merged_keys.append(key)
            merge_indices.append(position)
        return merged_keys, merge_indices, contributions, pushed

    def expand_beam(
        self,
        *,
        state: Mapping[ConfigKey, Tensor],
        h: Tensor,
        time_index: int,
        chronological_parent: StackSig | None,
    ) -> tuple[dict[ConfigKey, Tensor], set[StackSig]]:
        merged_keys, merge_indices, contributions, pushed = self._beam_expansion(
            state=state,
            h=h,
            time_index=time_index,
            chronological_parent=chronological_parent,
        )
        merged_weights = contributions.new_zeros(len(merged_keys)).index_add(
            0, torch.tensor(merge_indices, dtype=torch.long), contributions
        )
        candidates = {
            key: merged_weights[index] for index, key in enumerate(merged_keys)
        }
        return candidates, pushed

    def expand_and_select_beam(
        self,
        *,
        state: Mapping[ConfigKey, Tensor],
        h: Tensor,
        time_index: int,
        chronological_parent: StackSig | None,
        beam_size: int,
    ) -> tuple[dict[ConfigKey, Tensor], set[StackSig]]:
        merged_keys, merge_indices, contributions, pushed = self._beam_expansion(
            state=state,
            h=h,
            time_index=time_index,
            chronological_parent=chronological_parent,
        )
        with torch.no_grad():
            merged_weights = contributions.new_zeros(len(merged_keys)).index_add(
                0,
                torch.tensor(merge_indices, dtype=torch.long),
                contributions,
            )
            structural_order = sorted(
                range(len(merged_keys)), key=merged_keys.__getitem__
            )
            ranked_positions = np.argsort(
                -merged_weights.numpy()[structural_order], kind="stable"
            )[:beam_size]
            selected_indices = [
                structural_order[int(index)] for index in ranked_positions
            ]
        selected_positions = {
            merge_index: result_index
            for result_index, merge_index in enumerate(selected_indices)
        }
        contribution_indices: list[int] = []
        selected_merge_indices: list[int] = []
        for contribution_index, merge_index in enumerate(merge_indices):
            selected_position = selected_positions.get(merge_index)
            if selected_position is not None:
                contribution_indices.append(contribution_index)
                selected_merge_indices.append(selected_position)
        selected = contributions.new_zeros(len(selected_indices)).index_add(
            0,
            torch.tensor(selected_merge_indices, dtype=torch.long),
            contributions[torch.tensor(contribution_indices, dtype=torch.long)],
        )
        selected = selected / selected.sum()
        return {
            merged_keys[merge_index]: selected[result_index]
            for result_index, merge_index in enumerate(selected_indices)
        }, pushed

    @staticmethod
    def next_chronological_parent(
        *,
        state: Mapping[ConfigKey, Tensor],
        pushed: set[StackSig],
        current: StackSig,
    ) -> StackSig:
        retained_push_keys = sorted(key for key in state if key[1] in pushed)
        allocated: set[StackSig] = set()
        for _, stack in retained_push_keys:
            if stack not in allocated:
                current = stack
                allocated.add(stack)
        return current

    def run_weighted_stack(self, tokens: Sequence[int], beam_size: int) -> Tensor:
        zero = self.E.new_zeros(HIDDEN)
        h = zero
        r = zero
        state: dict[ConfigKey, Tensor] = {(0, ()): self.E.new_tensor(1.0)}
        payloads: dict[int, Tensor] = {}
        chronological_parent: StackSig = ()
        logits_by_time: list[Tensor] = []
        for time_index, token in enumerate(tokens[:-1]):
            h = self.common_hidden(token, h, r)
            raw_payload = self.W_v @ h
            payloads[time_index] = raw_payload / torch.clamp(
                torch.linalg.vector_norm(raw_payload), min=1.0
            )
            state, pushed = self.expand_and_select_beam(
                state=state,
                h=h,
                time_index=time_index,
                chronological_parent=chronological_parent,
                beam_size=beam_size,
            )
            # LINK32 uses this pointer; the other weighted-stack arms compute
            # the identical allocation chronology as a discarded control.
            chronological_parent = self.next_chronological_parent(
                state=state,
                pushed=pushed,
                current=chronological_parent,
            )
            read_weights = torch.stack(list(state.values()))
            read_payloads = torch.stack(
                [
                    payloads[stack[-1][1]] if stack else zero
                    for _, stack in state
                ]
            )
            r = (read_weights.unsqueeze(1) * read_payloads).sum(dim=0)
            logits_by_time.append(self.output_logits(h, r))
        return F.cross_entropy(
            torch.stack(logits_by_time),
            torch.tensor(tokens[1:], dtype=torch.long),
            reduction="sum",
        )

    def run_hard3(self, tokens: Sequence[int]) -> Tensor:
        zero = self.E.new_zeros(HIDDEN)
        h = zero
        r = zero
        lanes: list[tuple[int, StackSig, Tensor]] = [
            (q, (), self.E.new_tensor(1.0 / 3.0)) for q in range(3)
        ]
        payloads: dict[int, Tensor] = {}
        logits_by_time: list[Tensor] = []
        for time_index, token in enumerate(tokens[:-1]):
            h = self.common_hidden(token, h, r)
            raw_payload = self.W_v @ h
            payloads[time_index] = raw_payload / torch.clamp(
                torch.linalg.vector_norm(raw_payload), min=1.0
            )
            next_lanes: list[tuple[int, StackSig, Tensor]] = []
            unnormalized: list[Tensor] = []
            lane_reads: list[Tensor] = []
            probability_cache: dict[
                tuple[int, int], tuple[tuple[int, ...], Tensor]
            ] = {}
            base_logits = self.W_a @ h
            for q, stack, lane_weight in lanes:
                top_symbol = stack[-1][0] if stack else -1
                probability_key = (q, top_symbol)
                cached = probability_cache.get(probability_key)
                if cached is None:
                    cached = self.transition_probabilities(base_logits, q, top_symbol)
                    probability_cache[probability_key] = cached
                valid, probabilities = cached
                hard_local = int(torch.argmax(probabilities.detach()))
                possible_stacks: list[StackSig] = []
                possible_reads: list[Tensor] = []
                for action_index in valid:
                    action = ACTIONS[action_index]
                    if action.kind == "push":
                        assert action.symbol is not None
                        possible_stack = stack + ((action.symbol, time_index),)
                    elif action.kind == "pop":
                        possible_stack = stack[:-1]
                    else:
                        possible_stack = stack
                    possible_stacks.append(possible_stack)
                    possible_reads.append(
                        payloads[possible_stack[-1][1]] if possible_stack else zero
                    )
                next_stack = possible_stacks[hard_local]
                soft_read = torch.stack(possible_reads).mul(
                    probabilities.unsqueeze(1)
                ).sum(dim=0)
                hard_read = possible_reads[hard_local]
                # Hard deterministic state in the forward pass; the backward
                # pass follows the softmax-weighted alternative reads.
                lane_reads.append(hard_read + soft_read - soft_read.detach())
                weight = lane_weight * probabilities[hard_local]
                unnormalized.append(weight)
                next_lanes.append(
                    (ACTIONS[valid[hard_local]].next_q, next_stack, weight)
                )
            denominator = torch.stack(unnormalized).sum()
            lanes = [(q, stack, weight / denominator) for q, stack, weight in next_lanes]
            r = zero
            for (_, _, weight), lane_read in zip(lanes, lane_reads):
                r = r + weight * lane_read
            logits_by_time.append(self.output_logits(h, r))
        return F.cross_entropy(
            torch.stack(logits_by_time),
            torch.tensor(tokens[1:], dtype=torch.long),
            reduction="sum",
        )

    def run_control(self, tokens: Sequence[int]) -> Tensor:
        zero = self.E.new_zeros(HIDDEN)
        h = zero
        r = zero
        s = self.E.new_zeros(32) if self.arm == "RNN32" else None
        logits_by_time: list[Tensor] = []
        for token in tokens[:-1]:
            h = self.common_hidden(token, h, r)
            if self.arm == "MLP":
                r = self.W2 @ torch.tanh(self.W1 @ h + self.b1) + self.b2
                r = torch.cat((r[:12] * self.gains, r[12:]))
            elif self.arm == "RNN32":
                assert s is not None
                s = torch.tanh(self.A @ h + self.B @ s + self.b_s)
                r = self.C @ s + self.b_r
                r = torch.cat((r[:20] * self.gains, r[20:]))
            else:
                raise AssertionError(f"not a control arm: {self.arm}")
            logits_by_time.append(self.output_logits(h, r))
        return F.cross_entropy(
            torch.stack(logits_by_time),
            torch.tensor(tokens[1:], dtype=torch.long),
            reduction="sum",
        )

    def sequence_loss(self, tokens: Sequence[int]) -> Tensor:
        if self.arm in BEAM_BY_ARM:
            return self.run_weighted_stack(tokens, BEAM_BY_ARM[self.arm])
        if self.arm == "HARD3":
            return self.run_hard3(tokens)
        return self.run_control(tokens)


def read_manifest(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("lineage") != LINEAGE:
        raise ValueError("invalid manifest")
    return value


def verify_manifest_sources(
    *, project_root: Path, manifest: Mapping[str, object]
) -> None:
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("manifest source hashes missing")
    required = {
        "candidate",
        "evaluator",
        "evaluator_tests",
        "locked_evaluation",
        "locked_evaluation_tests",
        "preflight",
        "preflight_tests",
        "preregistration",
        "reference",
        "reference_tests",
        "training",
        "training_tests",
    }
    if not required.issubset(source_hashes):
        raise ValueError("manifest source hashes incomplete")
    root = project_root.resolve()
    for label, raw_record in source_hashes.items():
        if not isinstance(raw_record, dict):
            raise ValueError(f"invalid source record: {label}")
        relative = raw_record.get("path")
        expected_hash = raw_record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"source path/hash missing: {label}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"source escapes project root: {label}")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"frozen source hash mismatch: {label}")


def verify_preflight_receipt(
    *, project_root: Path, manifest: Mapping[str, object]
) -> None:
    manifest_sources = manifest.get("source_hashes")
    record = manifest.get("preflight_receipt")
    if not isinstance(manifest_sources, dict) or not isinstance(record, dict):
        raise ValueError("manifest preflight receipt missing")
    relative = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError("manifest preflight path/hash missing")
    root = project_root.resolve()
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or sha256_file(path) != expected_hash
    ):
        raise ValueError("preflight receipt hash mismatch")
    preflight = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(preflight, dict)
        or preflight.get("schema") != "cpkv-topk-001-preflight-v1"
        or preflight.get("lineage") != LINEAGE
        or preflight.get("status") != "pass"
        or preflight.get("seed") != 1701
        or preflight.get("base_sample_count") != 512
        or preflight.get("projection_limit_hours") != 5.7
    ):
        raise ValueError("preflight receipt contract mismatch")
    families = preflight.get("families")
    if not isinstance(families, dict) or set(families) != {"D1", "A1", "A2"}:
        raise ValueError("preflight family ledger mismatch")
    for family, raw in families.items():
        hours = raw.get("projected_all_arm_cpu_hours") if isinstance(raw, dict) else None
        if (
            not isinstance(raw, dict)
            or raw.get("status") != "pass"
            or isinstance(hours, bool)
            or not isinstance(hours, (int, float))
            or not math.isfinite(float(hours))
            or float(hours) > 5.7
            or raw.get("projection_limit_hours") != 5.7
            or raw.get("actual_freeze_limit_hours") != 6.0
        ):
            raise ValueError(f"preflight resource gate failed: {family}")
    preflight_sources = preflight.get("source_hashes")
    required = {"preflight", "preflight_tests", "training", "locked_evaluation"}
    if not isinstance(preflight_sources, dict) or set(preflight_sources) != required:
        raise ValueError("preflight source seal mismatch")
    for label in required:
        if preflight_sources[label] != manifest_sources.get(label):
            raise ValueError(f"preflight source binding mismatch: {label}")


def verify_runtime(manifest: Mapping[str, object]) -> None:
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("manifest environment missing")
    expected = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch_installed": torch.__version__,
    }
    for name, actual in expected.items():
        if environment.get(name) != actual:
            raise ValueError(
                f"runtime mismatch for {name}: "
                f"manifest={environment.get(name)!r} current={actual!r}"
            )
    if environment.get("torch_required") != torch.__version__:
        raise ValueError("current torch does not satisfy frozen requirement")


def load_sequences(
    *, project_root: Path, record: Mapping[str, object], token_to_id: Mapping[str, int]
) -> list[list[int]]:
    relative = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError("split path/hash missing")
    path = (project_root / relative).resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError("split escapes project root")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"split hash mismatch: {path}")
    sequences: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record_value = json.loads(line)
            tokens = record_value.get("tokens")
            if not isinstance(tokens, list):
                raise ValueError(f"invalid token record in {path}")
            sequences.append([token_to_id[str(token)] for token in tokens])
    return sequences


def learning_rate(step: int) -> float:
    if not 1 <= step <= TOTAL_STEPS:
        raise ValueError(f"step outside schedule: {step}")
    if step <= WARMUP_STEPS:
        return 3e-4 * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1.0 + math.cos(math.pi * progress))


def optimizer_for(model: AcquisitionModel) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for _, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def evaluate_nll(model: AcquisitionModel, sequences: Sequence[Sequence[int]]) -> float:
    total_loss = 0.0
    total_targets = 0
    model.eval()
    with torch.no_grad():
        for tokens in sequences:
            total_loss += float(model.sequence_loss(tokens))
            total_targets += len(tokens) - 1
    return total_loss / total_targets


def model_arrays(model: AcquisitionModel) -> dict[str, np.ndarray]:
    return {
        name: parameter.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, parameter in model.named_parameters()
    }


def earliest_minimum_epoch(values: Sequence[float]) -> int:
    minimum = min(values)
    for index, value in enumerate(values, start=1):
        if value <= minimum + 1e-8:
            return index
    raise AssertionError("unreachable empty development-loss list")


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def smoke_test(model: AcquisitionModel, sequences: Sequence[Sequence[int]]) -> dict[str, object]:
    optimizer = optimizer_for(model)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    losses = [model.sequence_loss(tokens) for tokens in sequences]
    targets = sum(len(tokens) - 1 for tokens in sequences)
    loss = torch.stack(losses).sum() / targets
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    elapsed = time.perf_counter() - started
    detached_loss = float(loss.detach())
    if not math.isfinite(detached_loss) or not math.isfinite(gradient_norm):
        raise AssertionError("nonfinite smoke loss or gradient")
    return {
        "status": "pass",
        "arm": model.arm,
        "loss": detached_loss,
        "gradient_norm": gradient_norm,
        "sequences": len(sequences),
        "targets": targets,
        "wall_seconds": elapsed,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def train(
    *,
    project_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
    family: str,
    arm: str,
    seed: int,
) -> dict[str, object]:
    token_maps = manifest.get("token_maps")
    fixed_splits = manifest.get("fixed_splits")
    if not isinstance(token_maps, dict) or not isinstance(fixed_splits, dict):
        raise ValueError("manifest data is incomplete")
    tokens = token_maps.get(family)
    if not isinstance(tokens, list):
        raise ValueError(f"family missing from manifest: {family}")
    token_to_id = {str(token): index for index, token in enumerate(tokens)}
    train_record = fixed_splits.get(f"{family}/train/{seed}")
    dev_record = fixed_splits.get(f"{family}/development")
    if not isinstance(train_record, dict) or not isinstance(dev_record, dict):
        raise ValueError("manifest split missing")
    training = load_sequences(
        project_root=project_root, record=train_record, token_to_id=token_to_id
    )
    development = load_sequences(
        project_root=project_root, record=dev_record, token_to_id=token_to_id
    )
    if len(training) != TRAIN_EXAMPLES or len(development) != 5_000:
        raise ValueError("split count mismatch")

    model = AcquisitionModel(vocabulary_size=len(tokens), arm=arm, model_seed=seed)
    optimizer = optimizer_for(model)
    development_nlls: list[float] = []
    epoch_checkpoints: list[dict[str, object]] = []
    step = 0
    started = time.perf_counter()
    cpu_started = time.process_time()
    output_root = project_root / "runtime" / "cpkv-topk-001" / family / arm / str(seed)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        generator = np.random.Generator(np.random.PCG64DXSM(seed + 100_000 + epoch))
        permutation = generator.permutation(len(training))
        for batch_start in range(0, len(training), BATCH_SIZE):
            indices = permutation[batch_start : batch_start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            batch_losses: list[Tensor] = []
            targets = 0
            for index in indices.tolist():
                sequence = training[index]
                batch_losses.append(model.sequence_loss(sequence))
                targets += len(sequence) - 1
            loss = torch.stack(batch_losses).sum() / targets
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step += 1
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
        dev_nll = evaluate_nll(model, development)
        development_nlls.append(dev_nll)
        epoch_path = output_root / "epochs" / f"epoch-{epoch:02d}.npz"
        atomic_save_npz(epoch_path, model_arrays(model))
        epoch_checkpoints.append(
            {
                "epoch": epoch,
                "path": epoch_path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(epoch_path),
                "development_nll": dev_nll,
            }
        )
        print(json.dumps({"family": family, "arm": arm, "seed": seed, "epoch": epoch, "development_nll": dev_nll, "step": step}), flush=True)

    if step != TOTAL_STEPS or len(epoch_checkpoints) != EPOCHS:
        raise AssertionError(f"training did not complete: step={step}")
    best_epoch = earliest_minimum_epoch(development_nlls)
    best_nll = development_nlls[best_epoch - 1]
    selected_checkpoint = epoch_checkpoints[best_epoch - 1]
    weights_path = project_root / str(selected_checkpoint["path"])
    run_path = output_root / "run.json"
    manifest_hash = manifest_sha256(manifest_path)
    run_record: dict[str, object] = {
        "schema": "cpkv-topk-001-run-v1",
        "lineage": LINEAGE,
        "manifest_sha256": manifest_hash,
        "family": family,
        "arm": arm,
        "seed": seed,
        "completed_epochs": EPOCHS,
        "optimizer_steps": step,
        "training_examples_seen": EPOCHS * len(training),
        "development_nlls": development_nlls,
        "selected_epoch": best_epoch,
        "development_nll": best_nll,
        "epoch_checkpoints": epoch_checkpoints,
        "training_data_sha256": train_record["sha256"],
        "development_data_sha256": dev_record["sha256"],
        "weights_path": weights_path.relative_to(project_root).as_posix(),
        "weights_sha256": sha256_file(weights_path),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    _atomic_write_text(run_path, json.dumps(run_record, indent=2, sort_keys=True) + "\n")
    return {
        "run_path": run_path.relative_to(project_root).as_posix(),
        "run_sha256": sha256_file(run_path),
        **run_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", choices=("D1", "A1", "A2"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=range(1701, 1711), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-sequences", type=int, default=4)
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
    manifest = read_manifest(manifest_path)
    verify_runtime(manifest)
    verify_manifest_sources(project_root=project_root, manifest=manifest)
    verify_preflight_receipt(project_root=project_root, manifest=manifest)
    token_maps = manifest["token_maps"]
    fixed_splits = manifest["fixed_splits"]
    tokens = token_maps[args.family]
    token_to_id = {str(token): index for index, token in enumerate(tokens)}
    training_record = fixed_splits[f"{args.family}/train/{args.seed}"]
    training = load_sequences(
        project_root=project_root,
        record=training_record,
        token_to_id=token_to_id,
    )
    model = AcquisitionModel(
        vocabulary_size=len(tokens), arm=args.arm, model_seed=args.seed
    )
    if args.smoke:
        if not 1 <= args.smoke_sequences <= len(training):
            raise ValueError("smoke sequence count outside training split")
        print(
            json.dumps(
                smoke_test(model, training[: args.smoke_sequences]),
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = train(
        project_root=project_root,
        manifest_path=manifest_path,
        manifest=manifest,
        family=args.family,
        arm=args.arm,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
