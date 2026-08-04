#!/usr/bin/env python3
"""Learned structured screen for staggered Power-Evidence Attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "results" / "staggered-power-evidence-learning-preregistration.md"
STAGE0 = ROOT / "results" / "staggered-power-evidence-stage0.json"
STAGE0_SHA = "c9f0874d19f4c520a9f70c2b2eef8adc43475abcf46d6e42f286e43e45ef72aa"
COMPARISON_STAGE0 = ROOT / "results" / "inclusive-power-attention-stage0.json"
COMPARISON_STAGE0_SHA = "98217c897f358647f81eaf27358c03b4a00e8e114306f8e2b6f20ed2e0935659"
INTEGRITY = ROOT / "results" / "staggered-power-evidence-learning-integrity.json"
DEFAULT_OUTPUT = ROOT / "results" / "staggered-power-evidence-learning.json"

SEEDS = (1103, 2207, 3719, 6673, 9011)
ARMS = (
    "ordinary",
    "temperature2",
    "pea_staggered",
    "ipa_staggered",
    "pea_same_offset",
    "ipa_same_offset",
)
CANDIDATES = ("pea_staggered", "ipa_staggered")
TASKS = ("anchor_same", "anchor_boundary", "exact_copy", "distributed")

SEQ = 128
CHUNK = 16
KEYS = 16
TYPE_COUNT = 4
DIN = KEYS + 1 + TYPE_COUNT
HEADS = 4
QK = 16
VD = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class CompactDataset:
    key_ids: torch.Tensor
    key_strength: torch.Tensor
    labels: torch.Tensor
    types: torch.Tensor
    query_ids: torch.Tensor
    targets: torch.Tensor
    task_ids: torch.Tensor
    subtype_ids: torch.Tensor
    target_positions: torch.Tensor

    def __len__(self) -> int:
        return int(self.query_ids.shape[0])

    def index(self, ids: torch.Tensor) -> "CompactDataset":
        return CompactDataset(*(field[ids] for field in (
            self.key_ids, self.key_strength, self.labels, self.types,
            self.query_ids, self.targets, self.task_ids, self.subtype_ids,
            self.target_positions,
        )))


def _empty_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full(SEQ, -1, dtype=np.int16),
        np.zeros(SEQ, dtype=np.float32),
        np.zeros(SEQ, dtype=np.float32),
        np.zeros(SEQ, dtype=np.int8),
    )


def _place_pair(
    key_ids: np.ndarray,
    strengths: np.ndarray,
    labels: np.ndarray,
    types: np.ndarray,
    used: set[int],
    key: int,
    label: float,
    anchor: int,
) -> None:
    payload = anchor + 1
    if anchor in used or payload in used or not (0 <= anchor < SEQ - 1):
        raise ValueError("invalid pair placement")
    used.update((anchor, payload))
    key_ids[anchor] = key
    strengths[anchor] = 1.0
    types[anchor] = 1
    labels[payload] = label
    types[payload] = 2


def _random_free_anchor(rng: np.random.Generator, used: set[int]) -> int:
    for _ in range(1000):
        anchor = int(rng.integers(2, SEQ - 2))
        if anchor not in used and anchor + 1 not in used:
            return anchor
    raise RuntimeError("could not place record")


def _random_free_position(rng: np.random.Generator, used: set[int]) -> int:
    for _ in range(1000):
        position = int(rng.integers(2, SEQ - 2))
        if position not in used:
            return position
    raise RuntimeError("could not place token")


def generate_dataset(seed: int, count: int, balanced: bool) -> CompactDataset:
    rng = np.random.default_rng(seed)
    key_ids = np.full((count, SEQ), -1, dtype=np.int16)
    strengths = np.zeros((count, SEQ), dtype=np.float32)
    labels = np.zeros((count, SEQ), dtype=np.float32)
    types = np.zeros((count, SEQ), dtype=np.int8)
    query_ids = np.zeros(count, dtype=np.int16)
    targets = np.zeros(count, dtype=np.float32)
    task_ids = np.zeros(count, dtype=np.int8)
    subtype_ids = np.zeros(count, dtype=np.int8)
    target_positions = np.full(count, -1, dtype=np.int16)

    for row in range(count):
        task_id = row % len(TASKS) if balanced else int(rng.integers(len(TASKS)))
        task = TASKS[task_id]
        task_ids[row] = task_id
        keys = rng.choice(KEYS, size=8, replace=False)
        record_labels = rng.choice((-1.0, 1.0), size=8)
        target_record = int(rng.integers(8))
        target_key = int(keys[target_record])
        query_ids[row] = target_key
        targets[row] = 1.0 if record_labels[target_record] > 0 else 0.0
        used: set[int] = set()

        if task in ("anchor_same", "anchor_boundary"):
            if task == "anchor_same":
                base = int(rng.integers(2, 6)) * CHUNK
                anchor = base + int(rng.choice((2, 3, 4, 10, 11, 12)))
            else:
                base = int(rng.integers(2, 6)) * CHUNK
                anchor = base + (15 if bool(rng.integers(2)) else 7)
            _place_pair(key_ids[row], strengths[row], labels[row], types[row], used,
                        target_key, float(record_labels[target_record]), anchor)
            for record in range(8):
                if record == target_record:
                    continue
                position = _random_free_anchor(rng, used)
                _place_pair(key_ids[row], strengths[row], labels[row], types[row], used,
                            int(keys[record]), float(record_labels[record]), position)
        elif task == "exact_copy":
            edge = (row // len(TASKS)) % 2 == 0
            subtype_ids[row] = 1 if edge else 2
            if edge:
                if bool(rng.integers(2)):
                    target_position = int(rng.integers(2, 7))
                else:
                    target_position = int(rng.integers(121, 126))
            else:
                target_chunk = int(rng.integers(1, 7))
                target_position = target_chunk * CHUNK + int(rng.integers(2, 14))
            used.add(target_position)
            target_positions[row] = target_position
            key_ids[row, target_position] = target_key
            strengths[row, target_position] = 1.0
            labels[row, target_position] = float(record_labels[target_record])
            types[row, target_position] = 3
            for record in range(8):
                if record == target_record:
                    continue
                position = _random_free_position(rng, used)
                used.add(position)
                key_ids[row, position] = int(keys[record])
                strengths[row, position] = 1.0
                labels[row, position] = float(record_labels[record])
                types[row, position] = 3
        elif task == "distributed":
            # Four positive and four negative key labels make the aggregate
            # value exactly zero, so a key-blind reader has no label-majority
            # signal. Each key has four noisy, jointly sufficient shares in
            # one chunk and one target-independent strong spike elsewhere.
            record_labels[:] = -1.0
            record_labels[rng.choice(8, size=4, replace=False)] = 1.0
            targets[row] = 1.0 if record_labels[target_record] > 0 else 0.0
            spike_labels = np.full(8, -1.0)
            spike_labels[rng.choice(8, size=4, replace=False)] = 1.0
            chunk_order = rng.permutation(8)
            position_slots = {
                chunk: list(rng.permutation(np.arange(chunk * CHUNK + 2, chunk * CHUNK + 14)))
                for chunk in range(8)
            }
            for record in range(8):
                label = float(record_labels[record])
                consensus_chunk = int(chunk_order[record])
                noise = rng.normal(0.0, 8.0, size=4)
                noise -= noise.mean()
                shares = np.asarray(noise + 0.25 * label, dtype=np.float32)
                shares[-1] = np.float32(label - shares[:3].sum(dtype=np.float32))
                for share in shares:
                    position = int(position_slots[consensus_chunk].pop())
                    used.add(position)
                    key_ids[row, position] = int(keys[record])
                    strengths[row, position] = 0.5
                    labels[row, position] = float(share)
                    types[row, position] = 4
                spike_chunk = int(chunk_order[(record + 1) % 8])
                position = int(position_slots[spike_chunk].pop())
                used.add(position)
                key_ids[row, position] = int(keys[record])
                strengths[row, position] = 1.0
                labels[row, position] = float(spike_labels[record])
                types[row, position] = 4
        else:
            raise ValueError(task)

    return CompactDataset(*(
        torch.from_numpy(array) for array in
        (key_ids, strengths, labels, types, query_ids, targets, task_ids, subtype_ids, target_positions)
    ))


def best_threshold_accuracy(values: np.ndarray, targets: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    x = values[order]
    y = targets[order].astype(np.int64)
    positive_prefix = np.concatenate(([0], np.cumsum(y)))
    negative_prefix = np.arange(len(y) + 1) - positive_prefix
    cuts = np.concatenate((
        np.array([0]),
        np.where(x[:-1] != x[1:])[0] + 1,
        np.array([len(x)]),
    ))
    positive_total = positive_prefix[-1]
    negative_total = negative_prefix[-1]
    direct = negative_prefix[cuts] + (positive_total - positive_prefix[cuts])
    inverse = positive_prefix[cuts] + (negative_total - negative_prefix[cuts])
    return float(max(direct.max(), inverse.max()) / len(y))


def distributed_oracle_probe(dataset: CompactDataset) -> dict:
    moderate_rows = []
    spike_values = []
    targets = []
    global_sums = []
    consensus_correct = 0
    for row in range(len(dataset)):
        if int(dataset.task_ids[row]) != TASKS.index("distributed"):
            continue
        query_key = int(dataset.query_ids[row])
        mask = (dataset.types[row] == 4) & (dataset.key_ids[row] == query_key)
        strengths = dataset.key_strength[row, mask].numpy()
        values = dataset.labels[row, mask].numpy()
        moderate = values[strengths == 0.5]
        spike = values[strengths == 1.0]
        if len(moderate) != 4 or len(spike) != 1:
            raise RuntimeError("distributed oracle geometry mismatch")
        target = int(dataset.targets[row] >= 0.5)
        moderate_rows.append(moderate)
        spike_values.append(float(spike[0]))
        targets.append(target)
        all_values = dataset.labels[row, dataset.types[row] == 4]
        global_sums.append(float(all_values.double().sum()))
        consensus_correct += int((moderate.sum() >= 0) == bool(target))
    moderate_array = np.asarray(moderate_rows)
    target_array = np.asarray(targets)
    oracles = {
        "pooled_moderate": best_threshold_accuracy(
            moderate_array.reshape(-1), np.repeat(target_array, 4)
        ),
        "maximum_moderate": best_threshold_accuracy(moderate_array.max(axis=1), target_array),
        "minimum_moderate": best_threshold_accuracy(moderate_array.min(axis=1), target_array),
        "strong_spike": best_threshold_accuracy(np.asarray(spike_values), target_array),
    }
    return {
        "count": len(targets),
        "single_occurrence_threshold_oracles": oracles,
        "best_single_occurrence_threshold_accuracy": max(oracles.values()),
        "consensus_sum_accuracy": consensus_correct / len(targets),
        "keyblind_global_value_sum_max_abs": max(abs(value) for value in global_sums),
    }


def distributed_oracle_valid(audit: dict) -> bool:
    return (
        audit["best_single_occurrence_threshold_accuracy"] <= 0.56
        and audit["consensus_sum_accuracy"] >= 0.999
        and audit["keyblind_global_value_sum_max_abs"] <= 1e-5
    )


def to_features(batch: CompactDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key_ids = batch.key_ids.to(device=device, dtype=torch.long)
    valid = key_ids >= 0
    safe_ids = key_ids.clamp_min(0)
    keys = F.one_hot(safe_ids, num_classes=KEYS).float()
    keys = keys * valid[..., None] * batch.key_strength.to(device)[..., None]
    label = batch.labels.to(device)[..., None]
    token_types = F.one_hot(batch.types.to(device=device, dtype=torch.long), num_classes=TYPE_COUNT + 1)[..., 1:].float()
    context = torch.cat((keys, label, token_types), dim=-1)
    query = torch.zeros((len(batch), DIN), device=device)
    query.scatter_(1, batch.query_ids.to(device=device, dtype=torch.long)[:, None], 1.0)
    targets = batch.targets.to(device)
    return context, query, targets


def group_indices(origin: int) -> list[torch.Tensor]:
    positions = torch.arange(SEQ)
    labels = torch.div(positions + origin, CHUNK, rounding_mode="floor")
    return [positions[labels == value] for value in torch.unique(labels)]


GROUPS = {origin: group_indices(origin) for origin in (0, 8)}


def transform_head(scores: torch.Tensor, arm: str, origin: int) -> torch.Tensor:
    if arm == "ordinary":
        return scores
    if arm == "temperature2":
        return 2.0 * scores
    result = scores.clone()
    for cpu_ids in GROUPS[origin]:
        ids = cpu_ids.to(scores.device)
        block = scores[:, ids]
        maximum = block.amax(dim=-1, keepdim=True)
        shifted = torch.exp(block - maximum)
        l1 = shifted.sum(dim=-1)
        if arm == "pea2":
            lift = maximum[:, 0] + torch.log(shifted.square().sum(dim=-1) / l1)
        elif arm == "ipa2":
            lift = maximum[:, 0] + torch.log(l1) - math.log(ids.numel())
        else:
            raise ValueError(arm)
        result[:, ids] = result[:, ids] + lift[:, None]
    return result


class LearnedRead(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.q = nn.Linear(DIN, HEADS * QK, bias=False)
        self.k = nn.Linear(DIN, HEADS * QK, bias=False)
        self.v = nn.Linear(DIN, HEADS * VD, bias=False)
        self.out = nn.Linear(HEADS * VD, 1, bias=False)

    def raw_scores(self, context: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        q = self.q(query).view(batch, HEADS, QK)
        k = self.k(context).view(batch, SEQ, HEADS, QK).transpose(1, 2)
        v = self.v(context).view(batch, SEQ, HEADS, VD).transpose(1, 2)
        scores = torch.einsum("bhd,bhnd->bhn", q, k) / math.sqrt(QK)
        return scores, v

    def forward(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        ablate_to_p1: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        scores, v = self.raw_scores(context, query)
        transformed = []
        for head in range(HEADS):
            if head >= 2 or ablate_to_p1:
                transformed.append(scores[:, head])
                continue
            if self.arm == "ordinary":
                mode, origin = "ordinary", 0
            elif self.arm == "temperature2":
                mode, origin = "temperature2", 0
            elif self.arm == "pea_staggered":
                mode, origin = "pea2", 0 if head == 0 else 8
            elif self.arm == "ipa_staggered":
                mode, origin = "ipa2", 0 if head == 0 else 8
            elif self.arm == "pea_same_offset":
                mode, origin = "pea2", 0
            elif self.arm == "ipa_same_offset":
                mode, origin = "ipa2", 0
            else:
                raise ValueError(self.arm)
            transformed.append(transform_head(scores[:, head], mode, origin))
        weights = torch.softmax(torch.stack(transformed, dim=1), dim=-1)
        head_output = torch.einsum("bhn,bhnv->bhv", weights, v)
        logits = self.out(head_output.reshape(batch, HEADS * VD))[:, 0]
        return (logits, scores, weights) if return_diagnostics else logits


@torch.no_grad()
def evaluate(model: LearnedRead, dataset: CompactDataset, device: torch.device, batch_size: int, ablate: bool = False) -> dict:
    model.eval()
    losses = []
    correct = 0
    count = 0
    score_blocks = []
    target_attention_blocks = []
    for start in range(0, len(dataset), batch_size):
        ids = torch.arange(start, min(start + batch_size, len(dataset)))
        indexed = dataset.index(ids)
        context, query, target = to_features(indexed, device)
        logits, raw_scores, attention_weights = model(context, query, ablate_to_p1=ablate, return_diagnostics=True)
        losses.append(F.binary_cross_entropy_with_logits(logits, target, reduction="none").cpu())
        score_blocks.append(raw_scores.detach().abs().transpose(0, 1).reshape(HEADS, -1).cpu())
        target_positions = indexed.target_positions
        if bool((target_positions >= 0).any()):
            batch_rows = torch.arange(target_positions.numel(), device=device)
            target_attention_blocks.append(
                attention_weights[batch_rows, :, target_positions.to(device=device, dtype=torch.long)].detach().cpu()
            )
        correct += int(((logits >= 0) == (target >= 0.5)).sum())
        count += target.numel()
    per_example = torch.cat(losses)
    absolute_scores = torch.cat(score_blocks, dim=1).float()
    subtype_ids = dataset.subtype_ids
    subgroups = {}
    if bool((subtype_ids > 0).any()):
        predictions = []
        for start in range(0, len(dataset), batch_size):
            ids = torch.arange(start, min(start + batch_size, len(dataset)))
            context, query, _ = to_features(dataset.index(ids), device)
            predictions.append((model(context, query, ablate_to_p1=ablate) >= 0).cpu())
        prediction = torch.cat(predictions)
        target_attention = torch.cat(target_attention_blocks)
        target = dataset.targets >= 0.5
        for subtype, name in ((1, "edge"), (2, "interior")):
            mask = subtype_ids == subtype
            subgroups[name] = {
                "accuracy": float((prediction[mask] == target[mask]).float().mean()),
                "count": int(mask.sum()),
                "target_attention_mean_by_head": target_attention[mask].mean(dim=0).tolist(),
            }
    return {
        "loss": float(per_example.mean()),
        "accuracy": correct / count,
        "count": count,
        "per_example_loss": per_example.tolist(),
        "raw_score_rms_by_head": absolute_scores.square().mean(dim=1).sqrt().tolist(),
        "raw_score_abs_q99_by_head": torch.quantile(absolute_scores, 0.99, dim=1).tolist(),
        "subgroups": subgroups,
    }


def train_one(
    arm: str,
    seed: int,
    train_data: CompactDataset,
    validation: dict[str, CompactDataset],
    device: torch.device,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = LearnedRead(arm).to(device)
    initial_hash = hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.0)
    order_rng = torch.Generator().manual_seed(seed + 404)
    order = torch.randperm(len(train_data), generator=order_rng)
    cursor = 0
    loss_trace = []
    max_gradient_norm = 0.0
    nonfinite = False
    model.train()
    for step in range(steps):
        if cursor + batch_size > len(order):
            order = torch.randperm(len(train_data), generator=order_rng)
            cursor = 0
        ids = order[cursor:cursor + batch_size]
        cursor += batch_size
        context, query, target = to_features(train_data.index(ids), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(context, query)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        if not bool(torch.isfinite(loss)):
            nonfinite = True
            break
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(norm):
            nonfinite = True
            break
        max_gradient_norm = max(max_gradient_norm, norm)
        optimizer.step()
        if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
            loss_trace.append({"step": step + 1, "loss": float(loss)})

    evaluations = {task: evaluate(model, data, device, eval_batch_size) for task, data in validation.items()}
    ablation = (
        {task: evaluate(model, data, device, eval_batch_size, ablate=True) for task, data in validation.items()}
        if arm in CANDIDATES else {}
    )
    return {
        "arm": arm,
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_hash,
        "loss_trace": loss_trace,
        "max_preclip_gradient_norm": max_gradient_norm,
        "nonfinite": nonfinite,
        "evaluations": evaluations,
        "p1_ablation": ablation,
    }


def summarize(
    runs: dict[str, dict[str, dict]],
    dataset_audits: dict[str, dict],
    prerequisites_valid: bool,
    formal_configuration_valid: bool,
) -> tuple[dict, dict]:
    summary: dict[str, dict] = {}
    for arm in ARMS:
        summary[arm] = {}
        for task in TASKS:
            accuracies = [runs[str(seed)][arm]["evaluations"][task]["accuracy"] for seed in SEEDS]
            losses = [runs[str(seed)][arm]["evaluations"][task]["loss"] for seed in SEEDS]
            summary[arm][task] = {
                "accuracies": accuracies,
                "mean_accuracy": float(np.mean(accuracies)),
                "losses": losses,
                "mean_loss": float(np.mean(losses)),
            }
            if runs[str(SEEDS[0])][arm]["evaluations"][task]["subgroups"]:
                summary[arm][task]["subgroups"] = {
                    name: {
                        "accuracies": [runs[str(seed)][arm]["evaluations"][task]["subgroups"][name]["accuracy"] for seed in SEEDS],
                        "mean_accuracy": float(np.mean([
                            runs[str(seed)][arm]["evaluations"][task]["subgroups"][name]["accuracy"] for seed in SEEDS
                        ])),
                        "target_attention_mean_by_head": np.mean([
                            runs[str(seed)][arm]["evaluations"][task]["subgroups"][name]["target_attention_mean_by_head"]
                            for seed in SEEDS
                        ], axis=0).tolist(),
                    }
                    for name in ("edge", "interior")
                }
        per_seed_overall = [
            float(np.mean([runs[str(seed)][arm]["evaluations"][task]["accuracy"] for task in TASKS]))
            for seed in SEEDS
        ]
        summary[arm]["overall"] = {"accuracies": per_seed_overall, "mean_accuracy": float(np.mean(per_seed_overall))}

    ordinary = summary["ordinary"]
    temperature = summary["temperature2"]
    anchor_tasks = ("anchor_same", "anchor_boundary")

    parameter_counts = {runs[str(seed)][arm]["parameter_count"] for seed in SEEDS for arm in ARMS}
    initialization_equal = all(
        len({runs[str(seed)][arm]["initial_state_sha256"] for arm in ARMS}) == 1
        for seed in SEEDS
    )
    def evaluation_finite(evaluation: dict) -> bool:
        return (
            math.isfinite(evaluation["loss"])
            and math.isfinite(evaluation["accuracy"])
            and all(math.isfinite(value) for value in evaluation["per_example_loss"])
            and all(math.isfinite(value) for value in evaluation["raw_score_rms_by_head"])
            and all(math.isfinite(value) for value in evaluation["raw_score_abs_q99_by_head"])
            and all(math.isfinite(group["accuracy"]) for group in evaluation["subgroups"].values())
            and all(
                all(math.isfinite(value) for value in group["target_attention_mean_by_head"])
                for group in evaluation["subgroups"].values()
            )
        )

    finite = all(
        not runs[str(seed)][arm]["nonfinite"]
        and math.isfinite(runs[str(seed)][arm]["max_preclip_gradient_norm"])
        and all(evaluation_finite(runs[str(seed)][arm]["evaluations"][task]) for task in TASKS)
        and (
            all(evaluation_finite(runs[str(seed)][arm]["p1_ablation"][task]) for task in TASKS)
            if arm in CANDIDATES else True
        )
        for seed in SEEDS for arm in ARMS
    )
    global_gates = {
        "formal_configuration_valid": formal_configuration_valid,
        "stage0_prerequisites_valid": prerequisites_valid,
        "equal_parameter_counts": len(parameter_counts) == 1,
        "equal_initialization_within_seed": initialization_equal,
        "all_reported_metrics_finite": finite,
        "distributed_single_occurrence_shortcut_absent": all(
            distributed_oracle_valid(dataset_audits[str(seed)])
            for seed in SEEDS
        ),
    }

    candidate_gates = {}
    candidate_diagnostics = {}
    derived_diagnostics_finite = True
    for candidate in CANDIDATES:
        candidate_summary = summary[candidate]
        same_offset = "pea_same_offset" if candidate == "pea_staggered" else "ipa_same_offset"
        anchor_gain = all(
            candidate_summary[task]["mean_accuracy"] - max(ordinary[task]["mean_accuracy"], temperature[task]["mean_accuracy"]) >= 0.10
            and sum(
                runs[str(seed)][candidate]["evaluations"][task]["accuracy"]
                > max(runs[str(seed)]["ordinary"]["evaluations"][task]["accuracy"], runs[str(seed)]["temperature2"]["evaluations"][task]["accuracy"])
                for seed in SEEDS
            ) >= 4
            for task in anchor_tasks
        )
        boundary_wins = [
            runs[str(seed)][candidate]["evaluations"]["anchor_boundary"]["accuracy"]
            > runs[str(seed)][same_offset]["evaluations"]["anchor_boundary"]["accuracy"]
            for seed in SEEDS
        ]
        boundary_recovery = (
            candidate_summary["anchor_boundary"]["mean_accuracy"]
            - summary[same_offset]["anchor_boundary"]["mean_accuracy"] >= 0.05
            and sum(boundary_wins) >= 4
        )
        protected = all(
            candidate_summary[task]["mean_accuracy"] >= ordinary[task]["mean_accuracy"] - 0.02
            and all(
                runs[str(seed)][candidate]["evaluations"][task]["accuracy"]
                >= runs[str(seed)]["ordinary"]["evaluations"][task]["accuracy"] - 0.04
                for seed in SEEDS
            )
            for task in ("exact_copy", "distributed")
        )
        overall = (
            candidate_summary["overall"]["mean_accuracy"] > max(ordinary["overall"]["mean_accuracy"], temperature["overall"]["mean_accuracy"])
            and sum(
                candidate_summary["overall"]["accuracies"][i]
                > max(ordinary["overall"]["accuracies"][i], temperature["overall"]["accuracies"][i])
                for i in range(len(SEEDS))
            ) >= 4
        )
        ablation_drops = []
        for seed in SEEDS:
            run = runs[str(seed)][candidate]
            full = np.mean([run["evaluations"][task]["accuracy"] for task in anchor_tasks])
            ablated = np.mean([run["p1_ablation"][task]["accuracy"] for task in anchor_tasks])
            ablation_drops.append(float(full - ablated))
        causal = float(np.mean(ablation_drops)) >= 0.05 and sum(drop >= 0.05 for drop in ablation_drops) >= 4

        score_ratios = []
        score_scale_ok = True
        for seed in SEEDS:
            seed_record = {}
            for task in TASKS:
                task_record = {}
                for metric in ("raw_score_rms_by_head", "raw_score_abs_q99_by_head"):
                    ratios = []
                    for head in (0, 1):
                        numerator = runs[str(seed)][candidate]["evaluations"][task][metric][head]
                        denominator = max(
                            runs[str(seed)]["ordinary"]["evaluations"][task][metric][head],
                            runs[str(seed)]["temperature2"]["evaluations"][task][metric][head],
                            1e-12,
                        )
                        ratios.append(numerator / denominator)
                    task_record[metric] = ratios
                    score_scale_ok &= max(ratios) <= 2.0
                seed_record[task] = task_record
            score_ratios.append(seed_record)

        edge_gaps = []
        ordinary_edge_gaps = []
        target_attention_pair_distortions = []
        affected_head_target_attention_ratios = []
        affected_head_log_distortions = []
        for seed in SEEDS:
            groups = runs[str(seed)][candidate]["evaluations"]["exact_copy"]["subgroups"]
            ordinary_groups = runs[str(seed)]["ordinary"]["evaluations"]["exact_copy"]["subgroups"]
            edge_gaps.append(groups["edge"]["accuracy"] - groups["interior"]["accuracy"])
            ordinary_edge_gaps.append(ordinary_groups["edge"]["accuracy"] - ordinary_groups["interior"]["accuracy"])
            candidate_edge = np.mean(groups["edge"]["target_attention_mean_by_head"][:2])
            candidate_interior = np.mean(groups["interior"]["target_attention_mean_by_head"][:2])
            ordinary_edge = np.mean(ordinary_groups["edge"]["target_attention_mean_by_head"][:2])
            ordinary_interior = np.mean(ordinary_groups["interior"]["target_attention_mean_by_head"][:2])
            candidate_head_edge = groups["edge"]["target_attention_mean_by_head"][1]
            candidate_head_interior = groups["interior"]["target_attention_mean_by_head"][1]
            ordinary_head_edge = ordinary_groups["edge"]["target_attention_mean_by_head"][1]
            ordinary_head_interior = ordinary_groups["interior"]["target_attention_mean_by_head"][1]
            operands = (
                candidate_edge, candidate_interior, ordinary_edge, ordinary_interior,
                candidate_head_edge, candidate_head_interior, ordinary_head_edge, ordinary_head_interior,
            )
            if all(math.isfinite(value) and value > 0.0 for value in operands):
                target_attention_pair_distortions.append(float(np.log(
                    (candidate_edge / candidate_interior) / (ordinary_edge / ordinary_interior)
                )))
                affected_head_target_attention_ratios.append(float(candidate_head_edge / candidate_head_interior))
                affected_head_log_distortions.append(float(np.log(
                    (candidate_head_edge / candidate_head_interior)
                    / (ordinary_head_edge / ordinary_head_interior)
                )))
            else:
                target_attention_pair_distortions.append(None)
                affected_head_target_attention_ratios.append(None)
                affected_head_log_distortions.append(None)
        derived_finite = all(
            value is not None and math.isfinite(value)
            for value in target_attention_pair_distortions + affected_head_target_attention_ratios + affected_head_log_distortions
        )
        derived_diagnostics_finite &= derived_finite
        partial_chunk_guard = (
            derived_finite
            and abs(float(np.mean(target_attention_pair_distortions))) <= math.log(1.03)
            and all(abs(distortion) <= math.log(1.08) for distortion in target_attention_pair_distortions)
            and abs(float(np.mean(affected_head_log_distortions))) <= math.log(1.05)
            and all(abs(distortion) <= math.log(1.15) for distortion in affected_head_log_distortions)
        )

        candidate_gates[candidate] = {
            "anchor_gain_over_ordinary_and_temperature": anchor_gain,
            "staggering_recovers_boundary": boundary_recovery,
            "copy_and_balanced_distributed_noninferior": protected,
            "overall_gain": overall,
            "router_power_causally_used": causal,
            "raw_score_scale_bounded": score_scale_ok,
            "partial_chunk_sawtooth_guard": partial_chunk_guard,
        }
        candidate_diagnostics[candidate] = {
            "p1_ablation_anchor_accuracy_drops": ablation_drops,
            "raw_score_ratios": score_ratios,
            "exact_copy_edge_minus_interior": edge_gaps,
            "ordinary_exact_copy_edge_minus_interior": ordinary_edge_gaps,
            "candidate_pair_target_attention_log_distortion": target_attention_pair_distortions,
            "origin8_head_target_attention_edge_over_interior": affected_head_target_attention_ratios,
            "origin8_head_ordinary_relative_log_distortion": affected_head_log_distortions,
        }

    pea_anchor_mean = float(np.mean([summary["pea_staggered"][task]["mean_accuracy"] for task in anchor_tasks]))
    ipa_anchor_mean = float(np.mean([summary["ipa_staggered"][task]["mean_accuracy"] for task in anchor_tasks]))
    pea_generic_mean = float(np.mean([summary["pea_staggered"][task]["mean_accuracy"] for task in ("exact_copy", "distributed")]))
    ipa_generic_mean = float(np.mean([summary["ipa_staggered"][task]["mean_accuracy"] for task in ("exact_copy", "distributed")]))
    selection_gates = {
        "pea_l2_anchor_margin_over_ipa": pea_anchor_mean - ipa_anchor_mean >= 0.05,
        "pea_generic_noninferior_to_ipa": pea_generic_mean >= ipa_generic_mean - 0.01,
    }
    global_gates["all_reported_metrics_finite"] &= derived_diagnostics_finite
    global_pass = all(global_gates.values())
    candidate_passes = {candidate: all(gates.values()) for candidate, gates in candidate_gates.items()}
    selected_candidate = None
    if global_pass:
        if candidate_passes["pea_staggered"] and all(selection_gates.values()):
            selected_candidate = "pea_staggered"
        elif candidate_passes["ipa_staggered"]:
            selected_candidate = "ipa_staggered"

    decision = {
        "global_gates": global_gates,
        "candidate_gates": candidate_gates,
        "candidate_passes": candidate_passes,
        "selection_gates": selection_gates,
        "selected_candidate": selected_candidate,
        "diagnostics": candidate_diagnostics,
        "learning_pass": selected_candidate is not None,
    }
    return summary, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--train-examples", type=int, default=65536)
    parser.add_argument("--validation-examples", type=int, default=4096)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if args.formal and (args.steps, args.batch_size, args.eval_batch_size, args.train_examples, args.validation_examples) != (800, 512, 1024, 65536, 4096):
        raise ValueError("formal arguments differ from preregistration")
    if args.formal and args.output.resolve() != DEFAULT_OUTPUT.resolve():
        raise ValueError("formal run must use the preregistered output path")
    if not args.formal and args.output.resolve() == DEFAULT_OUTPUT.resolve():
        raise ValueError("non-formal run cannot write the formal output path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    stage0_hash = sha256(STAGE0)
    stage0 = json.loads(STAGE0.read_text())
    if stage0_hash != STAGE0_SHA or not stage0["stage0_pass"]:
        raise RuntimeError("invalid Stage-0 prerequisite")
    comparison_stage0_hash = sha256(COMPARISON_STAGE0)
    comparison_stage0 = json.loads(COMPARISON_STAGE0.read_text())
    if comparison_stage0_hash != COMPARISON_STAGE0_SHA or not comparison_stage0["comparison_pass"]:
        raise RuntimeError("invalid IPA comparison Stage-0 prerequisite")
    integrity_valid = False
    integrity_hash = None
    if args.formal:
        manifest = json.loads(INTEGRITY.read_text())
        observed = {
            "source_sha256": sha256(Path(__file__)),
            "preregistration_sha256": sha256(PREREG),
            "stage0_sha256": stage0_hash,
            "comparison_stage0_sha256": comparison_stage0_hash,
        }
        integrity_valid = all(observed[name] == manifest[name] for name in observed)
        if not integrity_valid:
            raise RuntimeError(f"formal integrity mismatch: {observed}")
        integrity_hash = sha256(INTEGRITY)
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name()
    if args.formal and "H100" not in device_name:
        raise RuntimeError(f"formal run requires H100, found {device_name}")
    runs: dict[str, dict[str, dict]] = {}
    dataset_audits: dict[str, dict] = {}
    for seed in SEEDS:
        train_data = generate_dataset(seed + 100_000, args.train_examples, balanced=True)
        validation = {}
        for task_id, task in enumerate(TASKS):
            raw = generate_dataset(seed + 200_000 + task_id * 10_000, args.validation_examples * 4, balanced=True)
            ids = torch.where(raw.task_ids == task_id)[0]
            validation[task] = raw.index(ids)
            if len(validation[task]) != args.validation_examples:
                raise RuntimeError("validation slice count mismatch")
        dataset_audits[str(seed)] = distributed_oracle_probe(validation["distributed"])
        if args.formal and not distributed_oracle_valid(dataset_audits[str(seed)]):
            raise RuntimeError(f"distributed generator oracle failed before training: {dataset_audits[str(seed)]}")
        runs[str(seed)] = {}
        for arm in ARMS:
            runs[str(seed)][arm] = train_one(
                arm, seed, train_data, validation, device,
                args.steps, args.batch_size, args.eval_batch_size,
            )
            print(seed, arm, {task: runs[str(seed)][arm]["evaluations"][task]["accuracy"] for task in TASKS}, flush=True)
    formal_configuration_valid = (
        args.formal
        and args.output.resolve() == DEFAULT_OUTPUT.resolve()
        and "H100" in device_name
        and integrity_valid
    )
    summary, decision = summarize(
        runs,
        dataset_audits,
        prerequisites_valid=True,
        formal_configuration_valid=formal_configuration_valid,
    )
    result = {
        "schema": "staggered-power-evidence-learning-v2",
        "formal": args.formal,
        "args": vars(args) | {"output": str(args.output)},
        "source_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREG),
        "stage0_sha256": stage0_hash,
        "comparison_stage0_sha256": comparison_stage0_hash,
        "integrity_manifest_sha256": integrity_hash,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device_name,
        },
        "model": {"sequence": SEQ, "chunk": CHUNK, "input": DIN, "heads": HEADS, "qk": QK, "value": VD},
        "runs": runs,
        "dataset_audits": dataset_audits,
        "summary": summary,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
