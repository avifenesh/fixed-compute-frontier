#!/usr/bin/env python3
"""Matched learned structured screen for address-factored transport attention."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
STAGE0 = ROOT / "results" / "address-factored-transport-stage0.json"
STAGE0_SHA = "3efc18684ab8f1af34742c24f97138e60ad7b60b8e3ebc7a9a8949da4e2defbd"
PREREGISTRATION = ROOT / "results" / "address-factored-transport-learning-preregistration.md"
PREREGISTRATION_SHA = "fb931993d65e34f4e0e493d73c4df8939683c750f16d9d5bbb7cb7f49bcf2739"
DEFAULT_OUTPUT = ROOT / "results" / "address-factored-transport-learning.json"

SEEDS = (1201, 3253, 4969, 7753, 9901)
ARMS = (
    "split12",
    "split12_temp2",
    "hybrid_afta",
    "hybrid_common1",
    "hybrid_ordinary0",
)
TASKS = ("shared_record", "independent_record", "ordinary_record")


@dataclass(frozen=True)
class Config:
    context_length: int = 83
    records: int = 16
    cell_width: int = 5
    relations: int = 4
    key_dim: int = 64
    head_dim: int = 16
    total_width: int = 192
    train_steps: int = 1000
    batch_size: int = 256
    eval_examples: int = 8192
    train_noise: float = 1.25
    hard_noise: float = 1.75
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    independence_bound: float = 0.06

    @property
    def input_dim(self) -> int:
        return self.relations * self.key_dim + self.relations + self.relations


@dataclass
class Batch:
    context: torch.Tensor
    query: torch.Tensor
    labels: torch.Tensor
    task_ids: torch.Tensor


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_hash(batch: Batch) -> str:
    digest = hashlib.sha256()
    for tensor in (batch.context, batch.query, batch.labels, batch.task_ids):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rademacher(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    value = torch.randint(0, 2, shape, generator=generator, device=device)
    return value.to(torch.float32) * 2.0 - 1.0


def make_batch(
    config: Config,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    *,
    task_id: int | None,
    noise: float,
) -> Batch:
    if task_id is None:
        task_ids = torch.arange(batch_size, device=device) % len(TASKS)
        task_ids = task_ids[
            torch.randperm(batch_size, generator=generator, device=device)
        ]
    else:
        task_ids = torch.full((batch_size,), task_id, dtype=torch.long, device=device)

    context = torch.zeros(
        (batch_size, config.context_length, config.input_dim), device=device
    )
    query = torch.zeros((batch_size, config.input_dim), device=device)
    codes = rademacher(
        (batch_size, config.records, config.key_dim), generator, device
    )
    tuple_ids = torch.rand(
        (batch_size, config.records), generator=generator, device=device
    ).argsort(dim=-1)
    bit_positions = torch.arange(config.relations, device=device)
    bits = (
        ((tuple_ids[:, :, None] >> bit_positions[None, None, :]) & 1).to(torch.float32)
        * 2.0
        - 1.0
    )
    bases = torch.arange(config.records, device=device) * config.cell_width
    repeated_codes = codes[:, :, None, :].expand(
        -1, -1, config.relations, -1
    ).reshape(batch_size, config.records, config.relations * config.key_dim)
    context[:, bases, : config.relations * config.key_dim] = repeated_codes

    payload_start = config.relations * config.key_dim
    type_start = payload_start + config.relations
    for relation in range(config.relations):
        positions = bases + relation
        context[:, positions, payload_start + relation] = bits[:, :, relation]
        context[:, positions, type_start + relation] = 1.0

    shared_mask = task_ids.ne(1)
    common_targets = torch.randint(
        0, config.records, (batch_size,), generator=generator, device=device
    )
    independent_targets = torch.randint(
        0,
        config.records,
        (batch_size, config.relations),
        generator=generator,
        device=device,
    )
    targets = independent_targets.clone()
    targets[shared_mask] = common_targets[shared_mask].unsqueeze(1).expand(
        -1, config.relations
    )

    batch_ids = torch.arange(batch_size, device=device)[:, None]
    relation_ids = torch.arange(config.relations, device=device)[None, :]
    query_codes = codes[batch_ids, targets, :]
    query_codes = query_codes + noise * torch.randn(
        query_codes.shape, generator=generator, device=device
    )
    query[:, : config.relations * config.key_dim] = query_codes.reshape(
        batch_size, config.relations * config.key_dim
    )

    ordinary_rows = task_ids.eq(2)
    if ordinary_rows.any():
        context[ordinary_rows, :, payload_start : payload_start + config.relations] = 0.0
        ordinary_bits = bits[ordinary_rows]
        ordinary_context = context[ordinary_rows]
        for relation in range(config.relations):
            ordinary_context[:, bases, payload_start + relation] = ordinary_bits[:, :, relation]
        context[ordinary_rows] = ordinary_context

    target_bits = bits[batch_ids, targets, relation_ids]
    labels = torch.zeros((batch_size,), dtype=torch.long, device=device)
    for relation in range(config.relations):
        labels += target_bits[:, relation].gt(0).long() << relation
    return Batch(context=context, query=query, labels=labels, task_ids=task_ids)


def transported_read(
    attention: torch.Tensor, values: torch.Tensor, offset: int
) -> torch.Tensor:
    length = attention.shape[-1]
    source_start = max(0, -offset)
    source_end = min(length, length - offset)
    shifted_values = torch.zeros_like(values)
    shifted_values[:, source_start:source_end] = values[
        :, source_start + offset : source_end + offset
    ]
    return torch.einsum("bl,blg->bg", attention, shifted_values)


class AttentionClassifier(nn.Module):
    def __init__(self, config: Config, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.config = config
        self.arm = arm
        self.q_proj = nn.Linear(config.input_dim, config.total_width, bias=False)
        self.k_proj = nn.Linear(config.input_dim, config.total_width, bias=False)
        self.v_proj = nn.Linear(config.input_dim, config.total_width, bias=False)
        self.o_proj = nn.Linear(config.total_width, config.total_width, bias=False)
        self.norm = nn.RMSNorm(config.total_width)
        self.readout = nn.Linear(config.total_width, 16)

    def forward(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        *,
        ablate_wide: bool = False,
        return_scores: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)
        outputs: list[torch.Tensor] = []
        raw_scores: list[torch.Tensor] = []
        width = self.config.total_width
        head = self.config.head_dim

        if self.arm.startswith("split12"):
            temperature = 2.0 if self.arm == "split12_temp2" else 1.0
            for index in range(width // head):
                start = index * head
                score = torch.einsum(
                    "bg,blg->bl", q[:, start : start + head], k[:, :, start : start + head]
                ) / math.sqrt(head)
                raw_scores.append(score)
                attention = torch.softmax(score * temperature, dim=-1)
                outputs.append(
                    transported_read(attention, v[:, :, start : start + head], index % 4)
                )
        else:
            wide = 4 * head
            score = torch.einsum("bg,blg->bl", q[:, :wide], k[:, :, :wide]) / math.sqrt(wide)
            raw_scores.append(score)
            attention = torch.softmax(score, dim=-1)
            if self.arm == "hybrid_afta":
                offsets = (0, 1, 2, 3)
            elif self.arm == "hybrid_common1":
                offsets = (1, 1, 1, 1)
            elif self.arm == "hybrid_ordinary0":
                offsets = (0, 0, 0, 0)
            else:
                raise AssertionError(self.arm)
            for relation, offset in enumerate(offsets):
                start = relation * head
                value = transported_read(attention, v[:, :, start : start + head], offset)
                outputs.append(torch.zeros_like(value) if ablate_wide else value)
            for index in range(8):
                start = wide + index * head
                score = torch.einsum(
                    "bg,blg->bl", q[:, start : start + head], k[:, :, start : start + head]
                ) / math.sqrt(head)
                raw_scores.append(score)
                attention = torch.softmax(score, dim=-1)
                outputs.append(
                    transported_read(attention, v[:, :, start : start + head], index % 4)
                )
        joined = torch.cat(outputs, dim=-1)
        logits = self.readout(self.norm(self.o_proj(joined)))
        return logits, raw_scores if return_scores else []


@torch.inference_mode()
def evaluate(
    model: AttentionClassifier,
    config: Config,
    *,
    seed: int,
    task_id: int,
    noise: float,
    device: torch.device,
    ablate_wide: bool = False,
    query_ablation: str | None = None,
    collect_scores: bool = True,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    correct = 0
    total_loss = 0.0
    score_values: list[list[torch.Tensor]] | None = None
    labels_all = []
    processed = 0
    while processed < config.eval_examples:
        size = min(config.batch_size, config.eval_examples - processed)
        batch = make_batch(
            config, size, generator, device, task_id=task_id, noise=noise
        )
        if query_ablation == "zero":
            model_query = torch.zeros_like(batch.query)
        elif query_ablation == "permute_examples":
            model_query = torch.roll(batch.query, shifts=1, dims=0)
        elif query_ablation is None:
            model_query = batch.query
        else:
            raise ValueError(query_ablation)
        logits, scores = model(
            batch.context,
            model_query,
            ablate_wide=ablate_wide,
            return_scores=collect_scores,
        )
        total_loss += float(F.cross_entropy(logits, batch.labels, reduction="sum").item())
        correct += int(logits.argmax(dim=-1).eq(batch.labels).sum().item())
        labels_all.append(batch.labels.cpu())
        if collect_scores:
            if score_values is None:
                score_values = [[] for _ in scores]
            for index, score in enumerate(scores):
                score_values[index].append(score.detach().float().cpu())
        processed += size
    score_stats = []
    if collect_scores:
        assert score_values is not None
        for chunks in score_values:
            flat = torch.cat(chunks).flatten()
            score_stats.append(
                {
                    "rms": float(flat.square().mean().sqrt().item()),
                    "abs_q99": float(torch.quantile(flat.abs(), 0.99).item()),
                }
            )
    labels = torch.cat(labels_all)
    class_counts = torch.bincount(labels, minlength=16)
    return {
        "accuracy": correct / config.eval_examples,
        "loss": total_loss / config.eval_examples,
        "class_prior_max": float(class_counts.max().item() / config.eval_examples),
        "score_stats": score_stats,
    }


@torch.inference_mode()
def audit_dataset(
    config: Config,
    *,
    seed: int,
    task_id: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Stream an empirical key-blind shortcut audit without retaining contexts."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    features = config.relations * config.key_dim
    sum_x = torch.zeros(features, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros_like(sum_x)
    sum_y = torch.zeros(config.relations, dtype=torch.float64, device=device)
    sum_y2 = torch.zeros_like(sum_y)
    sum_xy = torch.zeros(features, config.relations, dtype=torch.float64, device=device)
    class_counts = torch.zeros(16, dtype=torch.long, device=device)
    processed = 0
    while processed < config.eval_examples:
        size = min(config.batch_size, config.eval_examples - processed)
        batch = make_batch(
            config,
            size,
            generator,
            device,
            task_id=task_id,
            noise=config.train_noise,
        )
        x = batch.query[:, :features].to(torch.float64)
        shifts = torch.arange(config.relations, device=device)
        y = ((batch.labels[:, None] >> shifts[None, :]) & 1).to(torch.float64)
        sum_x += x.sum(dim=0)
        sum_x2 += x.square().sum(dim=0)
        sum_y += y.sum(dim=0)
        sum_y2 += y.square().sum(dim=0)
        sum_xy += x.T @ y
        class_counts += torch.bincount(batch.labels, minlength=16)
        processed += size
    count = float(processed)
    mean_x = sum_x / count
    mean_y = sum_y / count
    covariance = sum_xy / count - mean_x[:, None] * mean_y[None, :]
    variance_x = (sum_x2 / count - mean_x.square()).clamp_min(0.0)
    variance_y = (sum_y2 / count - mean_y.square()).clamp_min(0.0)
    denominator = torch.sqrt(variance_x[:, None] * variance_y[None, :])
    correlation = torch.where(
        denominator > 0,
        covariance / denominator,
        torch.zeros_like(covariance),
    )
    return {
        "examples": processed,
        "class_prior_max": float(class_counts.max().item() / processed),
        "query_label_max_abs_correlation": float(correlation.abs().max().item()),
    }


def symbolic_ledger(config: Config, arm: str) -> dict[str, int]:
    hybrid = arm.startswith("hybrid_")
    if arm.startswith("split12"):
        offsets = (0, 1, 2, 3) * 3
        wide_offsets: tuple[int, ...] = ()
    elif arm == "hybrid_afta":
        wide_offsets = (0, 1, 2, 3)
        offsets = wide_offsets + (0, 1, 2, 3) * 2
    elif arm == "hybrid_common1":
        wide_offsets = (1, 1, 1, 1)
        offsets = wide_offsets + (0, 1, 2, 3) * 2
    elif arm == "hybrid_ordinary0":
        wide_offsets = (0, 0, 0, 0)
        offsets = wide_offsets + (0, 1, 2, 3) * 2
    else:
        raise ValueError(arm)
    maps = 9 if hybrid else 12
    return {
        "q_width": config.total_width,
        "k_width": config.total_width,
        "v_width": config.total_width,
        "output_width": config.total_width,
        "qk_scalar_macs_per_query": config.context_length * config.total_width,
        "pv_scalar_macs_per_query": config.context_length * config.total_width,
        "kv_cache_scalars_per_token": 2 * config.total_width,
        "attention_maps_per_query": maps,
        "attention_score_scalars_per_query": config.context_length * maps,
        "softmax_elements_per_query": config.context_length * maps,
        "materialized_attention_bytes_fp32_per_query": 4 * config.context_length * maps,
        "transported_group_reductions": len(offsets),
        "logical_nonzero_pv_terms_per_query": config.head_dim
        * sum(config.context_length - offset for offset in offsets),
        "nonzero_shift_groups": sum(offset != 0 for offset in offsets),
        "wide_value_channel_groups": config.relations if hybrid else 0,
        "distinct_wide_offsets": len(set(wide_offsets)),
    }


def train_arm(
    arm: str,
    config: Config,
    seed: int,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[AttentionClassifier, str]:
    model = AttentionClassifier(config, arm).to(device)
    model.load_state_dict(base_state)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 100_000 + 17)
    data_trace = hashlib.sha256()
    for step in range(config.train_steps):
        batch = make_batch(
            config,
            config.batch_size,
            generator,
            device,
            task_id=None,
            noise=config.train_noise,
        )
        data_trace.update(step.to_bytes(8, byteorder="little"))
        data_trace.update(generator.get_state().cpu().contiguous().numpy().tobytes())
        if step in (0, config.train_steps - 1):
            data_trace.update(batch_hash(batch).encode())
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.context, batch.query)
        loss = F.cross_entropy(logits, batch.labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss for {arm} seed {seed} step {step}")
        loss.backward()
        optimizer.step()
    return model, data_trace.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = Config()
    if sha256(STAGE0) != STAGE0_SHA or not json.loads(STAGE0.read_text())["pass"]:
        raise RuntimeError("frozen Stage 0 prerequisite mismatch")
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA:
        raise RuntimeError("frozen learned-screen preregistration mismatch")

    torch.use_deterministic_algorithms(True)
    runs: list[dict[str, Any]] = []
    initialization_hashes: dict[str, dict[str, str]] = {}
    parameter_counts: dict[str, int] = {}
    ledgers = {arm: symbolic_ledger(config, arm) for arm in ARMS}
    diagnostic_hashes: dict[str, dict[str, dict[str, str]]] = {}
    training_data_traces: dict[str, dict[str, str]] = {}
    dataset_audits: dict[str, dict[str, dict[str, float | int]]] = {}

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        template = AttentionClassifier(config, ARMS[0]).to(device)
        base_state = copy.deepcopy(template.state_dict())
        base_hash = tensor_hash(base_state)
        initialization_hashes[str(seed)] = {}
        diagnostic_hashes[str(seed)] = {}
        training_data_traces[str(seed)] = {}
        dataset_audits[str(seed)] = {}

        for task_id, task in enumerate(TASKS):
            dataset_audits[str(seed)][task] = audit_dataset(
                config,
                seed=seed * 10_000 + 700 + task_id,
                task_id=task_id,
                device=device,
            )

        for arm in ARMS:
            probe = AttentionClassifier(config, arm).to(device)
            probe.load_state_dict(base_state)
            initialization_hashes[str(seed)][arm] = tensor_hash(probe.state_dict())
            parameter_counts[arm] = sum(parameter.numel() for parameter in probe.parameters())
            if initialization_hashes[str(seed)][arm] != base_hash:
                raise AssertionError("initialization mismatch")
            diagnostic_hashes[str(seed)][arm] = {}
            for diagnostic_task_id, diagnostic_task in ((None, "mixed"), *enumerate(TASKS)):
                diagnostic_generator = torch.Generator(device=device)
                diagnostic_generator.manual_seed(
                    seed * 1_000_000 + 31 + (3 if diagnostic_task_id is None else diagnostic_task_id)
                )
                diagnostic = make_batch(
                    config,
                    32,
                    diagnostic_generator,
                    device,
                    task_id=diagnostic_task_id,
                    noise=config.train_noise,
                )
                diagnostic_hashes[str(seed)][arm][diagnostic_task] = batch_hash(diagnostic)
            model, data_trace = train_arm(arm, config, seed, base_state, device)
            training_data_traces[str(seed)][arm] = data_trace
            for task_id, task in enumerate(TASKS):
                metrics = evaluate(
                    model,
                    config,
                    seed=seed * 10_000 + 500 + task_id,
                    task_id=task_id,
                    noise=config.train_noise,
                    device=device,
                )
                runs.append(
                    {"seed": seed, "arm": arm, "task": task, "noise": config.train_noise, **metrics}
                )
                for query_ablation in ("zero", "permute_examples"):
                    ablated_query_metrics = evaluate(
                        model,
                        config,
                        seed=seed * 10_000 + 500 + task_id,
                        task_id=task_id,
                        noise=config.train_noise,
                        device=device,
                        query_ablation=query_ablation,
                        collect_scores=False,
                    )
                    runs.append(
                        {
                            "seed": seed,
                            "arm": f"{arm}_query_{query_ablation}",
                            "base_arm": arm,
                            "task": task,
                            "noise": config.train_noise,
                            "query_ablation": query_ablation,
                            **ablated_query_metrics,
                        }
                    )
            hard = evaluate(
                model,
                config,
                seed=seed * 10_000 + 900,
                task_id=0,
                noise=config.hard_noise,
                device=device,
            )
            runs.append(
                {"seed": seed, "arm": arm, "task": "shared_record_hard", "noise": config.hard_noise, **hard}
            )
            if arm == "hybrid_afta":
                ablated = evaluate(
                    model,
                    config,
                    seed=seed * 10_000 + 500,
                    task_id=0,
                    noise=config.train_noise,
                    device=device,
                    ablate_wide=True,
                )
                runs.append(
                    {"seed": seed, "arm": "hybrid_afta_wide_ablated", "task": "shared_record", "noise": config.train_noise, **ablated}
                )

    def rows(arm: str, task: str) -> list[dict[str, Any]]:
        return [run for run in runs if run["arm"] == arm and run["task"] == task]

    summaries: dict[str, Any] = {}
    for arm in ARMS:
        summaries[arm] = {}
        for task in (*TASKS, "shared_record_hard"):
            selected = rows(arm, task)
            summaries[arm][task] = {
                "mean_accuracy": mean([run["accuracy"] for run in selected]),
                "accuracies": [run["accuracy"] for run in selected],
                "mean_loss": mean([run["loss"] for run in selected]),
            }
        summaries[arm]["overall"] = mean(
            [summaries[arm][task]["mean_accuracy"] for task in TASKS]
        )

    candidate = "hybrid_afta"
    competitors = [arm for arm in ARMS if arm != candidate]
    best_shared = max(
        competitors, key=lambda arm: summaries[arm]["shared_record"]["mean_accuracy"]
    )
    best_hard = max(
        competitors, key=lambda arm: summaries[arm]["shared_record_hard"]["mean_accuracy"]
    )
    candidate_shared = rows(candidate, "shared_record")
    candidate_independent = rows(candidate, "independent_record")
    baseline_independent = rows("split12", "independent_record")
    candidate_ordinary = rows(candidate, "ordinary_record")
    baseline_ordinary = rows("split12", "ordinary_record")
    ablated_rows = rows("hybrid_afta_wide_ablated", "shared_record")

    score_guard = True
    for task in (*TASKS, "shared_record_hard"):
        candidate_task = rows(candidate, task)
        baseline_task = rows("split12", task)
        for cand, base in zip(candidate_task, baseline_task, strict=True):
            wide_stats = cand["score_stats"][0]
            baseline_rms = float(np.median([item["rms"] for item in base["score_stats"]]))
            baseline_q99 = float(np.median([item["abs_q99"] for item in base["score_stats"]]))
            score_guard &= 0.5 <= wide_stats["rms"] / baseline_rms <= 2.0
            score_guard &= 0.5 <= wide_stats["abs_q99"] / baseline_q99 <= 2.0

    class_prior_ok = all(
        audit["class_prior_max"] <= 0.075
        for per_seed in dataset_audits.values()
        for audit in per_seed.values()
    )
    independence_ok = all(
        audit["query_label_max_abs_correlation"] <= config.independence_bound
        for per_seed in dataset_audits.values()
        for audit in per_seed.values()
    )
    address_ablation_rows = [
        run
        for run in runs
        if run.get("query_ablation") in ("zero", "permute_examples")
    ]
    expected_address_ablations = len(SEEDS) * len(ARMS) * len(TASKS) * 2
    diagnostic_data_equal = all(
        len({per_arm[task] for per_arm in per_seed.values()}) == 1
        for per_seed in diagnostic_hashes.values()
        for task in ("mixed", *TASKS)
    )
    training_data_equal = all(
        len(set(per_seed.values())) == 1 for per_seed in training_data_traces.values()
    )
    core_ledger_fields = (
        "q_width",
        "k_width",
        "v_width",
        "output_width",
        "qk_scalar_macs_per_query",
        "pv_scalar_macs_per_query",
        "kv_cache_scalars_per_token",
    )
    core_ledger_equal = all(
        len({ledger[field] for ledger in ledgers.values()}) == 1
        for field in core_ledger_fields
    )
    max_record_position = (config.records - 1) * config.cell_width + config.relations - 1
    formal_config = (
        config.relations == 4
        and config.records == 2**config.relations
        and config.total_width == 12 * config.head_dim
        and config.total_width == 4 * config.head_dim + 8 * config.head_dim
        and max_record_position < config.context_length
        and config.input_dim
        == config.relations * config.key_dim + 2 * config.relations
    )
    stage0_valid = sha256(STAGE0) == STAGE0_SHA and json.loads(STAGE0.read_text())["pass"]
    gates = {
        "integrity": (
            stage0_valid
            and formal_config
            and len(set(parameter_counts.values())) == 1
            and all(len(set(per_seed.values())) == 1 for per_seed in initialization_hashes.values())
            and diagnostic_data_equal
            and training_data_equal
            and core_ledger_equal
            and all(math.isfinite(run["accuracy"]) and math.isfinite(run["loss"]) for run in runs)
        ),
        "shared_gain": (
            summaries[candidate]["shared_record"]["mean_accuracy"]
            - summaries[best_shared]["shared_record"]["mean_accuracy"] >= 0.05
            and all(
                cand["accuracy"]
                > max(
                    next(
                        run["accuracy"]
                        for run in rows(arm, "shared_record")
                        if run["seed"] == cand["seed"]
                    )
                    for arm in competitors
                )
                for cand in candidate_shared
            )
        ),
        "hard_shared_gain": (
            summaries[candidate]["shared_record_hard"]["mean_accuracy"]
            - summaries[best_hard]["shared_record_hard"]["mean_accuracy"] >= 0.05
        ),
        "independent_protection": (
            summaries[candidate]["independent_record"]["mean_accuracy"]
            >= summaries["split12"]["independent_record"]["mean_accuracy"] - 0.01
            and all(cand["accuracy"] >= base["accuracy"] - 0.02 for cand, base in zip(candidate_independent, baseline_independent, strict=True))
        ),
        "ordinary_protection": (
            summaries[candidate]["ordinary_record"]["mean_accuracy"]
            >= summaries["split12"]["ordinary_record"]["mean_accuracy"] - 0.01
            and all(cand["accuracy"] >= base["accuracy"] - 0.02 for cand, base in zip(candidate_ordinary, baseline_ordinary, strict=True))
        ),
        "overall_gain": summaries[candidate]["overall"] - summaries["split12"]["overall"] >= 0.02,
        "causal_use": all(cand["accuracy"] - ablated["accuracy"] >= 0.05 for cand, ablated in zip(candidate_shared, ablated_rows, strict=True)),
        "score_guard": bool(score_guard),
        "no_shortcut": (
            class_prior_ok
            and independence_ok
            and len(address_ablation_rows) == expected_address_ablations
            and all(run["accuracy"] <= 0.075 for run in address_ablation_rows)
        ),
    }
    result = {
        "schema": "address-factored-transport-learning-v1",
        "stage0_sha256": STAGE0_SHA,
        "preregistration_sha256": PREREGISTRATION_SHA,
        "source_sha256": sha256(Path(__file__)),
        "config": asdict(config) | {"input_dim": config.input_dim},
        "seeds": SEEDS,
        "arms": ARMS,
        "tasks": TASKS,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "parameter_counts": parameter_counts,
        "symbolic_ledgers": ledgers,
        "initialization_hashes": initialization_hashes,
        "diagnostic_data_hashes": diagnostic_hashes,
        "training_data_trace_hashes": training_data_traces,
        "dataset_audits": dataset_audits,
        "runs": runs,
        "summary": summaries,
        "best_shared_competitor": best_shared,
        "best_hard_competitor": best_hard,
        "gates": gates,
        "pass": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "pass": result["pass"], "summary": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
