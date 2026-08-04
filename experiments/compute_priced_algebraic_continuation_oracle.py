#!/usr/bin/env python3
"""Oracle gradient-structure gate for compute-priced algebraic continuation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaConfig


VOCABULARY = 49_152
WIDTHS = (192, 384, 768)
LAYERS = 6
HEAD_DIM = 64
SEQUENCE_LENGTH = 256
BATCH_SIZE = 4
MEASUREMENT_BATCH_SIZE = 16
STEPS = 512
CHECKPOINTS = (0, 64, 512)
SAMPLED_LAYERS = (0, 3, 5)
BLOCK = 16
QUALITY = 0.90
SEED = 260_730
RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 384)
KRONECKER_RANKS = (1, 2, 4, 8)
SPARSE_FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 0.90)
MIXED_SPARSE_FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.35)
PREREGISTRATION = Path("results/compute-priced-algebraic-continuation-oracle-preregistration.md")
TEST_SOURCE = Path("tests/test_compute_priced_algebraic_continuation_oracle.py")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TokenStream:
    def __init__(self, path: Path, sequence_length: int) -> None:
        self.tokens = np.memmap(path, mode="r", dtype=np.uint16)
        self.width = sequence_length + 1

    def batch(
        self, index: int, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first = index * batch_size * self.width
        last = first + batch_size * self.width
        if last > self.tokens.size:
            raise IndexError("token stream exhausted")
        host = np.asarray(self.tokens[first:last], dtype=np.int64).reshape(
            batch_size, self.width
        )
        sequence = torch.from_numpy(host).to(device)
        return sequence[:, :-1], sequence[:, 1:]


def factor_pair(value: int) -> tuple[int, int]:
    """Return near-balanced integer factors in ascending order."""
    best = (1, value)
    for lower in range(1, math.isqrt(value) + 1):
        if value % lower == 0:
            best = (lower, value // lower)
    return best


def direction_metrics(gradient: torch.Tensor, direction: torch.Tensor) -> dict[str, float]:
    gradient32 = gradient.float()
    direction32 = direction.float()
    gradient_energy = float(gradient32.square().sum())
    direction_energy = float(direction32.square().sum())
    dot = float((gradient32 * direction32).sum())
    cosine_squared = (
        0.0
        if gradient_energy == 0.0 or direction_energy == 0.0
        else dot * dot / (gradient_energy * direction_energy)
    )
    return {
        "gradient_energy": gradient_energy,
        "direction_energy": direction_energy,
        "dot": dot,
        "cosine_squared": min(1.0, max(0.0, cosine_squared)),
    }


def block_sparse_direction(
    matrix: torch.Tensor, fraction: float, block: int = BLOCK
) -> tuple[torch.Tensor, int]:
    rows, columns = matrix.shape
    if rows % block or columns % block:
        raise ValueError("matrix dimensions must be divisible by block size")
    row_blocks, column_blocks = rows // block, columns // block
    blocks = matrix.reshape(row_blocks, block, column_blocks, block).permute(
        0, 2, 1, 3
    )
    energy = blocks.float().square().sum(dim=(-1, -2))
    count = max(1, math.ceil(fraction * energy.numel()))
    selected = torch.topk(energy.flatten(), count, sorted=False).indices
    block_mask = torch.zeros(energy.numel(), dtype=torch.bool, device=matrix.device)
    block_mask[selected] = True
    block_mask = block_mask.reshape(row_blocks, column_blocks)
    mask = (
        block_mask[:, :, None, None]
        .expand(row_blocks, column_blocks, block, block)
        .permute(0, 2, 1, 3)
        .reshape(rows, columns)
    )
    return matrix * mask, count * block * block


def kronecker_approximation(
    matrix: torch.Tensor, rank: int
) -> tuple[torch.Tensor, float, dict[str, int]]:
    rows, columns = matrix.shape
    row_low, row_high = factor_pair(rows)
    column_low, column_high = factor_pair(columns)
    m1, m2 = row_high, row_low
    n1, n2 = column_low, column_high
    rearranged = (
        matrix.reshape(m1, m2, n1, n2)
        .permute(0, 2, 1, 3)
        .reshape(m1 * n1, m2 * n2)
    )
    u, singular, vh = torch.linalg.svd(rearranged, full_matrices=False)
    effective_rank = min(rank, singular.numel())
    approximation = (
        (u[:, :effective_rank] * singular[:effective_rank])
        @ vh[:effective_rank]
    )
    restored = (
        approximation.reshape(m1, n1, m2, n2)
        .permute(0, 2, 1, 3)
        .reshape(rows, columns)
    )
    cost = effective_rank * (1.0 / m1 + 1.0 / n2)
    return restored, cost, {
        "rank": effective_rank,
        "m1": m1,
        "m2": m2,
        "n1": n1,
        "n2": n2,
    }


def candidate_record(
    gradient: torch.Tensor,
    direction: torch.Tensor,
    family: str,
    cost_ratio: float,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "family": family,
        "cost_ratio": float(cost_ratio),
        **metadata,
        **direction_metrics(gradient, direction),
    }


def analyze_matrix(
    gradient: torch.Tensor, *, include_candidates: bool = True
) -> dict[str, Any]:
    gradient = gradient.detach().float()
    rows, columns = gradient.shape
    dense_cost = rows * columns
    candidates: list[dict[str, Any]] = [
        candidate_record(gradient, gradient, "dense", 1.0)
    ]

    u, singular, vh = torch.linalg.svd(gradient, full_matrices=False)
    valid_ranks = [rank for rank in RANKS if rank <= singular.numel()]
    low_rank_directions: dict[int, torch.Tensor] = {}
    for rank in valid_ranks:
        low_rank = (u[:, :rank] * singular[:rank]) @ vh[:rank]
        low_rank_directions[rank] = low_rank
        candidates.append(
            candidate_record(
                gradient,
                low_rank,
                "low_rank",
                rank * (rows + columns) / dense_cost,
                rank=rank,
            )
        )

    for fraction in SPARSE_FRACTIONS:
        sparse, nonzero = block_sparse_direction(gradient, fraction)
        candidates.append(
            candidate_record(
                gradient,
                sparse,
                "block_sparse",
                nonzero / dense_cost,
                requested_fraction=fraction,
                nonzero=nonzero,
            )
        )

    for rank, low_rank in low_rank_directions.items():
        if rank > 128:
            continue
        residual = gradient - low_rank
        low_rank_cost = rank * (rows + columns) / dense_cost
        for fraction in MIXED_SPARSE_FRACTIONS:
            sparse_residual, nonzero = block_sparse_direction(residual, fraction)
            direction = low_rank + sparse_residual
            candidates.append(
                candidate_record(
                    gradient,
                    direction,
                    "low_rank_plus_block",
                    low_rank_cost + nonzero / dense_cost,
                    rank=rank,
                    requested_fraction=fraction,
                    nonzero=nonzero,
                )
            )

    row_low, row_high = factor_pair(rows)
    column_low, column_high = factor_pair(columns)
    m1, m2 = row_high, row_low
    n1, n2 = column_low, column_high
    rearranged = (
        gradient.reshape(m1, m2, n1, n2)
        .permute(0, 2, 1, 3)
        .reshape(m1 * n1, m2 * n2)
    )
    kron_u, kron_singular, kron_vh = torch.linalg.svd(
        rearranged, full_matrices=False
    )
    for requested_rank in KRONECKER_RANKS:
        rank = min(requested_rank, kron_singular.numel())
        rearranged_approximation = (
            (kron_u[:, :rank] * kron_singular[:rank]) @ kron_vh[:rank]
        )
        kronecker = (
            rearranged_approximation.reshape(m1, n1, m2, n2)
            .permute(0, 2, 1, 3)
            .reshape(rows, columns)
        )
        cost = rank * (1.0 / m1 + 1.0 / n2)
        candidates.append(
            candidate_record(
                gradient,
                kronecker,
                "kronecker",
                cost,
                rank=rank,
                m1=m1,
                m2=m2,
                n1=n1,
                n2=n2,
            )
        )

    admissible = [row for row in candidates if row["cosine_squared"] >= QUALITY]
    best = min(admissible, key=lambda row: row["cost_ratio"])
    low_rank_admissible = [
        row
        for row in candidates
        if row["family"] in {"low_rank", "dense"}
        and row["cosine_squared"] >= QUALITY
    ]
    best_low_rank = min(low_rank_admissible, key=lambda row: row["cost_ratio"])
    return {
        "shape": [rows, columns],
        "gradient_energy": float(gradient.square().sum()),
        "singular_values": [float(value) for value in singular[: min(32, len(singular))]],
        "best_at_quality": best,
        "best_low_rank_at_quality": best_low_rank,
        "candidates": candidates if include_candidates else [],
    }


def model_config(width: int) -> LlamaConfig:
    if width % HEAD_DIM or width % 3:
        raise ValueError("width must be divisible by head dimension and three")
    heads = width // HEAD_DIM
    return LlamaConfig(
        vocab_size=VOCABULARY,
        hidden_size=width,
        intermediate_size=width * 8 // 3,
        num_hidden_layers=LAYERS,
        num_attention_heads=heads,
        num_key_value_heads=max(1, heads // 3),
        head_dim=HEAD_DIM,
        max_position_embeddings=2048,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        initializer_range=1.0 / math.sqrt(width),
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        use_cache=False,
    )


def selected_matrix_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    selected = {}
    suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for name, parameter in model.named_parameters():
        if not any(name.startswith(f"model.layers.{layer}.") for layer in SAMPLED_LAYERS):
            continue
        if name.endswith(suffixes):
            selected[name] = parameter
    expected = len(SAMPLED_LAYERS) * len(suffixes)
    if len(selected) != expected:
        raise RuntimeError(f"selected {len(selected)} matrices, expected {expected}")
    return selected


def causal_loss(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def learning_rate_multiplier(step: int) -> float:
    if step < 32:
        return (step + 1) / 32
    progress = (step - 32) / max(STEPS - 33, 1)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def aggregate_matrix_results(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dense_macs = sum(value["shape"][0] * value["shape"][1] for value in rows.values())
    mixed_macs = sum(
        value["best_at_quality"]["cost_ratio"] * value["shape"][0] * value["shape"][1]
        for value in rows.values()
    )
    low_rank_macs = sum(
        value["best_low_rank_at_quality"]["cost_ratio"]
        * value["shape"][0]
        * value["shape"][1]
        for value in rows.values()
    )
    total_gradient_energy = sum(value["gradient_energy"] for value in rows.values())
    retained_descent = sum(
        value["gradient_energy"] * value["best_at_quality"]["cosine_squared"]
        for value in rows.values()
    )
    family_counts: dict[str, int] = {}
    for value in rows.values():
        family = value["best_at_quality"]["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    mixed_ratio = mixed_macs / dense_macs
    low_rank_ratio = low_rank_macs / dense_macs
    return {
        "matrices": len(rows),
        "dense_macs": dense_macs,
        "mixed_cost_ratio": mixed_ratio,
        "low_rank_only_cost_ratio": low_rank_ratio,
        "mixed_saving_vs_low_rank": 1.0 - mixed_ratio / low_rank_ratio,
        "gradient_energy_weighted_cosine_squared": retained_descent
        / total_gradient_energy,
        "selected_family_counts": family_counts,
    }


def frontier_projection(aggregate: dict[str, Any]) -> dict[str, float]:
    width = 4096
    layers = 32
    context = 4096
    vocabulary = VOCABULARY
    projection = layers * 10.5 * width * width
    fixed = layers * 2 * context * width + vocabulary * width
    projection_fraction = projection / (projection + fixed)
    structured = aggregate["mixed_cost_ratio"]
    refresh_interval = 64
    whole_step_ratio = (
        1.0
        - projection_fraction
        + projection_fraction * structured
        + projection_fraction * (1.0 - structured) / refresh_interval
    )
    quality = aggregate["gradient_energy_weighted_cosine_squared"]
    return {
        "projection_fraction": projection_fraction,
        "whole_step_cost_ratio_with_refresh": whole_step_ratio,
        "optimistic_descent_per_compute": quality / whole_step_ratio,
    }


def measure_checkpoint(
    model: torch.nn.Module,
    validation: TokenStream,
    device: torch.device,
    width: int,
    step: int,
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    inputs, targets = validation.batch(0, MEASUREMENT_BATCH_SIZE, device)
    loss = causal_loss(model, inputs, targets)
    loss.backward()
    matrices = {}
    for name, parameter in selected_matrix_parameters(model).items():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        matrices[name] = analyze_matrix(parameter.grad)
        print(
            json.dumps(
                {
                    "width": width,
                    "step": step,
                    "matrix": name,
                    "family": matrices[name]["best_at_quality"]["family"],
                    "cost": matrices[name]["best_at_quality"]["cost_ratio"],
                }
            ),
            flush=True,
        )
    model.zero_grad(set_to_none=True)
    aggregate = aggregate_matrix_results(matrices)
    aggregate["frontier_projection"] = frontier_projection(aggregate)
    return {
        "width": width,
        "step": step,
        "loss": float(loss.detach()),
        "aggregate": aggregate,
        "matrices": matrices,
    }


def isotropic_controls(device: torch.device) -> dict[str, Any]:
    rows = {}
    generator = torch.Generator(device=device).manual_seed(SEED + 900_000)
    for width in WIDTHS:
        intermediate = width * 8 // 3
        kv_width = width // 3
        for label, shape in {
            "square": (width, width),
            "kv": (kv_width, width),
            "up": (intermediate, width),
            "down": (width, intermediate),
        }.items():
            gradient = torch.randn(shape, generator=generator, device=device)
            rows[f"width_{width}_{label}"] = analyze_matrix(
                gradient, include_candidates=False
            )
    return {"matrices": rows, "aggregate": aggregate_matrix_results(rows)}


def train_width(
    width: int,
    train: TokenStream,
    validation: TokenStream,
    device: torch.device,
) -> dict[str, Any]:
    seed = SEED + width
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AutoModelForCausalLM.from_config(
        model_config(width), attn_implementation="sdpa"
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    measurements = []
    losses = []
    started = time.perf_counter()
    for step in range(STEPS + 1):
        if step in CHECKPOINTS:
            measurements.append(
                measure_checkpoint(model, validation, device, width, step)
            )
        if step == STEPS:
            break
        model.train()
        inputs, targets = train.batch(step, BATCH_SIZE, device)
        optimizer.param_groups[0]["lr"] = 3e-4 * learning_rate_multiplier(step)
        optimizer.zero_grad(set_to_none=True)
        loss = causal_loss(model, inputs, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError(f"non-finite training at width {width}, step {step}")
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 64 == 0:
            print(
                json.dumps(
                    {"width": width, "training_step": step + 1, "loss": losses[-1]}
                ),
                flush=True,
            )
    torch.cuda.synchronize()
    result = {
        "width": width,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "measurements": measurements,
        "training": {
            "initial_loss": measurements[0]["loss"],
            "terminal_loss": measurements[-1]["loss"],
            "mean_step_loss": float(np.mean(losses)),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(width_results: list[dict[str, Any]], isotropic: dict[str, Any]) -> dict[str, Any]:
    cells = {
        (row["width"], measurement["step"]): measurement["aggregate"]
        for row in width_results
        for measurement in row["measurements"]
    }
    cheap_cells = sum(value["mixed_cost_ratio"] <= 0.35 for value in cells.values())
    distinct_cells = sum(
        value["mixed_saving_vs_low_rank"] >= 0.25 for value in cells.values()
    )
    scaling_checks = {}
    for step in CHECKPOINTS:
        costs = [cells[(width, step)]["mixed_cost_ratio"] for width in WIDTHS]
        scaling_checks[str(step)] = costs[0] >= costs[1] >= costs[2] and costs[2] < costs[0]
    widest_efficiencies = [
        cells[(WIDTHS[-1], step)]["frontier_projection"]["optimistic_descent_per_compute"]
        for step in CHECKPOINTS
    ]
    gates = {
        "cost_at_most_35_percent_in_seven_of_nine_cells": cheap_cells >= 7,
        "mixed_beats_low_rank_by_25_percent_in_six_cells": distinct_cells >= 6,
        "structured_fraction_decreases_with_width_every_checkpoint": all(
            scaling_checks.values()
        ),
        "isotropic_control_requires_85_percent_dense_cost": isotropic["aggregate"][
            "mixed_cost_ratio"
        ]
        >= 0.85,
        "widest_projected_efficiency_at_least_2x_every_checkpoint": all(
            value >= 2.0 for value in widest_efficiencies
        ),
        "all_models_learned": all(
            row["training"]["terminal_loss"] < row["training"]["initial_loss"]
            for row in width_results
        ),
    }
    return {
        "cheap_cells": cheap_cells,
        "distinct_cells": distinct_cells,
        "scaling_checks": scaling_checks,
        "widest_projected_efficiencies": widest_efficiencies,
        "gates": gates,
        "advance": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/block-algebra-scratch-v4576/train.uint16.bin"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/block-algebra-scratch-v4576/validation.uint16.bin"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/compute-priced-algebraic-continuation-oracle.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    train = TokenStream(args.train_file, SEQUENCE_LENGTH)
    validation = TokenStream(args.validation_file, SEQUENCE_LENGTH)
    width_results = [
        train_width(width, train, validation, device) for width in WIDTHS
    ]
    isotropic = isotropic_controls(device)
    payload = {
        "schema": "compute-priced-algebraic-continuation-oracle-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "test_sha256": sha256_file(TEST_SOURCE),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "protocol": {
            "widths": WIDTHS,
            "layers": LAYERS,
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "measurement_batch_size": MEASUREMENT_BATCH_SIZE,
            "steps": STEPS,
            "checkpoints": CHECKPOINTS,
            "sampled_layers": SAMPLED_LAYERS,
            "quality": QUALITY,
            "seed": SEED,
        },
        "width_results": width_results,
        "isotropic_controls": isotropic,
    }
    payload["decision"] = decide(width_results, isotropic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
