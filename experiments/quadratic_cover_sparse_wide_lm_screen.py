#!/usr/bin/env python3
"""Frozen quality screen for a 2:4 sparse-wide quadratic mask code."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.cycle_factor_ffn_lm_screen import D, M, SEED
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger, write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


WIDTH = 1792
ARMS = ("random_sparse_wide", "quadratic_cover_sparse_wide")
PREDECESSOR = Path("results/cycle-factor-ffn-lm-screen.json")
PREDECESSOR_SHA256 = "c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029"
PREREGISTRATION = Path("results/quadratic-cover-sparse-wide-preregistration.md")

SUBSETS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _symmetric_42_codebook() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    same = [(subset, subset) for subset in SUBSETS for _ in range(3)]
    overlap_one = [
        (left, right)
        for left in SUBSETS
        for right in SUBSETS
        if len(set(left).intersection(right)) == 1
    ]
    result = same + overlap_one
    if len(result) != 42:
        raise AssertionError("bad symmetric codebook")
    return result


def _balanced_28_codebook() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    # Integer solution with 14 appearances of every gate/up coordinate and ten
    # supports for every degree-two monomial.  Together with 42 full copies of
    # the symmetric code this exactly fills WIDTH=1792.
    counts = (
        (2, (0, 1), (0, 1)),
        (2, (0, 1), (0, 2)),
        (2, (0, 1), (0, 3)),
        (2, (0, 1), (1, 2)),
        (2, (0, 2), (0, 2)),
        (2, (0, 3), (0, 3)),
        (2, (0, 3), (1, 3)),
        (2, (1, 2), (1, 2)),
        (2, (1, 2), (1, 3)),
        (2, (1, 3), (1, 3)),
        (4, (2, 3), (0, 2)),
        (2, (2, 3), (1, 3)),
        (2, (2, 3), (2, 3)),
    )
    result = []
    for count, left, right in counts:
        result.extend([(left, right)] * count)
    if len(result) != 28:
        raise AssertionError("bad remainder codebook")
    return result


def quadratic_cover_codebook() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    result = _symmetric_42_codebook() * 42 + _balanced_28_codebook()
    if len(result) != WIDTH:
        raise AssertionError("codebook does not fill the sparse width")
    return result


def degree_two_support_counts(
    codebook: list[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for left in range(4):
        for right in range(left, 4):
            count = 0
            for gate, up in codebook:
                if left == right:
                    count += int(left in gate and left in up)
                else:
                    count += int(
                        (left in gate and right in up)
                        or (right in gate and left in up)
                    )
            result[f"{left},{right}"] = count
    return result


def _local_permutation(layer_index: int, group_index: int) -> tuple[int, int, int, int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(91_019 + 10_007 * layer_index + 1_009 * group_index)
    return tuple(torch.randperm(4, generator=generator).tolist())


def quadratic_cover_masks(layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    codebook = quadratic_cover_codebook()
    gate = torch.zeros(WIDTH, D, dtype=torch.bool)
    up = torch.zeros_like(gate)
    for group in range(D // 4):
        permutation = _local_permutation(layer_index, group)
        offset = (97 * group + 193 * layer_index) % WIDTH
        for row in range(WIDTH):
            left, right = codebook[(row + offset) % WIDTH]
            base = 4 * group
            gate[row, [base + permutation[index] for index in left]] = True
            up[row, [base + permutation[index] for index in right]] = True
    return gate, up


def random_2of4_mask(rows: int, columns: int, seed: int) -> torch.Tensor:
    if columns % 4:
        raise ValueError("2:4 K dimension must be divisible by four")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    scores = torch.rand(rows, columns // 4, 4, generator=generator)
    selected = torch.topk(scores, 2, dim=-1, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    return mask.reshape(rows, columns)


def down_mask(layer_index: int) -> torch.Tensor:
    return random_2of4_mask(D, WIDTH, 313_337 + 10_007 * layer_index)


def serving_ledger() -> dict[str, int | bool | float]:
    dense_coordinates = 3 * D * M
    sparse_coordinates = 3 * D * WIDTH
    nonzeros = sparse_coordinates // 2
    groups = sparse_coordinates // 4
    value_bytes = 2 * nonzeros
    metadata_bytes = groups // 2  # four bits per group of four
    candidate_bytes = value_bytes + metadata_bytes
    baseline_bytes = 2 * dense_coordinates
    return {
        "baseline_width": M,
        "candidate_width": WIDTH,
        "width_ratio": WIDTH / M,
        "baseline_ffn_bytes_per_layer": baseline_bytes,
        "candidate_value_bytes_per_layer": value_bytes,
        "candidate_metadata_bytes_per_layer": metadata_bytes,
        "candidate_ffn_bytes_per_layer": candidate_bytes,
        "storage_slack_bytes_per_layer": baseline_bytes - candidate_bytes,
        "baseline_matrix_macs_per_token": dense_coordinates,
        "candidate_nonzero_macs_per_token": nonzeros,
        "nonzero_mac_ratio": nonzeros / dense_coordinates,
        "fits_bytes": candidate_bytes <= baseline_bytes,
        "fits_nonzero_macs": nonzeros <= dense_coordinates,
    }


class MaskedLinear(nn.Module):
    def __init__(self, rows: int, columns: int, mask: torch.Tensor, std: float) -> None:
        super().__init__()
        if mask.shape != (rows, columns):
            raise ValueError("mask shape mismatch")
        self.weight = nn.Parameter(torch.empty(rows, columns))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight * self.mask)


class SparseWideSwiGLU(nn.Module):
    def __init__(self, arm: str, layer_index: int, std: float) -> None:
        super().__init__()
        if arm == "quadratic_cover_sparse_wide":
            gate_mask, up_mask = quadratic_cover_masks(layer_index)
        elif arm == "random_sparse_wide":
            gate_mask = random_2of4_mask(WIDTH, D, 101_003 + 20_011 * layer_index)
            up_mask = random_2of4_mask(WIDTH, D, 101_009 + 20_011 * layer_index)
        else:
            raise ValueError(arm)
        shared_down_mask = down_mask(layer_index)
        input_std = std * math.sqrt(2.0)
        output_std = std * math.sqrt(2.0 * M / WIDTH)
        self.gate_proj = MaskedLinear(WIDTH, D, gate_mask, input_std)
        self.up_proj = MaskedLinear(WIDTH, D, up_mask, input_std)
        self.down_proj = MaskedLinear(D, WIDTH, shared_down_mask, output_std)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def build_arm(device: torch.device, arm: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    for layer_index, layer in enumerate(model.model.layers):
        layer.mlp = SparseWideSwiGLU(
            arm, layer_index, model.config.initializer_range
        ).to(device)
    model.config.use_cache = False
    return model


@torch.no_grad()
def diagnostics(model: nn.Module, validation: TokenFile, device: torch.device) -> dict[str, float]:
    model.eval()
    inputs, _ = validation.batch(0, 2, device)
    captured = []
    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        for layer in model.model.layers
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    rms = [float(value.float().square().mean().sqrt()) for value in captured]
    nonfinite = [float((~torch.isfinite(value)).float().mean()) for value in captured]
    return {
        "activation_rms_layer_median": float(np.median(rms)),
        "activation_rms_layer_max": max(rms),
        "nonfinite_fraction_layer_max": max(nonfinite),
    }


def train_arm(arm, args, train, validation, device):
    model = build_arm(device, arm, args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay, fused=True,
    )
    optimizer.param_groups[0]["base_lr"] = args.learning_rate
    evaluations = {"0": evaluate(model, validation, args.eval_batches, args.eval_batch_size, device)}
    initial_diagnostics = diagnostics(model, validation, device)
    optimizer.zero_grad(set_to_none=True)
    losses, norms, durations = [], [], []
    for step in range(args.steps):
        model.train()
        optimizer.param_groups[0]["lr"] = args.learning_rate * lr_multiplier(
            step, args.steps, args.warmup_steps
        )
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            inputs, targets = train.batch(
                step * args.gradient_accumulation + micro,
                args.micro_batch_size,
                device,
            )
            value = causal_loss(model, inputs, targets)
            (value / args.gradient_accumulation).backward()
            accumulated += float(value.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"nonfinite {arm} step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        norms.append(norm)
        durations.append(time.perf_counter() - started)
    evaluations[str(args.steps)] = evaluate(
        model, validation, args.eval_batches, args.eval_batch_size, device
    )
    terminal_diagnostics = diagnostics(model, validation, device)
    result = {
        "arm": arm,
        "training_shadow_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "evaluations": evaluations,
        "diagnostics": {"initial": initial_diagnostics, "terminal": terminal_diagnostics},
        "train": {
            "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "max_gradient_norm": max(norms),
            "elapsed_seconds": sum(durations),
            "tokens_per_second": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length / sum(durations),
            "nonfinite": False,
        },
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results, predecessor, protocol_valid):
    if set(results) != set(ARMS):
        return {"complete": False, "missing": sorted(set(ARMS) - set(results))}
    step = "305"
    dense = predecessor["arms"]["full_swiglu"]
    losses = {"full_swiglu": dense["evaluations"][step]["loss"]}
    losses.update({arm: results[arm]["evaluations"][step]["loss"] for arm in ARMS})
    comparisons = {
        arm: paired_loss_interval(results[arm], dense, step) for arm in ARMS
    }
    comparisons["cover_minus_random"] = paired_loss_interval(
        results["quadratic_cover_sparse_wide"], results["random_sparse_wide"], step
    )
    dense_loss = losses["full_swiglu"]
    sparse_pass = {}
    for arm in ARMS:
        improvement = (dense_loss - losses[arm]) / dense_loss
        sparse_pass[arm] = improvement >= 0.0005 and comparisons[arm]["upper_95"] < 0
    cover_gain = (
        losses["random_sparse_wide"] - losses["quadratic_cover_sparse_wide"]
    ) / losses["random_sparse_wide"]
    cover_pass = cover_gain >= 0.0005 and comparisons["cover_minus_random"]["upper_95"] < 0
    ledger = serving_ledger()
    finite = all(
        not result["train"]["nonfinite"]
        and result["diagnostics"]["terminal"]["nonfinite_fraction_layer_max"] == 0
        for result in results.values()
    )
    gates = {
        "protocol_integrity_valid": protocol_valid,
        "exact_sparse_ledger_fits": bool(ledger["fits_bytes"] and ledger["fits_nonzero_macs"]),
        "finite": finite,
        "random_sparse_beats_dense": sparse_pass["random_sparse_wide"],
        "quadratic_cover_beats_dense": sparse_pass["quadratic_cover_sparse_wide"],
        "quadratic_cover_beats_random_sparse": cover_pass,
    }
    return {
        "complete": True,
        "losses": losses,
        "relative_improvement_vs_dense": {
            arm: (dense_loss - losses[arm]) / dense_loss for arm in ARMS
        },
        "relative_cover_improvement_vs_random": cover_gain,
        "paired_intervals": comparisons,
        "gates": gates,
        "advance_sparse_wide_to_50m": protocol_valid and finite and bool(ledger["fits_bytes"] and ledger["fits_nonzero_macs"]) and any(sparse_pass.values()),
        "advance_quadratic_cover_refinement": protocol_valid and finite and sparse_pass["quadratic_cover_sparse_wide"] and cover_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/quadratic-cover-sparse-wide-lm-screen.json"))
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    expected = (
        args.steps, args.sequence_length, args.micro_batch_size,
        args.gradient_accumulation, args.eval_batch_size, args.eval_batches,
        args.warmup_steps, args.learning_rate, args.weight_decay,
        args.gradient_clip, args.seed,
    ) == (305, 512, 32, 2, 32, 128, 30, 3e-4, 0.1, 1.0, SEED)
    expected = (
        expected
        and torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
        and "H100" in torch.cuda.get_device_name()
    )
    if sha256_file(PREDECESSOR) != PREDECESSOR_SHA256:
        raise RuntimeError("frozen predecessor changed")
    predecessor = json.loads(PREDECESSOR.read_text())
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol_valid = expected and predecessor["protocol_valid"] and data_ledger["valid"]
    train = TokenFile(args.train_file, args.sequence_length)
    validation = TokenFile(args.validation_file, args.sequence_length)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    payload = {
        "schema": "quadratic-cover-sparse-wide-lm-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "predecessor_sha256": sha256_file(PREDECESSOR),
        "protocol_valid": protocol_valid,
        "data_ledger": data_ledger,
        "serving_ledger": serving_ledger(),
        "quadratic_cover": {
            "rows": len(quadratic_cover_codebook()),
            "degree_two_support_counts": degree_two_support_counts(quadratic_cover_codebook()),
        },
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, train, validation, device)
        payload["decision"] = decide(payload["arms"], predecessor, protocol_valid)
        write_payload(args.output, payload)
        print(json.dumps({"finished_arm": arm, "loss": payload["arms"][arm]["evaluations"]["305"]["loss"]}), flush=True)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

