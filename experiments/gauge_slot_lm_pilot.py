#!/usr/bin/env python3
"""Matched LM pilot for gauge-canonical, storage-funded G2 attention."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from experiments.gauge_slot_attention import (
    GaugeSlotAttention,
    replace_llama_attention,
)
from experiments.reflex_swiglu_lm_screen import (
    MODEL,
    MODEL_REVISION,
    TokenFile,
    evaluate,
    lr_multiplier,
    paired_loss_interval,
    validate_data_ledger,
    write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss, scratch_config


ARMS = ("raw_baseline", "canonical_bilinear_control", "gauge_slot_g2")
OUTPUT = Path("results/gauge-slot-lm-pilot.json")


def build_model(
    device: torch.device, arm: str
) -> tuple[nn.Module, list[GaugeSlotAttention]]:
    if arm not in ARMS:
        raise ValueError(f"invalid arm: {arm}")
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules: list[GaugeSlotAttention] = []
    if arm != "raw_baseline":
        modules = replace_llama_attention(model, arm)
    return model, modules


def optimizer_for(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = learning_rate
    return optimizer


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


@torch.no_grad()
def attention_diagnostics(
    model: nn.Module,
    modules: list[GaugeSlotAttention],
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    inputs, _ = validation_file.batch(0, 2, device)
    if modules:
        for module in modules:
            module.record_diagnostics = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, use_cache=False)
        records = [module.last_diagnostics for module in modules]
        for module in modules:
            module.record_diagnostics = False
        if any(record is None for record in records):
            raise RuntimeError("missing gauge-slot diagnostic")
        typed = [record for record in records if record is not None]
        summary = {
            key + "_layer_median": float(np.median([
                record[key] for record in typed
            ]))
            for key in typed[0]
        }
        summary["coefficient_abs_max_layer_max"] = max(
            record["coefficient_abs_max"] for record in typed
        )
        summary["key_abs_max_layer_max"] = max(
            record["key_abs_max"] for record in typed
        )
        summary["key_nonfinite_fraction_layer_max"] = max(
            record["key_nonfinite_fraction"] for record in typed
        )
        return summary

    captured: list[torch.Tensor] = []
    handles = [
        layer.self_attn.k_proj.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach())
        )
        for layer in model.model.layers
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    records = [value.float() for value in captured]
    return {
        "coefficient_rms_layer_median": 0.0,
        "coefficient_abs_max_layer_median": 0.0,
        "key_rms_layer_median": float(np.median([
            float(value.square().mean().sqrt()) for value in records
        ])),
        "key_abs_max_layer_median": float(np.median([
            float(value.abs().max()) for value in records
        ])),
        "key_nonfinite_fraction_layer_median": float(np.median([
            float((~torch.isfinite(value)).float().mean()) for value in records
        ])),
        "coefficient_abs_max_layer_max": 0.0,
        "key_abs_max_layer_max": max(float(value.abs().max()) for value in records),
        "key_nonfinite_fraction_layer_max": max(
            float((~torch.isfinite(value)).float().mean()) for value in records
        ),
    }


def evaluate_ablation(
    model: nn.Module,
    modules: list[GaugeSlotAttention],
    validation_file: TokenFile,
    args: argparse.Namespace,
    device: torch.device,
    coordinate: str,
) -> dict[str, Any]:
    saved = [module.key_packed.detach().clone() for module in modules]
    with torch.no_grad():
        for module in modules:
            indices = module.pivot_indices.reshape(-1, 2)
            if coordinate == "both":
                chosen = indices.reshape(-1)
            elif coordinate == "alpha":
                chosen = indices[:, 0]
            elif coordinate == "beta":
                chosen = indices[:, 1]
            else:
                raise ValueError(f"unknown ablation {coordinate}")
            module.key_packed.reshape(-1)[chosen] = 0.0
    result = evaluate(
        model,
        validation_file,
        args.eval_batches,
        args.eval_batch_size,
        device,
    )
    with torch.no_grad():
        for module, value in zip(modules, saved):
            module.key_packed.copy_(value)
    return result


def train_arm(
    arm: str,
    args: argparse.Namespace,
    train_file: TokenFile,
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(device, arm)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_tensors = len(list(model.parameters()))
    optimizer = optimizer_for(model, args.learning_rate, args.weight_decay)
    diagnostics = {
        "initial": attention_diagnostics(model, modules, validation_file, device)
    }
    evaluations = {
        "0": evaluate(
            model,
            validation_file,
            args.eval_batches,
            args.eval_batch_size,
            device,
        )
    }
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    gradient_norms: list[float] = []
    durations: list[float] = []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(
                batch_index, args.micro_batch_size, device
            )
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        ))
        if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite {arm} step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        gradient_norms.append(gradient_norm)
        durations.append(time.perf_counter() - started)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(
                model,
                validation_file,
                args.eval_batches,
                args.eval_batch_size,
                device,
            )
            print(json.dumps({
                "arm": arm,
                "step": step + 1,
                "train_loss": accumulated,
                "validation_loss": evaluations[str(step + 1)]["loss"],
                "gradient_norm": gradient_norm,
            }), flush=True)
    diagnostics["terminal"] = attention_diagnostics(
        model, modules, validation_file, device
    )
    ablations: dict[str, Any] = {}
    if arm == "gauge_slot_g2":
        for coordinate in ("alpha", "beta", "both"):
            ablations[f"zero_{coordinate}"] = evaluate_ablation(
                model, modules, validation_file, args, device, coordinate
            )
    prediction_tokens = (
        args.steps
        * args.gradient_accumulation
        * args.micro_batch_size
        * args.sequence_length
    )
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "parameter_tensors": parameter_tensors,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "packed_attention_values_per_layer": [
            module.chart_parameter_count() for module in modules
        ],
        "atlas_indices_per_layer": 0 if not modules else int(
            modules[0].pivot_columns.numel()
        ),
        "atlas_index_bits_per_layer": 0 if not modules else int(
            modules[0].pivot_columns.numel()
            * math.ceil(math.log2(modules[0].hidden_size))
        ),
        "runtime_scope": (
            "quality reference only; packed-kernel latency and metadata residency "
            "are separate admission gates"
        ),
        "evaluations": evaluations,
        "attention_diagnostics": diagnostics,
        "terminal_ablations": ablations,
        "train": {
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "max_loss": max(losses),
            "max_preclip_gradient_norm": max(gradient_norms),
            "elapsed_seconds": sum(durations),
            "tokens_per_second": prediction_tokens / sum(durations),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nonfinite": False,
        },
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if set(results) != set(ARMS):
        return {"complete": False, "missing_arms": sorted(set(ARMS) - set(results))}
    terminal = str(args.steps)
    losses = {
        arm: result["evaluations"][terminal]["loss"]
        for arm, result in results.items()
    }
    candidate = results["gauge_slot_g2"]
    control = results["canonical_bilinear_control"]
    baseline = results["raw_baseline"]
    candidate_vs_baseline = paired_loss_interval(candidate, baseline, terminal)
    candidate_vs_control = paired_loss_interval(candidate, control, terminal)
    full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
    ablated = {"evaluations": {
        terminal: candidate["terminal_ablations"]["zero_both"]
    }}
    full_vs_ablation = paired_loss_interval(full, ablated, terminal)
    all_diagnostics = [
        value
        for result in results.values()
        for phase in ("initial", "terminal")
        for value in result["attention_diagnostics"][phase].values()
    ]
    gates = {
        "all_parameter_counts_equal": len({
            result["total_parameters"] for result in results.values()
        }) == 1,
        "all_parameter_tensor_counts_equal": len({
            result["parameter_tensors"] for result in results.values()
        }) == 1,
        "all_optimizer_state_bytes_equal": len({
            result["optimizer_state_bytes"] for result in results.values()
        }) == 1,
        "all_training_and_diagnostics_finite": (
            all(not result["train"]["nonfinite"] for result in results.values())
            and all(math.isfinite(value) for value in all_diagnostics)
        ),
        "all_key_nonfinite_fractions_zero": all(
            result["attention_diagnostics"][phase]["key_nonfinite_fraction_layer_max"] == 0.0
            for result in results.values()
            for phase in ("initial", "terminal")
        ),
        "canonical_control_initial_nll_matches_raw_within_1e_4": abs(
            control["evaluations"]["0"]["loss"]
            - baseline["evaluations"]["0"]["loss"]
        ) <= 1e-4,
        "candidate_initial_nll_matches_control_within_1e_7": abs(
            candidate["evaluations"]["0"]["loss"]
            - control["evaluations"]["0"]["loss"]
        ) <= 1e-7,
        "canonical_control_terminal_noninferior_to_raw_within_0_05_percent": (
            losses["canonical_bilinear_control"] - losses["raw_baseline"]
        ) / losses["raw_baseline"] <= 0.0005,
        "candidate_terminal_improves_raw_by_0_025_percent": (
            losses["raw_baseline"] - losses["gauge_slot_g2"]
        ) / losses["raw_baseline"] >= 0.00025,
        "candidate_terminal_improves_control_by_0_01_percent": (
            losses["canonical_bilinear_control"] - losses["gauge_slot_g2"]
        ) / losses["canonical_bilinear_control"] >= 0.0001,
        "paired_interval_favors_candidate_vs_raw": candidate_vs_baseline["upper_95"] < 0.0,
        "paired_interval_favors_candidate_vs_control": candidate_vs_control["upper_95"] < 0.0,
        "zeroing_both_coefficients_hurts_0_01_percent": (
            ablated["evaluations"][terminal]["loss"] - losses["gauge_slot_g2"]
        ) / losses["gauge_slot_g2"] >= 0.0001,
        "paired_interval_favors_full_vs_zero_both": full_vs_ablation["upper_95"] < 0.0,
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "relative_candidate_improvement": {
            reference: (losses[reference] - losses["gauge_slot_g2"])
            / losses[reference]
            for reference in ("raw_baseline", "canonical_bilinear_control")
        },
        "paired_intervals": {
            "candidate_vs_raw": candidate_vs_baseline,
            "candidate_vs_control": candidate_vs_control,
            "full_vs_zero_both": full_vs_ablation,
        },
        "gates": gates,
        "pilot_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps not in args.eval_steps:
        raise ValueError("terminal step must be present in --eval-steps")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 is required")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": "gauge-slot-lm-pilot-development-v1",
        "candidate": "gauge-canonical-packed-G2-attention",
        "scope": "matched 10M-token from-scratch quality pilot; no runtime admission",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "data_ledger": data_ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(payload["arms"], args)
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--eval-steps", type=parse_steps, default=[61, 305])
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2207)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
