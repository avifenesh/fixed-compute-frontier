#!/usr/bin/env python3
"""Exact-error oracle for phase-adaptive factored linear backpropagation."""

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
from transformers import AutoModelForCausalLM

from experiments.compute_priced_algebraic_continuation_oracle import (
    BATCH_SIZE,
    LAYERS,
    MEASUREMENT_BATCH_SIZE,
    SEED,
    SEQUENCE_LENGTH,
    SAMPLED_LAYERS,
    TokenStream,
    VOCABULARY,
    WIDTHS,
    causal_loss,
    learning_rate_multiplier,
    model_config,
    sha256_file,
)


STEPS = 512
CHECKPOINTS = (0, 16, 64, 256, 512)
QUALITY = 0.99
COARSE_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
PREREGISTRATION = Path(
    "results/phase-adaptive-error-factored-backprop-oracle-preregistration.md"
)
TEST_SOURCE = Path("tests/test_phase_adaptive_error_factored_backprop_oracle.py")
DEPENDENCY_SOURCE = Path(
    "experiments/compute_priced_algebraic_continuation_oracle.py"
)


LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def selected_linear_modules(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    selected = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name.startswith(f"model.layers.{layer}.") for layer in SAMPLED_LAYERS):
            continue
        if name.endswith(LINEAR_SUFFIXES):
            selected[name] = module
    expected = len(SAMPLED_LAYERS) * len(LINEAR_SUFFIXES)
    if len(selected) != expected:
        raise RuntimeError(f"selected {len(selected)} modules, expected {expected}")
    return selected


def combined_direction_metrics(
    exact_input: torch.Tensor,
    exact_weight: torch.Tensor,
    approximate_input: torch.Tensor,
    approximate_weight: torch.Tensor,
) -> dict[str, float]:
    exact_energy = float(exact_input.square().sum() + exact_weight.square().sum())
    approximate_energy = float(
        approximate_input.square().sum() + approximate_weight.square().sum()
    )
    dot = float(
        (exact_input * approximate_input).sum()
        + (exact_weight * approximate_weight).sum()
    )
    cosine_squared = (
        0.0
        if exact_energy == 0.0 or approximate_energy == 0.0
        else dot * dot / (exact_energy * approximate_energy)
    )
    return {
        "exact_gradient_energy": exact_energy,
        "approximate_gradient_energy": approximate_energy,
        "dot": dot,
        "cosine_squared": min(1.0, max(0.0, cosine_squared)),
    }


def error_factored_backward_oracle(
    inputs: torch.Tensor,
    error: torch.Tensor,
    weight: torch.Tensor,
    quality: float = QUALITY,
) -> dict[str, Any]:
    inputs = inputs.detach().float().reshape(-1, inputs.shape[-1])
    error = error.detach().float().reshape(-1, error.shape[-1])
    weight = weight.detach().float()
    tokens, input_width = inputs.shape
    error_tokens, output_width = error.shape
    if tokens != error_tokens or tuple(weight.shape) != (output_width, input_width):
        raise ValueError("incompatible linear backward shapes")

    exact_input = error @ weight
    exact_weight = error.T @ inputs
    u, singular, vh = torch.linalg.svd(error, full_matrices=False)
    maximum_rank = singular.numel()
    error_energy = float(singular.square().sum())
    evaluated: dict[int, dict[str, Any]] = {}

    def evaluate_rank(rank: int) -> dict[str, Any]:
        rank = min(maximum_rank, rank)
        if rank in evaluated:
            return evaluated[rank]
        left = u[:, :rank] * singular[:rank]
        right = vh[:rank]
        approximate_input = left @ (right @ weight)
        approximate_weight = right.T @ (left.T @ inputs)
        metrics = combined_direction_metrics(
            exact_input,
            exact_weight,
            approximate_input,
            approximate_weight,
        )
        ideal_ratio = rank / tokens + rank / output_width
        charged_ratio = ideal_ratio + rank / input_width
        row = {
            "rank": rank,
            "rank_fraction_of_error_width": rank / min(tokens, output_width),
            "error_energy_retained": float(singular[:rank].square().sum())
            / max(error_energy, 1e-30),
            "ideal_backward_cost_ratio": ideal_ratio,
            "charged_backward_cost_ratio": min(1.0, charged_ratio),
            **metrics,
        }
        evaluated[rank] = row
        return row

    previous = 0
    upper = None
    for rank in COARSE_RANKS:
        if rank > maximum_rank:
            break
        row = evaluate_rank(rank)
        if row["cosine_squared"] >= quality:
            upper = rank
            break
        if row["charged_backward_cost_ratio"] >= 1.0:
            break
        previous = rank
    if upper is None and previous < maximum_rank:
        full = evaluate_rank(maximum_rank)
        if full["cosine_squared"] >= quality and full["charged_backward_cost_ratio"] < 1.0:
            upper = maximum_rank

    if upper is not None:
        low, high = previous + 1, upper
        while low < high:
            middle = (low + high) // 2
            if evaluate_rank(middle)["cosine_squared"] >= quality:
                high = middle
            else:
                low = middle + 1
        selected = evaluate_rank(low)
        if selected["charged_backward_cost_ratio"] >= 1.0:
            upper = None

    if upper is None:
        exact_energy = float(exact_input.square().sum() + exact_weight.square().sum())
        selected = {
            "rank": maximum_rank,
            "rank_fraction_of_error_width": 1.0,
            "error_energy_retained": 1.0,
            "ideal_backward_cost_ratio": 1.0,
            "charged_backward_cost_ratio": 1.0,
            "exact_gradient_energy": exact_energy,
            "approximate_gradient_energy": exact_energy,
            "dot": exact_energy,
            "cosine_squared": 1.0,
            "dense_fallback": True,
        }
    else:
        selected = {**selected, "dense_fallback": False}

    curve = [evaluated[rank] for rank in sorted(evaluated)]
    return {
        "tokens": tokens,
        "input_width": input_width,
        "output_width": output_width,
        "error_energy": error_energy,
        "selected": selected,
        "curve": curve,
    }


def aggregate(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dense_work = sum(
        row["tokens"] * row["input_width"] * row["output_width"]
        for row in rows.values()
    )
    charged_work = sum(
        row["selected"]["charged_backward_cost_ratio"]
        * row["tokens"]
        * row["input_width"]
        * row["output_width"]
        for row in rows.values()
    )
    total_energy = sum(row["selected"]["exact_gradient_energy"] for row in rows.values())
    retained = sum(
        row["selected"]["exact_gradient_energy"]
        * row["selected"]["cosine_squared"]
        for row in rows.values()
    )
    fallback_count = sum(row["selected"]["dense_fallback"] for row in rows.values())
    return {
        "matrices": len(rows),
        "dense_backward_macs": dense_work * 2,
        "charged_backward_cost_ratio": charged_work / dense_work,
        "energy_weighted_cosine_squared": retained / total_energy,
        "dense_fallback_count": fallback_count,
        "median_selected_rank": float(
            np.median([row["selected"]["rank"] for row in rows.values()])
        ),
        "maximum_selected_rank_fraction": max(
            row["selected"]["rank_fraction_of_error_width"] for row in rows.values()
        ),
    }


def frontier_projection(summary: dict[str, Any]) -> dict[str, float]:
    width = 4096
    layers = 32
    context = 4096
    projection = layers * 10.5 * width * width
    fixed = layers * 2 * context * width + VOCABULARY * width
    projection_fraction = projection / (projection + fixed)
    backward_ratio = summary["charged_backward_cost_ratio"]
    whole_step_ratio = (
        1.0
        - projection_fraction
        + projection_fraction * (1.0 + 2.0 * backward_ratio) / 3.0
    )
    return {
        "projection_fraction": projection_fraction,
        "whole_step_cost_ratio": whole_step_ratio,
        "optimistic_descent_per_compute": summary[
            "energy_weighted_cosine_squared"
        ]
        / whole_step_ratio,
    }


def capture_linear_backward(
    model: torch.nn.Module,
    validation: TokenStream,
    device: torch.device,
    width: int,
    step: int,
) -> dict[str, Any]:
    modules = selected_linear_modules(model)
    captured: dict[str, dict[str, torch.Tensor]] = {}
    handles = []

    def make_forward_hook(name: str):
        def hook(
            module: torch.nn.Module,
            arguments: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[name] = {"input": arguments[0].detach().float()}

            def error_hook(gradient: torch.Tensor) -> None:
                captured[name]["error"] = gradient.detach().float()

            output.register_hook(error_hook)

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_forward_hook(name)))

    model.train()
    model.zero_grad(set_to_none=True)
    inputs, targets = validation.batch(0, MEASUREMENT_BATCH_SIZE, device)
    loss = causal_loss(model, inputs, targets)
    loss.backward()
    for handle in handles:
        handle.remove()

    results = {}
    for name, module in modules.items():
        if name not in captured or "error" not in captured[name]:
            raise RuntimeError(f"missing capture for {name}")
        results[name] = error_factored_backward_oracle(
            captured[name]["input"],
            captured[name]["error"],
            module.weight,
        )
        selected = results[name]["selected"]
        print(
            json.dumps(
                {
                    "width": width,
                    "step": step,
                    "matrix": name,
                    "rank": selected["rank"],
                    "cost": selected["charged_backward_cost_ratio"],
                    "q": selected["cosine_squared"],
                    "fallback": selected["dense_fallback"],
                }
            ),
            flush=True,
        )
    model.zero_grad(set_to_none=True)
    summary = aggregate(results)
    summary["frontier_projection"] = frontier_projection(summary)
    return {
        "width": width,
        "step": step,
        "loss": float(loss.detach()),
        "aggregate": summary,
        "matrices": results,
    }


def train_width(
    width: int,
    train: TokenStream,
    validation: TokenStream,
    device: torch.device,
) -> dict[str, Any]:
    seed = SEED + 10_000 + width
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
                capture_linear_backward(model, validation, device, width, step)
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


def isotropic_controls(device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(SEED + 1_900_000)
    rows = {}
    tokens = MEASUREMENT_BATCH_SIZE * SEQUENCE_LENGTH
    for width in WIDTHS:
        intermediate = width * 8 // 3
        kv_width = width // 3
        shapes = {
            "square": (width, width),
            "kv": (kv_width, width),
            "up": (intermediate, width),
            "down": (width, intermediate),
        }
        for label, (output_width, input_width) in shapes.items():
            inputs = torch.randn(
                tokens, input_width, generator=generator, device=device
            )
            error = torch.randn(
                tokens, output_width, generator=generator, device=device
            )
            weight = torch.randn(
                output_width, input_width, generator=generator, device=device
            ) / math.sqrt(input_width)
            rows[f"width_{width}_{label}"] = error_factored_backward_oracle(
                inputs, error, weight
            )
    return {"matrices": rows, "aggregate": aggregate(rows)}


def decide(width_results: list[dict[str, Any]], isotropic: dict[str, Any]) -> dict[str, Any]:
    cells = {
        (row["width"], measurement["step"]): measurement["aggregate"]
        for row in width_results
        for measurement in row["measurements"]
    }
    step64_costs = [cells[(width, 64)]["charged_backward_cost_ratio"] for width in WIDTHS]
    scaling_checks = {}
    for step in (64, 256, 512):
        costs = [cells[(width, step)]["charged_backward_cost_ratio"] for width in WIDTHS]
        scaling_checks[str(step)] = costs[0] > costs[1] > costs[2]
    widest_efficiencies = {
        str(step): cells[(WIDTHS[-1], step)]["frontier_projection"][
            "optimistic_descent_per_compute"
        ]
        for step in (64, 256, 512)
    }
    all_cosines = [
        value["energy_weighted_cosine_squared"] for value in cells.values()
    ]
    all_energies_nonzero = all(
        row["selected"]["exact_gradient_energy"] > 1e-30
        for width_result in width_results
        for measurement in width_result["measurements"]
        for row in measurement["matrices"].values()
    )
    gates = {
        "step64_cost_at_most_20_20_8_percent": step64_costs[0] <= 0.20
        and step64_costs[1] <= 0.20
        and step64_costs[2] <= 0.08,
        "cost_strictly_decreases_with_width_after_warmup": all(
            scaling_checks.values()
        ),
        "all_aggregate_cosines_at_least_0p99": all(
            value >= QUALITY for value in all_cosines
        ),
        "isotropic_control_requires_85_percent_dense_backward": isotropic[
            "aggregate"
        ]["charged_backward_cost_ratio"]
        >= 0.85,
        "widest_projected_efficiency_at_least_2x_after_warmup": all(
            value >= 2.0 for value in widest_efficiencies.values()
        ),
        "all_models_learned_and_gradients_nondegenerate": all(
            row["training"]["terminal_loss"] < row["training"]["initial_loss"]
            for row in width_results
        )
        and all_energies_nonzero,
    }
    return {
        "step64_costs": step64_costs,
        "scaling_checks": scaling_checks,
        "widest_projected_efficiencies": widest_efficiencies,
        "isotropic_cost": isotropic["aggregate"]["charged_backward_cost_ratio"],
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
        default=Path("results/phase-adaptive-error-factored-backprop-oracle.json"),
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
        "schema": "phase-adaptive-error-factored-backprop-oracle-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "test_sha256": sha256_file(TEST_SOURCE),
        "dependency_sha256": sha256_file(DEPENDENCY_SOURCE),
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
            "training_batch_size": BATCH_SIZE,
            "measurement_batch_size": MEASUREMENT_BATCH_SIZE,
            "steps": STEPS,
            "checkpoints": CHECKPOINTS,
            "sampled_layers": SAMPLED_LAYERS,
            "quality": QUALITY,
            "seed_base": SEED + 10_000,
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
