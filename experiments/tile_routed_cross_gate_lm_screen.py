#!/usr/bin/env python3
"""Matched continuation screen for tile-routed cross-gate SwiGLU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

from experiments.reflex_swiglu_lm_screen import (
    MODEL,
    MODEL_REVISION,
    TokenFile,
    evaluate,
    lr_multiplier,
    paired_loss_interval,
    sha256_file,
    validate_data_ledger,
    write_payload,
)


VALID_ARMS = (
    "baseline",
    "static_local",
    "static_balanced",
    "global_routed",
    "tile_routed",
    "tile_routed_reindexed",
)
TILE_SIZE = 256
ROUTES = 4
TEMPERATURE = 1.0
PREREGISTRATION = Path(
    "results/tile-routed-cross-gate-lm-screen-preregistration.md"
)
INTEGRITY_MANIFEST = Path("results/tile-routed-cross-gate-integrity-manifest.json")
TEST_SOURCE = Path("tests/test_tile_routed_cross_gate_lm_screen.py")
REINDEX_SEED = 20_260_727


def effective_alpha(beta: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.tanh(beta / 2.0)


def hard_straight_through(probabilities: torch.Tensor) -> torch.Tensor:
    index = probabilities.argmax(dim=-1)
    hard = F.one_hot(index, num_classes=probabilities.shape[-1]).to(
        probabilities.dtype
    )
    # Subtract first so the forward value is bit-exactly hard even in BF16.
    return hard + (probabilities - probabilities.detach())


def cyclic_candidates(up: torch.Tensor, tile_size: int = TILE_SIZE) -> torch.Tensor:
    if up.shape[-1] % tile_size:
        raise ValueError("intermediate width must be divisible by tile size")
    tiled = up.reshape(*up.shape[:-1], up.shape[-1] // tile_size, tile_size)
    return torch.stack(
        [torch.roll(tiled, shifts=-route, dims=-1) for route in range(ROUTES)],
        dim=-2,
    )


def routed_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    arm: str,
    temperature: float = TEMPERATURE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the hard-forward routed value stream and hard route indices."""
    canonical_arm = "tile_routed" if arm == "tile_routed_reindexed" else arm
    if canonical_arm == "baseline":
        return up, None
    candidates = cyclic_candidates(up)
    if canonical_arm == "static_local":
        return candidates[..., 1, :].reshape_as(up), torch.ones(
            *up.shape[:-1], up.shape[-1] // TILE_SIZE,
            dtype=torch.long,
            device=up.device,
        )
    if canonical_arm == "static_balanced":
        tiles = up.shape[-1] // TILE_SIZE
        routes = torch.arange(tiles, device=up.device) % ROUTES
        routes = routes.reshape(*([1] * (up.ndim - 1)), tiles).expand(
            *up.shape[:-1], tiles
        )
        selected = torch.gather(
            candidates,
            -2,
            routes.unsqueeze(-1).unsqueeze(-1).expand(
                *routes.shape, 1, TILE_SIZE
            ),
        ).squeeze(-2)
        return selected.reshape_as(up), routes

    gate_tiles = gate.reshape(
        *gate.shape[:-1], gate.shape[-1] // TILE_SIZE, TILE_SIZE
    )
    if canonical_arm == "global_routed":
        logits = gate[..., :ROUTES] / temperature
        probabilities = torch.softmax(logits.float(), dim=-1).to(gate.dtype)
        weights = hard_straight_through(probabilities)
        weights = weights.unsqueeze(-2).expand(
            *weights.shape[:-1], gate.shape[-1] // TILE_SIZE, ROUTES
        )
    elif canonical_arm == "tile_routed":
        logits = gate_tiles[..., :ROUTES] / temperature
        probabilities = torch.softmax(logits.float(), dim=-1).to(gate.dtype)
        weights = hard_straight_through(probabilities)
    else:
        raise ValueError(f"unknown arm {arm}")
    selected = (candidates * weights.unsqueeze(-1)).sum(dim=-2).reshape_as(up)
    return selected, weights.detach().argmax(dim=-1)


class CrossGateMLP(nn.Module):
    def __init__(self, original: nn.Module, arm: str, reindex_seed: int) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"unknown arm {arm}")
        if original.gate_proj.out_features % TILE_SIZE:
            raise ValueError("model intermediate width is not tile aligned")
        self.arm = "tile_routed" if arm == "tile_routed_reindexed" else arm
        self.logically_reindexed = arm == "tile_routed_reindexed"
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        self.beta = nn.Parameter(
            torch.zeros((), dtype=self.gate_proj.weight.dtype, device=self.gate_proj.weight.device)
        )
        if self.logically_reindexed:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(reindex_seed)
            permutation = torch.randperm(
                self.gate_proj.out_features, generator=generator
            ).to(self.gate_proj.weight.device)
            inverse = torch.argsort(permutation)
        else:
            permutation = torch.empty(
                0, dtype=torch.long, device=self.gate_proj.weight.device
            )
            inverse = permutation
        self.register_buffer("reindex_permutation", permutation, persistent=False)
        self.register_buffer("reindex_inverse", inverse, persistent=False)
        self.record_routes = False
        self.last_routes: torch.Tensor | None = None

    def activation(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.logically_reindexed:
            working_gate = gate[..., self.reindex_permutation]
            working_up = up[..., self.reindex_permutation]
        else:
            working_gate = gate
            working_up = up
        selected, routes = routed_up(working_gate, working_up, self.arm)
        alpha = effective_alpha(self.beta).to(gate.dtype)
        if self.arm == "baseline":
            effective_up = working_up + 0.0 * alpha
        else:
            effective_up = working_up + alpha * (selected - working_up)
        if self.record_routes and routes is not None:
            self.last_routes = routes.detach().cpu()
        activation = self.act_fn(working_gate) * effective_up
        if self.logically_reindexed:
            activation = activation[..., self.reindex_inverse]
        return activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(self.activation(gate, up))


def install_arm(model: nn.Module, arm: str) -> list[CrossGateMLP]:
    modules = []
    for layer_index, layer in enumerate(model.model.layers):
        module = CrossGateMLP(layer.mlp, arm, REINDEX_SEED + layer_index)
        layer.mlp = module
        modules.append(module)
    return modules


def causal_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def make_optimizer(
    model: nn.Module, shared_lr: float, beta_lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    beta_parameters, shared_parameters = [], []
    for name, parameter in model.named_parameters():
        (beta_parameters if name.endswith(".beta") else shared_parameters).append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": shared_parameters,
                "lr": shared_lr,
                "base_lr": shared_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": beta_parameters,
                "lr": beta_lr,
                "base_lr": beta_lr,
                "weight_decay": 0.0,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )


def beta_gradient_stats(modules: list[CrossGateMLP]) -> dict[str, float | int]:
    values = [module.beta.grad.detach().float() for module in modules if module.beta.grad is not None]
    if not values:
        return {"coordinates": 0}
    tensor = torch.stack(values).abs()
    return {
        "coordinates": tensor.numel(),
        "mean_abs": float(tensor.mean()),
        "max_abs": float(tensor.max()),
    }


@torch.no_grad()
def route_diagnostics(
    model: nn.Module,
    modules: list[CrossGateMLP],
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    tiles = modules[0].gate_proj.out_features // TILE_SIZE
    totals = np.zeros((len(modules), tiles, ROUTES), dtype=np.int64)
    for module in modules:
        module.record_routes = True
    for batch_index in range(2):
        inputs, _ = validation_file.batch(batch_index, 8, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, use_cache=False)
        for layer, module in enumerate(modules):
            if module.last_routes is not None:
                values = module.last_routes.numpy().reshape(-1, tiles)
                for tile in range(tiles):
                    totals[layer, tile] += np.bincount(
                        values[:, tile], minlength=ROUTES
                    )
    for module in modules:
        module.record_routes = False
    global_counts = totals.sum(axis=(0, 1))
    global_loads = global_counts / max(1, global_counts.sum())
    cell_entropies = []
    cells_with_two_routes_above_5_percent = 0
    for counts in totals.reshape(-1, ROUTES):
        probabilities = counts / max(1, counts.sum())
        nonzero = probabilities[probabilities > 0]
        cell_entropies.append(
            float(-(nonzero * np.log(nonzero)).sum() / np.log(ROUTES))
        )
        cells_with_two_routes_above_5_percent += int(
            np.count_nonzero(probabilities >= 0.05) >= 2
        )
    alphas = np.asarray(
        [float(effective_alpha(module.beta.detach().float())) for module in modules]
    )
    return {
        "per_layer_tile_route_counts": totals.tolist(),
        "global_route_counts": global_counts.tolist(),
        "global_route_loads": global_loads.tolist(),
        "median_layer_tile_normalized_entropy": float(np.median(cell_entropies)),
        "minimum_layer_tile_normalized_entropy": float(np.min(cell_entropies)),
        "fraction_layer_tiles_with_two_routes_above_5_percent":
            cells_with_two_routes_above_5_percent / len(cell_entropies),
        "alpha_mean_abs": float(np.mean(np.abs(alphas))),
        "alpha_max_abs": float(np.max(np.abs(alphas))),
        "alpha_saturation_fraction": float(np.mean(np.abs(alphas) > 1.9)),
    }


def validate_protocol(args: argparse.Namespace, arms: tuple[str, ...]) -> dict[str, Any]:
    expected = {
        "arms": VALID_ARMS,
        "device_contains": "H100",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 128,
        "steps": 320,
        "eval_steps": [16, 80, 320],
        "warmup_steps": 16,
        "shared_lr": 1e-4,
        "beta_lr": 5e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 131,
        "tile_size": TILE_SIZE,
        "routes": ROUTES,
        "temperature": TEMPERATURE,
        "reindex_seed": REINDEX_SEED,
    }
    actual = {
        "arms": arms,
        "device_contains": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup_steps,
        "shared_lr": args.shared_lr,
        "beta_lr": args.beta_lr,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "tile_size": TILE_SIZE,
        "routes": ROUTES,
        "temperature": TEMPERATURE,
        "reindex_seed": REINDEX_SEED,
    }
    checks = {
        key: (expected_value in actual[key] if key == "device_contains" else actual[key] == expected_value)
        for key, expected_value in expected.items()
    }
    if args.strict_protocol and not all(checks.values()):
        failed = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key, valid in checks.items()
            if not valid
        }
        raise ValueError(f"invalid experiment protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


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
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    modules = install_arm(model, arm)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = make_optimizer(model, args.shared_lr, args.beta_lr, args.weight_decay)
    evaluations = {
        "0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    }
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    step_losses, step_seconds, gradient_norms = [], [], []
    beta_gradients: dict[str, Any] = {}
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * multiplier
        started = time.perf_counter()
        accumulated_loss = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(batch_index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation
        if step in {0, 15, 79, 319}:
            beta_gradients[str(step + 1)] = beta_gradient_stats(modules)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(norm) or not math.isfinite(accumulated_loss):
            raise RuntimeError(f"non-finite training state in {arm} at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_losses.append(accumulated_loss)
        step_seconds.append(time.perf_counter() - started)
        gradient_norms.append(norm)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step + 1,
                        "train_loss": accumulated_loss,
                        "validation_loss": evaluations[str(step + 1)]["loss"],
                        "gradient_norm": norm,
                    }
                ),
                flush=True,
            )
    diagnostics = route_diagnostics(model, modules, validation_file, device)
    prediction_tokens = (
        args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    )
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "added_beta_parameters": len(modules),
        "evaluations": evaluations,
        "train": {
            "steps": args.steps,
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(step_losses)),
            "max_loss": max(step_losses),
            "final_loss": step_losses[-1],
            "max_preclip_gradient_norm": max(gradient_norms),
            "elapsed_seconds": sum(step_seconds),
            "tokens_per_second": prediction_tokens / sum(step_seconds),
            "median_step_seconds_after_five": float(np.median(step_seconds[5:])),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nonfinite": False,
        },
        "beta_gradient_stats": beta_gradients,
        "route_diagnostics": diagnostics,
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], integrity_valid: bool) -> dict[str, Any]:
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing_arms": sorted(set(VALID_ARMS) - set(results))}
    terminal = "320"
    baseline = results["baseline"]
    candidates = {
        name: results[name]
        for name in ("tile_routed", "tile_routed_reindexed")
    }
    control_name = min(
        ("static_local", "static_balanced", "global_routed"),
        key=lambda name: results[name]["evaluations"][terminal]["loss"],
    )
    control = results[control_name]
    baseline_loss = baseline["evaluations"][terminal]["loss"]
    candidate_losses = {
        name: value["evaluations"][terminal]["loss"]
        for name, value in candidates.items()
    }
    worse_candidate_name = max(candidate_losses, key=candidate_losses.get)
    worse_candidate = candidates[worse_candidate_name]
    worse_candidate_loss = candidate_losses[worse_candidate_name]
    control_loss = control["evaluations"][terminal]["loss"]
    versus_baseline = {
        name: paired_loss_interval(value, baseline, terminal)
        for name, value in candidates.items()
    }
    versus_control = paired_loss_interval(worse_candidate, control, terminal)
    initial_losses = [result["evaluations"]["0"]["loss"] for result in results.values()]
    endpoint_difference = max(initial_losses) - min(initial_losses)
    diagnostics = {
        name: value["route_diagnostics"] for name, value in candidates.items()
    }
    reindex_relative_difference = abs(
        candidate_losses["tile_routed"] - candidate_losses["tile_routed_reindexed"]
    ) / min(candidate_losses.values())
    gates = {
        "integrity_protocol_valid": integrity_valid,
        "identical_parameter_counts": len({value["total_parameters"] for value in results.values()}) == 1,
        "exact_initial_loss_endpoint": endpoint_difference <= 1e-7,
        "all_training_finite": all(not value["train"]["nonfinite"] for value in results.values()),
        "bounded_training": all(
            value["train"]["max_preclip_gradient_norm"] <= 100.0
            and value["train"]["max_loss"] <= 20.0
            for value in results.values()
        ),
        "both_tile_arms_at_least_0p05_percent_better_than_baseline": all(
            (baseline_loss - loss) / baseline_loss >= 0.0005
            for loss in candidate_losses.values()
        ),
        "paired_intervals_favor_both_tile_arms_vs_baseline": all(
            interval["upper_95"] < 0.0 for interval in versus_baseline.values()
        ),
        "worse_tile_arm_at_least_0p025_percent_better_than_best_control":
            (control_loss - worse_candidate_loss) / control_loss >= 0.00025,
        "paired_interval_favors_tile_vs_best_control": versus_control["upper_95"] < 0.0,
        "both_tile_arms_have_within_tile_token_variation": all(
            record["fraction_layer_tiles_with_two_routes_above_5_percent"] >= 0.75
            and record["median_layer_tile_normalized_entropy"] >= 0.50
            for record in diagnostics.values()
        ),
        "both_tile_arm_alphas_are_live": all(
            record["alpha_mean_abs"] >= 1e-4 for record in diagnostics.values()
        ),
        "tile_arm_alphas_not_saturated": all(
            record["alpha_saturation_fraction"] == 0.0
            for record in diagnostics.values()
        ),
        "gain_is_reindex_robust_within_0p025_percent":
            reindex_relative_difference < 0.00025,
    }
    return {
        "complete": True,
        "terminal_step": 320,
        "best_control": control_name,
        "baseline_terminal_loss": baseline_loss,
        "tile_terminal_losses": candidate_losses,
        "worse_tile_arm": worse_candidate_name,
        "best_control_terminal_loss": control_loss,
        "tile_relative_improvements_vs_baseline": {
            name: (baseline_loss - loss) / baseline_loss
            for name, loss in candidate_losses.items()
        },
        "worse_tile_relative_improvement_vs_best_control":
            (control_loss - worse_candidate_loss) / control_loss,
        "tile_reindex_relative_terminal_difference": reindex_relative_difference,
        "endpoint_max_loss_difference": endpoint_difference,
        "paired_terminal_intervals": {
            "tile_arms_vs_baseline": versus_baseline,
            "tile_vs_best_control": versus_control,
        },
        "gates": gates,
        "advance_to_fused_replication": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if arms != VALID_ARMS:
        raise ValueError(f"strict screen requires arms in order {VALID_ARMS}")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol = validate_protocol(args, arms)
    required_train = args.steps * args.gradient_accumulation * args.micro_batch_size
    required_validation = args.eval_batches * args.eval_batch_size
    if train_file.sequence_count < required_train or validation_file.sequence_count < required_validation:
        raise ValueError("token files are too short for the frozen protocol")
    integrity_manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    integrity_checks = {
        "source": integrity_manifest.get("source_sha256") == sha256_file(Path(__file__)),
        "preregistration": integrity_manifest.get("preregistration_sha256")
            == sha256_file(PREREGISTRATION),
        "test": integrity_manifest.get("test_sha256") == sha256_file(TEST_SOURCE),
    }
    if not all(integrity_checks.values()):
        raise ValueError(f"invalid frozen integrity manifest: {integrity_checks}")
    preregistration_sha = sha256_file(PREREGISTRATION)
    source_sha = sha256_file(Path(__file__))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "candidate": "tile-routed cross-gate SwiGLU",
        "scope": "matched one-seed continuation-learning screen",
        "source_sha256": source_sha,
        "preregistration_sha256": preregistration_sha,
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
        "integrity_checks": integrity_checks,
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "data_ledger": data_ledger,
        "experiment_protocol": protocol,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(
            payload["arms"],
            data_ledger["valid"] and protocol["valid"] and all(integrity_checks.values()),
        )
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/reflex-lm-screen/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/reflex-lm-screen/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/reflex-swiglu-lm-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/tile-routed-cross-gate-lm-screen.json"))
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--eval-steps", type=parse_steps, default=parse_steps("16,80,320"))
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--shared-lr", type=float, default=1e-4)
    parser.add_argument("--beta-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
