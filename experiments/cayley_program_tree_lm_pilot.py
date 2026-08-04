#!/usr/bin/env python3
"""10M-token language pilot for the full-budget Cayley program tree."""

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
import transformers

from experiments.cayley_program_tree import CayleyProgramTreeMLP, exact_ledger
from experiments.coalesced_attention_ffn_lm_screen import (
    build_model as build_parallel_model,
)
from experiments.reflex_swiglu_lm_screen import (
    TokenFile,
    lr_multiplier,
    paired_loss_interval,
    sha256_file,
    write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


SEED = 815
STEPS = 305
PREREGISTRATION = Path("results/cayley-program-tree-lm-pilot-preregistration.md")
EXPECTED_DATA_HASHES = {
    "train": "1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a",
    "validation": "889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_candidate(
    device: torch.device, seed: int = SEED
) -> tuple[nn.Module, list[CayleyProgramTreeMLP]]:
    seed_everything(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules = []
    for layer_index, layer in enumerate(model.model.layers):
        replacement = CayleyProgramTreeMLP(
            seed=seed + 10_000 * layer_index
        ).to(device)
        layer.mlp = replacement
        modules.append(replacement)
    return model, modules


@torch.no_grad()
def evaluate(
    model: nn.Module,
    token_file: TokenFile,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses = []
    started = time.perf_counter()
    for index in range(batches):
        inputs, targets = token_file.batch(index, batch_size, device)
        losses.append(float(causal_loss(model, inputs, targets)))
    torch.cuda.synchronize()
    return {
        "loss": float(np.mean(losses)),
        "per_batch_loss": losses,
        "prediction_tokens": batches * batch_size * token_file.sequence_length,
        "elapsed_seconds": time.perf_counter() - started,
    }


def router_auxiliary(modules: list[CayleyProgramTreeMLP]) -> torch.Tensor:
    return torch.stack([module.last_router_auxiliary for module in modules]).mean()


@torch.no_grad()
def activation_diagnostics(
    model: nn.Module,
    modules: list[CayleyProgramTreeMLP],
    validation: TokenFile,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    captured: list[torch.Tensor] = []
    handles = [
        module.register_forward_hook(
            lambda _module, _arguments, output: captured.append(output.detach())
        )
        for module in modules
    ]
    inputs, _ = validation.batch(0, 2, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    rms = [float(output.float().square().mean().sqrt()) for output in captured]
    nonfinite = [float((~torch.isfinite(output)).float().mean()) for output in captured]
    return {
        "median_layer_output_rms": float(np.median(rms)),
        "maximum_layer_output_rms": max(rms),
        "maximum_nonfinite_fraction": max(nonfinite),
    }


@torch.no_grad()
def route_diagnostics(
    model: nn.Module,
    modules: list[CayleyProgramTreeMLP],
    validation: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    inputs, _ = validation.batch(0, 2, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    rows = []
    for layer_index, module in enumerate(modules):
        for depth, (nodes, bits) in enumerate(
            zip(module.last_nodes, module.last_bits, strict=True)
        ):
            count = bits.numel()
            one_fraction = float(bits.float().mean())
            probabilities = torch.tensor(
                [1.0 - one_fraction, one_fraction], dtype=torch.float64
            )
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum()
            reachable = 2**depth
            unique_nodes = int(torch.unique(nodes).numel())
            rows.append(
                {
                    "layer": layer_index,
                    "depth": depth,
                    "zero_count": int(count - bits.sum()),
                    "one_count": int(bits.sum()),
                    "maximum_bit_load": max(one_fraction, 1.0 - one_fraction),
                    "route_perplexity": float(entropy.exp()),
                    "unique_nodes_visited": unique_nodes,
                    "reachable_nodes": reachable,
                    "node_coverage": unique_nodes / reachable,
                }
            )
    deep_rows = [row for row in rows if row["depth"] >= 6]
    return {
        "rows": rows,
        "median_route_perplexity": float(
            np.median([row["route_perplexity"] for row in rows])
        ),
        "maximum_bit_load": max(row["maximum_bit_load"] for row in rows),
        "minimum_deep_node_coverage": min(
            row["node_coverage"] for row in deep_rows
        ),
    }


def baseline_is_compatible(
    baseline_payload: dict[str, Any], train_hash: str, validation_hash: str
) -> bool:
    arm = baseline_payload.get("arms", {}).get("full_swiglu", {})
    environment = baseline_payload.get("environment", {})
    arguments = baseline_payload.get("arguments", {})
    return bool(
        baseline_payload.get("protocol_valid")
        and baseline_payload.get("data", {}).get("train_sha256") == train_hash
        and baseline_payload.get("data", {}).get("validation_sha256")
        == validation_hash
        and environment.get("torch") == torch.__version__
        and environment.get("cuda") == torch.version.cuda
        and environment.get("transformers") == transformers.__version__
        and environment.get("gpu") == torch.cuda.get_device_name()
        and arguments.get("steps") == 305
        and arguments.get("sequence_length") == 512
        and arguments.get("micro_batch_size") == 8
        and arguments.get("gradient_accumulation") == 8
        and arguments.get("eval_batch_size") == 32
        and arguments.get("eval_batches") == 128
        and arguments.get("warmup_steps") == 30
        and arguments.get("learning_rate") == 3e-4
        and arguments.get("weight_decay") == 0.1
        and arguments.get("gradient_clip") == 1.0
        and arguments.get("router_aux_weight") == 0.01
        and arguments.get("seed") == SEED
        and arm.get("train", {}).get("prediction_tokens") == 9_994_240
        and len(arm.get("evaluations", {}).get(str(STEPS), {}).get("per_batch_loss", []))
        == 128
    )


def train_candidate(
    args: argparse.Namespace,
    train: TokenFile,
    validation: TokenFile,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print(json.dumps({"status": "building_candidate"}), flush=True)
    model, modules = build_candidate(device, args.seed)
    model.config.use_cache = False
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    ffn_parameters = sum(module.parameter_count() for module in modules)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer.param_groups[0]["base_lr"] = args.learning_rate
    print(json.dumps({"status": "initial_evaluation"}), flush=True)
    evaluations = {
        "0": evaluate(
            model, validation, args.eval_batches, args.eval_batch_size, device
        )
    }
    print(
        json.dumps(
            {"status": "initial_evaluation_complete", "loss": evaluations["0"]["loss"]}
        ),
        flush=True,
    )
    initial_activation_diagnostics = activation_diagnostics(
        model, modules, validation, device
    )
    optimizer.zero_grad(set_to_none=True)
    language_losses = []
    auxiliaries = []
    norms = []
    durations = []
    for step in range(args.steps):
        model.train()
        optimizer.param_groups[0]["lr"] = args.learning_rate * lr_multiplier(
            step, args.steps, args.warmup_steps
        )
        started = time.perf_counter()
        accumulated_language = 0.0
        accumulated_auxiliary = 0.0
        for micro in range(args.gradient_accumulation):
            index = step * args.gradient_accumulation + micro
            inputs, targets = train.batch(index, args.micro_batch_size, device)
            language = causal_loss(model, inputs, targets)
            auxiliary = router_auxiliary(modules)
            objective = language + args.router_aux_weight * auxiliary
            (objective / args.gradient_accumulation).backward()
            accumulated_language += float(language.detach()) / args.gradient_accumulation
            accumulated_auxiliary += float(auxiliary.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated_language) or not math.isfinite(norm):
            raise RuntimeError(f"nonfinite program tree at step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        language_losses.append(accumulated_language)
        auxiliaries.append(accumulated_auxiliary)
        norms.append(norm)
        durations.append(time.perf_counter() - started)
        if step == 0 or (step + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "language_loss": accumulated_language,
                        "router_auxiliary": accumulated_auxiliary,
                        "seconds": durations[-1],
                    }
                ),
                flush=True,
            )
    print(json.dumps({"status": "terminal_evaluation"}), flush=True)
    evaluations[str(args.steps)] = evaluate(
        model, validation, args.eval_batches, args.eval_batch_size, device
    )
    diagnostics = route_diagnostics(model, modules, validation, device)
    terminal_activation_diagnostics = activation_diagnostics(
        model, modules, validation, device
    )
    for module in modules:
        module.force_bits = (0,) * module.depth
    fixed = evaluate(
        model, validation, min(args.eval_batches, 32), args.eval_batch_size, device
    )
    for module in modules:
        module.force_bits = None
    prediction_tokens = (
        args.steps
        * args.gradient_accumulation
        * args.micro_batch_size
        * args.sequence_length
    )
    result = {
        "total_parameters": total_parameters,
        "ffn_parameters": ffn_parameters,
        "evaluations": evaluations,
        "activation_diagnostics": {
            "initial": initial_activation_diagnostics,
            "terminal": terminal_activation_diagnostics,
        },
        "route_diagnostics": diagnostics,
        "all_zero_path_evaluation": fixed,
        "train": {
            "prediction_tokens": prediction_tokens,
            "mean_language_loss": float(np.mean(language_losses)),
            "final_language_loss": language_losses[-1],
            "mean_router_auxiliary": float(np.mean(auxiliaries)),
            "max_gradient_norm": max(norms),
            "elapsed_seconds": sum(durations),
            "tokens_per_second": prediction_tokens / sum(durations),
            "nonfinite": False,
        },
    }
    checkpoint = {
        "model": model.state_dict(),
        "optimizer_step": args.steps,
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, checkpoint


def decide(
    candidate: dict[str, Any], baseline: dict[str, Any], protocol_valid: bool
) -> dict[str, Any]:
    baseline_arm = baseline["arms"]["full_swiglu"]
    key = str(STEPS)
    candidate_loss = candidate["evaluations"][key]["loss"]
    baseline_loss = baseline_arm["evaluations"][key]["loss"]
    comparison = paired_loss_interval(candidate, baseline_arm, key)
    margin = 0.005 * baseline_loss
    dynamic_first32 = float(
        np.mean(candidate["evaluations"][key]["per_batch_loss"][:32])
    )
    fixed_loss = candidate["all_zero_path_evaluation"]["loss"]
    diagnostics = candidate["route_diagnostics"]
    ledger = exact_ledger()
    gates = {
        "protocol_valid": protocol_valid,
        "candidate_ffn_not_larger": (
            candidate["ffn_parameters"]
            <= 12 * ledger["baseline_swiglu_parameters"]
        ),
        "candidate_model_not_larger": (
            candidate["total_parameters"] <= baseline_arm["total_parameters"]
        ),
        "finite": not candidate["train"]["nonfinite"],
        "activations_finite": (
            candidate["activation_diagnostics"]["initial"]["maximum_nonfinite_fraction"]
            == 0.0
            and candidate["activation_diagnostics"]["terminal"]["maximum_nonfinite_fraction"]
            == 0.0
        ),
        "nll_within_half_percent": candidate_loss - baseline_loss <= margin,
        "paired_upper_within_noninferiority_margin": comparison["upper_95"] <= margin,
        "ideal_active_ratio_below_nine_percent": ledger["ideal_active_ratio"] < 0.09,
        "median_route_perplexity_at_least_1p5": (
            diagnostics["median_route_perplexity"] >= 1.5
        ),
        "maximum_bit_load_below_90_percent": (
            diagnostics["maximum_bit_load"] < 0.90
        ),
        "minimum_deep_node_coverage_at_least_25_percent": (
            diagnostics["minimum_deep_node_coverage"] >= 0.25
        ),
        "all_zero_path_loses_half_percent": (
            (fixed_loss - dynamic_first32) / dynamic_first32 >= 0.005
        ),
    }
    return {
        "candidate_loss": candidate_loss,
        "baseline_loss": baseline_loss,
        "candidate_relative_nll": (candidate_loss - baseline_loss) / baseline_loss,
        "paired_interval": comparison,
        "dynamic_first32_loss": dynamic_first32,
        "all_zero_path_loss": fixed_loss,
        "all_zero_path_relative_loss": (fixed_loss - dynamic_first32) / dynamic_first32,
        "ledger": ledger,
        "gates": gates,
        "advance": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cayley-program-tree-lm-pilot.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cayley-program-tree-lm-pilot.pt"))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_hash = sha256_file(args.train_file)
    validation_hash = sha256_file(args.validation_file)
    baseline = json.loads(args.baseline_result.read_text())
    compatible_baseline = baseline_is_compatible(
        baseline, train_hash, validation_hash
    )
    protocol_valid = bool(
        compatible_baseline
        and args.steps == STEPS
        and args.sequence_length == 512
        and args.micro_batch_size == 8
        and args.gradient_accumulation == 8
        and args.eval_batch_size == 32
        and args.eval_batches == 128
        and args.warmup_steps == 30
        and args.learning_rate == 3e-4
        and args.weight_decay == 0.1
        and args.gradient_clip == 1.0
        and args.router_aux_weight == 0.01
        and args.seed == SEED
        and train_hash == EXPECTED_DATA_HASHES["train"]
        and validation_hash == EXPECTED_DATA_HASHES["validation"]
    )
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    train = TokenFile(args.train_file, args.sequence_length)
    validation = TokenFile(args.validation_file, args.sequence_length)
    payload: dict[str, Any] = {
        "schema": "cayley-program-tree-lm-pilot-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(Path("experiments/cayley_program_tree.py")),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "baseline_result_sha256": sha256_file(args.baseline_result),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "data": {
            "train_sha256": train_hash,
            "validation_sha256": validation_hash,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol_valid": protocol_valid,
        "candidate": None,
    }
    candidate, checkpoint = train_candidate(
        args, train, validation, device
    )
    payload["candidate"] = candidate
    payload["decision"] = decide(candidate, baseline, protocol_valid)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    payload["checkpoint_sha256"] = sha256_file(args.checkpoint)
    write_payload(args.output, payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
