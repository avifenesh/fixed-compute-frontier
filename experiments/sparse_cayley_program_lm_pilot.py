#!/usr/bin/env python3
"""Frozen 10M-token language pilot for sparse Cayley program FFNs."""

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
import torch.nn.functional as F
import transformers

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
from experiments.sparse_cayley_program import (
    SparseCayleyProgramMLP,
    logical_ledger,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


ARMS = ("full_swiglu", "narrow_swiglu", "static_cayley", "dynamic_cayley")
D = 384
M = 1024
NARROW_WIDTH = 32
SEED = 815
PREREGISTRATION = Path("results/sparse-cayley-program-lm-pilot-preregistration.md")
EXPECTED_DATA_HASHES = {
    "train": "1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a",
    "validation": "889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b",
}


class NarrowSwiGLU(nn.Module):
    def __init__(self, initializer_range: float) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(D, NARROW_WIDTH, bias=False)
        self.up_proj = nn.Linear(D, NARROW_WIDTH, bias=False)
        self.down_proj = nn.Linear(NARROW_WIDTH, D, bias=False)
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Match the baseline's initial aggregate feature-energy scale.
        activation = math.sqrt(M / NARROW_WIDTH) * (
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return self.down_proj(activation)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_arm(
    device: torch.device, arm: str, seed: int
) -> tuple[nn.Module, list[SparseCayleyProgramMLP]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    seed_everything(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules: list[SparseCayleyProgramMLP] = []
    if arm == "full_swiglu":
        return model, modules
    for layer_index, layer in enumerate(model.model.layers):
        if arm == "narrow_swiglu":
            layer.mlp = NarrowSwiGLU(model.config.initializer_range).to(device)
        else:
            replacement = SparseCayleyProgramMLP(
                D,
                seed=seed + 10_000 * layer_index,
                static_route=arm == "static_cayley",
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


@torch.no_grad()
def route_diagnostics(
    model: nn.Module,
    modules: list[SparseCayleyProgramMLP],
    validation: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    if not modules:
        return {}
    model.eval()
    inputs, _ = validation.batch(0, 2, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    rows = []
    for layer_index, module in enumerate(modules):
        for projection_index, projection in enumerate(module.program_projections()):
            for step, ids in enumerate(projection.last_route_ids):
                counts = torch.bincount(ids, minlength=projection.experts).float()
                probabilities = counts / counts.sum()
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                rows.append(
                    {
                        "layer": layer_index,
                        "projection": projection_index,
                        "step": step,
                        "route_perplexity": float(entropy.exp()),
                        "maximum_load": float(probabilities.max()),
                        "counts": counts.cpu().int().tolist(),
                    }
                )
    return {
        "rows": rows,
        "median_route_perplexity": float(np.median([row["route_perplexity"] for row in rows])),
        "maximum_expert_load": max(row["maximum_load"] for row in rows),
    }


def router_auxiliary(modules: list[SparseCayleyProgramMLP]) -> torch.Tensor:
    return torch.stack([module.router_auxiliary_loss() for module in modules]).mean()


def train_arm(
    arm: str,
    args: argparse.Namespace,
    train: TokenFile,
    validation: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    model, modules = build_arm(device, arm, args.seed)
    model.config.use_cache = False
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    ffn_parameters = sum(
        sum(parameter.numel() for parameter in layer.mlp.parameters())
        for layer in model.model.layers
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer.param_groups[0]["base_lr"] = args.learning_rate
    evaluations = {
        "0": evaluate(
            model, validation, args.eval_batches, args.eval_batch_size, device
        )
    }
    optimizer.zero_grad(set_to_none=True)
    losses = []
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
        accumulated = 0.0
        accumulated_language = 0.0
        accumulated_auxiliary = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train.batch(batch_index, args.micro_batch_size, device)
            language = causal_loss(model, inputs, targets)
            auxiliary = (
                router_auxiliary(modules)
                if modules
                else torch.zeros((), device=device)
            )
            objective = language + args.router_aux_weight * auxiliary
            (objective / args.gradient_accumulation).backward()
            accumulated += float(objective.detach()) / args.gradient_accumulation
            accumulated_language += float(language.detach()) / args.gradient_accumulation
            accumulated_auxiliary += float(auxiliary.detach()) / args.gradient_accumulation
        norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        )
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"nonfinite {arm} at step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        language_losses.append(accumulated_language)
        auxiliaries.append(accumulated_auxiliary)
        norms.append(norm)
        durations.append(time.perf_counter() - started)
        if step == 0 or (step + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step + 1,
                        "language_loss": accumulated_language,
                        "router_auxiliary": accumulated_auxiliary,
                        "seconds": durations[-1],
                    }
                ),
                flush=True,
            )
    evaluations[str(args.steps)] = evaluate(
        model, validation, args.eval_batches, args.eval_batch_size, device
    )
    diagnostics = route_diagnostics(model, modules, validation, device)
    fixed_route_evaluation = None
    if arm == "dynamic_cayley":
        for module in modules:
            for projection in module.program_projections():
                projection.force_route = 0
        fixed_route_evaluation = evaluate(
            model,
            validation,
            min(args.eval_batches, 32),
            args.eval_batch_size,
            device,
        )
        for module in modules:
            for projection in module.program_projections():
                projection.force_route = None
    prediction_tokens = (
        args.steps
        * args.gradient_accumulation
        * args.micro_batch_size
        * args.sequence_length
    )
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "ffn_parameters": ffn_parameters,
        "evaluations": evaluations,
        "route_diagnostics": diagnostics,
        "fixed_route_evaluation": fixed_route_evaluation,
        "train": {
            "prediction_tokens": prediction_tokens,
            "mean_objective": float(np.mean(losses)),
            "mean_language_loss": float(np.mean(language_losses)),
            "final_language_loss": language_losses[-1],
            "mean_router_auxiliary": float(np.mean(auxiliaries)),
            "max_gradient_norm": max(norms),
            "elapsed_seconds": sum(durations),
            "tokens_per_second": prediction_tokens / sum(durations),
            "nonfinite": False,
        },
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], steps: int) -> dict[str, Any]:
    if set(results) != set(ARMS):
        return {"complete": False, "missing": sorted(set(ARMS) - set(results))}
    key = str(steps)
    losses = {arm: result["evaluations"][key]["loss"] for arm, result in results.items()}
    candidate = losses["dynamic_cayley"]
    full = losses["full_swiglu"]
    narrow = losses["narrow_swiglu"]
    static = losses["static_cayley"]
    gap = narrow - full
    recovered = (narrow - candidate) / gap if gap > 0 else float("-inf")
    comparisons = {
        arm: paired_loss_interval(results["dynamic_cayley"], results[arm], key)
        for arm in ("full_swiglu", "narrow_swiglu", "static_cayley")
    }
    diagnostics = results["dynamic_cayley"]["route_diagnostics"]
    fixed = results["dynamic_cayley"]["fixed_route_evaluation"]["loss"]
    ledger = logical_ledger()
    gates = {
        "equal_ffn_parameters": (
            results["dynamic_cayley"]["ffn_parameters"]
            == results["narrow_swiglu"]["ffn_parameters"]
        ),
        "candidate_total_below_70_percent_full": (
            results["dynamic_cayley"]["total_parameters"]
            <= 0.70 * results["full_swiglu"]["total_parameters"]
        ),
        "finite": all(not result["train"]["nonfinite"] for result in results.values()),
        "recovers_25_percent_of_gap": recovered >= 0.25,
        "beats_narrow_by_2_percent": (narrow - candidate) / narrow >= 0.02,
        "paired_upper_below_zero_vs_narrow": comparisons["narrow_swiglu"]["upper_95"] < 0,
        "beats_static_by_1_percent": (static - candidate) / static >= 0.01,
        "fixed_route_ablation_loses_1_percent": (fixed - candidate) / candidate >= 0.01,
        "route_perplexity_at_least_2": diagnostics["median_route_perplexity"] >= 2.0,
        "maximum_route_load_below_80_percent": diagnostics["maximum_expert_load"] < 0.80,
        "ideal_active_ratio_below_7_percent": ledger["candidate_to_baseline_ideal_ratio"] < 0.07,
    }
    return {
        "complete": True,
        "losses": losses,
        "gap_recovered": recovered,
        "relative_improvements": {
            "over_narrow": (narrow - candidate) / narrow,
            "over_static": (static - candidate) / static,
        },
        "fixed_route_relative_loss": (fixed - candidate) / candidate,
        "paired_intervals": comparisons,
        "ledger": ledger,
        "gates": gates,
        "advance": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/sparse-cayley-program-lm-pilot.json"))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
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
    protocol_valid = (
        args.steps == 305
        and args.sequence_length == 512
        and args.micro_batch_size * args.gradient_accumulation == 64
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
        "schema": "sparse-cayley-program-lm-pilot-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(Path("experiments/sparse_cayley_program.py")),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
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
        "arguments": vars(args) | {
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "output": str(args.output),
        },
        "protocol_valid": protocol_valid,
        "arms": {},
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if (
            previous.get("source_sha256") == payload["source_sha256"]
            and previous.get("data") == payload["data"]
            and previous.get("arguments") == payload["arguments"]
        ):
            payload["arms"] = previous.get("arms", {})
    for arm in args.arms:
        if arm in payload["arms"]:
            print(json.dumps({"skipping_completed_arm": arm}), flush=True)
            continue
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, train, validation, device)
        payload["decision"] = decide(payload["arms"], args.steps)
        write_payload(args.output, payload)
        print(
            json.dumps(
                {
                    "finished_arm": arm,
                    "loss": payload["arms"][arm]["evaluations"][str(args.steps)]["loss"],
                }
            ),
            flush=True,
        )
    payload["decision"] = decide(payload["arms"], args.steps)
    write_payload(args.output, payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
