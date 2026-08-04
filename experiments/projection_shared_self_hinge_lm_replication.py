#!/usr/bin/env python3
"""Independent 50M-token replication of projection-shared self-hinge."""

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
import transformers

from experiments.projection_shared_coupling_lm_screen import (
    BOUND,
    build_model,
    causal_loss,
    diagnostics,
)
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


VALID_ARMS = ("raw_baseline", "folded_null", "self_hinge")
PREREGISTRATION = Path("results/projection-shared-self-hinge-lm-replication-preregistration.md")


@torch.no_grad()
def replication_diagnostics(model, modules, validation_file, device):
    result = diagnostics(model, modules, validation_file, device)
    if not modules:
        return result
    if modules[0].arm == "self_hinge":
        maxima = []
        for module in modules:
            alpha, beta = module.coefficients(torch.float32)
            maximum = torch.cat((alpha, beta)).abs().max()
            maxima.append(float((1.0 + maximum) / (1.0 - maximum)))
        result["maximum_chart_condition"] = max(maxima)
    elif modules[0].arm == "folded_null":
        result["maximum_chart_condition"] = 1.0
    return result


def train_arm(arm, args, train_file, validation_file, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(device, arm)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=args.weight_decay, fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    chart_diagnostics = {"initial": replication_diagnostics(model, modules, validation_file, device)}
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    losses, durations, norms = [], [], []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"non-finite training in {arm} at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        norms.append(norm)
        durations.append(time.perf_counter() - started)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(json.dumps({"arm": arm, "step": step + 1,
                              "validation_loss": evaluations[str(step + 1)]["loss"]}), flush=True)
    chart_diagnostics["terminal"] = replication_diagnostics(
        model, modules, validation_file, device
    )
    ablations = {}
    if arm == "self_hinge":
        for module in modules:
            module.ablation_mode = "zero"
        ablations["zero"] = evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
        for module in modules:
            module.ablation_mode = "full"
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm, "total_parameters": total, "trainable_parameters": trainable,
        "evaluations": evaluations, "diagnostics": chart_diagnostics,
        "terminal_ablations": ablations,
        "train": {
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(losses)), "final_loss": losses[-1],
            "max_loss": max(losses), "max_preclip_gradient_norm": max(norms),
            "elapsed_seconds": sum(durations),
            "tokens_per_second": prediction_tokens / sum(durations),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "nonfinite": False,
        },
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results, protocol_valid):
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing": sorted(set(VALID_ARMS) - set(results))}
    candidate = results["self_hinge"]
    losses = {
        step: {arm: result["evaluations"][step]["loss"] for arm, result in results.items()}
        for step in ("305", "1525")
    }
    intervals = {}
    for step in ("305", "1525"):
        for control in ("raw_baseline", "folded_null"):
            intervals[f"self_vs_{control}_{step}"] = paired_loss_interval(
                candidate, results[control], step
            )
    full_wrapper = {"evaluations": {"1525": candidate["evaluations"]["1525"]}}
    intervals["full_vs_zero_1525"] = paired_loss_interval(
        full_wrapper,
        {"evaluations": {"1525": candidate["terminal_ablations"]["zero"]}},
        "1525",
    )
    initial_losses = [result["evaluations"]["0"]["loss"] for result in results.values()]
    terminal_diag = candidate["diagnostics"]["terminal"]
    full_loss = losses["1525"]["self_hinge"]
    zero_loss = candidate["terminal_ablations"]["zero"]["loss"]
    gates = {
        "protocol_valid": protocol_valid,
        "identical_total_parameters": len({r["total_parameters"] for r in results.values()}) == 1,
        "identical_trainable_parameters": len({r["trainable_parameters"] for r in results.values()}) == 1,
        "initial_losses_match_2e_5": max(initial_losses) - min(initial_losses) <= 2e-5,
        "finite_training": all(not r["train"]["nonfinite"] for r in results.values()),
        "self_no_worse_both_controls_at_10m": all(
            losses["305"]["self_hinge"] <= losses["305"][control]
            for control in ("raw_baseline", "folded_null")
        ),
        "self_0p005_percent_better_both_at_50m": all(
            (losses["1525"][control] - full_loss) / losses["1525"][control] >= 0.00005
            for control in ("raw_baseline", "folded_null")
        ),
        "paired_intervals_favor_self_at_50m": all(
            intervals[f"self_vs_{control}_1525"]["upper_95"] < 0
            for control in ("raw_baseline", "folded_null")
        ),
        "gamma_live": terminal_diag["coefficient_abs_mean"] >= 0.002,
        "gamma_bounded": terminal_diag["coefficient_abs_max_layer_median"] < BOUND,
        "chart_condition_below_1p2": terminal_diag["maximum_chart_condition"] < 1.2,
        "zero_ablation_hurts_0p005_percent": (zero_loss - full_loss) / full_loss >= 0.00005,
        "zero_interval_favors_full": intervals["full_vs_zero_1525"]["upper_95"] < 0,
        "signs_not_collapsed": 0.1 <= terminal_diag["input_positive_fraction_layer_median"] <= 0.9,
        "rms_within_5_percent": 0.95 <= terminal_diag["output_rms_layer_median"] / terminal_diag["input_rms_layer_median"] <= 1.05,
    }
    return {
        "complete": True, "losses": losses, "paired_intervals": intervals,
        "terminal_candidate_diagnostics": terminal_diag,
        "terminal_zero_loss": zero_loss, "gates": gates,
        "replicated": all(gates.values()),
    }


def validate_protocol(args):
    expected = {
        "device": "NVIDIA H100 80GB HBM3", "torch": "2.5.1+cu124", "cuda": "12.4",
        "transformers": "4.57.6", "steps": 1525, "eval_steps": [305, 1525],
        "sequence_length": 512, "micro_batch_size": 32, "gradient_accumulation": 2,
        "eval_batch_size": 32, "eval_batches": 128, "warmup_steps": 100,
        "learning_rate": 3e-4, "weight_decay": 0.1, "seed": 811,
    }
    actual = {
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": transformers.__version__,
        **{key: getattr(args, key) for key in expected if key not in {"device", "torch", "cuda", "transformers"}},
    }
    checks = {key: actual[key] == value for key, value in expected.items()}
    if args.strict_protocol and not all(checks.values()):
        raise ValueError({key: {"expected": expected[key], "actual": actual[key]} for key, passed in checks.items() if not passed})
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def parse_steps(text):
    return [int(value) for value in text.split(",") if value]


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    protocol = validate_protocol(args)
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    ledger = validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    payload = {
        "schema": "projection-shared-self-hinge-lm-replication-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "data_ledger": ledger, "protocol": protocol, "model": MODEL,
        "model_revision": MODEL_REVISION, "bound": BOUND, "arms": {},
    }
    for arm in VALID_ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, torch.device("cuda")
        )
        payload["decision"] = decide(payload["arms"], protocol["valid"] and ledger["valid"])
        write_payload(args.output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/projection-shared-self-hinge-lm-replication.json"))
    parser.add_argument("--steps", type=int, default=1525)
    parser.add_argument("--eval-steps", type=parse_steps, default=parse_steps("305,1525"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
