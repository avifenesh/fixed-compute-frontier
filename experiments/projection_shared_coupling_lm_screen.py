#!/usr/bin/env python3
"""Matched 10M-token LM screen for projection-shared coupling norms."""

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
from experiments.triangular_microdepth_lm_screen import HIDDEN_SIZE, scratch_config


VALID_ARMS = ("raw_baseline", "folded_null", "self_hinge", "one_way", "two_way")
BOUND = 0.25
PREREGISTRATION = Path("results/projection-shared-coupling-lm-development-preregistration.md")


class CouplingRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float, arm: str) -> None:
        super().__init__()
        if arm not in set(VALID_ARMS) - {"raw_baseline"}:
            raise ValueError(arm)
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.arm = arm
        self.theta_alpha = nn.Parameter(torch.zeros(hidden_size // 2))
        self.theta_beta = nn.Parameter(torch.zeros(hidden_size // 2))
        self.ablation_mode = "full"
        self.record_diagnostics = False
        self.last_diagnostics = None

    @staticmethod
    def bounded(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (BOUND * torch.tanh(theta / BOUND)).to(dtype)

    def coefficients(self, dtype: torch.dtype):
        return self.bounded(self.theta_alpha, dtype), self.bounded(
            self.theta_beta, dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        normalized = values * torch.rsqrt(
            values.square().mean(-1, keepdim=True) + self.epsilon
        )
        normalized = normalized.to(input_dtype)
        alpha, beta = self.coefficients(input_dtype)
        mode = self.ablation_mode

        if self.arm == "folded_null" or mode == "zero":
            output = normalized + (alpha.sum() + beta.sum()) * 0.0
        elif self.arm == "self_hinge":
            gamma = torch.cat((alpha, beta))
            output = normalized + gamma * normalized.abs()
        else:
            a, b = normalized.chunk(2, dim=-1)
            if mode == "shifted":
                source_a = torch.roll(a, shifts=1, dims=-1)
            else:
                source_a = a
            b_prime = b + alpha * source_a.abs()
            if self.arm == "one_way" or mode == "one_way":
                a_prime = a + beta.sum() * 0.0
            else:
                a_prime = a + beta * b_prime.abs()
            output = torch.cat((a_prime, b_prime), dim=-1)

        if self.record_diagnostics:
            with torch.no_grad():
                flat_alpha = alpha.float()
                flat_beta = beta.float()
                self.last_diagnostics = {
                    "alpha_abs_mean": float(flat_alpha.abs().mean()),
                    "beta_abs_mean": float(flat_beta.abs().mean()),
                    "coefficient_abs_max": float(
                        torch.maximum(flat_alpha.abs().max(), flat_beta.abs().max())
                    ),
                    "input_positive_fraction": float((normalized > 0).float().mean()),
                    "input_rms": float(normalized.float().square().mean().sqrt()),
                    "output_rms": float(output.float().square().mean().sqrt()),
                }
        return output

    @torch.no_grad()
    def maximum_chart_condition(self) -> float:
        alpha, beta = self.coefficients(torch.float32)
        signs = torch.tensor(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            device=alpha.device,
        )
        sa = signs[:, 0, None]
        sb = signs[:, 1, None]
        j00 = 1.0 + beta[None, :] * sb * alpha[None, :] * sa
        j01 = beta[None, :] * sb
        j10 = alpha[None, :] * sa
        j11 = torch.ones_like(j00)
        matrices = torch.stack((j00, j01, j10, j11), dim=-1).reshape(-1, 2, 2)
        return float(torch.linalg.cond(matrices).max())


def build_model(device: torch.device, arm: str):
    config = scratch_config()
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules = []
    if arm == "raw_baseline":
        return model, modules
    for layer in model.model.layers:
        for attribute in ("input_layernorm", "post_attention_layernorm"):
            source = getattr(layer, attribute)
            epsilon = getattr(source, "variance_epsilon", config.rms_norm_eps)
            replacement = CouplingRMSNorm(HIDDEN_SIZE, epsilon, arm).to(device)
            setattr(layer, attribute, replacement)
            modules.append(replacement)
    return model, modules


@torch.no_grad()
def diagnostics(model, modules, validation_file, device):
    model.eval()
    inputs, _ = validation_file.batch(0, 2, device)
    if not modules:
        return {"raw_baseline": True}
    for module in modules:
        module.record_diagnostics = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    if any(record is None for record in records):
        raise RuntimeError("missing coupling diagnostics")
    result = {
        key + "_layer_median": float(np.median([record[key] for record in records]))
        for key in records[0]
    }
    result["coefficient_abs_mean"] = float(
        np.mean(
            [
                0.5 * (record["alpha_abs_mean"] + record["beta_abs_mean"])
                for record in records
            ]
        )
    )
    result["maximum_chart_condition"] = max(
        module.maximum_chart_condition() for module in modules
    )
    return result


def causal_loss(model, inputs, targets):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def train_arm(arm, args, train_file, validation_file, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(device, arm)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    chart_diagnostics = {"initial": diagnostics(model, modules, validation_file, device)}
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
    evaluations[str(args.steps)] = evaluate(
        model, validation_file, args.eval_batches, args.eval_batch_size, device
    )
    chart_diagnostics["terminal"] = diagnostics(model, modules, validation_file, device)

    ablations = {}
    if arm == "two_way":
        for mode in ("zero", "one_way", "shifted"):
            for module in modules:
                module.ablation_mode = mode
            ablations[mode] = evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
        for module in modules:
            module.ablation_mode = "full"

    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "evaluations": evaluations,
        "diagnostics": chart_diagnostics,
        "terminal_ablations": ablations,
        "train": {
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "max_loss": max(losses),
            "max_preclip_gradient_norm": max(norms),
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


def decide(results, protocol_valid):
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing": sorted(set(VALID_ARMS) - set(results))}
    step = "305"
    losses = {arm: result["evaluations"][step]["loss"] for arm, result in results.items()}
    controls = ("folded_null", "self_hinge", "one_way")
    best_control = min(controls, key=lambda arm: losses[arm])
    candidate = results["two_way"]
    intervals = {
        "two_way_vs_raw": paired_loss_interval(candidate, results["raw_baseline"], step),
        "two_way_vs_best_control": paired_loss_interval(candidate, results[best_control], step),
    }
    full_wrapper = {"evaluations": {step: candidate["evaluations"][step]}}
    intervals["full_vs_zero"] = paired_loss_interval(
        full_wrapper, {"evaluations": {step: candidate["terminal_ablations"]["zero"]}}, step
    )
    initial_losses = [result["evaluations"]["0"]["loss"] for result in results.values()]
    terminal_diag = candidate["diagnostics"]["terminal"]
    zero_loss = candidate["terminal_ablations"]["zero"]["loss"]
    gates = {
        "protocol_valid": protocol_valid,
        "identical_total_parameters": len({r["total_parameters"] for r in results.values()}) == 1,
        "identical_trainable_parameters": len({r["trainable_parameters"] for r in results.values()}) == 1,
        "initial_losses_match_2e_5": max(initial_losses) - min(initial_losses) <= 2e-5,
        "finite_training": all(not r["train"]["nonfinite"] for r in results.values()),
        "two_way_0p025_percent_better_raw": (losses["raw_baseline"] - losses["two_way"]) / losses["raw_baseline"] >= 0.00025,
        "two_way_0p01_percent_better_best_control": (losses[best_control] - losses["two_way"]) / losses[best_control] >= 0.0001,
        "paired_interval_favors_two_way_vs_raw": intervals["two_way_vs_raw"]["upper_95"] < 0,
        "paired_interval_favors_two_way_vs_best_control": intervals["two_way_vs_best_control"]["upper_95"] < 0,
        "coefficients_live": terminal_diag["coefficient_abs_mean"] >= 0.002,
        "coefficients_bounded": terminal_diag["coefficient_abs_max_layer_median"] < BOUND,
        "chart_condition_below_2": terminal_diag["maximum_chart_condition"] < 2,
        "zero_ablation_hurts": zero_loss > losses["two_way"],
        "zero_interval_favors_full": intervals["full_vs_zero"]["upper_95"] < 0,
        "signs_not_collapsed": 0.1 <= terminal_diag["input_positive_fraction_layer_median"] <= 0.9,
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "best_control": best_control,
        "paired_intervals": intervals,
        "terminal_candidate_diagnostics": terminal_diag,
        "terminal_candidate_ablation_losses": {
            mode: value["loss"] for mode, value in candidate["terminal_ablations"].items()
        },
        "gates": gates,
        "advance_to_50m_replication": all(gates.values()),
    }


def validate_protocol(args):
    expected = {
        "arms": VALID_ARMS,
        "device": "NVIDIA H100 80GB HBM3",
        "torch": "2.5.1+cu124",
        "cuda": "12.4",
        "transformers": "4.57.6",
        "steps": 305,
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 128,
        "warmup_steps": 30,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "seed": 809,
    }
    actual = {
        "arms": tuple(value.strip() for value in args.arms.split(",") if value.strip()),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        **{
            key: getattr(args, key)
            for key in expected
            if key not in {"arms", "device", "torch", "cuda", "transformers"}
            and hasattr(args, key)
        },
    }
    checks = {key: actual[key] == value for key, value in expected.items()}
    if args.strict_protocol and not all(checks.values()):
        raise ValueError({key: {"expected": expected[key], "actual": actual[key]} for key, passed in checks.items() if not passed})
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if arms != VALID_ARMS:
        raise ValueError(f"strict arm order is {VALID_ARMS}")
    protocol = validate_protocol(args)
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload = {
        "schema": "projection-shared-coupling-lm-development-v1",
        "scope": "matched one-seed 10M-token scratch screen",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "data_ledger": ledger,
        "protocol": protocol,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "bound": BOUND,
        "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(payload["arms"], protocol["valid"] and ledger["valid"])
        write_payload(args.output, payload)
        print(
            json.dumps(
                {
                    "finished_arm": arm,
                    "terminal_loss": payload["arms"][arm]["evaluations"][str(args.steps)]["loss"],
                }
            ),
            flush=True,
        )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/projection-shared-coupling-lm-development.json"))
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
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
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
