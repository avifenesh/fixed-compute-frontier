#!/usr/bin/env python3
"""Matched 10M-token screen for training-only gauge scaffolding."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
    MODEL, MODEL_REVISION, TokenFile, evaluate, lr_multiplier,
    paired_loss_interval, sha256_file, validate_data_ledger, write_payload,
)
from experiments.triangular_microdepth_lm_screen import HIDDEN_SIZE, scratch_config


VALID_ARMS = (
    "raw_baseline", "scaffold_null", "scaffold_scale", "scaffold_bias",
    "scaffold_hinge", "scaffold_centered",
)
SCAFFOLD_ARMS = VALID_ARMS[1:]
BOUND = 0.25
CENTER = math.sqrt(2.0 / math.pi)
PREREGISTRATION = Path("results/train-time-gauge-scaffold-lm-preregistration.md")


class ScaffoldRMSNorm(nn.Module):
    def __init__(self, source: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in SCAFFOLD_ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.weight = nn.Parameter(source.weight.detach().clone())
        half = source.weight.numel() // 2
        self.theta_alpha = nn.Parameter(torch.zeros(half, device=source.weight.device))
        self.theta_beta = nn.Parameter(torch.zeros(half, device=source.weight.device))
        self.epsilon = source.variance_epsilon
        self.ablation_mode = "full"
        self.record_diagnostics = False
        self.last_diagnostics = None

    def gamma(self, dtype: torch.dtype) -> torch.Tensor:
        theta = torch.cat((self.theta_alpha, self.theta_beta))
        return (BOUND * torch.tanh(theta / BOUND)).to(dtype)

    def chart(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        mode = self.ablation_mode
        if mode == "zero" or self.arm == "scaffold_null":
            return z + gamma.sum() * 0.0
        if mode == "scale" or self.arm == "scaffold_scale":
            return z + gamma * z
        if mode == "bias" or self.arm == "scaffold_bias":
            return z + gamma
        if mode == "hinge" or self.arm == "scaffold_hinge":
            return z + gamma * z.abs()
        if mode == "centered" or self.arm == "scaffold_centered":
            return z + gamma * (z.abs() - CENTER)
        if mode == "bias_only":
            return z - gamma * CENTER
        raise ValueError((self.arm, mode))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        z = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.epsilon)
        z = z.to(input_dtype)
        gamma = self.gamma(z.dtype)
        charted = self.chart(z, gamma)
        output = self.weight * charted
        if self.record_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "gamma_abs_mean": float(gamma.float().abs().mean()),
                    "gamma_abs_max": float(gamma.float().abs().max()),
                    "gain_abs_mean": float(self.weight.float().abs().mean()),
                    "input_positive_fraction": float((z > 0).float().mean()),
                    "input_rms": float(z.float().square().mean().sqrt()),
                    "output_rms": float(output.float().square().mean().sqrt()),
                    "output_mean": float(output.float().mean()),
                    "input_dtype": str(hidden_states.dtype),
                    "output_dtype": str(output.dtype),
                }
        return output


def hash_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor).tobytes())
    return digest.hexdigest()


def make_base_state(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    config = scratch_config()
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    del model
    return config, state, hash_state(state)


def build_model(config, state, device, arm):
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.load_state_dict(state, strict=True)
    modules = []
    if arm != "raw_baseline":
        for layer in model.model.layers:
            for attribute in ("input_layernorm", "post_attention_layernorm"):
                replacement = ScaffoldRMSNorm(getattr(layer, attribute), arm)
                setattr(layer, attribute, replacement)
                modules.append(replacement)
    model = model.to(device)
    model.config.use_cache = False
    return model, modules


@torch.no_grad()
def diagnostics(model, modules, validation_file, device):
    if not modules:
        return {"raw_baseline": True}
    model.eval()
    inputs, _ = validation_file.batch(0, 2, device)
    for module in modules:
        module.record_diagnostics = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    numeric = [key for key, value in records[0].items() if not isinstance(value, str)]
    result = {
        key + "_layer_median": float(np.median([record[key] for record in records]))
        for key in numeric
    }
    result["input_dtypes"] = sorted({record["input_dtype"] for record in records})
    result["output_dtypes"] = sorted({record["output_dtype"] for record in records})
    return result


def causal_loss(model, inputs, targets):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def train_arm(arm, args, config, state, train_file, validation_file, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, modules = build_model(config, state, device, arm)
    total = sum(parameter.numel() for parameter in model.parameters())
    scaffold_gain_count = sum(module.weight.numel() for module in modules)
    served = total - scaffold_gain_count
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    no_decay_ids = {id(parameter) for parameter in no_decay}
    decay = [parameter for parameter in model.parameters() if id(parameter) not in no_decay_ids]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    chart_diagnostics = {"initial": diagnostics(model, modules, validation_file, device)}
    torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True)
    losses, durations, norms = [], [], []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
        started = time.perf_counter(); accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"non-finite {arm} step {step + 1}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        losses.append(accumulated); norms.append(norm); durations.append(time.perf_counter() - started)
    evaluations[str(args.steps)] = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    chart_diagnostics["terminal"] = diagnostics(model, modules, validation_file, device)
    ablations = {}
    if arm in {"scaffold_hinge", "scaffold_centered"}:
        for mode in ("zero", "bias_only", "centered", "hinge"):
            for module in modules:
                module.ablation_mode = mode
            ablations[mode] = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
        for module in modules:
            module.ablation_mode = "full"
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm, "training_parameters": total, "served_parameters_after_export": served,
        "scaffold_gain_parameters_removed": scaffold_gain_count,
        "evaluations": evaluations, "diagnostics": chart_diagnostics,
        "terminal_ablations": ablations,
        "optimizer_parameter_counts": {"decay": sum(p.numel() for p in decay), "no_decay": sum(p.numel() for p in no_decay)},
        "train": {"prediction_tokens": prediction_tokens, "mean_loss": float(np.mean(losses)),
                  "final_loss": losses[-1], "max_loss": max(losses),
                  "max_preclip_gradient_norm": max(norms), "elapsed_seconds": sum(durations),
                  "tokens_per_second": prediction_tokens / sum(durations),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "nonfinite": False},
    }
    del optimizer, modules, model; gc.collect(); torch.cuda.empty_cache()
    return result


def decide(results, valid):
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing": sorted(set(VALID_ARMS) - set(results))}
    step = "305"
    losses = {arm: result["evaluations"][step]["loss"] for arm, result in results.items()}
    winner = min(("scaffold_hinge", "scaffold_centered"), key=lambda arm: losses[arm])
    controls = ("scaffold_null", "scaffold_scale", "scaffold_bias")
    best_control = min(controls, key=lambda arm: losses[arm])
    intervals = {
        "winner_vs_raw": paired_loss_interval(results[winner], results["raw_baseline"], step),
        "winner_vs_best_control": paired_loss_interval(results[winner], results[best_control], step),
    }
    full_wrapper = {"evaluations": {step: results[winner]["evaluations"][step]}}
    intervals["winner_full_vs_zero"] = paired_loss_interval(
        full_wrapper, {"evaluations": {step: results[winner]["terminal_ablations"]["zero"]}}, step
    )
    initial = [result["evaluations"]["0"]["loss"] for result in results.values()]
    diag = results[winner]["diagnostics"]["terminal"]
    full_loss = losses[winner]
    zero_loss = results[winner]["terminal_ablations"]["zero"]["loss"]
    scaffold_counts = {results[arm]["training_parameters"] for arm in SCAFFOLD_ARMS}
    served_counts = {result["served_parameters_after_export"] for result in results.values()}
    gates = {
        "protocol_and_data_valid": valid,
        "initial_losses_match_2e_5": max(initial) - min(initial) <= 2e-5,
        "finite_training": all(not result["train"]["nonfinite"] for result in results.values()),
        "scaffold_training_counts_equal": len(scaffold_counts) == 1,
        "exported_served_counts_equal_raw": len(served_counts) == 1,
        "winner_0p005_percent_better_raw": (losses["raw_baseline"] - full_loss) / losses["raw_baseline"] >= 0.00005,
        "winner_0p005_percent_better_best_control": (losses[best_control] - full_loss) / losses[best_control] >= 0.00005,
        "paired_interval_favors_winner_vs_raw": intervals["winner_vs_raw"]["upper_95"] < 0,
        "paired_interval_favors_winner_vs_best_control": intervals["winner_vs_best_control"]["upper_95"] < 0,
        "gamma_live": diag["gamma_abs_mean_layer_median"] >= 0.002,
        "gamma_bounded": diag["gamma_abs_max_layer_median"] < BOUND,
        "signs_not_collapsed": 0.1 <= diag["input_positive_fraction_layer_median"] <= 0.9,
        "rms_within_5_percent": 0.95 <= diag["output_rms_layer_median"] / diag["input_rms_layer_median"] <= 1.05,
        "zero_ablation_hurts_0p005_percent": (zero_loss - full_loss) / full_loss >= 0.00005,
        "zero_interval_favors_full": intervals["winner_full_vs_zero"]["upper_95"] < 0,
    }
    return {
        "complete": True, "losses": losses, "selected_nonlinear": winner,
        "best_control": best_control, "paired_intervals": intervals,
        "winner_terminal_diagnostics": diag,
        "winner_ablation_losses": {mode: value["loss"] for mode, value in results[winner]["terminal_ablations"].items()},
        "gates": gates, "advance_to_50m": all(gates.values()),
    }


def protocol(args):
    expected = {"device": "NVIDIA H100 80GB HBM3", "torch": "2.5.1+cu124", "cuda": "12.4",
                "transformers": "4.57.6", "steps": 305, "sequence_length": 512,
                "micro_batch_size": 32, "gradient_accumulation": 2, "eval_batch_size": 32,
                "eval_batches": 128, "warmup_steps": 30, "learning_rate": 3e-4,
                "weight_decay": 0.1, "seed": 812}
    actual = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers": transformers.__version__,
              **{key: getattr(args, key) for key in expected if key not in {"device", "torch", "cuda", "transformers"}}}
    checks = {key: actual[key] == value for key, value in expected.items()}
    if args.strict_protocol and not all(checks.values()):
        raise ValueError({key: {"expected": expected[key], "actual": actual[key]} for key, ok in checks.items() if not ok})
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def run(args):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    protocol_result = protocol(args)
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    ledger = validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    config, state, state_hash = make_base_state(args.seed)
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
    payload = {"schema": "train-time-gauge-scaffold-lm-v1", "source_sha256": sha256_file(Path(__file__)),
               "preregistration_sha256": sha256_file(PREREGISTRATION), "base_state_sha256": state_hash,
               "data_ledger": ledger, "protocol": protocol_result, "model": MODEL,
               "model_revision": MODEL_REVISION, "bound": BOUND, "center": CENTER, "arms": {}}
    for arm in VALID_ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(arm, args, config, state, train_file, validation_file, torch.device("cuda"))
        payload["decision"] = decide(payload["arms"], protocol_result["valid"] and ledger["valid"])
        write_payload(args.output, payload)
        print(json.dumps({"finished_arm": arm, "loss": payload["arms"][arm]["evaluations"][str(args.steps)]["loss"]}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/train-time-gauge-scaffold-lm-screen.json"))
    parser.add_argument("--steps", type=int, default=305); parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32); parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32); parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=30); parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1); parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(); payload = run(args); print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__": main()
