#!/usr/bin/env python3
"""Matched 50M-token LM screen for exact-budget self-product FFN."""

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

from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
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
from experiments.triangular_microdepth_lm_screen import (
    BASELINE_INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
    causal_loss,
)


VALID_ARMS = ("parallel_swiglu", "parallel_wide_silu", "parallel_self_product")
WIDE = 3 * BASELINE_INTERMEDIATE_SIZE // 2
PLAIN_SCALE = math.sqrt(BASELINE_INTERMEDIATE_SIZE / WIDE)
SELF_PRODUCT_SCALE = 0.44642046792894413
PREREGISTRATION = Path("results/self-product-ffn-lm-preregistration.md")
TEST_SOURCE = Path("tests/test_self_product_ffn_lm_screen.py")
STAGE0_SOURCE = Path("experiments/generator_edge_ffn_gate.py")
STAGE0_RESULT = Path("results/generator-edge-ffn-stage0.json")
H100_SOURCE = Path("experiments/self_product_ffn_h100.py")
H100_RESULT = Path("results/self-product-ffn-h100.json")
H100_PREREGISTRATION = Path("results/self-product-ffn-h100-preregistration.md")
H100_TEST = Path("tests/test_self_product_ffn_h100.py")
INTEGRITY_MANIFEST = Path("results/self-product-ffn-lm-integrity-manifest.json")


class WideAtomMLP(nn.Module):
    def __init__(self, initializer_range: float, arm: str) -> None:
        super().__init__()
        if arm not in {"parallel_wide_silu", "parallel_self_product"}:
            raise ValueError(f"invalid wide atom arm: {arm}")
        self.arm = arm
        self.generator_proj = nn.Linear(HIDDEN_SIZE, WIDE, bias=False)
        self.down_proj = nn.Linear(WIDE, HIDDEN_SIZE, bias=False)
        for projection in (self.generator_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=initializer_range)
        self.ablation_plain = False
        self.deployment_folded = False
        self.record_diagnostics = False
        self.diagnostic_reference_rms: float | None = None
        self.last_diagnostics: dict[str, float] | None = None

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        generated = self.generator_proj(hidden_states)
        if self.arm == "parallel_wide_silu":
            activation = F.silu(generated)
            if not self.deployment_folded:
                activation = PLAIN_SCALE * activation
        elif self.ablation_plain:
            activation = F.silu(generated)
            if not self.deployment_folded:
                activation = SELF_PRODUCT_SCALE * activation
        else:
            activation = F.silu(generated) * generated
            if not self.deployment_folded:
                activation = SELF_PRODUCT_SCALE * activation
        if self.record_diagnostics:
            with torch.no_grad():
                values = activation.float()
                absolute = values.abs()
                self.last_diagnostics = {
                    "activation_rms": float(values.square().mean().sqrt()),
                    "activation_abs_max": float(absolute.max()),
                    "activation_abs_p99": float(torch.quantile(absolute.flatten(), 0.99)),
                    "activation_nonfinite_fraction": float((~torch.isfinite(values)).float().mean()),
                }
                if self.diagnostic_reference_rms is not None:
                    self.last_diagnostics["activation_outside_4x_reference_rms_fraction"] = float(
                        (absolute > 4.0 * self.diagnostic_reference_rms).float().mean()
                    )
        return activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden_states))

    def folded_down_weight(self) -> torch.Tensor:
        if self.deployment_folded:
            return self.down_proj.weight
        scale = PLAIN_SCALE if self.arm == "parallel_wide_silu" else SELF_PRODUCT_SCALE
        return self.down_proj.weight * scale

    def fold_for_deployment(self) -> None:
        if self.deployment_folded:
            return
        with torch.no_grad():
            self.down_proj.weight.copy_(self.folded_down_weight())
        self.deployment_folded = True


def build_model(device: torch.device, arm: str) -> tuple[nn.Module, list[WideAtomMLP]]:
    if arm not in VALID_ARMS:
        raise ValueError(f"invalid arm: {arm}")
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules: list[WideAtomMLP] = []
    if arm == "parallel_swiglu":
        return model, modules
    for layer in model.model.layers:
        replacement = WideAtomMLP(model.config.initializer_range, arm).to(device)
        layer.mlp = replacement
        modules.append(replacement)
    return model, modules


def ffn_dense_parameters(arm: str) -> int:
    if arm == "parallel_swiglu":
        return 3 * HIDDEN_SIZE * BASELINE_INTERMEDIATE_SIZE
    return 2 * HIDDEN_SIZE * WIDE


@torch.no_grad()
def activation_diagnostics(
    model: nn.Module,
    modules: list[WideAtomMLP],
    validation_file: TokenFile,
    device: torch.device,
    reference_rms: float | None = None,
) -> dict[str, float]:
    model.eval()
    inputs, _ = validation_file.batch(0, 2, device)
    if modules:
        for module in modules:
            module.record_diagnostics = True
            module.diagnostic_reference_rms = reference_rms
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, use_cache=False)
        records = [module.last_diagnostics for module in modules]
        for module in modules:
            module.record_diagnostics = False
            module.diagnostic_reference_rms = None
        if any(record is None for record in records):
            raise RuntimeError("missing wide atom diagnostics")
        typed = [record for record in records if record is not None]
    else:
        captured: list[list[torch.Tensor]] = [[] for _ in model.model.layers]
        handles = [
            layer.mlp.down_proj.register_forward_pre_hook(
                lambda _module, args, index=index: captured[index].append(args[0].detach())
            )
            for index, layer in enumerate(model.model.layers)
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=inputs, use_cache=False)
        for handle in handles:
            handle.remove()
        if any(len(values) != 1 for values in captured):
            raise RuntimeError("missing SwiGLU activation diagnostic")
        typed = []
        for values in captured:
            activation = values[0].float()
            absolute = activation.abs()
            record = {
                "activation_rms": float(activation.square().mean().sqrt()),
                "activation_abs_max": float(absolute.max()),
                "activation_abs_p99": float(torch.quantile(absolute.flatten(), 0.99)),
                "activation_nonfinite_fraction": float((~torch.isfinite(activation)).float().mean()),
            }
            if reference_rms is not None:
                record["activation_outside_4x_reference_rms_fraction"] = float(
                    (absolute > 4.0 * reference_rms).float().mean()
                )
            typed.append(record)
    summary = {
        key + "_layer_median": float(np.median([record[key] for record in typed]))
        for key in typed[0]
    }
    summary["activation_abs_max_layer_max"] = max(record["activation_abs_max"] for record in typed)
    summary["activation_abs_p99_layer_max"] = max(record["activation_abs_p99"] for record in typed)
    summary["activation_nonfinite_fraction_layer_max"] = max(
        record["activation_nonfinite_fraction"] for record in typed
    )
    if reference_rms is not None:
        summary["activation_outside_4x_reference_rms_fraction_layer_max"] = max(
            record["activation_outside_4x_reference_rms_fraction"] for record in typed
        )
    return summary


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
        "steps": 1525,
        "eval_steps": [305, 1525],
        "warmup_steps": 100,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 1907,
        "prediction_tokens": 49_971_200,
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
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
        "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
    }
    checks = {
        key: (expected_value in actual[key] if key == "device_contains" else actual[key] == expected_value)
        for key, expected_value in expected.items()
    }
    if args.strict_protocol and not all(checks.values()):
        failed = {key: {"expected": expected[key], "actual": actual[key]} for key, valid in checks.items() if not valid}
        raise ValueError(f"invalid protocol: {json.dumps(failed, sort_keys=True)}")
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def train_arm(
    arm: str,
    args: argparse.Namespace,
    train_file: TokenFile,
    validation_file: TokenFile,
    device: torch.device,
    reference_rms: dict[str, float] | None,
) -> dict[str, Any]:
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
    optimizer.param_groups[0]["base_lr"] = args.learning_rate
    diagnostics = {"initial": activation_diagnostics(
        model, modules, validation_file, device,
        None if reference_rms is None else reference_rms["initial"],
    )}
    evaluations = {"0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)}
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    durations: list[float] = []
    norms: list[float] = []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        optimizer.param_groups[0]["lr"] = args.learning_rate * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(norm) or not math.isfinite(accumulated):
            raise RuntimeError(f"nonfinite {arm} at step {step + 1}")
        optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
        losses.append(accumulated); norms.append(norm); durations.append(time.perf_counter() - started)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
            print(json.dumps({
                "arm": arm,
                "step": step + 1,
                "train_loss": accumulated,
                "validation_loss": evaluations[str(step + 1)]["loss"],
                "gradient_norm": norm,
            }), flush=True)
    diagnostics["terminal"] = activation_diagnostics(
        model, modules, validation_file, device,
        None if reference_rms is None else reference_rms["terminal"],
    )
    ablations: dict[str, Any] = {}
    if arm in {"parallel_wide_silu", "parallel_self_product"}:
        for module in modules:
            module.fold_for_deployment()
        ablations["folded_deployment"] = evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
    if arm == "parallel_self_product":
        for module in modules:
            module.ablation_plain = True
        ablations["plain_silu"] = evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
        for module in modules:
            module.ablation_plain = False
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "ffn_dense_parameters_per_layer": ffn_dense_parameters(arm),
        "evaluations": evaluations,
        "activation_diagnostics": diagnostics,
        "terminal_ablations": ablations,
        "train": {
            "steps": args.steps,
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(losses)),
            "max_loss": max(losses),
            "final_loss": losses[-1],
            "max_preclip_gradient_norm": max(norms),
            "elapsed_seconds": sum(durations),
            "tokens_per_second": prediction_tokens / sum(durations),
            "median_step_seconds_after_five": float(np.median(durations[5:])),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "nonfinite": False,
        },
    }
    del optimizer, modules, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def decide(
    results: dict[str, Any], integrity_valid: bool, h100_feasibility_pass: bool
) -> dict[str, Any]:
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing_arms": sorted(set(VALID_ARMS) - set(results))}
    early, terminal = "305", "1525"
    losses = {arm: result["evaluations"][terminal]["loss"] for arm, result in results.items()}
    early_losses = {arm: result["evaluations"][early]["loss"] for arm, result in results.items()}
    candidate = results["parallel_self_product"]
    wide = results["parallel_wide_silu"]
    unfolded_candidate_eval = candidate["evaluations"][terminal]
    folded_candidate_eval = candidate["terminal_ablations"]["folded_deployment"]
    unfolded_wide_eval = wide["evaluations"][terminal]
    folded_wide_eval = wide["terminal_ablations"]["folded_deployment"]
    losses["parallel_self_product"] = folded_candidate_eval["loss"]
    losses["parallel_wide_silu"] = folded_wide_eval["loss"]
    folded_candidate = {"evaluations": {terminal: folded_candidate_eval}}
    folded_wide = {"evaluations": {terminal: folded_wide_eval}}
    comparisons = {
        "parallel_swiglu": paired_loss_interval(folded_candidate, results["parallel_swiglu"], terminal),
        "parallel_wide_silu": paired_loss_interval(folded_candidate, folded_wide, terminal),
    }
    full = folded_candidate
    ablated_eval = candidate["terminal_ablations"]["plain_silu"]
    ablated = {"evaluations": {terminal: ablated_eval}}
    ablation_interval = paired_loss_interval(full, ablated, terminal)
    unfolded = {"evaluations": {terminal: unfolded_candidate_eval}}
    folded_vs_unfolded = paired_loss_interval(folded_candidate, unfolded, terminal)
    unfolded_wide = {"evaluations": {terminal: unfolded_wide_eval}}
    folded_vs_unfolded_wide = paired_loss_interval(folded_wide, unfolded_wide, terminal)
    diagnostic_values = [
        value for result in results.values() for phase in ("initial", "terminal")
        for value in result["activation_diagnostics"][phase].values()
    ]
    gates = {
        "integrity_protocol_valid": integrity_valid,
        "folded_h100_feasibility_pass": h100_feasibility_pass,
        "folded_vs_unfolded_nll_within_0p01_percent": abs(
            folded_candidate_eval["loss"] - unfolded_candidate_eval["loss"]
        ) / unfolded_candidate_eval["loss"] <= 0.0001,
        "wide_folded_vs_unfolded_nll_within_0p01_percent": abs(
            folded_wide_eval["loss"] - unfolded_wide_eval["loss"]
        ) / unfolded_wide_eval["loss"] <= 0.0001,
        "identical_total_parameters": len({result["total_parameters"] for result in results.values()}) == 1,
        "identical_trainable_parameters": len({result["trainable_parameters"] for result in results.values()}) == 1,
        "identical_ffn_dense_ledgers": len({result["ffn_dense_parameters_per_layer"] for result in results.values()}) == 1,
        "all_training_finite": all(not result["train"]["nonfinite"] for result in results.values()),
        "all_activation_diagnostics_finite": all(math.isfinite(value) for value in diagnostic_values),
        "all_activation_nonfinite_fractions_zero": all(
            result["activation_diagnostics"][phase]["activation_nonfinite_fraction_layer_max"] == 0.0
            for result in results.values() for phase in ("initial", "terminal")
        ),
        "candidate_activation_outside_4x_baseline_rms_at_most_1_percent": all(
            candidate["activation_diagnostics"][phase]["activation_outside_4x_reference_rms_fraction_layer_max"] <= 0.01
            for phase in ("initial", "terminal")
        ),
        "candidate_early_noninferior_within_0p05_percent": (early_losses["parallel_self_product"] - early_losses["parallel_swiglu"]) / early_losses["parallel_swiglu"] <= 0.0005,
        "candidate_terminal_0p05_percent_better_than_swiglu": (losses["parallel_swiglu"] - losses["parallel_self_product"]) / losses["parallel_swiglu"] >= 0.0005,
        "paired_interval_favors_candidate_vs_swiglu": comparisons["parallel_swiglu"]["upper_95"] < 0.0,
        "candidate_terminal_0p025_percent_better_than_wide_silu": (losses["parallel_wide_silu"] - losses["parallel_self_product"]) / losses["parallel_wide_silu"] >= 0.00025,
        "paired_interval_favors_candidate_vs_wide_silu": comparisons["parallel_wide_silu"]["upper_95"] < 0.0,
        "plain_ablation_hurts_0p01_percent": (ablated_eval["loss"] - losses["parallel_self_product"]) / losses["parallel_self_product"] >= 0.0001,
        "paired_interval_favors_full_vs_plain_ablation": ablation_interval["upper_95"] < 0.0,
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "early_losses": early_losses,
        "candidate_relative_improvements": {
            control: (losses[control] - losses["parallel_self_product"]) / losses[control]
            for control in ("parallel_swiglu", "parallel_wide_silu")
        },
        "paired_terminal_intervals": comparisons | {
            "full_vs_plain_ablation": ablation_interval,
            "folded_vs_unfolded": folded_vs_unfolded,
            "wide_folded_vs_unfolded": folded_vs_unfolded_wide,
        },
        "terminal_ablation_losses": {
            "unfolded": unfolded_candidate_eval["loss"],
            "folded_deployment": losses["parallel_self_product"],
            "plain_silu": ablated_eval["loss"],
            "wide_unfolded": unfolded_wide_eval["loss"],
            "wide_folded_deployment": losses["parallel_wide_silu"],
        },
        "gates": gates,
        "advance_to_second_seed_subject_to_folded_h100": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    arms = tuple(value for value in args.arms.split(",") if value)
    if arms != VALID_ARMS:
        raise ValueError(f"strict arms/order required: {VALID_ARMS}")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    protocol = validate_protocol(args, arms)
    if train_file.sequence_count < args.steps * args.gradient_accumulation * args.micro_batch_size:
        raise ValueError("train file too short")
    if validation_file.sequence_count < args.eval_batches * args.eval_batch_size:
        raise ValueError("validation file too short")
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__), "preregistration": PREREGISTRATION, "test": TEST_SOURCE,
        "stage0_source": STAGE0_SOURCE, "stage0_result": STAGE0_RESULT,
        "h100_source": H100_SOURCE, "h100_result": H100_RESULT,
        "h100_preregistration": H100_PREREGISTRATION, "h100_test": H100_TEST,
    }
    integrity_checks = {name: manifest.get(name + "_sha256") == sha256_file(path) for name, path in paths.items()}
    if not all(integrity_checks.values()):
        raise ValueError(f"invalid integrity manifest: {integrity_checks}")
    h100_payload = json.loads(H100_RESULT.read_text())
    h100_embedded_expected = {
        "source": sha256_file(H100_SOURCE),
        "preregistration": sha256_file(H100_PREREGISTRATION),
        "lm_source": sha256_file(Path(__file__)),
        "lm_preregistration": sha256_file(PREREGISTRATION),
        "test": sha256_file(H100_TEST),
    }
    h100_embedded_checks = {
        key: h100_payload.get("hashes", {}).get(key) == value
        for key, value in h100_embedded_expected.items()
    }
    if not all(h100_embedded_checks.values()):
        raise ValueError(f"stale folded H100 artifact: {h100_embedded_checks}")
    h100_feasibility_pass = bool(h100_payload.get("h100_feasibility_pass"))
    if not h100_feasibility_pass:
        raise ValueError("frozen folded H100 feasibility gate did not pass")
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "candidate": "variance-matched-self-product-ffn",
        "scope": "matched one-seed 50M-token from-scratch screen",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
        "integrity_checks": integrity_checks,
        "folded_h100_result_sha256": sha256_file(H100_RESULT),
        "folded_h100_feasibility_pass": h100_feasibility_pass,
        "folded_h100_embedded_hash_checks": h100_embedded_checks,
        "data_ledger": data_ledger,
        "experiment_protocol": protocol,
        "base_config": MODEL,
        "base_config_revision": MODEL_REVISION,
        "candidate_shape": {"width": WIDE, "plain_scale": PLAIN_SCALE, "self_product_scale": SELF_PRODUCT_SCALE},
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        reference_rms = None
        if "parallel_swiglu" in payload["arms"]:
            baseline_diagnostics = payload["arms"]["parallel_swiglu"]["activation_diagnostics"]
            reference_rms = {phase: baseline_diagnostics[phase]["activation_rms_layer_median"] for phase in ("initial", "terminal")}
        payload["arms"][arm] = train_arm(arm, args, train_file, validation_file, device, reference_rms)
        payload["decision"] = decide(
            payload["arms"],
            data_ledger["valid"] and protocol["valid"] and all(integrity_checks.values()),
            h100_feasibility_pass,
        )
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/self-product-ffn-lm-screen.json"))
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1525)
    parser.add_argument("--eval-steps", type=parse_steps, default=parse_steps("305,1525"))
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
