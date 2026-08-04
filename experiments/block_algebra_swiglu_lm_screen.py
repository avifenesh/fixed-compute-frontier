#!/usr/bin/env python3
"""Matched from-scratch LM screen for rank-three block-algebra SwiGLU."""

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
from transformers import AutoConfig, AutoModelForCausalLM

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


VALID_ARMS = ("baseline", "split_complex", "complex")
PREREGISTRATION = Path("results/block-algebra-swiglu-lm-screen-preregistration.md")
INTEGRITY_MANIFEST = Path("results/block-algebra-swiglu-integrity-manifest.json")
TEST_SOURCE = Path("tests/test_block_algebra_swiglu_lm_screen.py")
ALGEBRA_GATE_SOURCE = Path("experiments/block_algebra_swiglu_gate.py")
ALGEBRA_GATE_TEST = Path("tests/test_block_algebra_swiglu_gate.py")
ALGEBRA_GATE_RESULT = Path("results/block-algebra-swiglu-stage0.json")
HIDDEN_SIZE = 384
INTERMEDIATE_SIZE = 1024
LAYERS = 12
ATTENTION_HEADS = 6
KV_HEADS = 2
HEAD_DIM = 64


def block_product(
    activated_gate: torch.Tensor, up: torch.Tensor, arm: str
) -> torch.Tensor:
    if arm == "baseline":
        return activated_gate * up
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown arm {arm}")
    if activated_gate.shape[-1] % 2:
        raise ValueError("block algebra requires an even intermediate width")
    gate_pairs = activated_gate.reshape(*activated_gate.shape[:-1], -1, 2)
    up_pairs = up.reshape(*up.shape[:-1], -1, 2)
    g0, g1 = gate_pairs.unbind(dim=-1)
    u0, u1 = up_pairs.unbind(dim=-1)
    if arm == "split_complex":
        first = g0 * u0 + g1 * u1
    else:
        first = g0 * u0 - g1 * u1
    second = g0 * u1 + g1 * u0
    return torch.stack((first, second), dim=-1).flatten(-2) / math.sqrt(2.0)


class BlockAlgebraMLP(nn.Module):
    def __init__(self, original: nn.Module, arm: str) -> None:
        super().__init__()
        if arm not in VALID_ARMS:
            raise ValueError(f"unknown arm {arm}")
        if original.gate_proj.out_features % 2:
            raise ValueError("model intermediate width must be even")
        self.arm = arm
        self.gate_proj = original.gate_proj
        self.up_proj = original.up_proj
        self.down_proj = original.down_proj
        self.act_fn = original.act_fn
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def activation(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        activated_gate = self.act_fn(gate)
        result = block_product(activated_gate, up, self.arm)
        if self.record_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "activated_gate_rms": float(activated_gate.float().square().mean().sqrt()),
                    "up_rms": float(up.float().square().mean().sqrt()),
                    "activation_rms": float(result.float().square().mean().sqrt()),
                    "activation_abs_max": float(result.float().abs().max()),
                }
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(self.activation(gate, up))


def install_arm(model: nn.Module, arm: str) -> list[BlockAlgebraMLP]:
    modules = []
    for layer in model.model.layers:
        replacement = BlockAlgebraMLP(layer.mlp, arm)
        layer.mlp = replacement
        modules.append(replacement)
    return modules


def scratch_config() -> transformers.PretrainedConfig:
    config = AutoConfig.from_pretrained(MODEL, revision=MODEL_REVISION)
    config.hidden_size = HIDDEN_SIZE
    config.intermediate_size = INTERMEDIATE_SIZE
    config.num_hidden_layers = LAYERS
    config.num_attention_heads = ATTENTION_HEADS
    config.num_key_value_heads = KV_HEADS
    config.head_dim = HEAD_DIM
    config.initializer_range = 1.0 / math.sqrt(HIDDEN_SIZE)
    config.dtype = "float32"
    config.use_cache = False
    return config


def build_model(device: torch.device, arm: str) -> tuple[nn.Module, list[BlockAlgebraMLP]]:
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    return model, install_arm(model, arm)


def causal_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


@torch.no_grad()
def activation_diagnostics(
    model: nn.Module,
    modules: list[BlockAlgebraMLP],
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    for module in modules:
        module.record_diagnostics = True
    inputs, _ = validation_file.batch(0, 2, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    torch.cuda.synchronize()
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    if any(record is None for record in records):
        raise RuntimeError("missing activation diagnostics")
    typed = [record for record in records if record is not None]
    return {
        key + "_layer_median": float(np.median([record[key] for record in typed]))
        for key in (
            "activated_gate_rms",
            "up_rms",
            "activation_rms",
            "activation_abs_max",
        )
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
        "steps": 1525,
        "eval_steps": [305, 1525],
        "warmup_steps": 100,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": 223,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "layers": LAYERS,
        "attention_heads": ATTENTION_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
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
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "layers": LAYERS,
        "attention_heads": ATTENTION_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "prediction_tokens": (
            args.steps
            * args.gradient_accumulation
            * args.micro_batch_size
            * args.sequence_length
        ),
    }
    checks = {
        key: (
            expected_value in actual[key]
            if key == "device_contains"
            else actual[key] == expected_value
        )
        for key, expected_value in expected.items()
    }
    if args.strict_protocol and not all(checks.values()):
        failed = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key, valid in checks.items()
            if not valid
        }
        raise ValueError(f"invalid experiment protocol: {json.dumps(failed, sort_keys=True)}")
    return {
        "valid": all(checks.values()),
        "strict": args.strict_protocol,
        "checks": checks,
        "expected": expected,
        "actual": actual,
    }


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
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
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
    diagnostics = {
        "initial": activation_diagnostics(
            model, modules, validation_file, device
        )
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
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    step_losses, step_seconds, gradient_norms = [], [], []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * multiplier
        started = time.perf_counter()
        accumulated_loss = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(
                batch_index, args.micro_batch_size, device
            )
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation
        norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        )
        if not math.isfinite(norm) or not math.isfinite(accumulated_loss):
            raise RuntimeError(
                f"non-finite training state in {arm} at step {step + 1}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_losses.append(accumulated_loss)
        step_seconds.append(time.perf_counter() - started)
        gradient_norms.append(norm)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(
                model,
                validation_file,
                args.eval_batches,
                args.eval_batch_size,
                device,
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
    diagnostics["terminal"] = activation_diagnostics(
        model, modules, validation_file, device
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
        "trainable_parameters": trainable_parameters,
        "evaluations": evaluations,
        "activation_diagnostics": diagnostics,
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
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], integrity_valid: bool) -> dict[str, Any]:
    if set(results) != set(VALID_ARMS):
        return {
            "complete": False,
            "missing_arms": sorted(set(VALID_ARMS) - set(results)),
        }
    terminal = "1525"
    baseline = results["baseline"]
    split = results["split_complex"]
    complex_result = results["complex"]
    losses = {
        name: result["evaluations"][terminal]["loss"]
        for name, result in results.items()
    }
    complex_vs_baseline = paired_loss_interval(
        complex_result, baseline, terminal
    )
    complex_vs_split = paired_loss_interval(complex_result, split, terminal)
    baseline_activation_rms = baseline["activation_diagnostics"]["initial"][
        "activation_rms_layer_median"
    ]
    initialization_rms_ratios = {
        name: result["activation_diagnostics"]["initial"][
            "activation_rms_layer_median"
        ]
        / baseline_activation_rms
        for name, result in results.items()
    }
    gates = {
        "integrity_protocol_valid": integrity_valid,
        "identical_parameter_counts": len(
            {result["total_parameters"] for result in results.values()}
        )
        == 1,
        "identical_trainable_parameter_counts": len(
            {result["trainable_parameters"] for result in results.values()}
        )
        == 1,
        "initial_activation_rms_matched_within_25_percent": all(
            0.75 <= ratio <= 1.25 for ratio in initialization_rms_ratios.values()
        ),
        "all_training_finite": all(
            not result["train"]["nonfinite"] for result in results.values()
        ),
        "bounded_training": all(
            result["train"]["max_preclip_gradient_norm"] <= 100.0
            and result["train"]["max_loss"] <= 20.0
            for result in results.values()
        ),
        "complex_at_least_0p1_percent_better_than_baseline": (
            losses["baseline"] - losses["complex"]
        )
        / losses["baseline"]
        >= 0.001,
        "complex_at_least_0p1_percent_better_than_split_control": (
            losses["split_complex"] - losses["complex"]
        )
        / losses["split_complex"]
        >= 0.001,
        "paired_interval_favors_complex_vs_baseline": complex_vs_baseline[
            "upper_95"
        ]
        < 0.0,
        "paired_interval_favors_complex_vs_split": complex_vs_split["upper_95"]
        < 0.0,
    }
    return {
        "complete": True,
        "terminal_step": 1525,
        "terminal_losses": losses,
        "complex_relative_improvement_vs_baseline": (
            losses["baseline"] - losses["complex"]
        )
        / losses["baseline"],
        "complex_relative_improvement_vs_split": (
            losses["split_complex"] - losses["complex"]
        )
        / losses["split_complex"],
        "paired_terminal_intervals": {
            "complex_vs_baseline": complex_vs_baseline,
            "complex_vs_split": complex_vs_split,
        },
        "initial_activation_rms_ratios_to_baseline": initialization_rms_ratios,
        "gates": gates,
        "advance_to_fused_h100_gate": all(gates.values()),
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
        args.data_manifest,
        args.train_file,
        args.validation_file,
        args.sequence_length,
    )
    protocol = validate_protocol(args, arms)
    required_train = (
        args.steps * args.gradient_accumulation * args.micro_batch_size
    )
    required_validation = args.eval_batches * args.eval_batch_size
    if (
        train_file.sequence_count < required_train
        or validation_file.sequence_count < required_validation
    ):
        raise ValueError("token files are too short for the frozen protocol")
    integrity_manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    integrity_checks = {
        "source": integrity_manifest.get("source_sha256")
        == sha256_file(Path(__file__)),
        "preregistration": integrity_manifest.get("preregistration_sha256")
        == sha256_file(PREREGISTRATION),
        "test": integrity_manifest.get("test_sha256") == sha256_file(TEST_SOURCE),
        "algebra_gate_source": integrity_manifest.get("algebra_gate_source_sha256")
        == sha256_file(ALGEBRA_GATE_SOURCE),
        "algebra_gate_test": integrity_manifest.get("algebra_gate_test_sha256")
        == sha256_file(ALGEBRA_GATE_TEST),
        "algebra_gate_result": integrity_manifest.get("algebra_gate_result_sha256")
        == sha256_file(ALGEBRA_GATE_RESULT),
    }
    if not all(integrity_checks.values()):
        raise ValueError(f"invalid frozen integrity manifest: {integrity_checks}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "candidate": "rank-three complex block-algebra SwiGLU",
        "scope": "matched one-seed 50M-token from-scratch screen",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
        "integrity_checks": integrity_checks,
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "data_ledger": data_ledger,
        "experiment_protocol": protocol,
        "base_config": MODEL,
        "base_config_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(
            payload["arms"],
            data_ledger["valid"]
            and protocol["valid"]
            and all(integrity_checks.values()),
        )
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/block-algebra-scratch/train.uint16.bin"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/block-algebra-scratch/validation.uint16.bin"),
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path("results/block-algebra-scratch-data-manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/block-algebra-swiglu-lm-screen.json"),
    )
    parser.add_argument("--arms", default=",".join(VALID_ARMS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1525)
    parser.add_argument(
        "--eval-steps", type=parse_steps, default=parse_steps("305,1525")
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument(
        "--strict-protocol", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
