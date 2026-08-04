#!/usr/bin/env python3
"""Matched 50M-token LM screen for block-triangular FFN microdepth."""

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
from experiments.triangular_microdepth_gate import (
    matched_width,
    packed_edges,
    partial_bias_indices,
)


VALID_ARMS = ("baseline", "plain_silu", "linear_reparam", "triangular")
PREREGISTRATION = Path("results/triangular-microdepth-lm-preregistration.md")
INTEGRITY_MANIFEST = Path("results/triangular-microdepth-integrity-manifest.json")
TEST_SOURCE = Path("tests/test_triangular_microdepth_lm_screen.py")
ALGEBRA_SOURCE = Path("experiments/triangular_microdepth_gate.py")
ALGEBRA_TEST = Path("tests/test_triangular_microdepth_gate.py")
ALGEBRA_RESULT = Path("results/triangular-microdepth-stage0.json")
FUSED_SOURCE = Path("experiments/fused_rank_one_program_h100.py")
FUSED_CPP_SOURCE = Path("experiments/csrc/fused_rank_one.cpp")
FUSED_CUDA_SOURCE = Path("experiments/csrc/fused_rank_one_cuda.cu")
FUSED_RESULT = Path("results/fused-rank-one-program-h100-v3.json")

HIDDEN_SIZE = 384
BASELINE_INTERMEDIATE_SIZE = 1024
GROUP_SIZE = 8
CANDIDATE_WIDTH, BIAS_COUNT = matched_width(
    HIDDEN_SIZE, BASELINE_INTERMEDIATE_SIZE, GROUP_SIZE
)
GROUPS = CANDIDATE_WIDTH // GROUP_SIZE
COUPLING_PARAMETERS = GROUPS * packed_edges(GROUP_SIZE)
BIAS_INDICES = partial_bias_indices(CANDIDATE_WIDTH, GROUP_SIZE, BIAS_COUNT).tolist()
ACTIVATION_SCALE = math.sqrt(BASELINE_INTERMEDIATE_SIZE / CANDIDATE_WIDTH)
LAYERS = 12
ATTENTION_HEADS = 6
KV_HEADS = 2
HEAD_DIM = 64


class TriangularMicrodepthMLP(nn.Module):
    def __init__(self, arm: str, initializer_range: float) -> None:
        super().__init__()
        if arm not in {"linear_reparam", "triangular"}:
            raise ValueError(f"invalid microdepth arm {arm}")
        self.arm = arm
        self.up_proj = nn.Linear(HIDDEN_SIZE, CANDIDATE_WIDTH, bias=False)
        self.down_proj = nn.Linear(CANDIDATE_WIDTH, HIDDEN_SIZE, bias=False)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=initializer_range)
        self.packed_coupling = nn.Parameter(
            torch.zeros(GROUPS, packed_edges(GROUP_SIZE))
        )
        self.partial_bias = nn.Parameter(torch.zeros(BIAS_COUNT))
        self.register_buffer(
            "bias_indices", torch.tensor(BIAS_INDICES, dtype=torch.long), persistent=False
        )
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None
        self.ablation_mode = "full"

    def _base(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.up_proj(hidden_states)
        partial_bias = self.partial_bias.new_zeros(CANDIDATE_WIDTH).scatter(
            0, self.bias_indices, self.partial_bias
        )
        return (base + partial_bias).unflatten(-1, (GROUPS, GROUP_SIZE))

    def _linear_reparam(self, base: torch.Tensor) -> torch.Tensor:
        plain = F.silu(base)
        values = []
        for i in range(GROUP_SIZE):
            value = plain[..., i]
            if i:
                start = i * (i - 1) // 2
                value = value + torch.sum(
                    torch.stack(values, dim=-1)
                    * self.packed_coupling[:, start : start + i],
                    dim=-1,
                )
            values.append(value)
        return torch.stack(values, dim=-1)

    def _triangular(self, base: torch.Tensor) -> torch.Tensor:
        values = []
        one_hop_values = F.silu(base) if self.ablation_mode == "one_hop" else None
        for i in range(GROUP_SIZE):
            preactivation = base[..., i]
            if i:
                start = i * (i - 1) // 2
                previous = (
                    one_hop_values[..., :i]
                    if one_hop_values is not None
                    else torch.stack(values, dim=-1)
                )
                preactivation = preactivation + torch.sum(
                    previous * self.packed_coupling[:, start : start + i],
                    dim=-1,
                )
            values.append(F.silu(preactivation))
        return torch.stack(values, dim=-1)

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self._base(hidden_states)
        if self.arm == "linear_reparam":
            activation = self._linear_reparam(base)
        else:
            activation = self._triangular(base)
        activation = activation.flatten(-2) * ACTIVATION_SCALE
        if self.record_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "base_rms": float(base.float().square().mean().sqrt()),
                    "activation_rms": float(
                        activation.float().square().mean().sqrt()
                    ),
                    "activation_abs_max": float(activation.float().abs().max()),
                    "coupling_rms": float(
                        self.packed_coupling.float().square().mean().sqrt()
                    ),
                    "bias_rms": float(
                        self.partial_bias.float().square().mean().sqrt()
                    ),
                }
        return activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden_states))


class PlainSiluMLP(nn.Module):
    """Strong exact-budget two-projection control: 2*384*1536 = 3*384*1024."""

    WIDTH = 3 * BASELINE_INTERMEDIATE_SIZE // 2
    SCALE = math.sqrt(BASELINE_INTERMEDIATE_SIZE / WIDTH)

    def __init__(self, initializer_range: float) -> None:
        super().__init__()
        self.up_proj = nn.Linear(HIDDEN_SIZE, self.WIDTH, bias=False)
        self.down_proj = nn.Linear(self.WIDTH, HIDDEN_SIZE, bias=False)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(hidden_states)) * self.SCALE)


def scratch_config() -> transformers.PretrainedConfig:
    config = AutoConfig.from_pretrained(MODEL, revision=MODEL_REVISION)
    config.hidden_size = HIDDEN_SIZE
    config.intermediate_size = BASELINE_INTERMEDIATE_SIZE
    config.num_hidden_layers = LAYERS
    config.num_attention_heads = ATTENTION_HEADS
    config.num_key_value_heads = KV_HEADS
    config.head_dim = HEAD_DIM
    config.initializer_range = 1.0 / math.sqrt(HIDDEN_SIZE)
    config.dtype = "float32"
    config.use_cache = False
    return config


def build_model(
    device: torch.device, arm: str
) -> tuple[nn.Module, list[TriangularMicrodepthMLP]]:
    config = scratch_config()
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules = []
    if arm == "plain_silu":
        for layer in model.model.layers:
            layer.mlp = PlainSiluMLP(config.initializer_range).to(device)
    elif arm != "baseline":
        for layer in model.model.layers:
            replacement = TriangularMicrodepthMLP(arm, config.initializer_range).to(
                device
            )
            layer.mlp = replacement
            modules.append(replacement)
    return model, modules


def causal_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


@torch.no_grad()
def activation_diagnostics(
    model: nn.Module,
    modules: list[TriangularMicrodepthMLP],
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
            raise RuntimeError("missing microdepth diagnostics")
        typed = [record for record in records if record is not None]
        return {
            key + "_layer_median": float(np.median([record[key] for record in typed]))
            for key in (
                "base_rms",
                "activation_rms",
                "activation_abs_max",
                "coupling_rms",
                "bias_rms",
            )
        }

    # Match ordinary controls at the actual down-projection activation site.
    captured: list[torch.Tensor] = []
    original = model.model.layers[0].mlp
    handle = original.down_proj.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    handle.remove()
    activation = captured[0]
    return {
        "activation_rms_layer_median": float(
            activation.float().square().mean().sqrt()
        ),
        "activation_abs_max_layer_median": float(activation.float().abs().max()),
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
        "seed": 277,
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
        "initial": activation_diagnostics(model, modules, validation_file, device)
    }
    evaluations = {
        "0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
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
            inputs, targets = train_file.batch(batch_index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation
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
    diagnostics["terminal"] = activation_diagnostics(
        model, modules, validation_file, device
    )
    ablations: dict[str, Any] = {}
    if arm == "triangular":
        saved_couplings = [
            module.packed_coupling.detach().clone() for module in modules
        ]
        with torch.no_grad():
            for module in modules:
                module.packed_coupling.zero_()
        ablations["h_zero"] = evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
        with torch.no_grad():
            for module, saved in zip(modules, saved_couplings, strict=True):
                module.packed_coupling.copy_(saved)
                module.ablation_mode = "one_hop"
        ablations["one_hop"] = evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
        for module in modules:
            module.ablation_mode = "full"
    prediction_tokens = (
        args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    )
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "evaluations": evaluations,
        "activation_diagnostics": diagnostics,
        "terminal_ablations": ablations,
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
        return {"complete": False, "missing_arms": sorted(set(VALID_ARMS) - set(results))}
    terminal = "1525"
    early = "305"
    losses = {
        name: result["evaluations"][terminal]["loss"] for name, result in results.items()
    }
    early_losses = {
        name: result["evaluations"][early]["loss"] for name, result in results.items()
    }
    triangular_vs_baseline = paired_loss_interval(
        results["triangular"], results["baseline"], terminal
    )
    triangular_vs_control = paired_loss_interval(
        results["triangular"], results["linear_reparam"], terminal
    )
    triangular_vs_plain = paired_loss_interval(
        results["triangular"], results["plain_silu"], terminal
    )
    triangular_result = results["triangular"]
    h_zero_evaluation = triangular_result["terminal_ablations"]["h_zero"]
    one_hop_evaluation = triangular_result["terminal_ablations"]["one_hop"]
    full_evaluation = triangular_result["evaluations"][terminal]
    full_wrapper = {"evaluations": {terminal: full_evaluation}}
    h_zero_wrapper = {"evaluations": {terminal: h_zero_evaluation}}
    one_hop_wrapper = {"evaluations": {terminal: one_hop_evaluation}}
    full_vs_h_zero = paired_loss_interval(full_wrapper, h_zero_wrapper, terminal)
    full_vs_one_hop = paired_loss_interval(full_wrapper, one_hop_wrapper, terminal)
    baseline_rms = results["baseline"]["activation_diagnostics"]["initial"][
        "activation_rms_layer_median"
    ]
    rms_ratios = {
        name: result["activation_diagnostics"]["initial"][
            "activation_rms_layer_median"
        ]
        / baseline_rms
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
            0.75 <= ratio <= 1.25 for ratio in rms_ratios.values()
        ),
        "all_training_finite": all(
            not result["train"]["nonfinite"] for result in results.values()
        ),
        "bounded_training": all(
            result["train"]["max_preclip_gradient_norm"] <= 100.0
            and result["train"]["max_loss"] <= 20.0
            for result in results.values()
        ),
        "triangular_not_worse_at_10m_tokens": all(
            early_losses["triangular"] <= early_losses[control]
            for control in ("baseline", "plain_silu", "linear_reparam")
        ),
        "triangular_at_least_0p1_percent_better_than_baseline": (
            losses["baseline"] - losses["triangular"]
        )
        / losses["baseline"]
        >= 0.001,
        "triangular_at_least_0p1_percent_better_than_linear_control": (
            losses["linear_reparam"] - losses["triangular"]
        )
        / losses["linear_reparam"]
        >= 0.001,
        "triangular_at_least_0p1_percent_better_than_plain_silu": (
            losses["plain_silu"] - losses["triangular"]
        )
        / losses["plain_silu"]
        >= 0.001,
        "paired_interval_favors_triangular_vs_baseline": triangular_vs_baseline[
            "upper_95"
        ]
        < 0.0,
        "paired_interval_favors_triangular_vs_linear_control": triangular_vs_control[
            "upper_95"
        ]
        < 0.0,
        "paired_interval_favors_triangular_vs_plain_silu": triangular_vs_plain[
            "upper_95"
        ]
        < 0.0,
        "zeroing_h_hurts_by_at_least_0p05_percent": (
            h_zero_evaluation["loss"] - full_evaluation["loss"]
        )
        / full_evaluation["loss"]
        >= 0.0005,
        "paired_interval_favors_full_vs_h_zero": full_vs_h_zero["upper_95"] < 0.0,
        "removing_multihop_hurts_by_at_least_0p02_percent": (
            one_hop_evaluation["loss"] - full_evaluation["loss"]
        )
        / full_evaluation["loss"]
        >= 0.0002,
        "paired_interval_favors_full_vs_one_hop": full_vs_one_hop["upper_95"] < 0.0,
    }
    return {
        "complete": True,
        "terminal_step": 1525,
        "terminal_losses": losses,
        "early_losses": early_losses,
        "triangular_relative_improvement_vs_baseline": (
            losses["baseline"] - losses["triangular"]
        )
        / losses["baseline"],
        "triangular_relative_improvement_vs_linear_control": (
            losses["linear_reparam"] - losses["triangular"]
        )
        / losses["linear_reparam"],
        "paired_terminal_intervals": {
            "triangular_vs_baseline": triangular_vs_baseline,
            "triangular_vs_plain_silu": triangular_vs_plain,
            "triangular_vs_linear_control": triangular_vs_control,
            "full_vs_h_zero": full_vs_h_zero,
            "full_vs_one_hop": full_vs_one_hop,
        },
        "terminal_ablation_losses": {
            "full": full_evaluation["loss"],
            "h_zero": h_zero_evaluation["loss"],
            "one_hop": one_hop_evaluation["loss"],
        },
        "initial_activation_rms_ratios_to_baseline": rms_ratios,
        "gates": gates,
        "advance_to_second_seed": all(gates.values()),
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
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "test": TEST_SOURCE,
        "algebra_source": ALGEBRA_SOURCE,
        "algebra_test": ALGEBRA_TEST,
        "algebra_result": ALGEBRA_RESULT,
        "fused_source": FUSED_SOURCE,
        "fused_cpp_source": FUSED_CPP_SOURCE,
        "fused_cuda_source": FUSED_CUDA_SOURCE,
        "fused_result": FUSED_RESULT,
    }
    integrity_checks = {
        name: integrity_manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(integrity_checks.values()):
        raise ValueError(f"invalid frozen integrity manifest: {integrity_checks}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "candidate": "block-triangular-microdepth-ffn",
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
        "candidate_shape": {
            "width": CANDIDATE_WIDTH,
            "group_size": GROUP_SIZE,
            "groups": GROUPS,
            "coupling_parameters_per_layer": COUPLING_PARAMETERS,
            "partial_bias_parameters_per_layer": BIAS_COUNT,
            "activation_scale": ACTIVATION_SCALE,
        },
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
            data_ledger["valid"] and protocol["valid"] and all(integrity_checks.values()),
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
        default=Path("results/triangular-microdepth-lm-screen.json"),
    )
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
    parser.add_argument("--seed", type=int, default=277)
    parser.add_argument(
        "--strict-protocol", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
