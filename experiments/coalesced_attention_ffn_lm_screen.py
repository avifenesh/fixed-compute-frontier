#!/usr/bin/env python3
"""Matched 50M-token LM screen for projection-coalesced attention/FFN."""

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
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.llama.modeling_llama import LlamaRMSNorm, apply_rotary_pos_emb

from experiments.coalesced_attention_ffn_gate import resource_ledger
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
    ATTENTION_HEADS,
    BASELINE_INTERMEDIATE_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    KV_HEADS,
    causal_loss,
    scratch_config,
)


VALID_ARMS = (
    "sequential_baseline",
    "parallel_baseline",
    "coalesced_1024",
    "coalesced_1344",
)
LEDGER = resource_ledger(
    HIDDEN_SIZE,
    BASELINE_INTERMEDIATE_SIZE,
    ATTENTION_HEADS,
    KV_HEADS,
    HEAD_DIM,
)
CANDIDATE_WIDTH = LEDGER.candidate_width
PREREGISTRATION = Path("results/coalesced-attention-ffn-lm-preregistration.md")
STAGE0_PREREGISTRATION = Path("results/coalesced-attention-ffn-stage0-preregistration.md")
STAGE0_SOURCE = Path("experiments/coalesced_attention_ffn_gate.py")
STAGE0_TEST = Path("tests/test_coalesced_attention_ffn_gate.py")
STAGE0_RESULT = Path("results/coalesced-attention-ffn-stage0.json")
TEST_SOURCE = Path("tests/test_coalesced_attention_ffn_lm_screen.py")
INTEGRITY_MANIFEST = Path("results/coalesced-attention-ffn-lm-integrity-manifest.json")


class ParallelDecoderLayer(nn.Module):
    """Independent baseline branches sharing one pre-norm and residual."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.input_layernorm = source.input_layernorm
        self.self_attn = source.self_attn
        self.mlp = source.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del position_ids, use_cache
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention, _ = self.self_attn(
            hidden_states=normalized,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return residual + attention + self.mlp(normalized)


class CoalescedDecoderLayer(nn.Module):
    """Q/K/V and SwiGLU share two banks; O and down share one matrix."""

    def __init__(self, config: transformers.PretrainedConfig, layer_idx: int, width: int) -> None:
        super().__init__()
        if width < HIDDEN_SIZE + 2 * LEDGER.kv_dim:
            raise ValueError("coalesced width is too small for frozen slices")
        self.config = config
        self.layer_idx = layer_idx
        self.width = width
        self.head_dim = HEAD_DIM
        self.num_key_value_groups = ATTENTION_HEADS // KV_HEADS
        self.scaling = HEAD_DIM**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = True
        self.input_layernorm = LlamaRMSNorm(HIDDEN_SIZE, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(HIDDEN_SIZE, width, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, width, bias=False)
        self.down_proj = nn.Linear(width, HIDDEN_SIZE, bias=False)
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=config.initializer_range)
        self.zero_attention_local_products = False
        self.record_diagnostics = False
        self.diagnostic_reference_rms: float | None = None
        self.last_diagnostics: dict[str, float] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del position_ids, use_cache
        if position_embeddings is None:
            raise ValueError("RoPE position embeddings are required")
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        gate = self.gate_proj(normalized)
        up = self.up_proj(normalized)
        batch, tokens, _ = gate.shape
        query = gate[..., :HIDDEN_SIZE].view(
            batch, tokens, ATTENTION_HEADS, HEAD_DIM
        ).transpose(1, 2)
        key_start = HIDDEN_SIZE
        value_start = key_start + LEDGER.kv_dim
        key = up[..., key_start:value_start].view(
            batch, tokens, KV_HEADS, HEAD_DIM
        ).transpose(1, 2)
        value = up[..., value_start:value_start + LEDGER.kv_dim].view(
            batch, tokens, KV_HEADS, HEAD_DIM
        ).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )
        attention, _ = sdpa_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attention = attention.reshape(batch, tokens, HIDDEN_SIZE)
        local = F.silu(gate) * up
        if self.zero_attention_local_products:
            local = torch.cat(
                (
                    torch.zeros_like(local[..., : HIDDEN_SIZE + 2 * LEDGER.kv_dim]),
                    local[..., HIDDEN_SIZE + 2 * LEDGER.kv_dim :],
                ),
                dim=-1,
            )
        joint = local.clone()
        joint[..., :HIDDEN_SIZE] = joint[..., :HIDDEN_SIZE] + attention
        if self.record_diagnostics:
            with torch.no_grad():
                float_local = local.float()
                float_joint = joint.float()
                shared = float_local[..., : HIDDEN_SIZE + 2 * LEDGER.kv_dim]
                finite = torch.isfinite(float_joint)
                absolute = float_joint.abs()
                self.last_diagnostics = {
                    "attention_rms": float(attention.float().square().mean().sqrt()),
                    "local_rms": float(float_local.square().mean().sqrt()),
                    "joint_rms": float(float_joint.square().mean().sqrt()),
                    "joint_abs_max": float(float_joint.abs().max()),
                    "joint_abs_p99": float(torch.quantile(absolute.flatten(), 0.99)),
                    "joint_nonfinite_fraction": float((~finite).float().mean()),
                    "shared_local_rms": float(shared.square().mean().sqrt()),
                    "independent_local_rms": float(
                        float_local[..., HIDDEN_SIZE + 2 * LEDGER.kv_dim :]
                        .square().mean().sqrt()
                    ),
                }
                if self.diagnostic_reference_rms is not None:
                    self.last_diagnostics["joint_outside_4x_reference_rms_fraction"] = float(
                        (absolute > 4.0 * self.diagnostic_reference_rms).float().mean()
                    )
        return residual + self.down_proj(joint)


def build_model(
    device: torch.device, arm: str
) -> tuple[nn.Module, list[CoalescedDecoderLayer]]:
    if arm not in VALID_ARMS:
        raise ValueError(f"invalid arm: {arm}")
    config = scratch_config()
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules: list[CoalescedDecoderLayer] = []
    if arm == "sequential_baseline":
        return model, modules
    if arm == "parallel_baseline":
        for index, source in enumerate(model.model.layers):
            model.model.layers[index] = ParallelDecoderLayer(source).to(device)
        return model, modules
    width = BASELINE_INTERMEDIATE_SIZE if arm == "coalesced_1024" else CANDIDATE_WIDTH
    for index in range(len(model.model.layers)):
        replacement = CoalescedDecoderLayer(config, index, width).to(device)
        model.model.layers[index] = replacement
        modules.append(replacement)
    return model, modules


def dense_projection_parameters(arm: str) -> int:
    if arm in {"sequential_baseline", "parallel_baseline"}:
        return LEDGER.baseline_dense_parameters
    width = BASELINE_INTERMEDIATE_SIZE if arm == "coalesced_1024" else CANDIDATE_WIDTH
    return 3 * HIDDEN_SIZE * width


@torch.no_grad()
def activation_diagnostics(
    model: nn.Module,
    modules: list[CoalescedDecoderLayer],
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
            raise RuntimeError("missing coalesced diagnostics")
        typed = [record for record in records if record is not None]
        summary = {
            key + "_layer_median": float(np.median([record[key] for record in typed]))
            for key in typed[0]
        }
        summary["joint_abs_max_layer_max"] = max(record["joint_abs_max"] for record in typed)
        summary["joint_abs_p99_layer_max"] = max(record["joint_abs_p99"] for record in typed)
        summary["joint_nonfinite_fraction_layer_max"] = max(
            record["joint_nonfinite_fraction"] for record in typed
        )
        if reference_rms is not None:
            summary["joint_outside_4x_reference_rms_fraction_layer_max"] = max(
                record["joint_outside_4x_reference_rms_fraction"] for record in typed
            )
        return summary

    captured_attention: list[list[torch.Tensor]] = [[] for _ in model.model.layers]
    captured_local: list[list[torch.Tensor]] = [[] for _ in model.model.layers]
    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda _module, args, index=index: captured_attention[index].append(args[0].detach())
        ))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda _module, args, index=index: captured_local[index].append(args[0].detach())
        ))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    for handle in handles:
        handle.remove()
    if any(len(values) != 1 for values in captured_attention + captured_local):
        raise RuntimeError("missing baseline activation diagnostic")
    records = []
    for attention_values, local_values in zip(captured_attention, captured_local, strict=True):
        joint = torch.cat((attention_values[0], local_values[0]), dim=-1).float()
        absolute = joint.abs()
        record = {
            "joint_rms": float(joint.square().mean().sqrt()),
            "joint_abs_max": float(absolute.max()),
            "joint_abs_p99": float(torch.quantile(absolute.flatten(), 0.99)),
            "joint_nonfinite_fraction": float((~torch.isfinite(joint)).float().mean()),
        }
        if reference_rms is not None:
            record["joint_outside_4x_reference_rms_fraction"] = float(
                (absolute > 4.0 * reference_rms).float().mean()
            )
        records.append(record)
    summary = {
        key + "_layer_median": float(np.median([record[key] for record in records]))
        for key in records[0]
    }
    summary["joint_abs_max_layer_max"] = max(record["joint_abs_max"] for record in records)
    summary["joint_abs_p99_layer_max"] = max(record["joint_abs_p99"] for record in records)
    summary["joint_nonfinite_fraction_layer_max"] = max(
        record["joint_nonfinite_fraction"] for record in records
    )
    if reference_rms is not None:
        summary["joint_outside_4x_reference_rms_fraction_layer_max"] = max(
            record["joint_outside_4x_reference_rms_fraction"] for record in records
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
        "seed": 1601,
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
        raise ValueError(f"invalid experiment protocol: {json.dumps(failed, sort_keys=True)}")
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
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
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
            raise RuntimeError(f"non-finite training state in {arm} at step {step + 1}")
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
    if arm == "coalesced_1344":
        for module in modules:
            module.zero_attention_local_products = True
        ablations["zero_attention_local_products"] = evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
        for module in modules:
            module.zero_attention_local_products = False
    prediction_tokens = args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length
    result = {
        "arm": arm,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "dense_projection_parameters_per_layer": dense_projection_parameters(arm),
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
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(results: dict[str, Any], integrity_valid: bool) -> dict[str, Any]:
    if set(results) != set(VALID_ARMS):
        return {"complete": False, "missing_arms": sorted(set(VALID_ARMS) - set(results))}
    early, terminal = "305", "1525"
    losses = {name: result["evaluations"][terminal]["loss"] for name, result in results.items()}
    early_losses = {name: result["evaluations"][early]["loss"] for name, result in results.items()}
    candidate = results["coalesced_1344"]
    comparisons = {
        control: paired_loss_interval(candidate, results[control], terminal)
        for control in ("sequential_baseline", "parallel_baseline", "coalesced_1024")
    }
    full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
    ablated_eval = candidate["terminal_ablations"]["zero_attention_local_products"]
    ablated = {"evaluations": {terminal: ablated_eval}}
    ablation_interval = paired_loss_interval(full, ablated, terminal)
    baseline_rms = results["sequential_baseline"]["activation_diagnostics"]["initial"]["joint_rms_layer_median"]
    candidate_initial = candidate["activation_diagnostics"]["initial"]
    all_diagnostic_values = [
        value
        for result in results.values()
        for phase in ("initial", "terminal")
        for value in result["activation_diagnostics"][phase].values()
    ]
    gates = {
        "integrity_protocol_valid": integrity_valid,
        "candidate_dense_projection_parameters_lower": candidate["dense_projection_parameters_per_layer"] < results["sequential_baseline"]["dense_projection_parameters_per_layer"],
        "all_training_finite": all(not result["train"]["nonfinite"] for result in results.values()),
        "all_activation_diagnostics_finite": all(math.isfinite(value) for value in all_diagnostic_values),
        "all_activation_nonfinite_fractions_zero": all(
            result["activation_diagnostics"][phase]["joint_nonfinite_fraction_layer_max"] == 0.0
            for result in results.values()
            for phase in ("initial", "terminal")
        ),
        "bounded_training": all(result["train"]["max_preclip_gradient_norm"] <= 100.0 and result["train"]["max_loss"] <= 20.0 for result in results.values()),
        "candidate_initial_joint_rms_within_2x_baseline": candidate_initial["joint_rms_layer_median"] <= 2.0 * baseline_rms,
        "candidate_activation_outside_4x_baseline_rms_at_most_1_percent": all(
            candidate["activation_diagnostics"][phase]["joint_outside_4x_reference_rms_fraction_layer_max"] <= 0.01
            for phase in ("initial", "terminal")
        ),
        "candidate_terminal_0p05_percent_better_than_sequential": (losses["sequential_baseline"] - losses["coalesced_1344"]) / losses["sequential_baseline"] >= 0.0005,
        "paired_interval_favors_candidate_vs_sequential": comparisons["sequential_baseline"]["upper_95"] < 0.0,
        "candidate_terminal_0p025_percent_better_than_parallel": (losses["parallel_baseline"] - losses["coalesced_1344"]) / losses["parallel_baseline"] >= 0.00025,
        "candidate_terminal_0p025_percent_better_than_coalesced_1024": (losses["coalesced_1024"] - losses["coalesced_1344"]) / losses["coalesced_1024"] >= 0.00025,
        "candidate_early_noninferior_within_0p05_percent": (early_losses["coalesced_1344"] - early_losses["sequential_baseline"]) / early_losses["sequential_baseline"] <= 0.0005,
        "zeroing_attention_local_products_hurts_0p01_percent": (ablated_eval["loss"] - losses["coalesced_1344"]) / losses["coalesced_1344"] >= 0.0001,
        "paired_interval_favors_full_vs_ablation": ablation_interval["upper_95"] < 0.0,
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "early_losses": early_losses,
        "candidate_relative_improvements": {
            control: (losses[control] - losses["coalesced_1344"]) / losses[control]
            for control in ("sequential_baseline", "parallel_baseline", "coalesced_1024")
        },
        "paired_terminal_intervals": comparisons | {"full_vs_zero_attention_local_products": ablation_interval},
        "terminal_ablation_losses": {
            "full": losses["coalesced_1344"],
            "zero_attention_local_products": ablated_eval["loss"],
        },
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
    data_ledger = validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    protocol = validate_protocol(args, arms)
    required_train = args.steps * args.gradient_accumulation * args.micro_batch_size
    required_validation = args.eval_batches * args.eval_batch_size
    if train_file.sequence_count < required_train or validation_file.sequence_count < required_validation:
        raise ValueError("token files are too short for the frozen protocol")
    integrity_manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "stage0_preregistration": STAGE0_PREREGISTRATION,
        "stage0_source": STAGE0_SOURCE,
        "stage0_test": STAGE0_TEST,
        "stage0_result": STAGE0_RESULT,
        "test": TEST_SOURCE,
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
        "candidate": "projection-coalesced-attention-ffn",
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
        "ledger": LEDGER.__dict__,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in arms:
        print(json.dumps({"starting_arm": arm}), flush=True)
        reference_rms = None
        if "sequential_baseline" in payload["arms"]:
            baseline_diagnostics = payload["arms"]["sequential_baseline"]["activation_diagnostics"]
            reference_rms = {
                phase: baseline_diagnostics[phase]["joint_rms_layer_median"]
                for phase in ("initial", "terminal")
            }
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device, reference_rms
        )
        payload["decision"] = decide(payload["arms"], data_ledger["valid"] and protocol["valid"] and all(integrity_checks.values()))
        write_payload(args.output, payload)
    return payload


def parse_steps(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/coalesced-attention-ffn-lm-screen.json"))
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
    parser.add_argument("--seed", type=int, default=1601)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
