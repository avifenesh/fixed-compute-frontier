#!/usr/bin/env python3
"""Quality gate for a W8 prefill base plus decode-only W4 residual FFN."""

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

from experiments.bbcm_matched_learning_screen import (
    exact_symmetric_decode, fake_quant_symmetric, initial_symmetric_scale,
)
from experiments.coalesced_attention_ffn_lm_screen import build_model as build_parallel_model
from experiments.cycle_factor_ffn_lm_screen import D, M, SEED
from experiments.reflex_swiglu_lm_screen import (
    TokenFile, evaluate, lr_multiplier, paired_loss_interval, sha256_file,
    validate_data_ledger, write_payload,
)
from experiments.triangular_microdepth_lm_screen import causal_loss


BRANCH_WIDTH = 1984
PREDECESSOR = Path("results/cycle-factor-ffn-lm-screen.json")
PREDECESSOR_SHA256 = "c66dfe6f5a6e3a1c4032932b9cbd2c491812af2e86a314e9cc6d46e83aba4029"
PREREGISTRATION = Path("results/phase-elastic-precision-ffn-preregistration.md")


def serving_ledger() -> dict[str, int | float | bool]:
    baseline_coordinates = 3 * D * M
    branch_coordinates = 3 * D * BRANCH_WIDTH
    baseline_bytes = 2 * baseline_coordinates
    base_code_bytes = baseline_coordinates
    base_scale_bytes = 2 * (2 * M + D)
    branch_code_bytes = branch_coordinates // 2
    branch_scale_bytes = 2 * (2 * BRANCH_WIDTH + D)
    candidate_bytes = base_code_bytes + base_scale_bytes + branch_code_bytes + branch_scale_bytes
    return {
        "dimension": D,
        "base_width": M,
        "decode_residual_width": BRANCH_WIDTH,
        "residual_width_ratio": BRANCH_WIDTH / M,
        "baseline_bf16_ffn_bytes_per_layer": baseline_bytes,
        "base_w8_code_bytes_per_layer": base_code_bytes,
        "base_bf16_scale_bytes_per_layer": base_scale_bytes,
        "residual_w4_code_bytes_per_layer": branch_code_bytes,
        "residual_bf16_scale_bytes_per_layer": branch_scale_bytes,
        "candidate_ffn_bytes_per_layer": candidate_bytes,
        "storage_slack_bytes_per_layer": baseline_bytes - candidate_bytes,
        "prefill_base_matrix_macs_per_token": baseline_coordinates,
        "decode_full_matrix_macs_per_token": baseline_coordinates + branch_coordinates,
        "decode_to_baseline_mac_ratio": (baseline_coordinates + branch_coordinates) / baseline_coordinates,
        "fits_bf16_bytes": candidate_bytes <= baseline_bytes,
    }


class QuantizedWeight(nn.Module):
    def __init__(self, target: torch.Tensor, maximum_code: int) -> None:
        super().__init__()
        target = target.detach().float()
        scale = initial_symmetric_scale(target, maximum_code)
        self.shadow = nn.Parameter(target.clone())
        self.log_scale = nn.Parameter(scale.log())
        self.maximum_code = maximum_code

    def forward(self) -> torch.Tensor:
        return fake_quant_symmetric(self.shadow, self.log_scale, self.maximum_code)

    @torch.no_grad()
    def exact(self) -> torch.Tensor:
        return exact_symmetric_decode(self.shadow, self.log_scale, self.maximum_code)

    @torch.no_grad()
    def audit(self) -> dict[str, Any]:
        scale = self.log_scale.exp().to(torch.bfloat16).to(torch.float32)
        normalized = self.shadow.detach().float() / scale
        codes = torch.round(normalized).clamp(-self.maximum_code, self.maximum_code)
        return {
            "code_min": int(codes.min()),
            "code_max": int(codes.max()),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
            "all_finite": bool(torch.isfinite(codes).all() and torch.isfinite(scale).all()),
            "output_rows": int(codes.shape[0]),
            "coordinates": codes.numel(),
        }


class PhaseElasticMLP(nn.Module):
    def __init__(self, original: nn.Module, layer_index: int, std: float) -> None:
        super().__init__()
        self.base_gate = QuantizedWeight(original.gate_proj.weight, 127)
        self.base_up = QuantizedWeight(original.up_proj.weight, 127)
        self.base_down = QuantizedWeight(original.down_proj.weight, 127)
        gate = torch.empty(BRANCH_WIDTH, D, device=original.gate_proj.weight.device)
        up = torch.empty_like(gate)
        down = torch.empty(D, BRANCH_WIDTH, device=gate.device)
        nn.init.normal_(gate, mean=0.0, std=std)
        nn.init.normal_(up, mean=0.0, std=std)
        nn.init.normal_(down, mean=0.0, std=std * 0.05 * math.sqrt(M / BRANCH_WIDTH))
        self.branch_gate = QuantizedWeight(gate, 7)
        self.branch_up = QuantizedWeight(up, 7)
        self.branch_down = QuantizedWeight(down, 7)
        self.layer_index = layer_index
        self.mode = "train"
        self.last_branch_active = False
        self.last_base_rms = 0.0
        self.last_branch_rms = 0.0

    def set_mode(self, mode: str) -> None:
        if mode not in {"train", "base", "full"}:
            raise ValueError(mode)
        self.mode = mode

    def _branch_is_active(self) -> bool:
        if self.mode == "base":
            return False
        if self.mode == "full":
            return True
        if not self.training:
            raise RuntimeError("evaluation requires explicit base or full mode")
        return bool(torch.rand((), device=self.base_gate.shadow.device) < 0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_activation = F.silu(F.linear(hidden_states, self.base_gate())) * F.linear(
            hidden_states, self.base_up()
        )
        base = F.linear(base_activation, self.base_down())
        active = self._branch_is_active()
        self.last_branch_active = active
        if not active:
            return base
        branch_activation = F.silu(F.linear(hidden_states, self.branch_gate())) * F.linear(
            hidden_states, self.branch_up()
        )
        branch = F.linear(branch_activation, self.branch_down())
        return base + branch

    @torch.no_grad()
    def audit(self) -> dict[str, Any]:
        return {
            name: module.audit()
            for name, module in (
                ("base_gate", self.base_gate), ("base_up", self.base_up),
                ("base_down", self.base_down), ("branch_gate", self.branch_gate),
                ("branch_up", self.branch_up), ("branch_down", self.branch_down),
            )
        }


def set_mode(modules: list[PhaseElasticMLP], mode: str) -> None:
    for module in modules:
        module.set_mode(mode)


def build_candidate(device: torch.device, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, _ = build_parallel_model(device, "parallel_baseline")
    modules = []
    for layer_index, layer in enumerate(model.model.layers):
        replacement = PhaseElasticMLP(
            layer.mlp, layer_index, model.config.initializer_range
        ).to(device)
        layer.mlp = replacement
        modules.append(replacement)
    model.config.use_cache = False
    return model, modules


@torch.no_grad()
def evaluate_mode(model, modules, mode, validation, batches, batch_size, device):
    set_mode(modules, mode)
    result = evaluate(model, validation, batches, batch_size, device)
    set_mode(modules, "train")
    return result


def train(args, train_file, validation, device):
    model, modules = build_candidate(device, args.seed)
    scale_parameters = [
        parameter for name, parameter in model.named_parameters()
        if "log_scale" in name
    ]
    scale_ids = {id(parameter) for parameter in scale_parameters}
    ordinary_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in scale_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": ordinary_parameters, "weight_decay": args.weight_decay},
            {"params": scale_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    evaluations = {
        "0": {
            "base": evaluate_mode(model, modules, "base", validation, args.eval_batches, args.eval_batch_size, device),
            "full": evaluate_mode(model, modules, "full", validation, args.eval_batches, args.eval_batch_size, device),
        }
    }
    optimizer.zero_grad(set_to_none=True)
    step_losses, step_seconds, gradient_norms = [], [], []
    branch_active = np.zeros(len(modules), dtype=np.int64)
    for step in range(args.steps):
        model.train()
        set_mode(modules, "train")
        scheduled_lr = args.learning_rate * lr_multiplier(
            step, args.steps, args.warmup_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            inputs, targets = train_file.batch(
                step * args.gradient_accumulation + micro,
                args.micro_batch_size,
                device,
            )
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
            branch_active += np.asarray([module.last_branch_active for module in modules], dtype=np.int64)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
            raise RuntimeError(f"nonfinite step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_losses.append(accumulated)
        gradient_norms.append(gradient_norm)
        step_seconds.append(time.perf_counter() - started)
    evaluations[str(args.steps)] = {
        "base": evaluate_mode(model, modules, "base", validation, args.eval_batches, args.eval_batch_size, device),
        "full": evaluate_mode(model, modules, "full", validation, args.eval_batches, args.eval_batch_size, device),
    }
    audits = [module.audit() for module in modules]
    result = {
        "training_shadow_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "evaluations": evaluations,
        "branch_training_active_fraction_per_layer": (
            branch_active / (args.steps * args.gradient_accumulation)
        ).tolist(),
        "quantized_weight_audits": audits,
        "train": {
            "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
            "mean_loss": float(np.mean(step_losses)),
            "final_loss": step_losses[-1],
            "max_loss": max(step_losses),
            "max_preclip_gradient_norm": max(gradient_norms),
            "elapsed_seconds": sum(step_seconds),
            "tokens_per_second": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length / sum(step_seconds),
            "nonfinite": False,
        },
    }
    del optimizer, modules, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def decide(candidate, predecessor, protocol_valid):
    step = "305"
    dense = predecessor["arms"]["full_swiglu"]
    dense_loss = dense["evaluations"][step]["loss"]
    full = candidate["evaluations"][step]["full"]
    base = candidate["evaluations"][step]["base"]
    full_wrapper = {"evaluations": {step: full}}
    base_wrapper = {"evaluations": {step: base}}
    full_dense_interval = paired_loss_interval(full_wrapper, dense, step)
    base_dense_interval = paired_loss_interval(base_wrapper, dense, step)
    full_base_interval = paired_loss_interval(full_wrapper, base_wrapper, step)
    full_gain = (dense_loss - full["loss"]) / dense_loss
    base_regret = (base["loss"] - dense_loss) / dense_loss
    branch_gain = (base["loss"] - full["loss"]) / base["loss"]
    audits = [entry for layer in candidate["quantized_weight_audits"] for entry in layer.values()]
    ledger = serving_ledger()
    gates = {
        "protocol_integrity_valid": protocol_valid,
        "exact_storage_fits": bool(ledger["fits_bf16_bytes"]),
        "all_codes_scales_finite_and_bounded": all(
            audit["all_finite"] and audit["code_min"] >= -127 and audit["code_max"] <= 127
            for audit in audits
        ),
        "training_finite": not candidate["train"]["nonfinite"],
        "dropout_exercised_both_paths": min(candidate["branch_training_active_fraction_per_layer"]) >= 0.4 and max(candidate["branch_training_active_fraction_per_layer"]) <= 0.6,
        "full_beats_dense_by_0p1_percent": full_gain >= 0.001 and full_dense_interval["upper_95"] < 0,
        "base_noninferior_dense_within_0p05_percent": base_regret <= 0.0005 and base_dense_interval["upper_95"] <= 0.0005 * dense_loss,
        "decode_branch_improves_base_by_0p1_percent": branch_gain >= 0.001 and full_base_interval["upper_95"] < 0,
    }
    return {
        "losses": {"dense_bf16": dense_loss, "candidate_base_prefill": base["loss"], "candidate_full_decode": full["loss"]},
        "relative_full_gain_vs_dense": full_gain,
        "relative_base_regret_vs_dense": base_regret,
        "relative_decode_branch_gain_vs_base": branch_gain,
        "paired_intervals": {
            "full_minus_dense": full_dense_interval,
            "base_minus_dense": base_dense_interval,
            "full_minus_base": full_base_interval,
        },
        "gates": gates,
        "advance_to_physical_h100_gate": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase-elastic-precision-ffn-lm-screen.json"))
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
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    expected = (
        args.steps, args.sequence_length, args.micro_batch_size,
        args.gradient_accumulation, args.eval_batch_size, args.eval_batches,
        args.warmup_steps, args.learning_rate, args.weight_decay,
        args.gradient_clip, args.seed,
    ) == (305, 512, 32, 2, 32, 128, 30, 3e-4, 0.1, 1.0, SEED)
    expected = (
        expected and torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and transformers.__version__ == "4.57.6"
        and "H100" in torch.cuda.get_device_name()
    )
    if sha256_file(PREDECESSOR) != PREDECESSOR_SHA256:
        raise RuntimeError("frozen predecessor changed")
    predecessor = json.loads(PREDECESSOR.read_text())
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    protocol_valid = expected and predecessor["protocol_valid"] and data_ledger["valid"]
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    candidate = train(
        args, TokenFile(args.train_file, args.sequence_length),
        TokenFile(args.validation_file, args.sequence_length), torch.device("cuda")
    )
    decision = decide(candidate, predecessor, protocol_valid)
    payload = {
        "schema": "phase-elastic-precision-ffn-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "predecessor_sha256": sha256_file(PREDECESSOR),
        "protocol_valid": protocol_valid,
        "data_ledger": data_ledger,
        "serving_ledger": serving_ledger(),
        "candidate": candidate,
        "decision": decision,
    }
    write_payload(args.output, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
