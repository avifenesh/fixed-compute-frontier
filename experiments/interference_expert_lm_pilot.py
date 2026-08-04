#!/usr/bin/env python3
"""Matched scratch-LM kill test for low-bit interference experts.

The candidate replaces each W8 gate/up projection by two W4 bases with one
shared scale.  A route bit derived from the already-computed primary gate
selects constructive or destructive interference.  The bases retain ordinary
long-K INT4 reductions; no expert dispatch or mid-reduction decoding is used.

This is a fake-quantized capability screen.  It does not claim a production
H100 latency result.
"""

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
from transformers import AutoModelForCausalLM

from experiments.bbcm_matched_learning_screen import (
    bf16_ste,
    fake_quant_symmetric,
    initial_symmetric_scale,
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
from experiments.triangular_microdepth_lm_screen import (
    BASELINE_INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
    causal_loss,
    scratch_config,
)


ARMS = (
    "raw_bf16",
    "w8a8",
    "w8a4",
    "w4a4_fixed_plus",
    "w4a4_interference",
)
OUTPUT = Path("results/interference-expert-lm-pilot-development.json")
GROUP_SIZE = 32
GROUPS = BASELINE_INTERMEDIATE_SIZE // GROUP_SIZE
ROUTE_TEMPERATURE = 0.25


def fake_quant_activation(value: torch.Tensor, maximum_code: int) -> torch.Tensor:
    maximum = value.detach().float().abs().amax(dim=-1, keepdim=True)
    scale = (maximum / maximum_code).clamp_min(torch.finfo(torch.bfloat16).tiny)
    scale = scale.to(torch.bfloat16).to(torch.float32)
    normalized = value / scale
    hard = torch.round(normalized).clamp(-maximum_code, maximum_code)
    codes = normalized + (hard - normalized).detach()
    return codes * scale


class QuantizedWeight(nn.Module):
    def __init__(self, target: torch.Tensor, maximum_code: int) -> None:
        super().__init__()
        target = target.detach().float()
        scale = initial_symmetric_scale(target, maximum_code)
        self.shadow = nn.Parameter(target.clone())
        self.log_scale = nn.Parameter(scale.log())
        self.maximum_code = maximum_code

    def weight(self) -> torch.Tensor:
        return fake_quant_symmetric(self.shadow, self.log_scale, self.maximum_code)

    @torch.no_grad()
    def code_diagnostics(self) -> dict[str, float]:
        scale = self.log_scale.exp().to(torch.bfloat16).float()
        codes = torch.round(self.shadow.detach().float() / scale).clamp(
            -self.maximum_code, self.maximum_code
        )
        decoded = codes * scale
        error = decoded - self.shadow.detach().float()
        return {
            "code_nonzero_fraction": float((codes != 0).float().mean()),
            "code_saturation_fraction": float(
                (codes.abs() == self.maximum_code).float().mean()
            ),
            "relative_weight_error": float(
                torch.linalg.vector_norm(error)
                / torch.linalg.vector_norm(self.shadow.detach().float()).clamp_min(1e-12)
            ),
        }


class SharedScaleInterferenceWeight(nn.Module):
    """Two signed-W4 bases and one BF16 scale per output row."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        target = target.detach().float()
        scale = initial_symmetric_scale(target, 7)
        self.primary_shadow = nn.Parameter(target.clone())
        # Exact zero forward starts both effective experts at the same W4 model.
        # STE gradients can then move the secondary codes across thresholds.
        self.secondary_shadow = nn.Parameter(torch.zeros_like(target))
        self.log_scale = nn.Parameter(scale.log())

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = bf16_ste(self.log_scale.exp())
        primary_normalized = self.primary_shadow / scale
        secondary_normalized = self.secondary_shadow / scale
        primary_hard = torch.round(primary_normalized).clamp(-7, 7)
        secondary_hard = torch.round(secondary_normalized).clamp(-7, 7)
        primary_codes = primary_normalized + (
            primary_hard - primary_normalized
        ).detach()
        secondary_codes = secondary_normalized + (
            secondary_hard - secondary_normalized
        ).detach()
        return primary_codes * scale, secondary_codes * scale

    @torch.no_grad()
    def code_diagnostics(self) -> dict[str, float]:
        scale = self.log_scale.exp().to(torch.bfloat16).float()
        primary = torch.round(self.primary_shadow.detach().float() / scale).clamp(-7, 7)
        secondary = torch.round(self.secondary_shadow.detach().float() / scale).clamp(-7, 7)
        primary_decoded = primary * scale
        secondary_decoded = secondary * scale
        return {
            "primary_code_nonzero_fraction": float((primary != 0).float().mean()),
            "secondary_code_nonzero_fraction": float((secondary != 0).float().mean()),
            "primary_saturation_fraction": float((primary.abs() == 7).float().mean()),
            "secondary_saturation_fraction": float((secondary.abs() == 7).float().mean()),
            "secondary_to_primary_rms": float(
                secondary_decoded.square().mean().sqrt()
                / primary_decoded.square().mean().sqrt().clamp_min(1e-12)
            ),
        }


class QuantizedSwiGLU(nn.Module):
    def __init__(self, source: nn.Module, input_maximum_code: int) -> None:
        super().__init__()
        self.gate = QuantizedWeight(source.gate_proj.weight, 127)
        self.up = QuantizedWeight(source.up_proj.weight, 127)
        self.down = QuantizedWeight(source.down_proj.weight, 127)
        self.input_maximum_code = input_maximum_code
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        quantized_input = fake_quant_activation(
            hidden_states, self.input_maximum_code
        )
        gate = F.linear(quantized_input, self.gate.weight())
        up = F.linear(quantized_input, self.up.weight())
        activation = F.silu(gate) * up
        quantized_activation = fake_quant_activation(activation, 127)
        output = F.linear(quantized_activation, self.down.weight())
        if self.record_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "activation_rms": float(activation.float().square().mean().sqrt()),
                    "input_relative_quantization_error": float(
                        torch.linalg.vector_norm(quantized_input.float() - hidden_states.float())
                        / torch.linalg.vector_norm(hidden_states.float()).clamp_min(1e-12)
                    ),
                    "activation_relative_quantization_error": float(
                        torch.linalg.vector_norm(quantized_activation.float() - activation.float())
                        / torch.linalg.vector_norm(activation.float()).clamp_min(1e-12)
                    ),
                    "route_positive_fraction": 1.0,
                    "route_entropy_bits": 0.0,
                    "secondary_code_nonzero_fraction": 0.0,
                    "secondary_to_primary_rms": 0.0,
                }
        return output


class InterferenceSwiGLU(nn.Module):
    def __init__(self, source: nn.Module, dynamic: bool) -> None:
        super().__init__()
        self.gate = SharedScaleInterferenceWeight(source.gate_proj.weight)
        self.up = SharedScaleInterferenceWeight(source.up_proj.weight)
        self.down = QuantizedWeight(source.down_proj.weight, 127)
        self.dynamic = dynamic
        self.routing_override = "full"
        self.record_diagnostics = False
        self.last_diagnostics: dict[str, float] | None = None

    def _route(self, primary_gate: torch.Tensor) -> torch.Tensor:
        logits = primary_gate.unflatten(-1, (GROUPS, GROUP_SIZE)).mean(dim=-1)
        soft = torch.tanh(logits / ROUTE_TEMPERATURE)
        hard = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
        route = hard.detach() - soft.detach() + soft if self.dynamic else torch.ones_like(logits)
        if self.routing_override == "fixed_plus":
            route = torch.ones_like(route)
        elif self.routing_override == "fixed_minus":
            route = -torch.ones_like(route)
        elif self.routing_override == "flip":
            route = -route
        elif self.routing_override == "zero_secondary" or self.routing_override == "full":
            pass
        else:
            raise ValueError(f"invalid routing override {self.routing_override}")
        return route.repeat_interleave(GROUP_SIZE, dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        quantized_input = fake_quant_activation(hidden_states, 7)
        gate_primary_weight, gate_secondary_weight = self.gate.weights()
        up_primary_weight, up_secondary_weight = self.up.weights()
        primary_gate = F.linear(quantized_input, gate_primary_weight)
        secondary_gate = F.linear(quantized_input, gate_secondary_weight)
        primary_up = F.linear(quantized_input, up_primary_weight)
        secondary_up = F.linear(quantized_input, up_secondary_weight)
        route = self._route(primary_gate)
        if self.routing_override == "zero_secondary":
            secondary_gate = torch.zeros_like(secondary_gate)
            secondary_up = torch.zeros_like(secondary_up)
        gate = primary_gate + route * secondary_gate
        up = primary_up + route * secondary_up
        activation = F.silu(gate) * up
        quantized_activation = fake_quant_activation(activation, 127)
        output = F.linear(quantized_activation, self.down.weight())
        if self.record_diagnostics:
            with torch.no_grad():
                hard_positive = route > 0
                probability = float(hard_positive.float().mean())
                entropy = 0.0
                if 0.0 < probability < 1.0:
                    entropy = -probability * math.log2(probability) - (
                        1.0 - probability
                    ) * math.log2(1.0 - probability)
                gate_codes = self.gate.code_diagnostics()
                up_codes = self.up.code_diagnostics()
                self.last_diagnostics = {
                    "activation_rms": float(activation.float().square().mean().sqrt()),
                    "input_relative_quantization_error": float(
                        torch.linalg.vector_norm(quantized_input.float() - hidden_states.float())
                        / torch.linalg.vector_norm(hidden_states.float()).clamp_min(1e-12)
                    ),
                    "activation_relative_quantization_error": float(
                        torch.linalg.vector_norm(quantized_activation.float() - activation.float())
                        / torch.linalg.vector_norm(activation.float()).clamp_min(1e-12)
                    ),
                    "route_positive_fraction": probability,
                    "route_entropy_bits": entropy,
                    "secondary_code_nonzero_fraction": 0.5
                    * (
                        gate_codes["secondary_code_nonzero_fraction"]
                        + up_codes["secondary_code_nonzero_fraction"]
                    ),
                    "secondary_to_primary_rms": 0.5
                    * (
                        gate_codes["secondary_to_primary_rms"]
                        + up_codes["secondary_to_primary_rms"]
                    ),
                }
        return output


def build_model(
    device: torch.device, arm: str
) -> tuple[nn.Module, list[nn.Module]]:
    if arm not in ARMS:
        raise ValueError(arm)
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    modules: list[nn.Module] = []
    if arm == "raw_bf16":
        return model, modules
    for layer in model.model.layers:
        if arm in {"w8a8", "w8a4"}:
            replacement = QuantizedSwiGLU(
                layer.mlp, input_maximum_code=127 if arm == "w8a8" else 7
            ).to(device)
        else:
            replacement = InterferenceSwiGLU(
                layer.mlp, dynamic=arm == "w4a4_interference"
            ).to(device)
        layer.mlp = replacement
        modules.append(replacement)
    return model, modules


@torch.no_grad()
def activation_diagnostics(
    model: nn.Module,
    modules: list[nn.Module],
    validation_file: TokenFile,
    device: torch.device,
) -> dict[str, float]:
    if not modules:
        return {
            "activation_rms_layer_median": float("nan"),
            "input_relative_quantization_error_layer_median": 0.0,
            "activation_relative_quantization_error_layer_median": 0.0,
            "route_positive_fraction_layer_mean": 1.0,
            "route_entropy_bits_layer_mean": 0.0,
            "secondary_code_nonzero_fraction_layer_mean": 0.0,
            "secondary_to_primary_rms_layer_mean": 0.0,
        }
    model.eval()
    for module in modules:
        module.record_diagnostics = True
    inputs, _ = validation_file.batch(0, 2, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=inputs, use_cache=False)
    records = [module.last_diagnostics for module in modules]
    for module in modules:
        module.record_diagnostics = False
    if any(record is None for record in records):
        raise RuntimeError("missing interference diagnostics")
    typed = [record for record in records if record is not None]
    return {
        "activation_rms_layer_median": float(np.median([r["activation_rms"] for r in typed])),
        "input_relative_quantization_error_layer_median": float(
            np.median([r["input_relative_quantization_error"] for r in typed])
        ),
        "activation_relative_quantization_error_layer_median": float(
            np.median([r["activation_relative_quantization_error"] for r in typed])
        ),
        "route_positive_fraction_layer_mean": float(
            np.mean([r["route_positive_fraction"] for r in typed])
        ),
        "route_entropy_bits_layer_mean": float(
            np.mean([r["route_entropy_bits"] for r in typed])
        ),
        "secondary_code_nonzero_fraction_layer_mean": float(
            np.mean([r["secondary_code_nonzero_fraction"] for r in typed])
        ),
        "secondary_to_primary_rms_layer_mean": float(
            np.mean([r["secondary_to_primary_rms"] for r in typed])
        ),
    }


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = args.learning_rate
    diagnostics = {"initial": activation_diagnostics(model, modules, validation_file, device)}
    evaluations = {
        "0": evaluate(model, validation_file, args.eval_batches, args.eval_batch_size, device)
    }
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    gradient_norms: list[float] = []
    durations: list[float] = []
    for step in range(args.steps):
        model.train()
        multiplier = lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            batch_index = step * args.gradient_accumulation + micro
            inputs, targets = train_file.batch(batch_index, args.micro_batch_size, device)
            loss = causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        )
        if not math.isfinite(accumulated) or not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite {arm} at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        gradient_norms.append(gradient_norm)
        durations.append(time.perf_counter() - started)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(json.dumps({
                "arm": arm,
                "step": step + 1,
                "validation_loss": evaluations[str(step + 1)]["loss"],
                "train_loss": accumulated,
                "gradient_norm": gradient_norm,
            }), flush=True)
    diagnostics["terminal"] = activation_diagnostics(
        model, modules, validation_file, device
    )
    ablations: dict[str, Any] = {}
    if arm == "w4a4_interference":
        typed_modules = [module for module in modules if isinstance(module, InterferenceSwiGLU)]
        for mode in ("fixed_plus", "fixed_minus", "flip", "zero_secondary"):
            for module in typed_modules:
                module.routing_override = mode
            ablations[mode] = evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
        for module in typed_modules:
            module.routing_override = "full"
    prediction_tokens = (
        args.steps
        * args.gradient_accumulation
        * args.micro_batch_size
        * args.sequence_length
    )
    result = {
        "arm": arm,
        "training_shadow_parameters": total_parameters,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "evaluations": evaluations,
        "activation_diagnostics": diagnostics,
        "terminal_ablations": ablations,
        "train": {
            "steps": args.steps,
            "prediction_tokens": prediction_tokens,
            "mean_loss": float(np.mean(losses)),
            "final_loss": losses[-1],
            "max_preclip_gradient_norm": max(gradient_norms),
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


def serving_ledger() -> dict[str, Any]:
    matrix_codes = 3 * HIDDEN_SIZE * BASELINE_INTERMEDIATE_SIZE
    scale_count = 2 * BASELINE_INTERMEDIATE_SIZE + HIDDEN_SIZE
    return {
        "w8_baseline": {
            "resident_code_bytes_per_layer": matrix_codes,
            "bf16_weight_scales_per_layer": scale_count,
            "ideal_gate_up_mma_k32_units": 2 * HIDDEN_SIZE * BASELINE_INTERMEDIATE_SIZE,
            "mathematical_products_per_layer": matrix_codes,
        },
        "interference": {
            "resident_code_bytes_per_layer": matrix_codes,
            "bf16_weight_scales_per_layer": scale_count,
            "ideal_gate_up_mma_k32_units": 2 * HIDDEN_SIZE * BASELINE_INTERMEDIATE_SIZE,
            "mathematical_products_per_layer": 5 * HIDDEN_SIZE * BASELINE_INTERMEDIATE_SIZE,
            "notes": "gate/up are four W4A4 products; down is one W8A8 product",
        },
        "exact_resident_bytes_equal": True,
        "ideal_tensor_instruction_units_equal": True,
        "protected_warning": "candidate has 5/3 as many mathematical scalar products and needs two gate/up accumulator banks",
    }


def decide(results: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if set(results) != set(ARMS):
        return {"complete": False, "missing_arms": sorted(set(ARMS) - set(results))}
    terminal = str(args.steps)
    losses = {arm: result["evaluations"][terminal]["loss"] for arm, result in results.items()}
    candidate = results["w4a4_interference"]
    intervals = {
        "candidate_vs_" + arm: paired_loss_interval(candidate, results[arm], terminal)
        for arm in ARMS[:-1]
    }
    diagnostics = candidate["activation_diagnostics"]["terminal"]
    ablations = candidate["terminal_ablations"]
    best_ablation_loss = min(result["loss"] for result in ablations.values())
    gates = {
        "finite_all_arms": all(not result["train"]["nonfinite"] for result in results.values()),
        "exact_served_weight_bytes_equal": serving_ledger()["exact_resident_bytes_equal"],
        "ideal_tensor_instruction_units_equal": serving_ledger()["ideal_tensor_instruction_units_equal"],
        "candidate_beats_w8a8": losses["w4a4_interference"] < losses["w8a8"],
        "candidate_beats_w8a4": losses["w4a4_interference"] < losses["w8a4"],
        "candidate_beats_fixed_plus": losses["w4a4_interference"] < losses["w4a4_fixed_plus"],
        "paired_interval_favors_candidate_vs_w8a8": intervals["candidate_vs_w8a8"]["upper_95"] < 0.0,
        "paired_interval_favors_candidate_vs_w8a4": intervals["candidate_vs_w8a4"]["upper_95"] < 0.0,
        "route_entropy_at_least_0p8_bits": diagnostics["route_entropy_bits_layer_mean"] >= 0.8,
        "secondary_codes_live": diagnostics["secondary_code_nonzero_fraction_layer_mean"] >= 0.01,
        "learned_route_beats_all_route_ablations": losses["w4a4_interference"] < best_ablation_loss,
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "relative_improvements_vs_w8a8": {
            arm: (losses["w8a8"] - loss) / losses["w8a8"]
            for arm, loss in losses.items()
        },
        "paired_intervals": intervals,
        "candidate_terminal_diagnostics": diagnostics,
        "candidate_terminal_ablation_losses": {
            mode: result["loss"] for mode, result in ablations.items()
        },
        "gates": gates,
        "advance_to_h100_kernel_and_longer_learning": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps not in args.eval_steps:
        raise ValueError("terminal step must be evaluated")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": "interference-expert-lm-pilot-development-v1",
        "candidate": "two W4 bases with activation-derived sum/difference route",
        "scope": "one-seed scratch-LM kill test with fake quantization",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "source_sha256": sha256_file(Path(__file__)),
        "data_ledger": data_ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "serving_ledger": serving_ledger(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(payload["arms"], args)
        write_payload(args.output, payload)
    return payload


def parse_steps(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=61)
    parser.add_argument("--eval-steps", type=parse_steps, default=[61])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2833)
    args = parser.parse_args()
    print(json.dumps(run(args)["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
