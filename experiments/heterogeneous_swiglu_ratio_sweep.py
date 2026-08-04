#!/usr/bin/env python3
"""Development sweep for a static heterogeneous SwiGLU/even-gate FFN.

This isolates three explanations for the first mixed-activation win:
activation RMS, the fraction of cubic/even channels, and mere SwiGLU output
scaling.  All arms retain the same learned-scalar and dense-matmul ledgers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import experiments.gaugebit_swiglu_lm_pilot as base
from experiments.reflex_swiglu_lm_screen import (
    MODEL,
    MODEL_REVISION,
    TokenFile,
    paired_loss_interval,
    sha256_file,
    validate_data_ledger,
    write_payload,
)
from experiments.triangular_microdepth_lm_screen import scratch_config


ARMS = (
    "raw_baseline",
    "canonical_null",
    "silu_x1p2507",
    "mixed25",
    "mixed50",
    "mixed75",
    "mixed50_rms",
    "all_even_rms",
)
OUTPUT = Path("results/heterogeneous-swiglu-ratio-sweep-development.json")
# Gaussian-reference RMS(phi0)/RMS(phi1), measured with 10M deterministic samples.
EVEN_RMS_SCALE = 0.6854475404927584
# RMS of an unscaled 50/50 mixture relative to all-SiLU under the same reference.
SILU_AMPLITUDE_CONTROL = 1.250678154532455


ARM_SETTINGS: dict[str, tuple[float, float, float]] = {
    # arm: (fraction even, even-channel multiplier, global activation multiplier)
    "canonical_null": (0.0, 1.0, 1.0),
    "silu_x1p2507": (0.0, 1.0, SILU_AMPLITUDE_CONTROL),
    "mixed25": (0.25, 1.0, 1.0),
    "mixed50": (0.50, 1.0, 1.0),
    "mixed75": (0.75, 1.0, 1.0),
    "mixed50_rms": (0.50, EVEN_RMS_SCALE, 1.0),
    "all_even_rms": (1.0, EVEN_RMS_SCALE, 1.0),
}


class StaticHeterogeneousMLP(base.GaugeBitMLP):
    def __init__(self, source, arm: str) -> None:
        super().__init__(source, "canonical_null")
        if arm not in ARM_SETTINGS:
            raise ValueError(f"invalid static arm: {arm}")
        self.static_arm = arm
        ratio, self.even_scale, self.global_scale = ARM_SETTINGS[arm]
        count = round(ratio * base.GROUPS)
        bits = torch.zeros(base.GROUPS, dtype=torch.bool)
        bits[:count] = True
        self.register_buffer("static_bits", bits, persistent=False)

    def hard_bits(self) -> torch.Tensor:
        return self.static_bits

    def group_selectors(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.static_bits.to(dtype) + 0.0 * self.betas().to(dtype))

    def activation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = F.linear(hidden_states, self.effective_up_weight())
        selectors = self.group_selectors(gate.dtype).repeat_interleave(base.GROUP_SIZE)
        ordinary = F.silu(gate)
        even = self.even_scale * gate * torch.tanh(gate)
        activation = self.global_scale * torch.lerp(ordinary, even, selectors) * up
        if self.record_diagnostics:
            with torch.no_grad():
                betas = self.betas().float()
                self.last_diagnostics = {
                    "activation_rms": float(activation.float().square().mean().sqrt()),
                    "activation_abs_max": float(activation.float().abs().max()),
                    "beta_mean": float(betas.mean()),
                    "beta_abs_mean": float(betas.abs().mean()),
                    "beta_abs_min": float(betas.abs().min()),
                    "hard_even_fraction": float(self.static_bits.float().mean()),
                    "soft_even_fraction": float(self.static_bits.float().mean()),
                }
        return activation


def build_model(device: torch.device, arm: str):
    if arm not in ARMS:
        raise ValueError(f"invalid ratio-sweep arm: {arm}")
    model = AutoModelForCausalLM.from_config(
        scratch_config(), attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    base.initialize_chart_pivots(model)
    modules: list[StaticHeterogeneousMLP] = []
    if arm != "raw_baseline":
        for layer in model.model.layers:
            replacement = StaticHeterogeneousMLP(layer.mlp, arm).to(device)
            layer.mlp = replacement
            modules.append(replacement)
    return model, modules


def decide(results: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if set(results) != set(ARMS):
        return {"complete": False, "missing_arms": sorted(set(ARMS) - set(results))}
    terminal = str(args.steps)
    losses = {arm: result["evaluations"][terminal]["loss"] for arm, result in results.items()}
    best = min(ARMS, key=lambda arm: losses[arm])
    intervals = {
        arm + "_vs_raw": paired_loss_interval(results[arm], results["raw_baseline"], terminal)
        for arm in ARMS[1:]
    }
    initial_rms = {
        arm: result["activation_diagnostics"]["initial"]["activation_rms_layer_median"]
        for arm, result in results.items()
    }
    gates = {
        "equal_parameter_counts": len({r["total_parameters"] for r in results.values()}) == 1,
        "equal_parameter_tensor_counts": len({r["parameter_tensors"] for r in results.values()}) == 1,
        "equal_optimizer_state_bytes": len({r["optimizer_state_bytes"] for r in results.values()}) == 1,
        "canonical_null_initial_exact": abs(
            results["canonical_null"]["evaluations"]["0"]["loss"]
            - results["raw_baseline"]["evaluations"]["0"]["loss"]
        ) <= 1e-7,
        "best_is_heterogeneous_not_amplitude_control": best.startswith("mixed"),
        "best_improves_raw_0p10_percent": (
            losses["raw_baseline"] - losses[best]
        ) / losses["raw_baseline"] >= 0.001,
        "paired_interval_favors_best_vs_raw": (
            best != "raw_baseline" and intervals[best + "_vs_raw"]["upper_95"] < 0.0
        ),
        "rms_matched_mixture_improves_raw": losses["mixed50_rms"] < losses["raw_baseline"],
        "rms_matched_interval_favors_mixture": intervals["mixed50_rms_vs_raw"]["upper_95"] < 0.0,
    }
    return {
        "complete": True,
        "terminal_losses": losses,
        "relative_improvements_vs_raw": {
            arm: (losses["raw_baseline"] - loss) / losses["raw_baseline"]
            for arm, loss in losses.items()
        },
        "initial_activation_rms": initial_rms,
        "best_arm": best,
        "paired_intervals": intervals,
        "gates": gates,
        "advance_to_long_horizon": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps not in args.eval_steps:
        raise ValueError("terminal step must be included in eval steps")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 is required")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    # base.train_arm resolves build_model from its defining module.
    base.build_model = build_model
    payload: dict[str, Any] = {
        "schema": "heterogeneous-swiglu-ratio-sweep-development-v1",
        "candidate": "static mixed SiLU and cubic-even gated FFN",
        "scope": "one-seed 10M-token ratio/RMS falsification",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "source_sha256": sha256_file(Path(__file__)),
        "data_ledger": data_ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "activation_controls": {
            "even_rms_scale": EVEN_RMS_SCALE,
            "silu_amplitude_control": SILU_AMPLITUDE_CONTROL,
            "settings": ARM_SETTINGS,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        payload["arms"][arm] = base.train_arm(
            arm, args, train_file, validation_file, device
        )
        payload["decision"] = decide(payload["arms"], args)
        write_payload(args.output, payload)
    return payload


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
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--eval-steps", type=base.parse_steps, default=[61, 305])
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2729)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
