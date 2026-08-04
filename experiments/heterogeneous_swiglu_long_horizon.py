#!/usr/bin/env python3
"""Frozen five-seed 50M-token confirmation for static heterogeneous SwiGLU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

import experiments.gaugebit_swiglu_lm_pilot as training
import experiments.heterogeneous_swiglu_ratio_sweep as ratio
from experiments.reflex_swiglu_lm_screen import (
    MODEL,
    MODEL_REVISION,
    TokenFile,
    paired_loss_interval,
    sha256_file,
    validate_data_ledger,
    write_payload,
)


ARMS = ("raw_baseline", "canonical_null", "mixed50_rms")
SEEDS = (2741, 2753, 2767, 2789, 2801)
PREREGISTRATION = Path("results/heterogeneous-swiglu-long-horizon-preregistration.md")
INTEGRITY_MANIFEST = Path("results/heterogeneous-swiglu-long-horizon-integrity-manifest.json")
RATIO_SOURCE = Path("experiments/heterogeneous_swiglu_ratio_sweep.py")
ALGEBRA_SOURCE = Path("experiments/gaugebit_swiglu.py")
TEST_SOURCE = Path("tests/test_heterogeneous_swiglu_long_horizon.py")
TRAINING_SOURCE = Path("experiments/gaugebit_swiglu_lm_pilot.py")
REFLEX_SOURCE = Path("experiments/reflex_swiglu_lm_screen.py")
SCRATCH_SOURCE = Path("experiments/triangular_microdepth_lm_screen.py")
OUTPUT = Path("results/heterogeneous-swiglu-long-horizon.json")


def student_t_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 5:
        raise ValueError("frozen confirmation requires five seed values")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    half_width = 2.7764451051977987 * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - half_width,
        "upper_95": mean + half_width,
    }


def decide(seed_results: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    expected = {str(seed) for seed in SEEDS}
    if set(seed_results) != expected or any(
        set(result) != set(ARMS) for result in seed_results.values()
    ):
        return {
            "complete": False,
            "completed_seeds": sorted(seed_results),
            "expected_seeds": sorted(expected),
        }
    terminal = str(args.steps)
    early = "305"
    per_seed: dict[str, Any] = {}
    improvements: list[float] = []
    initial_exact = []
    initial_rms_matched = []
    terminal_wins = []
    early_wins = []
    paired_wins = []
    null_relative_changes = []
    for seed in SEEDS:
        key = str(seed)
        results = seed_results[key]
        losses = {
            arm: results[arm]["evaluations"][terminal]["loss"] for arm in ARMS
        }
        early_losses = {
            arm: results[arm]["evaluations"][early]["loss"] for arm in ARMS
        }
        control_arm = min(("raw_baseline", "canonical_null"), key=lambda arm: losses[arm])
        early_control = min(
            ("raw_baseline", "canonical_null"), key=lambda arm: early_losses[arm]
        )
        relative = (losses[control_arm] - losses["mixed50_rms"]) / losses[control_arm]
        interval = paired_loss_interval(
            results["mixed50_rms"], results[control_arm], terminal
        )
        exact = abs(
            results["raw_baseline"]["evaluations"]["0"]["loss"]
            - results["canonical_null"]["evaluations"]["0"]["loss"]
        ) <= 1e-7
        raw_rms = results["raw_baseline"]["activation_diagnostics"]["initial"][
            "activation_rms_layer_median"
        ]
        candidate_rms = results["mixed50_rms"]["activation_diagnostics"]["initial"][
            "activation_rms_layer_median"
        ]
        rms_matched = abs(candidate_rms / raw_rms - 1.0) <= 0.01
        improvement = relative
        improvements.append(improvement)
        initial_exact.append(exact)
        initial_rms_matched.append(rms_matched)
        terminal_wins.append(improvement > 0.0)
        early_wins.append(early_losses["mixed50_rms"] < early_losses[early_control])
        paired_wins.append(interval["upper_95"] < 0.0)
        null_relative_changes.append(
            (losses["canonical_null"] - losses["raw_baseline"]) / losses["raw_baseline"]
        )
        per_seed[key] = {
            "terminal_losses": losses,
            "early_losses": early_losses,
            "better_terminal_control": control_arm,
            "relative_improvement_vs_better_control": relative,
            "paired_candidate_vs_better_control": interval,
            "initial_exact": exact,
            "initial_activation_rms_ratio_to_raw": candidate_rms / raw_rms,
            "initial_activation_rms_matched": rms_matched,
            "terminal_win": terminal_wins[-1],
            "early_win": early_wins[-1],
            "paired_win": paired_wins[-1],
        }
    interval = student_t_interval(improvements)
    all_results = [arm for seed in seed_results.values() for arm in seed.values()]
    numeric_diagnostics = [
        value
        for result in all_results
        for phase in ("initial", "terminal")
        for value in result["activation_diagnostics"][phase].values()
    ]
    gates = {
        "integrity_valid": args.integrity_valid,
        "equal_parameter_counts": len({r["total_parameters"] for r in all_results}) == 1,
        "equal_parameter_tensor_counts": len({r["parameter_tensors"] for r in all_results}) == 1,
        "equal_optimizer_state_bytes": len({r["optimizer_state_bytes"] for r in all_results}) == 1,
        "all_raw_and_null_initial_losses_exact": all(initial_exact),
        "all_candidate_initial_activation_rms_matched_within_1_percent": all(initial_rms_matched),
        "all_training_and_diagnostics_finite": (
            all(not r["train"]["nonfinite"] for r in all_results)
            and all(math.isfinite(value) for value in numeric_diagnostics)
        ),
        "canonical_null_mean_noninferior_0p05_percent": float(np.mean(null_relative_changes)) <= 0.0005,
        "candidate_terminal_wins_at_least_4_of_5": sum(terminal_wins) >= 4,
        "candidate_mean_relative_improvement_at_least_0p10_percent": interval["mean"] >= 0.001,
        "seed_level_95_percent_lower_bound_above_zero": interval["lower_95"] > 0.0,
        "candidate_early_wins_at_least_4_of_5": sum(early_wins) >= 4,
        "paired_terminal_intervals_win_at_least_4_of_5": sum(paired_wins) >= 4,
    }
    return {
        "complete": True,
        "per_seed": per_seed,
        "seed_level_relative_improvement_interval": interval,
        "terminal_win_count": sum(terminal_wins),
        "early_win_count": sum(early_wins),
        "paired_terminal_win_count": sum(paired_wins),
        "canonical_null_mean_relative_change": float(np.mean(null_relative_changes)),
        "gates": gates,
        "confirmation_pass": all(gates.values()),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "seeds": list(SEEDS),
        "arms": list(ARMS),
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
        "device_contains": "H100",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
    }
    actual = {
        "seeds": list(args.seeds),
        "arms": list(ARMS),
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
        "device_contains": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
    }
    checks = {
        key: (value in actual[key] if key == "device_contains" else actual[key] == value)
        for key, value in expected.items()
    }
    return {"valid": all(checks.values()), "checks": checks, "expected": expected, "actual": actual}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 is required")
    protocol = validate_protocol(args)
    if not protocol["valid"]:
        raise ValueError(f"invalid frozen protocol: {protocol}")
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "ratio_source": RATIO_SOURCE,
        "algebra_source": ALGEBRA_SOURCE,
        "test": TEST_SOURCE,
        "training_source": TRAINING_SOURCE,
        "reflex_source": REFLEX_SOURCE,
        "scratch_source": SCRATCH_SOURCE,
        "data_manifest": args.data_manifest,
    }
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    integrity_checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(integrity_checks.values()):
        raise ValueError(f"integrity failure: {integrity_checks}")
    train_file = TokenFile(args.train_file, args.sequence_length)
    validation_file = TokenFile(args.validation_file, args.sequence_length)
    data_ledger = validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    if not data_ledger["valid"]:
        raise ValueError("invalid frozen data ledger")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    training.build_model = ratio.build_model
    args.integrity_valid = all(integrity_checks.values()) and protocol["valid"] and data_ledger["valid"]
    payload: dict[str, Any] = {
        "schema": "heterogeneous-swiglu-long-horizon-confirmation-v1",
        "candidate": "static 50/50 RMS-matched SiLU plus cubic-even SwiGLU",
        "scope": "five untouched seeds, 50M prediction tokens per arm and seed",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "protocol": protocol,
        "integrity_checks": integrity_checks,
        "integrity_manifest_sha256": sha256_file(INTEGRITY_MANIFEST),
        "data_ledger": data_ledger,
        "candidate_settings": ratio.ARM_SETTINGS["mixed50_rms"],
        "seeds": {},
    }
    for seed in args.seeds:
        seed_key = str(seed)
        payload["seeds"].setdefault(seed_key, {})
        for arm in ARMS:
            print(json.dumps({"starting_seed": seed, "starting_arm": arm}), flush=True)
            args.seed = seed
            payload["seeds"][seed_key][arm] = training.train_arm(
                arm, args, train_file, validation_file, device
            )
            payload["decision"] = decide(payload["seeds"], args)
            write_payload(args.output, payload)
    return payload


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/block-algebra-scratch/train.uint16.bin"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/block-algebra-scratch/validation.uint16.bin"))
    parser.add_argument("--data-manifest", type=Path, default=Path("results/block-algebra-scratch-data-manifest.json"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", type=parse_ints, default=list(SEEDS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1525)
    parser.add_argument("--eval-steps", type=parse_ints, default=[305, 1525])
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
