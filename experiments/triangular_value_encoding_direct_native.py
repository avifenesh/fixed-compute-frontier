#!/usr/bin/env python3
"""Five-seed direct-native 100M-token confirmation for TVE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import transformers
import triton

import experiments.triangular_value_encoding_lm_long_horizon as long_horizon
import experiments.triangular_value_encoding_lm_pilot as pilot


OUTPUT = Path("results/triangular-value-encoding-direct-native.json")
CHECKPOINT_DIR = Path("results/triangular-value-encoding-direct-native-checkpoints")
PREREGISTRATION = Path("results/triangular-value-encoding-direct-native-preregistration.md")
MANIFEST = Path("results/triangular-value-encoding-direct-native-integrity-manifest.json")
TRAIN_FILE = long_horizon.TRAIN_FILE
VALIDATION_FILE = long_horizon.VALIDATION_FILE
DATA_MANIFEST = long_horizon.DATA_MANIFEST
SEEDS = (8821, 9901, 11003, 12109, 13217)
ARMS = ("packed_raw_control", "triangular_value_encoding")
EVAL_STEPS = long_horizon.EVAL_STEPS
STEPS = long_horizon.STEPS
T_CRITICAL_95_DF4 = 2.7764451051977987

DEPENDENCIES = {
    "direct_native_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "prior_long_horizon_result": Path("results/triangular-value-encoding-lm-long-horizon.json"),
    "prior_long_horizon_decision": Path("results/triangular-value-encoding-lm-long-horizon-decision.md"),
    "prior_long_horizon_manifest": long_horizon.MANIFEST,
}


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == pilot.sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    prior_result = json.loads(DEPENDENCIES["prior_long_horizon_result"].read_text())
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": manifest.get("transformers_version") == transformers.__version__,
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "prior_formal_no_go_disclosed": prior_result["decision"]["long_horizon_pass"] is False,
        "prior_manifest_matches_bound_result": (
            prior_result["formal_integrity"]["manifest_sha256"]
            == pilot.sha256_file(DEPENDENCIES["prior_long_horizon_manifest"])
        ),
        "bound_long_horizon_integrity_valid": long_horizon.validate_integrity()["valid"],
    })
    if not all(checks.values()):
        raise ValueError(f"invalid direct-native integrity: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "manifest_sha256": pilot.sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "data_manifest": str(DATA_MANIFEST),
        "output": str(OUTPUT),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "seeds": list(SEEDS),
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 64,
        "steps": STEPS,
        "eval_steps": list(EVAL_STEPS),
        "warmup_steps": 50,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "prediction_tokens": 99_942_400,
        "formal": True,
        "hidden_size": 384,
        "layers": 12,
        "query_heads": 6,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 1024,
    }
    actual = {
        "device_contains": torch.cuda.get_device_name(0),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "data_manifest": str(args.data_manifest),
        "output": str(args.output),
        "checkpoint_dir": str(args.checkpoint_dir),
        "seeds": args.seeds,
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
        "prediction_tokens": (
            args.steps * args.gradient_accumulation * args.micro_batch_size
            * args.sequence_length
        ),
        "formal": args.formal,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
    }
    checks = {
        key: (value in actual[key] if key == "device_contains" else actual[key] == value)
        for key, value in expected.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid direct-native protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def seed_cluster_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,):
        raise ValueError("five frozen seed means required")
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    radius = T_CRITICAL_95_DF4 * standard_error
    mean = float(array.mean())
    return {
        "seed_means": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "t_critical_95_df4": T_CRITICAL_95_DF4,
        "lower_95": mean - radius,
        "upper_95": mean + radius,
    }


def paired_mean(left: dict[str, Any], right: dict[str, Any], step: str) -> float:
    left_losses = np.asarray(left["evaluations"][step]["per_batch_loss"], dtype=np.float64)
    right_losses = np.asarray(right["evaluations"][step]["per_batch_loss"], dtype=np.float64)
    if left_losses.shape != (64,) or right_losses.shape != (64,):
        raise ValueError("64 fixed paired batches required")
    return float((left_losses - right_losses).mean())


def finite_result(result: dict[str, Any]) -> bool:
    values = [
        value
        for evaluation in result["evaluations"].values()
        for value in (evaluation["loss"], *evaluation["per_batch_loss"])
    ]
    return all(math.isfinite(value) for value in values) and not result["train"]["nonfinite"]


def structural_valid(raw: dict[str, Any], candidate: dict[str, Any]) -> bool:
    equal_fields = (
        "total_parameters",
        "parameter_tensors",
        "model_buffer_values",
        "model_buffer_tensors",
        "state_dict_values",
        "state_dict_bytes",
        "optimizer_state_bytes",
        "dense_attention_values_per_layer",
        "metadata_bits",
    )
    return (
        raw["total_parameters"] == candidate["total_parameters"] == 37_758_336
        and all(raw[field] == candidate[field] for field in equal_fields)
        and finite_result(raw)
        and finite_result(candidate)
        and all(
            result["attention_diagnostics"][phase]["value_nonfinite_fraction_layer_max"] == 0
            for result in (raw, candidate)
            for phase in ("initial", "terminal")
        )
        and candidate["serving_diagnostics"]["terminal"]["coefficient_bf16_nonzero_fraction"] > 0
        and candidate["serving_diagnostics"]["terminal"]["coefficient_bf16_abs_max"] <= 0.5
        and candidate["serving_diagnostics"]["terminal"]["bf16_export_coefficient_bit_exact"]
        and candidate["terminal_serving_bridge"]
        == {"layers": 12, "all_layers_bit_exact": True, "max_abs": 0.0}
        and raw["metadata_bits"] == candidate["metadata_bits"] == 0
    )


def decide(
    seed_results: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    integrity: dict[str, Any],
    protocol: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    if set(seed_results) != {str(seed) for seed in SEEDS} or any(
        set(results) != set(ARMS) for results in seed_results.values()
    ):
        return {"complete": False, "completed_seeds": sorted(seed_results)}

    trajectories = {}
    for step in EVAL_STEPS:
        key = str(step)
        seed_means = [
            paired_mean(
                seed_results[str(seed)]["triangular_value_encoding"],
                seed_results[str(seed)]["packed_raw_control"],
                key,
            )
            for seed in SEEDS
        ]
        trajectories[key] = {
            "prediction_tokens": (
                step * args.gradient_accumulation * args.micro_batch_size
                * args.sequence_length
            ),
            "candidate_vs_raw_seed_cluster": seed_cluster_interval(seed_means),
        }

    terminal = str(STEPS)
    initial_ok = True
    structural_ok = True
    artifacts_ok = True
    raw_losses = []
    candidate_losses = []
    ablation_means = []
    per_seed = {}
    for seed in SEEDS:
        results = seed_results[str(seed)]
        raw = results["packed_raw_control"]
        candidate = results["triangular_value_encoding"]
        initial = pilot.paired_loss_interval(candidate, raw, "0")
        initial_bound = 0.0002 * raw["evaluations"]["0"]["loss"]
        seed_initial_ok = (
            initial["lower_95"] >= -initial_bound
            and initial["upper_95"] <= initial_bound
        )
        initial_ok = initial_ok and seed_initial_ok
        seed_structural = structural_valid(raw, candidate)
        structural_ok = structural_ok and seed_structural
        seed_artifacts = all(
            long_horizon.artifact_valid(results[arm], seed, arm, args.checkpoint_dir)
            for arm in ARMS
        )
        artifacts_ok = artifacts_ok and seed_artifacts
        disabled = {
            "evaluations": {
                terminal: candidate["terminal_ablations"]["disable_encoding"]
            }
        }
        full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
        ablation_means.append(paired_mean(full, disabled, terminal))
        raw_losses.append(raw["evaluations"][terminal]["loss"])
        candidate_losses.append(candidate["evaluations"][terminal]["loss"])
        per_seed[str(seed)] = {
            "initial_candidate_vs_raw": initial,
            "initial_within_0_02_percent": seed_initial_ok,
            "structural_valid": seed_structural,
            "artifacts_valid": seed_artifacts,
        }

    primary = trajectories[terminal]["candidate_vs_raw_seed_cluster"]
    ablation = seed_cluster_interval(ablation_means)
    mean_raw = float(np.mean(raw_losses))
    mean_candidate = float(np.mean(candidate_losses))
    relative_improvement = (mean_raw - mean_candidate) / mean_raw
    gates = {
        "integrity_protocol_and_data_valid": (
            integrity["valid"] and protocol["valid"] and ledger["valid"]
        ),
        "all_structural_and_numerical_invariants": structural_ok,
        "all_initial_intervals_within_0_02_percent": initial_ok,
        "all_artifact_paths_sizes_and_hashes_valid": artifacts_ok,
        "all_five_terminal_seed_means_favor_candidate": all(
            value < 0 for value in primary["seed_means"]
        ),
        "terminal_cluster_clears_0_01_percent": (
            primary["upper_95"] <= -0.0001 * mean_raw
        ),
        "mean_relative_improvement_at_least_0_05_percent": (
            relative_improvement >= 0.0005
        ),
        "terminal_ablation_cluster_clears_0_005_percent": (
            ablation["upper_95"] <= -0.00005 * mean_candidate
        ),
    }
    return {
        "complete": True,
        "primary_endpoint": "100M-token TVE versus packed raw transformer",
        "mean_terminal_losses": {
            "packed_raw_control": mean_raw,
            "triangular_value_encoding": mean_candidate,
        },
        "relative_terminal_improvement": relative_improvement,
        "terminal_ablation_seed_cluster": ablation,
        "trajectories": trajectories,
        "per_seed_diagnostics": per_seed,
        "gates": gates,
        "direct_native_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = [
        args.checkpoint_dir / f"seed-{seed}-{arm}.pt"
        for seed in args.seeds
        for arm in ARMS
    ] + [
        args.checkpoint_dir / f"seed-{seed}-candidate-bf16-attention.pt"
        for seed in args.seeds
    ]
    existing = [str(path) for path in frozen if path.exists()]
    if existing:
        raise FileExistsError(existing)
    protocol = validate_protocol(args)
    integrity = validate_integrity()
    ledger = pilot.validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    train_file = pilot.TokenFile(args.train_file, args.sequence_length)
    validation_file = pilot.TokenFile(args.validation_file, args.sequence_length)
    required_sequences = args.steps * args.gradient_accumulation * args.micro_batch_size
    if (
        train_file.sequence_count < required_sequences
        or validation_file.sequence_count < args.eval_batches * args.eval_batch_size
    ):
        raise ValueError("token files too short")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": "triangular-value-encoding-direct-native-formal-v1",
        "scope": "five untouched seeds, direct native raw primary endpoint",
        "disclosure": {
            "prior_long_horizon_formal_pass": False,
            "prior_candidate_won_all_three_vs_raw": True,
            "prior_seeds_excluded_from_this_decision": [5501, 6607, 7717],
        },
        "source_sha256": pilot.sha256_file(Path(__file__)),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
        },
        "architecture_constants": {
            "block_size": 16,
            "tau": 0.125,
            "runtime_metadata_bits": 0,
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "seeds": {},
        "decision": {"complete": False, "completed_seeds": []},
    }
    for seed in args.seeds:
        payload["seeds"][str(seed)] = {}
        for arm in ARMS:
            print(json.dumps({"starting_direct_native_seed": seed, "arm": arm}), flush=True)
            seed_args = argparse.Namespace(**vars(args))
            seed_args.seed = seed
            payload["seeds"][str(seed)][arm] = pilot.train_arm(
                arm, seed_args, train_file, validation_file, device
            )
            pilot.write_payload(args.output, payload)
        payload["decision"] = decide(
            payload["seeds"], args, integrity, protocol, ledger
        )
        pilot.write_payload(args.output, payload)
    return payload


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--seeds", type=parse_ints, default=list(SEEDS))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-steps", type=parse_ints, default=list(EVAL_STEPS))
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--formal", action="store_true", default=True)
    decision = run(parser.parse_args())["decision"]
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
