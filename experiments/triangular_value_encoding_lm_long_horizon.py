#!/usr/bin/env python3
"""Three-seed, 100M-token long-horizon falsification for TVE."""

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

import experiments.triangular_value_encoding_lm_pilot as pilot


OUTPUT = Path("results/triangular-value-encoding-lm-long-horizon.json")
CHECKPOINT_DIR = Path("results/triangular-value-encoding-lm-long-horizon-checkpoints")
PREREGISTRATION = Path("results/triangular-value-encoding-lm-long-horizon-preregistration.md")
MANIFEST = Path("results/triangular-value-encoding-lm-long-horizon-integrity-manifest.json")
TRAIN_FILE = Path("data/self-product-ffn-scale/train.uint16.bin")
VALIDATION_FILE = Path("data/self-product-ffn-scale/validation.uint16.bin")
DATA_MANIFEST = Path("results/self-product-ffn-scale-data-manifest.json")
SEEDS = (5501, 6607, 7717)
EVAL_STEPS = (305, 763, 1526, 3050)
STEPS = 3050
T_CRITICAL_95_DF2 = 4.302652729911275


DEPENDENCIES = {
    "pilot_source": Path("experiments/triangular_value_encoding_lm_pilot.py"),
    "attention_source": Path("experiments/triangular_value_encoding_attention.py"),
    "value_flow_source": Path("experiments/triangular_value_flow_attention.py"),
    "cache_executor_source": Path("experiments/triangular_value_encoding_cache_h100.py"),
    "cache_helper_source": Path("experiments/triangular_value_encoding_h100.py"),
    "gauge_slot_source": Path("experiments/gauge_slot_lm_pilot.py"),
    "gauge_slot_attention_source": Path("experiments/gauge_slot_attention.py"),
    "reflex_source": Path("experiments/reflex_swiglu_lm_screen.py"),
    "triangular_source": Path("experiments/triangular_microdepth_lm_screen.py"),
    "triangular_gate_source": Path("experiments/triangular_microdepth_gate.py"),
    "attention_test": Path("tests/test_triangular_value_encoding_attention.py"),
    "bridge_test": Path("tests/test_triangular_value_encoding_cache_h100.py"),
    "lm_test": Path("tests/test_triangular_value_encoding_lm_pilot.py"),
    "h100_result": Path("results/triangular-value-encoding-cache-h100-formal-v6.json"),
    "h100_preregistration": Path("results/triangular-value-encoding-cache-h100-preregistration-v6.md"),
    "h100_manifest": Path("results/triangular-value-encoding-cache-h100-integrity-manifest-v6.json"),
    "h100_formal_source": Path("experiments/triangular_value_encoding_cache_h100_formal_v6.py"),
    "data_manifest": DATA_MANIFEST,
    "preregistration": PREREGISTRATION,
}


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    paths = {"long_horizon_source": Path(__file__), **DEPENDENCIES}
    checks = {name: manifest.get(name + "_sha256") == pilot.sha256_file(path) for name, path in paths.items()}
    h100 = json.loads(DEPENDENCIES["h100_result"].read_text())
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": manifest.get("transformers_version") == transformers.__version__,
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "h100_all_gates_pass": h100.get("all_gates_pass") is True,
        "h100_integrity_valid": all(h100.get("integrity", {}).values()),
        "h100_manifest_binding_valid": h100.get("integrity_manifest_sha256") == pilot.sha256_file(DEPENDENCIES["h100_manifest"]),
    })
    if not all(checks.values()):
        raise ValueError(f"invalid long-horizon integrity: {checks}")
    return {"valid": True, "checks": checks, "manifest_sha256": pilot.sha256_file(MANIFEST)}


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
        "prediction_tokens": args.steps * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
        "formal": args.formal,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
    }
    checks = {key: (value in actual[key] if key == "device_contains" else actual[key] == value) for key, value in expected.items()}
    if not all(checks.values()):
        raise ValueError(f"invalid long-horizon protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def seed_cluster_interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError("three frozen seed means required")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    radius = T_CRITICAL_95_DF2 * standard_error
    return {
        "seed_means": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "t_critical_95_df2": T_CRITICAL_95_DF2,
        "lower_95": mean - radius,
        "upper_95": mean + radius,
    }


def paired_mean(candidate: dict[str, Any], reference: dict[str, Any], step: str) -> float:
    left = np.asarray(candidate["evaluations"][step]["per_batch_loss"], dtype=np.float64)
    right = np.asarray(reference["evaluations"][step]["per_batch_loss"], dtype=np.float64)
    if left.shape != right.shape or left.size != 64:
        raise ValueError("64 matched validation batches required")
    return float((left - right).mean())


def artifact_valid(result: dict[str, Any], seed: int, arm: str, checkpoint_dir: Path) -> bool:
    artifact = result["terminal_artifacts"]
    checkpoint = checkpoint_dir / f"seed-{seed}-{arm}.pt"
    valid = (
        artifact.get("saved") is True
        and artifact.get("checkpoint") == str(checkpoint)
        and checkpoint.exists()
        and checkpoint.stat().st_size == artifact.get("checkpoint_bytes")
        and pilot.sha256_file(checkpoint) == artifact.get("checkpoint_sha256")
    )
    if arm != "triangular_value_encoding":
        return valid
    export = checkpoint_dir / f"seed-{seed}-candidate-bf16-attention.pt"
    return valid and (
        artifact.get("bf16_attention_export") == str(export)
        and export.exists()
        and export.stat().st_size == artifact.get("bf16_attention_export_bytes")
        and pilot.sha256_file(export) == artifact.get("bf16_attention_export_sha256")
    )


def decide(seed_results: dict[str, dict[str, Any]], args: argparse.Namespace, integrity: dict[str, Any], protocol: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    expected_seeds = {str(seed) for seed in SEEDS}
    if set(seed_results) != expected_seeds or any(set(results) != set(pilot.ARMS) for results in seed_results.values()):
        return {"complete": False, "completed_seeds": sorted(seed_results)}

    trajectories: dict[str, Any] = {}
    for step in EVAL_STEPS:
        key = str(step)
        candidate_control = []
        candidate_raw = []
        for seed in SEEDS:
            results = seed_results[str(seed)]
            candidate_control.append(paired_mean(results["triangular_value_encoding"], results["canonical_value_control"], key))
            candidate_raw.append(paired_mean(results["triangular_value_encoding"], results["packed_raw_control"], key))
        trajectories[key] = {
            "prediction_tokens": step * args.gradient_accumulation * args.micro_batch_size * args.sequence_length,
            "candidate_vs_control_seed_cluster": seed_cluster_interval(candidate_control),
            "candidate_vs_raw_seed_cluster": seed_cluster_interval(candidate_raw),
        }

    terminal = str(STEPS)
    terminal_ablation = []
    terminal_control_losses = []
    terminal_candidate_losses = []
    structural_gate_names = {
        "exact_parameter_count",
        "all_parameter_counts_equal",
        "all_parameter_tensor_counts_equal",
        "all_buffer_counts_equal",
        "all_state_and_optimizer_bytes_equal",
        "all_metrics_finite",
        "all_value_nonfinite_fractions_zero",
        "candidate_initial_bit_exact_to_control",
        "some_coefficients_survive_bf16",
        "coefficients_bounded",
        "coefficient_export_bit_exact",
        "trained_candidate_serving_bridge_bit_exact",
        "zero_metadata",
        "formal_artifacts_saved",
    }
    per_seed_diagnostics = {}
    artifacts_ok = True
    structural_ok = True
    canonical_ok = True
    for seed in SEEDS:
        results = seed_results[str(seed)]
        candidate = results["triangular_value_encoding"]
        disabled = {"evaluations": {terminal: candidate["terminal_ablations"]["disable_encoding"]}}
        full = {"evaluations": {terminal: candidate["evaluations"][terminal]}}
        terminal_ablation.append(paired_mean(full, disabled, terminal))
        terminal_control_losses.append(results["canonical_value_control"]["evaluations"][terminal]["loss"])
        terminal_candidate_losses.append(candidate["evaluations"][terminal]["loss"])
        seed_args = argparse.Namespace(**vars(args)); seed_args.seed = seed
        diagnostic = pilot.decide(results, seed_args)
        per_seed_diagnostics[str(seed)] = diagnostic
        structural_ok = structural_ok and all(diagnostic["gates"][name] for name in structural_gate_names)
        canonical_ok = canonical_ok and diagnostic["gates"]["canonical_initial_nll_matches_raw_within_0_02_percent"] and diagnostic["gates"]["canonical_initial_paired_interval_within_0_02_percent"] and diagnostic["gates"]["control_terminal_noninferior_raw_within_0_05_percent"]
        artifacts_ok = artifacts_ok and all(artifact_valid(results[arm], seed, arm, args.checkpoint_dir) for arm in pilot.ARMS)

    primary = trajectories[terminal]["candidate_vs_control_seed_cluster"]
    ablation = seed_cluster_interval(terminal_ablation)
    mean_control = float(np.mean(terminal_control_losses))
    mean_candidate = float(np.mean(terminal_candidate_losses))
    gates = {
        "integrity_protocol_and_data_valid": integrity["valid"] and protocol["valid"] and ledger["valid"],
        "all_structural_and_numerical_invariants": structural_ok,
        "canonical_control_valid_for_all_seeds": canonical_ok,
        "all_artifact_paths_sizes_and_hashes_valid": artifacts_ok,
        "all_three_terminal_seed_means_favor_candidate": all(value < 0 for value in primary["seed_means"]),
        "terminal_seed_cluster_clears_0_01_percent": primary["upper_95"] <= -0.0001 * mean_control,
        "terminal_ablation_seed_cluster_clears_0_005_percent": ablation["upper_95"] <= -0.00005 * mean_candidate,
    }
    return {
        "complete": True,
        "primary_endpoint": "100M-token candidate versus gauge-canonical control",
        "mean_terminal_losses": {"canonical_value_control": mean_control, "triangular_value_encoding": mean_candidate},
        "relative_terminal_improvement": (mean_control - mean_candidate) / mean_control,
        "terminal_ablation_seed_cluster": ablation,
        "trajectories": trajectories,
        "per_seed_diagnostics": per_seed_diagnostics,
        "gates": gates,
        "long_horizon_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = [
        args.checkpoint_dir / f"seed-{seed}-{arm}.pt"
        for seed in args.seeds
        for arm in pilot.ARMS
    ] + [args.checkpoint_dir / f"seed-{seed}-candidate-bf16-attention.pt" for seed in args.seeds]
    existing = [str(path) for path in frozen if path.exists()]
    if existing:
        raise FileExistsError(existing)
    protocol = validate_protocol(args)
    integrity = validate_integrity()
    ledger = pilot.validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    train_file = pilot.TokenFile(args.train_file, args.sequence_length)
    validation_file = pilot.TokenFile(args.validation_file, args.sequence_length)
    required_sequences = args.steps * args.gradient_accumulation * args.micro_batch_size
    if train_file.sequence_count < required_sequences or validation_file.sequence_count < args.eval_batches * args.eval_batch_size:
        raise ValueError("token files too short")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": "triangular-value-encoding-lm-long-horizon-formal-v1",
        "scope": "preregistered three-seed 100M-token control-catch-up falsification",
        "source_sha256": pilot.sha256_file(Path(__file__)),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "triton": triton.__version__},
        "architecture_constants": {"block_size": 16, "tau": 0.125, "runtime_metadata_bits": 0},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "seeds": {},
        "decision": {"complete": False, "completed_seeds": []},
    }
    for seed in args.seeds:
        payload["seeds"][str(seed)] = {}
        for arm in pilot.ARMS:
            print(json.dumps({"starting_long_horizon_seed": seed, "arm": arm}), flush=True)
            seed_args = argparse.Namespace(**vars(args)); seed_args.seed = seed
            payload["seeds"][str(seed)][arm] = pilot.train_arm(arm, seed_args, train_file, validation_file, device)
            pilot.write_payload(args.output, payload)
        payload["decision"] = decide(payload["seeds"], args, integrity, protocol, ledger)
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
