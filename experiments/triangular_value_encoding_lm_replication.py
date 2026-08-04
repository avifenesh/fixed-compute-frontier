#!/usr/bin/env python3
"""Untouched-seed, dependency-complete replication of the TVE LM discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import transformers
import triton

import experiments.triangular_value_encoding_lm_pilot as pilot


OUTPUT = Path("results/triangular-value-encoding-lm-replication.json")
CHECKPOINT_DIR = Path("results/triangular-value-encoding-lm-replication-checkpoints")
PREREGISTRATION = Path("results/triangular-value-encoding-lm-replication-preregistration.md")
MANIFEST = Path("results/triangular-value-encoding-lm-replication-integrity-manifest.json")
TRAIN_FILE = Path("data/block-algebra-scratch/train.uint16.bin")
VALIDATION_FILE = Path("data/block-algebra-scratch/validation.uint16.bin")
DATA_MANIFEST = Path("results/block-algebra-scratch-data-manifest.json")
SEED = 4409


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


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "python_version": "3.11.10",
        "numpy_version": "2.1.2",
        "torch_version": "2.5.1+cu124",
        "cuda_version": "12.4",
        "transformers_version": "4.57.6",
        "triton_version": "3.1.0",
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "data_manifest": str(DATA_MANIFEST),
        "output": str(OUTPUT),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 64,
        "steps": 305,
        "eval_steps": [61, 305],
        "warmup_steps": 50,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "seed": SEED,
        "prediction_tokens": 9_994_240,
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
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "triton_version": triton.__version__,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "data_manifest": str(args.data_manifest),
        "output": str(args.output),
        "checkpoint_dir": str(args.checkpoint_dir),
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
        raise ValueError(f"invalid replication protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    paths = {"replication_source": Path(__file__), **DEPENDENCIES}
    checks = {name: manifest.get(name + "_sha256") == pilot.sha256_file(path) for name, path in paths.items()}
    h100 = json.loads(DEPENDENCIES["h100_result"].read_text())
    checks.update({
        "h100_all_gates_pass": h100["all_gates_pass"] is True,
        "h100_integrity_valid": all(h100["integrity"].values()),
        "h100_manifest_binding_valid": h100["integrity_manifest_sha256"] == pilot.sha256_file(DEPENDENCIES["h100_manifest"]),
    })
    if not all(checks.values()):
        raise ValueError(f"invalid replication integrity: {checks}")
    return {"valid": True, "checks": checks, "manifest_sha256": pilot.sha256_file(MANIFEST)}


def replication_decision(results: dict[str, Any], args, integrity, protocol, ledger) -> dict[str, Any]:
    decision = pilot.decide(results, args)
    if not decision.get("complete"):
        return decision
    candidate = results["triangular_value_encoding"]
    artifact = candidate["terminal_artifacts"]
    expected_export = args.checkpoint_dir / f"seed-{args.seed}-candidate-bf16-attention.pt"
    decision["gates"].update({
        "candidate_bf16_export_saved_and_hashed": (
            artifact.get("bf16_attention_export") == str(expected_export)
            and artifact.get("bf16_attention_export_bytes", 0) > 0
            and len(artifact.get("bf16_attention_export_sha256", "")) == 64
            and expected_export.exists()
            and pilot.sha256_file(expected_export) == artifact["bf16_attention_export_sha256"]
        ),
        "runtime_metadata_zero": all(result["metadata_bits"] == 0 for result in results.values()),
        "architecture_constants_frozen": all(
            module_values == 393_216
            for result in results.values()
            for module_values in result["dense_attention_values_per_layer"]
        ),
        "replication_integrity_protocol_and_data_valid": integrity["valid"] and protocol["valid"] and ledger["valid"],
    })
    decision["pilot_pass"] = all(decision["gates"].values())
    return decision


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    frozen = [args.output, *(args.checkpoint_dir / f"seed-{args.seed}-{arm}.pt" for arm in pilot.ARMS), args.checkpoint_dir / f"seed-{args.seed}-candidate-bf16-attention.pt"]
    existing = [str(path) for path in frozen if path.exists()]
    if existing:
        raise FileExistsError(existing)
    protocol = validate_protocol(args)
    integrity = validate_integrity()
    train_file = pilot.TokenFile(args.train_file, args.sequence_length)
    validation_file = pilot.TokenFile(args.validation_file, args.sequence_length)
    ledger = pilot.validate_data_ledger(args.data_manifest, args.train_file, args.validation_file, args.sequence_length)
    if train_file.sequence_count < args.steps * args.gradient_accumulation * args.micro_batch_size or validation_file.sequence_count < args.eval_batches * args.eval_batch_size:
        raise ValueError("token files too short")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    payload: dict[str, Any] = {
        "schema": "triangular-value-encoding-lm-replication-formal-v1",
        "candidate": "gauge-canonical triangular value encoding",
        "scope": "untouched-seed confirmatory replication after exploratory discovery",
        "disclosure": "A one-step smoke and seed-3307 discovery were observed before this replication was frozen.",
        "source_sha256": pilot.sha256_file(Path(__file__)),
        "pilot_source_sha256": pilot.sha256_file(DEPENDENCIES["pilot_source"]),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "model": pilot.MODEL,
        "model_revision": pilot.MODEL_REVISION,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__, "triton": triton.__version__},
        "architecture_constants": {"block_size": 16, "tau": 0.125, "runtime_metadata_bits": 0},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "arms": {},
    }
    for arm in pilot.ARMS:
        print(json.dumps({"starting_replication_arm": arm}), flush=True)
        payload["arms"][arm] = pilot.train_arm(arm, args, train_file, validation_file, device)
        payload["decision"] = replication_decision(payload["arms"], args, integrity, protocol, ledger)
        pilot.write_payload(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--eval-steps", type=pilot.parse_steps, default=[61, 305])
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--formal", action="store_true", default=True)
    decision = run(parser.parse_args())["decision"]
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
