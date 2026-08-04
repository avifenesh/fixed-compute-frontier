#!/usr/bin/env python3
"""Screen full-head TVE against the established block-16 transform."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import transformers
import triton
from transformers import AutoModelForCausalLM

import experiments.triangular_value_encoding_lm_long_horizon as long_horizon
import experiments.triangular_value_encoding_lm_pilot as pilot
from experiments.triangular_value_encoding_attention import replace_llama_attention


OUTPUT = Path("results/triangular-value-encoding-gauge-budget-screen.json")
PREREGISTRATION = Path(
    "results/triangular-value-encoding-gauge-budget-preregistration.md"
)
MANIFEST = Path("results/triangular-value-encoding-gauge-budget-integrity-manifest.json")
TRAIN_FILE = Path("data/self-product-ffn-scale/train.uint16.bin")
VALIDATION_FILE = Path("data/self-product-ffn-scale/validation.uint16.bin")
DATA_MANIFEST = Path("results/self-product-ffn-scale-data-manifest.json")
SEED = 14321
VARIANTS = {
    "canonical_control": {
        "arm": "canonical_value_control",
        "block_size": 16,
        "tau": 0.125,
        "coefficient_slots_per_layer": 0,
    },
    "tve_block16": {
        "arm": "triangular_value_encoding",
        "block_size": 16,
        "tau": 0.125,
        "coefficient_slots_per_layer": 960,
    },
    "tve_full_head64": {
        "arm": "triangular_value_encoding",
        "block_size": 64,
        "tau": 0.25,
        "coefficient_slots_per_layer": 4032,
    },
}
DEPENDENCIES = {
    "screen_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "prior_long_horizon_manifest": long_horizon.MANIFEST,
    "prior_long_horizon_result": Path(
        "results/triangular-value-encoding-lm-long-horizon.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    prior = json.loads(DEPENDENCIES["prior_long_horizon_result"].read_text())
    checks.update({
        "prior_manifest_matches_bound_result": (
            prior["formal_integrity"]["manifest_sha256"]
            == sha256_file(DEPENDENCIES["prior_long_horizon_manifest"])
        ),
        "bound_long_horizon_integrity_valid": long_horizon.validate_integrity()["valid"],
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": manifest.get("transformers_version")
        == transformers.__version__,
        "triton_version": manifest.get("triton_version") == triton.__version__,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid gauge-budget integrity: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "data_manifest": str(DATA_MANIFEST),
        "output": str(OUTPUT),
        "checkpoint_dir": "results/triangular-value-encoding-gauge-budget-checkpoints",
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
        "prediction_tokens": 9_994_240,
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
    if args.steps not in args.eval_steps:
        checks["terminal_evaluation_present"] = False
    if not all(checks.values()):
        raise ValueError(f"invalid gauge-budget protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def builder_for(
    name: str,
    specification: dict[str, Any],
    observed_slots: dict[str, list[int]],
):
    def build(device: torch.device, arm: str):
        if arm != specification["arm"]:
            raise ValueError((arm, specification["arm"]))
        model = AutoModelForCausalLM.from_config(
            pilot.scratch_config(), attn_implementation="sdpa"
        ).to(device)
        model.config.use_cache = False
        modules = replace_llama_attention(
            model, arm, block_size=specification["block_size"]
        )
        for module in modules:
            module.tau = specification["tau"]
        observed_slots[name] = [module.coefficient_values().numel() for module in modules]
        return model, modules

    return build


def paired(candidate: dict[str, Any], reference: dict[str, Any], step: str):
    return pilot.paired_loss_interval(candidate, reference, step)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    protocol = validate_protocol(args)
    integrity = validate_integrity()
    ledger = pilot.validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    train_file = pilot.TokenFile(args.train_file, args.sequence_length)
    validation_file = pilot.TokenFile(args.validation_file, args.sequence_length)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    payload = {
        "schema": "triangular-value-encoding-gauge-budget-screen-v1",
        "scope": "development screen of unused full-head gauge coordinates",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "variants": VARIANTS,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": {},
    }
    original_builder = pilot.build_model
    observed_slots: dict[str, list[int]] = {}
    try:
        for name, specification in VARIANTS.items():
            print(json.dumps({"starting_gauge_budget_variant": name}), flush=True)
            pilot.build_model = builder_for(name, specification, observed_slots)
            variant_args = argparse.Namespace(**vars(args))
            variant_args.seed = SEED
            variant_args.formal = False
            payload["results"][name] = pilot.train_arm(
                specification["arm"],
                variant_args,
                train_file,
                validation_file,
                device,
            )
    finally:
        pilot.build_model = original_builder

    results = payload["results"]
    control = results["canonical_control"]
    block16 = results["tve_block16"]
    full64 = results["tve_full_head64"]
    terminal = str(args.steps)
    comparisons = {
        "block16_vs_control": paired(block16, control, terminal),
        "full64_vs_control": paired(full64, control, terminal),
        "full64_vs_block16": paired(full64, block16, terminal),
    }
    initial_exact = all(
        result["evaluations"]["0"]["loss"]
        == control["evaluations"]["0"]["loss"]
        and result["evaluations"]["0"]["per_batch_loss"]
        == control["evaluations"]["0"]["per_batch_loss"]
        for result in (block16, full64)
    )
    state_fields = (
        "total_parameters",
        "parameter_tensors",
        "model_buffer_values",
        "model_buffer_tensors",
        "state_dict_values",
        "state_dict_bytes",
        "optimizer_state_bytes",
        "metadata_bits",
    )
    state_equal = all(
        len({result[field] for result in results.values()}) == 1
        for field in state_fields
    )
    full_diagnostics = full64["serving_diagnostics"]["terminal"]
    full_attention = full64["attention_diagnostics"]["terminal"]
    full_loss = full64["evaluations"][terminal]["loss"]
    control_loss = control["evaluations"][terminal]["loss"]
    gates = {
        "integrity_and_protocol_valid": integrity["valid"] and protocol["valid"],
        "data_ledger_valid": ledger["valid"],
        "initial_candidate_evaluations_bit_exact": initial_exact,
        "all_state_counts_and_bytes_equal": state_equal,
        "full64_finite": (
            not full64["train"]["nonfinite"]
            and full_attention["value_nonfinite_fraction_layer_max"] == 0
        ),
        "full64_coefficients_nonzero_and_bounded": (
            full_diagnostics["coefficient_bf16_nonzero_fraction"] > 0
            and full_diagnostics["coefficient_bf16_abs_max"] <= 0.5
        ),
        "full64_export_bit_exact": full_diagnostics[
            "bf16_export_coefficient_bit_exact"
        ],
        "full64_serving_bridge_bit_exact": full64["terminal_serving_bridge"]
        == {"layers": 12, "all_layers_bit_exact": True, "max_abs": 0.0},
        "full64_clears_0_025_percent_vs_control": comparisons[
            "full64_vs_control"
        ]["upper_95"] <= -0.00025 * control_loss,
        "full64_paired_upper_below_block16": comparisons[
            "full64_vs_block16"
        ]["upper_95"] < 0,
        "full64_terminal_mean_below_control": full_loss < control_loss,
        "candidate_slot_counts_exact": (
            observed_slots["tve_block16"] == [960] * 12
            and observed_slots["tve_full_head64"] == [4032] * 12
        ),
    }
    payload.update({
        "comparisons": comparisons,
        "terminal_losses": {
            name: result["evaluations"][terminal]["loss"]
            for name, result in results.items()
        },
        "observed_coefficient_slots_per_layer": observed_slots,
        "gates": gates,
        "promote_full_head": all(gates.values()),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("results/triangular-value-encoding-gauge-budget-checkpoints"),
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=305)
    parser.add_argument("--eval-steps", type=lambda text: [int(v) for v in text.split(",")], default=[61, 305])
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    payload = run(parser.parse_args())
    print(json.dumps({"gates": payload["gates"], "promote_full_head": payload["promote_full_head"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
