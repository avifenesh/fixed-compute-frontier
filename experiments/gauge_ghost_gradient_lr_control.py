#!/usr/bin/env python3
"""Fatal learning-rate control for the seed-21017 ghost-gradient gain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import transformers
import triton

import experiments.triangular_value_encoding_forward_mechanism_v2 as mechanism


OUTPUT = Path("results/gauge-ghost-gradient-lr-control.json")
PREREGISTRATION = Path("results/gauge-ghost-gradient-lr-control-preregistration.md")
MANIFEST = Path("results/gauge-ghost-gradient-lr-control-integrity-manifest.json")
PRIOR_RESULT = Path("results/triangular-value-encoding-forward-mechanism-v2.json")
PRIOR_MANIFEST = Path(
    "results/triangular-value-encoding-forward-mechanism-v2-integrity-manifest.json"
)
SEED = 21017
BASE_LR = 3e-4
LR_GRID = (
    1.5e-4,
    3e-4 / math.sqrt(2.0),
    BASE_LR,
    3e-4 * math.sqrt(2.0),
    6e-4,
)
NEW_LRS = tuple(lr for lr in LR_GRID if lr != BASE_LR)
EVAL_STEPS = (61, 152, 305)
STEPS = 305

DEPENDENCIES = {
    "control_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "mechanism_source": Path(
        "experiments/triangular_value_encoding_forward_mechanism_v2.py"
    ),
    "mechanism_preregistration": Path(
        "results/triangular-value-encoding-forward-mechanism-v2-preregistration.md"
    ),
    "prior_manifest": PRIOR_MANIFEST,
    "prior_result": PRIOR_RESULT,
    "pilot_source": Path("experiments/triangular_value_encoding_lm_pilot.py"),
    "attention_source": Path("experiments/triangular_value_encoding_attention.py"),
    "data_manifest": mechanism.DATA_MANIFEST,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def prior_arm_realized_valid(
    arm: dict[str, Any], expected_arm: str, canonical: dict[str, Any],
) -> bool:
    evaluations = arm.get("evaluations", {})
    train = arm.get("train", {})
    equal_fields = (
        "total_parameters", "parameter_tensors", "model_buffer_values",
        "model_buffer_tensors", "state_dict_values", "state_dict_bytes",
        "optimizer_state_bytes", "metadata_bits",
    )
    return (
        arm.get("arm") == expected_arm
        and finite(arm)
        and set(evaluations) == {"0", "61", "152", "305"}
        and all(
            len(evaluation.get("per_batch_loss", [])) == 64
            and evaluation.get("prediction_tokens") == 1_048_576
            and math.isclose(
                evaluation.get("loss", float("nan")),
                float(np.mean(evaluation.get("per_batch_loss", []))),
                rel_tol=0.0, abs_tol=1e-12,
            )
            for evaluation in evaluations.values()
        )
        and len(train.get("step_losses", [])) == STEPS
        and train.get("prediction_tokens") == 9_994_240
        and train.get("nonfinite") is False
        and train.get("tokens_per_second", 0) > 0
        and arm.get("total_parameters") == 37_758_336
        and all(arm.get(field) == canonical.get(field) for field in equal_fields)
        and arm.get("metadata_bits") == 0
        and arm.get("terminal_serving_bridge") == {
            "forward_semantics": "ordinary_canonical",
            "tve_cache_encoding_required": False,
        }
    )


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    prior = json.loads(PRIOR_RESULT.read_text())
    decision = prior.get("decision", {})
    prior_protocol = prior.get("experiment_protocol", {})
    prior_expected_protocol = {
        "device_contains": "H100",
        "train_file": "data/self-product-ffn-scale/train.uint16.bin",
        "validation_file": "data/self-product-ffn-scale/validation.uint16.bin",
        "data_manifest": "results/self-product-ffn-scale-data-manifest.json",
        "output": "results/triangular-value-encoding-forward-mechanism-v2.json",
        "export_dir": (
            "results/triangular-value-encoding-forward-mechanism-v2-exports"
        ),
        "seed": SEED,
        "arms": ["canonical_baseline", "backward_only_tve", "full_tve"],
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 64,
        "steps": STEPS,
        "eval_steps": list(EVAL_STEPS),
        "warmup_steps": 50,
        "learning_rate": BASE_LR,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "prediction_tokens": 9_994_240,
        "formal": True,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "flash_sdp_enabled": False,
        "memory_efficient_sdp_enabled": False,
        "cudnn_sdp_enabled": False,
        "math_sdp_enabled": True,
        "fp16_bf16_reduction_math_sdp_allowed": False,
        "hidden_size": 384,
        "layers": 12,
        "query_heads": 6,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 1024,
    }
    prior_actual_protocol = prior_protocol.get("actual", {})
    normalized_prior_actual = dict(prior_actual_protocol)
    normalized_prior_actual["device_contains"] = "H100"
    prior_arms = prior.get("arms", {})
    prior_canonical = prior_arms.get("canonical_baseline", {})
    prior_ghost = prior_arms.get("backward_only_tve", {})
    try:
        recomputed_prior_decision = mechanism.decide(
            prior_arms, prior.get("first_backward_probe", {})
        )
    except (KeyError, TypeError, ValueError):
        recomputed_prior_decision = None
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": (
            manifest.get("transformers_version") == transformers.__version__
        ),
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "prior_current_integrity_valid": mechanism.validate_integrity()["valid"],
        "prior_screen_valid": decision.get("screen_valid") is True,
        "prior_classification_exact": (
            decision.get("classification")
            == "backward_signal_captures_gain_this_seed"
        ),
        "prior_losses_exact": decision.get("terminal_losses") == {
            "backward_only_tve": 6.507063269615173,
            "canonical_baseline": 6.523809522390366,
            "full_tve": 6.509276621043682,
        },
        "prior_manifest_binding_exact": (
            prior.get("formal_integrity", {}).get("manifest_sha256")
            == sha256_file(PRIOR_MANIFEST)
        ),
        "prior_schema_exact": (
            prior.get("schema")
            == "triangular-value-encoding-forward-mechanism-screen-v2"
        ),
        "prior_source_exact": (
            prior.get("source_sha256")
            == sha256_file(DEPENDENCIES["mechanism_source"])
        ),
        "prior_protocol_valid": prior_protocol.get("valid") is True,
        "prior_protocol_expected_exact": (
            prior_protocol.get("expected") == prior_expected_protocol
        ),
        "prior_protocol_actual_exact": (
            "H100" in prior_actual_protocol.get("device_contains", "")
            and normalized_prior_actual == prior_expected_protocol
        ),
        "prior_protocol_checks_all_true": (
            set(prior_protocol.get("checks", {})) == set(prior_expected_protocol)
            and all(prior_protocol.get("checks", {}).values())
        ),
        "prior_deterministic_backend_exact": (
            prior.get("deterministic_attention") == {
                "cublas_workspace_config": ":4096:8",
                "cudnn_sdp_enabled": False,
                "deterministic_algorithms": True,
                "flash_sdp_enabled": False,
                "fp16_bf16_reduction_math_sdp_allowed": False,
                "math_sdp_enabled": True,
                "memory_efficient_sdp_enabled": False,
            }
        ),
        "prior_arm_identities_exact": (
            set(prior_arms)
            == {"canonical_baseline", "backward_only_tve", "full_tve"}
        ),
        "prior_canonical_realized_valid": prior_arm_realized_valid(
            prior_canonical, "canonical_baseline", prior_canonical
        ) and prior_canonical.get("terminal_artifact") is None,
        "prior_ghost_realized_valid": prior_arm_realized_valid(
            prior_ghost, "backward_only_tve", prior_canonical
        ) and prior_ghost.get("terminal_artifact") is not None,
        "prior_initial_state_exact": (
            prior_canonical.get("initial_state_sha256")
            == prior_ghost.get("initial_state_sha256")
            == "93f064bd09eb322a6ecc3a29c98d9233f9bbd552e1f8300f31ca972f8bac0a94"
            and prior_canonical.get("evaluations", {}).get("0", {}).get(
                "per_batch_loss"
            ) == prior_ghost.get("evaluations", {}).get("0", {}).get(
                "per_batch_loss"
            )
        ),
        "prior_decision_recomputed_exact": recomputed_prior_decision == decision,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid LR-control integrity: {checks}")
    return {
        "valid": True, "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = mechanism.pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "output": str(OUTPUT),
        "seed": SEED,
        "lr_grid": list(LR_GRID),
        "new_lrs": list(NEW_LRS),
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 64,
        "steps": STEPS,
        "eval_steps": list(EVAL_STEPS),
        "warmup_steps": 50,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "prediction_tokens": 9_994_240,
        "hidden_size": 384,
        "layers": 12,
        "query_heads": 6,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 1024,
        "formal": True,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "flash_sdp_enabled": False,
        "memory_efficient_sdp_enabled": False,
        "cudnn_sdp_enabled": False,
        "math_sdp_enabled": True,
        "fp16_bf16_reduction_math_sdp_allowed": False,
        "float32_matmul_precision": "high",
        "allow_tf32": True,
    }
    actual = {
        "device_contains": torch.cuda.get_device_name(0),
        "output": str(args.output),
        "seed": args.seed,
        "lr_grid": list(LR_GRID),
        "new_lrs": args.learning_rates,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup_steps,
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
        "formal": args.formal,
        **mechanism.configure_deterministic_attention(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    checks = {
        key: (
            expected_value in actual[key]
            if key == "device_contains" else actual[key] == expected_value
        )
        for key, expected_value in expected.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid LR-control protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def arm_valid(
    result: dict[str, Any], args: argparse.Namespace, expected_lr: float,
    prior_canonical: dict[str, Any],
) -> bool:
    if result.get("arm") != "canonical_baseline" or not finite(result):
        return False
    if set(result.get("evaluations", {})) != {"0", "61", "152", "305"}:
        return False
    for evaluation in result["evaluations"].values():
        losses = evaluation.get("per_batch_loss", [])
        if (
            len(losses) != 64 or evaluation.get("prediction_tokens") != 1_048_576
            or not math.isclose(
                evaluation.get("loss", float("nan")), float(np.mean(losses)),
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            return False
    train = result.get("train", {})
    equal_fields = (
        "total_parameters", "parameter_tensors", "model_buffer_values",
        "model_buffer_tensors", "state_dict_values", "state_dict_bytes",
        "optimizer_state_bytes", "metadata_bits",
    )
    return (
        result.get("total_parameters") == 37_758_336
        and all(
            result.get(field) == prior_canonical.get(field)
            for field in equal_fields
        )
        and result.get("control_seed") == SEED
        and result.get("control_learning_rate") == expected_lr
        and result.get("control_preregistration_sha256")
        == sha256_file(PREREGISTRATION)
        and result.get("terminal_serving_bridge") == {
            "forward_semantics": "ordinary_canonical",
            "tve_cache_encoding_required": False,
        }
        and len(train.get("step_losses", [])) == args.steps
        and train.get("prediction_tokens") == 9_994_240
        and train.get("nonfinite") is False
        and train.get("tokens_per_second", 0) > 0
        and result.get("terminal_artifact") is None
        and result.get("metadata_bits") == 0
    )


def lr_key(lr: float) -> str:
    return format(lr, ".17g")


def decide(results: dict[str, Any], prior: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    expected = {lr_key(lr) for lr in NEW_LRS}
    if set(results) != expected:
        return {"complete": False, "completed": sorted(results)}
    prior_arms = prior["arms"]
    base = prior_arms["canonical_baseline"]
    ghost = prior_arms["backward_only_tve"]
    terminal = "305"
    losses = {BASE_LR: base["evaluations"][terminal]["loss"]}
    losses.update({float(key): result["evaluations"][terminal]["loss"] for key, result in results.items()})
    best_lr = min(losses, key=losses.get)
    best_loss = losses[best_lr]
    ghost_loss = ghost["evaluations"][terminal]["loss"]
    base_loss = base["evaluations"][terminal]["loss"]
    original_gain = base_loss - ghost_loss
    lr_captured = (base_loss - best_loss) / original_gain
    ghost_margin = best_loss - ghost_loss
    relative_margin = ghost_margin / best_loss
    best_interior_loss = min(losses[lr] for lr in LR_GRID[1:-1])
    endpoint = min(losses[LR_GRID[0]], losses[LR_GRID[-1]]) <= best_interior_loss
    survives = (
        not endpoint
        and relative_margin >= 0.0005
        and ghost_margin >= 0.5 * original_gain
    )
    fatal = not endpoint and lr_captured >= 0.80
    if endpoint:
        classification = "endpoint_extension_required"
    elif fatal:
        classification = "learning_rate_explanation_fatal"
    elif survives:
        classification = "ghost_survives_global_lr_control"
    else:
        classification = "inconclusive_lr_control"
    prior_initial = base["evaluations"]["0"]["per_batch_loss"]
    structural = all(
        arm_valid(result, args, float(key), base)
        and result["initial_state_sha256"] == base["initial_state_sha256"]
        and result["evaluations"]["0"]["per_batch_loss"] == prior_initial
        for key, result in results.items()
    )
    return {
        "complete": True,
        "terminal_canonical_losses_by_lr": {
            lr_key(lr): losses[lr] for lr in LR_GRID
        },
        "ghost_terminal_loss": ghost_loss,
        "best_canonical_lr": best_lr,
        "best_canonical_loss": best_loss,
        "original_ghost_gain": original_gain,
        "fraction_of_ghost_gain_captured_by_lr": lr_captured,
        "ghost_margin_over_best_canonical": ghost_margin,
        "relative_ghost_margin_over_best_canonical": relative_margin,
        "best_is_grid_endpoint": endpoint,
        "best_interior_canonical_loss": best_interior_loss,
        "all_new_arms_structurally_valid": structural,
        "classification": classification if structural else "invalid_control",
        "ghost_survives_global_lr_control": structural and survives,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    determinism = mechanism.configure_deterministic_attention()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    integrity = validate_integrity()
    protocol = validate_protocol(args)
    ledger = mechanism.pilot.validate_data_ledger(
        mechanism.DATA_MANIFEST, mechanism.TRAIN_FILE,
        mechanism.VALIDATION_FILE, args.sequence_length,
    )
    prior = json.loads(PRIOR_RESULT.read_text())
    train_file = mechanism.pilot.TokenFile(mechanism.TRAIN_FILE, args.sequence_length)
    validation_file = mechanism.pilot.TokenFile(
        mechanism.VALIDATION_FILE, args.sequence_length
    )
    device = torch.device("cuda")
    fresh = {
        "schema": "gauge-ghost-gradient-lr-control-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "triton": triton.__version__,
        },
        "deterministic_attention": determinism,
        "results": {},
        "completed_lrs": [],
        "decision": {"complete": False},
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text())
        immutable = set(fresh) - {"results", "completed_lrs", "decision"}
        if set(payload) != set(fresh) or any(payload[key] != fresh[key] for key in immutable):
            raise ValueError("invalid LR-control resume binding")
        expected_prefix = [lr_key(lr) for lr in args.learning_rates]
        if payload.get("completed_lrs") != expected_prefix[:len(payload.get("completed_lrs", []))]:
            raise ValueError("invalid LR-control completion prefix")
        if set(payload.get("results", {})) != set(payload["completed_lrs"]):
            raise ValueError("LR-control results disagree with completion journal")
        if payload.get("decision") != {"complete": False}:
            raise ValueError("only incomplete LR control can resume")
        prior_canonical = prior["arms"]["canonical_baseline"]
        if not all(
            arm_valid(result, args, float(key), prior_canonical)
            for key, result in payload["results"].items()
        ):
            raise ValueError("invalid resumed LR arm")
    else:
        payload = fresh
    seed_args = argparse.Namespace(**vars(args))
    seed_args.export_dir = Path("results/gauge-ghost-gradient-lr-control-unused")
    for lr in args.learning_rates:
        key = lr_key(lr)
        if key in payload["results"]:
            continue
        seed_args.learning_rate = lr
        print(json.dumps({"starting_canonical_lr": lr}), flush=True)
        result = mechanism.train_arm(
            "canonical_baseline", seed_args, train_file, validation_file, device
        )
        result["control_seed"] = args.seed
        result["control_learning_rate"] = lr
        result["control_preregistration_sha256"] = sha256_file(PREREGISTRATION)
        if not arm_valid(
            result, args, lr, prior["arms"]["canonical_baseline"]
        ):
            raise RuntimeError(f"invalid canonical LR arm: {lr}")
        payload["results"][key] = result
        payload["completed_lrs"].append(key)
        mechanism.pilot.write_payload(args.output, payload)
    payload["decision"] = decide(payload["results"], prior, args)
    mechanism.pilot.write_payload(args.output, payload)
    return payload


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--learning-rates", type=parse_floats, default=list(NEW_LRS)
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-steps", type=lambda s: [int(v) for v in s.split(",")], default=list(EVAL_STEPS))
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--formal", action="store_true", default=True)
    payload = run(parser.parse_args())
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
