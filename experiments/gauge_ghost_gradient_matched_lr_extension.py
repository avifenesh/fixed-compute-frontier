#!/usr/bin/env python3
"""Upper extension of the matched canonical/ghost LR frontier."""

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

import experiments.gauge_ghost_gradient_matched_lr as matched
import experiments.gauge_ghost_gradient_replication as replication
import experiments.triangular_value_encoding_forward_mechanism_v2 as mechanism


OUTPUT = Path("results/gauge-ghost-gradient-matched-lr-extension.json")
PREREGISTRATION = Path(
    "results/gauge-ghost-gradient-matched-lr-extension-preregistration.md"
)
MANIFEST = Path(
    "results/gauge-ghost-gradient-matched-lr-extension-integrity-manifest.json"
)
PRIOR_RESULT = matched.OUTPUT
EXPORT_DIR = Path("results/gauge-ghost-gradient-matched-lr-extension-exports")
SEED = 21017
STEPS = 305
EVAL_STEPS = (61, 152, 305)
EXTENSION_LRS = (
    1.2e-3 * math.sqrt(2.0),
    2.4e-3,
    2.4e-3 * math.sqrt(2.0),
    4.8e-3,
)
NEW_ARMS = tuple(
    (arm, lr)
    for lr in reversed(EXTENSION_LRS)
    for arm in ("canonical_baseline", "backward_only_tve")
)
FULL_GRID = matched.LR_GRID + EXTENSION_LRS

DEPENDENCIES = {
    "extension_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "matched_source": Path(matched.__file__),
    "matched_preregistration": matched.PREREGISTRATION,
    "matched_manifest": matched.MANIFEST,
    "matched_result": PRIOR_RESULT,
    "replication_source": Path(replication.__file__),
    "mechanism_source": Path(mechanism.__file__),
    "mechanism_manifest": mechanism.MANIFEST,
    "mechanism_result": mechanism.OUTPUT,
    "data_manifest": mechanism.DATA_MANIFEST,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def lr_key(lr: float) -> str:
    return format(lr, ".17g")


def result_key(arm: str, lr: float) -> str:
    return f"{arm}@{lr_key(lr)}"


def finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def prior_args(prior: dict[str, Any]) -> argparse.Namespace:
    actual = dict(prior["experiment_protocol"]["actual"])
    actual["output"] = Path(actual["output"])
    actual["export_dir"] = Path(actual["export_dir"])
    actual["new_arms"] = tuple(tuple(item) for item in actual["new_arms"])
    return argparse.Namespace(**actual)


def validate_integrity(device: torch.device) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    prior = json.loads(PRIOR_RESULT.read_text())
    mechanism_result = json.loads(mechanism.OUTPUT.read_text())
    p_args = prior_args(prior)
    try:
        recomputed = matched.decide(
            prior.get("results", {}), p_args,
            mechanism_result["arms"]["canonical_baseline"], device,
        )
    except (KeyError, TypeError, ValueError):
        recomputed = None
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": manifest.get("transformers_version") == transformers.__version__,
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "matched_current_integrity_valid": matched.validate_integrity()["valid"],
        "prior_schema_exact": (
            prior.get("schema") == "gauge-ghost-gradient-matched-lr-v1"
        ),
        "prior_source_exact": (
            prior.get("source_sha256") == sha256_file(DEPENDENCIES["matched_source"])
        ),
        "prior_decision_recomputed_exact": recomputed == prior.get("decision"),
        "prior_endpoint_exact": (
            prior.get("decision", {}).get("classification")
            == "endpoint_extension_required"
            and prior.get("decision", {}).get("best_learning_rates") == {
                "backward_only_tve": 0.0012,
                "canonical_baseline": 0.0012,
            }
            and prior.get("decision", {}).get("best_losses") == {
                "backward_only_tve": 6.102745749056339,
                "canonical_baseline": 6.099084354937077,
            }
        ),
    })
    if not all(checks.values()):
        raise ValueError(f"invalid matched-LR extension integrity: {checks}")
    return {
        "valid": True, "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = mechanism.pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "output": str(OUTPUT),
        "export_dir": str(EXPORT_DIR),
        "seed": SEED,
        "extension_lrs": list(EXTENSION_LRS),
        "new_arms": [[arm, lr] for arm, lr in NEW_ARMS],
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
        "prediction_tokens_per_arm": 9_994_240,
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
        "export_dir": str(args.export_dir),
        "seed": args.seed,
        "extension_lrs": list(EXTENSION_LRS),
        "new_arms": [[arm, lr] for arm, lr in args.new_arms],
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
        "prediction_tokens_per_arm": (
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
        raise ValueError(f"invalid matched-LR extension protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def expected_export_path(args: argparse.Namespace, arm: str, lr: float) -> Path:
    return (
        args.export_dir / f"lr-{lr_key(lr)}"
        / f"seed-{SEED}-{arm}-bf16-attention.pt"
    )


def arm_valid(
    result: dict[str, Any], args: argparse.Namespace, arm: str, lr: float,
    prior_canonical: dict[str, Any], device: torch.device,
) -> bool:
    evaluations = result.get("evaluations", {})
    train = result.get("train", {})
    artifact = result.get("terminal_artifact")
    export_ok = artifact is None if arm == "canonical_baseline" else (
        isinstance(artifact, dict)
        and Path(artifact.get("path", "")) == expected_export_path(args, arm, lr)
        and expected_export_path(args, arm, lr).exists()
        and expected_export_path(args, arm, lr).stat().st_size == artifact.get("bytes")
        and sha256_file(expected_export_path(args, arm, lr)) == artifact.get("sha256")
        and replication._export_payload_valid(
            expected_export_path(args, arm, lr), SEED, device
        )
    )
    equal_fields = (
        "total_parameters", "parameter_tensors", "model_buffer_values",
        "model_buffer_tensors", "state_dict_values", "state_dict_bytes",
        "optimizer_state_bytes", "metadata_bits",
    )
    return (
        result.get("arm") == arm and finite(result)
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
        and result.get("total_parameters") == 37_758_336
        and all(result.get(field) == prior_canonical.get(field) for field in equal_fields)
        and result.get("extension_seed") == SEED
        and result.get("extension_learning_rate") == lr
        and result.get("extension_preregistration_sha256") == sha256_file(PREREGISTRATION)
        and result.get("initial_state_sha256") == prior_canonical.get("initial_state_sha256")
        and evaluations.get("0", {}).get("per_batch_loss")
        == prior_canonical.get("evaluations", {}).get("0", {}).get("per_batch_loss")
        and result.get("terminal_serving_bridge") == {
            "forward_semantics": "ordinary_canonical",
            "tve_cache_encoding_required": False,
        }
        and result.get("metadata_bits") == 0 and export_ok
    )


def paired_interval(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    left = np.asarray(candidate["evaluations"]["305"]["per_batch_loss"], dtype=np.float64)
    right = np.asarray(reference["evaluations"]["305"]["per_batch_loss"], dtype=np.float64)
    delta = left - right
    mean = float(np.mean(delta))
    standard_error = float(np.std(delta, ddof=1) / math.sqrt(delta.size))
    return {
        "ghost_minus_canonical_mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
        "upper_95": mean + 1.96 * standard_error,
    }


def decide(
    results: dict[str, Any], args: argparse.Namespace,
    prior_canonical: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    expected = {result_key(arm, lr) for arm, lr in NEW_ARMS}
    if set(results) != expected:
        return {"complete": False}
    structural = all(
        arm_valid(results[result_key(arm, lr)], args, arm, lr, prior_canonical, device)
        for arm, lr in NEW_ARMS
    )
    prior = json.loads(PRIOR_RESULT.read_text())
    frontier = matched.assemble_frontier(prior["results"])
    for key, result in results.items():
        arm, lr_text = key.split("@", 1)
        frontier[arm][lr_text] = result
    expected_lrs = {lr_key(lr) for lr in FULL_GRID}
    structural = structural and all(set(frontier[arm]) == expected_lrs for arm in frontier)
    losses = {
        arm: {key: value["evaluations"]["305"]["loss"] for key, value in values.items()}
        for arm, values in frontier.items()
    }
    best_keys = {arm: min(values, key=values.get) for arm, values in losses.items()}
    best_lrs = {arm: float(key) for arm, key in best_keys.items()}
    best_losses = {arm: losses[arm][best_keys[arm]] for arm in losses}
    endpoint_key = lr_key(FULL_GRID[-1])
    interior_keys = {lr_key(lr) for lr in FULL_GRID[:-1]}
    endpoint = any(
        losses[arm][endpoint_key] <= min(losses[arm][key] for key in interior_keys)
        for arm in losses
    )
    interval = paired_interval(
        frontier["backward_only_tve"][best_keys["backward_only_tve"]],
        frontier["canonical_baseline"][best_keys["canonical_baseline"]],
    )
    canonical_loss = best_losses["canonical_baseline"]
    ghost_loss = best_losses["backward_only_tve"]
    gain = canonical_loss / ghost_loss - 1.0
    regret = ghost_loss / canonical_loss - 1.0
    survives = not endpoint and gain >= 0.0005 and interval["upper_95"] < 0.0
    fails = not endpoint and regret >= 0.0005
    if endpoint:
        classification = "endpoint_extension_required"
    elif fails:
        classification = "ghost_fails_matched_lr_frontier"
    elif survives:
        classification = "ghost_survives_matched_lr_frontier"
    else:
        classification = "inconclusive_matched_lr_frontier"
    if not structural:
        classification = "invalid_control"
    return {
        "complete": True,
        "all_new_arms_structurally_valid": structural,
        "terminal_losses": losses,
        "best_learning_rates": best_lrs,
        "best_losses": best_losses,
        "best_is_grid_endpoint": endpoint,
        "paired_best_terminal_diagnostic": interval,
        "ghost_relative_gain": gain,
        "ghost_relative_regret": regret,
        "classification": classification,
        "ghost_survives_matched_lr_frontier": structural and survives,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    expected_exports = [
        expected_export_path(args, arm, lr)
        for arm, lr in args.new_arms if arm == "backward_only_tve"
    ]
    if args.output.exists() or any(path.exists() for path in expected_exports):
        raise FileExistsError([str(args.output), *(str(path) for path in expected_exports)])
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    determinism = mechanism.configure_deterministic_attention()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    integrity = validate_integrity(device)
    protocol = validate_protocol(args)
    ledger = mechanism.pilot.validate_data_ledger(
        mechanism.DATA_MANIFEST, mechanism.TRAIN_FILE,
        mechanism.VALIDATION_FILE, args.sequence_length,
    )
    mechanism_result = json.loads(mechanism.OUTPUT.read_text())
    prior_canonical = mechanism_result["arms"]["canonical_baseline"]
    train_file = mechanism.pilot.TokenFile(mechanism.TRAIN_FILE, args.sequence_length)
    validation_file = mechanism.pilot.TokenFile(mechanism.VALIDATION_FILE, args.sequence_length)
    payload = {
        "schema": "gauge-ghost-gradient-matched-lr-extension-v1",
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
        "decision": {"complete": False},
    }
    for arm, lr in args.new_arms:
        run_args = argparse.Namespace(**vars(args))
        run_args.learning_rate = lr
        run_args.export_dir = args.export_dir / f"lr-{lr_key(lr)}"
        print(json.dumps({"starting_arm": arm, "learning_rate": lr}), flush=True)
        result = mechanism.train_arm(arm, run_args, train_file, validation_file, device)
        result["extension_seed"] = args.seed
        result["extension_learning_rate"] = lr
        result["extension_preregistration_sha256"] = sha256_file(PREREGISTRATION)
        if not arm_valid(result, args, arm, lr, prior_canonical, device):
            raise RuntimeError(f"invalid extension arm: {result_key(arm, lr)}")
        payload["results"][result_key(arm, lr)] = result
    payload["decision"] = decide(payload["results"], args, prior_canonical, device)
    mechanism.pilot.write_payload(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--new-arms", default=NEW_ARMS)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-steps", default=list(EVAL_STEPS))
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--formal", action="store_true", default=True)
    payload = run(parser.parse_args())
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
