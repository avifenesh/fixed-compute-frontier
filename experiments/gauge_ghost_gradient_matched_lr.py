#!/usr/bin/env python3
"""Matched canonical/ghost LR frontier on the used development seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import transformers
import triton

import experiments.gauge_ghost_gradient_lr_control as lr_control
import experiments.gauge_ghost_gradient_replication as replication
import experiments.triangular_value_encoding_forward_mechanism_v2 as mechanism


OUTPUT = Path("results/gauge-ghost-gradient-matched-lr.json")
PREREGISTRATION = Path(
    "results/gauge-ghost-gradient-matched-lr-preregistration.md"
)
MANIFEST = Path(
    "results/gauge-ghost-gradient-matched-lr-integrity-manifest.json"
)
EXPORT_DIR = Path("results/gauge-ghost-gradient-matched-lr-exports")
JOURNAL_DIR = Path("results/gauge-ghost-gradient-matched-lr-journals")
QUARANTINE_DIR = Path("results/gauge-ghost-gradient-matched-lr-orphans")
LR_RESULT = Path("results/gauge-ghost-gradient-lr-control.json")
SEED = 21017
STEPS = 305
EVAL_STEPS = (61, 152, 305)
LR_GRID = (
    1.5e-4,
    3e-4 / math.sqrt(2.0),
    3e-4,
    3e-4 * math.sqrt(2.0),
    6e-4,
    6e-4 * math.sqrt(2.0),
    1.2e-3,
)
NEW_ARMS = (
    ("canonical_baseline", LR_GRID[5]),
    ("backward_only_tve", LR_GRID[5]),
    ("canonical_baseline", LR_GRID[6]),
    ("backward_only_tve", LR_GRID[6]),
    ("backward_only_tve", LR_GRID[0]),
    ("backward_only_tve", LR_GRID[1]),
    ("backward_only_tve", LR_GRID[3]),
    ("backward_only_tve", LR_GRID[4]),
)

DEPENDENCIES = {
    "matched_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "lr_control_source": Path(lr_control.__file__),
    "lr_control_preregistration": lr_control.PREREGISTRATION,
    "lr_control_manifest": lr_control.MANIFEST,
    "lr_control_result": LR_RESULT,
    "replication_source": Path(replication.__file__),
    "mechanism_source": Path(mechanism.__file__),
    "mechanism_preregistration": Path(
        "results/triangular-value-encoding-forward-mechanism-v2-preregistration.md"
    ),
    "mechanism_manifest": mechanism.MANIFEST,
    "mechanism_result": mechanism.OUTPUT,
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


def lr_key(lr: float) -> str:
    return format(lr, ".17g")


def result_key(arm: str, lr: float) -> str:
    return f"{arm}@{lr_key(lr)}"


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    lr_payload = json.loads(LR_RESULT.read_text())
    lr_decision = lr_payload.get("decision", {})
    mechanism_payload = json.loads(mechanism.OUTPUT.read_text())
    prior_canonical = mechanism_payload.get("arms", {}).get(
        "canonical_baseline", {}
    )
    lr_args = argparse.Namespace(
        **lr_payload.get("experiment_protocol", {}).get("actual", {})
    )
    lr_results = lr_payload.get("results", {})
    try:
        recomputed_lr_decision = lr_control.decide(
            lr_results, mechanism_payload, lr_args
        )
        reused_lr_arms_valid = all(
            lr_control.arm_valid(result, lr_args, float(key), prior_canonical)
            and result.get("initial_state_sha256")
            == prior_canonical.get("initial_state_sha256")
            and result.get("evaluations", {}).get("0", {}).get("per_batch_loss")
            == prior_canonical.get("evaluations", {}).get("0", {}).get(
                "per_batch_loss"
            )
            for key, result in lr_results.items()
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        recomputed_lr_decision = None
        reused_lr_arms_valid = False
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": manifest.get("transformers_version") == transformers.__version__,
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "lr_control_current_integrity_valid": lr_control.validate_integrity()["valid"],
        "lr_result_schema_exact": (
            lr_payload.get("schema") == "gauge-ghost-gradient-lr-control-v1"
        ),
        "lr_result_source_exact": (
            lr_payload.get("source_sha256")
            == sha256_file(DEPENDENCIES["lr_control_source"])
        ),
        "lr_result_classification_exact": (
            lr_decision.get("classification") == "endpoint_extension_required"
            and lr_decision.get("all_new_arms_structurally_valid") is True
            and lr_decision.get("best_canonical_lr") == 6e-4
            and lr_decision.get("best_canonical_loss") == 6.268180273473263
        ),
        "lr_result_decision_recomputed_exact": (
            recomputed_lr_decision == lr_decision
        ),
        "lr_result_reused_arms_valid": reused_lr_arms_valid,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid matched-LR integrity: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = mechanism.pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "output": str(OUTPUT),
        "export_dir": str(EXPORT_DIR),
        "journal_dir": str(JOURNAL_DIR),
        "quarantine_dir": str(QUARANTINE_DIR),
        "seed": SEED,
        "lr_grid": list(LR_GRID),
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
        "journal_dir": str(JOURNAL_DIR),
        "quarantine_dir": str(QUARANTINE_DIR),
        "seed": args.seed,
        "lr_grid": list(LR_GRID),
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
        raise ValueError(f"invalid matched-LR protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def expected_export_path(args: argparse.Namespace, arm: str, lr: float) -> Path:
    return (
        args.export_dir / f"lr-{lr_key(lr)}"
        / f"seed-{SEED}-{arm}-bf16-attention.pt"
    )


def artifact_valid(
    result: dict[str, Any], args: argparse.Namespace, arm: str, lr: float,
    device: torch.device,
) -> bool:
    artifact = result.get("terminal_artifact")
    if arm == "canonical_baseline":
        return artifact is None
    if not isinstance(artifact, dict):
        return False
    path = Path(artifact.get("path", ""))
    expected = expected_export_path(args, arm, lr)
    return (
        path == expected
        and path.exists()
        and path.stat().st_size == artifact.get("bytes")
        and sha256_file(path) == artifact.get("sha256")
        and replication._export_payload_valid(path, SEED, device)
    )


def arm_valid(
    result: dict[str, Any], args: argparse.Namespace, arm: str, lr: float,
    prior_canonical: dict[str, Any], device: torch.device,
) -> bool:
    evaluations = result.get("evaluations", {})
    train = result.get("train", {})
    equal_fields = (
        "total_parameters", "parameter_tensors", "model_buffer_values",
        "model_buffer_tensors", "state_dict_values", "state_dict_bytes",
        "optimizer_state_bytes", "metadata_bits",
    )
    return (
        result.get("arm") == arm
        and finite(result)
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
        and result.get("matched_lr_seed") == SEED
        and result.get("matched_learning_rate") == lr
        and result.get("matched_preregistration_sha256") == sha256_file(PREREGISTRATION)
        and result.get("initial_state_sha256") == prior_canonical.get("initial_state_sha256")
        and evaluations.get("0", {}).get("per_batch_loss")
        == prior_canonical.get("evaluations", {}).get("0", {}).get("per_batch_loss")
        and result.get("terminal_serving_bridge") == {
            "forward_semantics": "ordinary_canonical",
            "tve_cache_encoding_required": False,
        }
        and result.get("metadata_bits") == 0
        and artifact_valid(result, args, arm, lr, device)
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


def assemble_frontier(new_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mechanism_result = json.loads(mechanism.OUTPUT.read_text())
    lr_result = json.loads(LR_RESULT.read_text())
    frontier: dict[str, dict[str, Any]] = {
        "canonical_baseline": {}, "backward_only_tve": {},
    }
    frontier["canonical_baseline"][lr_key(LR_GRID[2])] = mechanism_result["arms"]["canonical_baseline"]
    frontier["backward_only_tve"][lr_key(LR_GRID[2])] = mechanism_result["arms"]["backward_only_tve"]
    for key, result in lr_result["results"].items():
        frontier["canonical_baseline"][key] = result
    for key, result in new_results.items():
        arm, lr_text = key.split("@", 1)
        frontier[arm][lr_text] = result
    return frontier


def decide(
    new_results: dict[str, Any], args: argparse.Namespace,
    prior_canonical: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    expected = {result_key(arm, lr) for arm, lr in NEW_ARMS}
    if set(new_results) != expected:
        return {"complete": False, "completed": sorted(new_results)}
    structural = all(
        arm_valid(
            new_results[result_key(arm, lr)], args, arm, lr, prior_canonical,
            device,
        )
        for arm, lr in NEW_ARMS
    )
    frontier = assemble_frontier(new_results)
    expected_lrs = {lr_key(lr) for lr in LR_GRID}
    structural = structural and all(
        set(frontier[arm]) == expected_lrs
        for arm in ("canonical_baseline", "backward_only_tve")
    )
    losses = {
        arm: {
            key: result["evaluations"]["305"]["loss"]
            for key, result in arm_results.items()
        }
        for arm, arm_results in frontier.items()
    }
    best_keys = {arm: min(arm_losses, key=arm_losses.get) for arm, arm_losses in losses.items()}
    best_lrs = {arm: float(key) for arm, key in best_keys.items()}
    best_results = {
        arm: frontier[arm][best_keys[arm]] for arm in frontier
    }
    best_losses = {arm: losses[arm][best_keys[arm]] for arm in losses}
    endpoint_keys = {lr_key(LR_GRID[0]), lr_key(LR_GRID[-1])}
    interior_keys = {lr_key(lr) for lr in LR_GRID[1:-1]}
    endpoint = any(
        min(losses[arm][key] for key in endpoint_keys)
        <= min(losses[arm][key] for key in interior_keys)
        for arm in losses
    )
    interval = paired_interval(
        best_results["backward_only_tve"], best_results["canonical_baseline"]
    )
    canonical_loss = best_losses["canonical_baseline"]
    ghost_loss = best_losses["backward_only_tve"]
    ghost_relative_gain = canonical_loss / ghost_loss - 1.0
    ghost_relative_regret = ghost_loss / canonical_loss - 1.0
    survives = (
        not endpoint
        and ghost_relative_gain >= 0.0005
        and interval["upper_95"] < 0.0
    )
    fails = not endpoint and ghost_relative_regret >= 0.0005
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
        "ghost_relative_gain": ghost_relative_gain,
        "ghost_relative_regret": ghost_relative_regret,
        "classification": classification,
        "ghost_survives_matched_lr_frontier": structural and survives,
    }


def work_order() -> list[str]:
    return [result_key(arm, lr) for arm, lr in NEW_ARMS]


def journal_path(arm: str, lr: float) -> Path:
    index = NEW_ARMS.index((arm, lr))
    return JOURNAL_DIR / f"work-{index:02d}-{arm}.json"


def journal_binding(arm: str, lr: float, status: str) -> dict[str, Any]:
    return {
        "schema": "gauge-ghost-gradient-matched-lr-journal-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity_manifest_sha256": sha256_file(MANIFEST),
        "arm": arm,
        "learning_rate": lr,
        "status": status,
    }


def write_journal(arm: str, lr: float, status: str) -> None:
    path = journal_path(arm, lr)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(journal_binding(arm, lr, status), indent=2, sort_keys=True)
        + "\n"
    )
    os.replace(temporary, path)


def read_journal(arm: str, lr: float) -> dict[str, Any] | None:
    path = journal_path(arm, lr)
    if not path.exists():
        return None
    journal = json.loads(path.read_text())
    if journal not in (
        journal_binding(arm, lr, "started"),
        journal_binding(arm, lr, "completed"),
    ):
        raise ValueError(f"invalid matched-LR journal: {path}")
    return journal


def quarantine_glob(arm: str, lr: float) -> str:
    index = NEW_ARMS.index((arm, lr))
    return f"work-{index:02d}-{arm}-orphan-*.pt"


def recovery_record(path: Path, arm: str, lr: float) -> dict[str, Any]:
    return {
        "work_key": result_key(arm, lr),
        "arm": arm,
        "learning_rate": lr,
        "quarantined_path": str(path),
        "sha256": sha256_file(path),
        "reason": "bound export existed before arm payload checkpoint",
    }


def recover_bound_orphan(
    payload: dict[str, Any], arm: str, lr: float,
    args: argparse.Namespace, device: torch.device,
) -> None:
    journal = read_journal(arm, lr)
    if journal is not None and journal["status"] == "completed":
        raise ValueError(f"completed journal without payload arm: {arm}/{lr}")
    if arm == "canonical_baseline":
        return
    export = expected_export_path(args, arm, lr)
    recorded = {
        recovery["quarantined_path"] for recovery in payload["recoveries"]
    }
    existing = sorted(QUARANTINE_DIR.glob(quarantine_glob(arm, lr)))
    if existing and (journal is None or journal["status"] != "started"):
        raise ValueError(f"unbound quarantined exports: {existing}")
    for path in existing:
        if not replication._export_payload_valid(path, SEED, device):
            raise ValueError(f"invalid quarantined export: {path}")
        if str(path) not in recorded:
            payload["recoveries"].append(recovery_record(path, arm, lr))
    if not export.exists():
        return
    if journal is None or journal["status"] != "started":
        raise ValueError(f"unbound orphan export: {export}")
    if not replication._export_payload_valid(export, SEED, device):
        raise ValueError(f"invalid orphan export: {export}")
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    quarantined = (
        QUARANTINE_DIR
        / f"work-{NEW_ARMS.index((arm, lr)):02d}-{arm}-orphan-{len(existing)}.pt"
    )
    if quarantined.exists():
        raise FileExistsError(quarantined)
    os.replace(export, quarantined)
    payload["recoveries"].append(recovery_record(quarantined, arm, lr))


def validate_resume_payload(
    payload: dict[str, Any], fresh: dict[str, Any], args: argparse.Namespace,
    prior_canonical: dict[str, Any], device: torch.device,
) -> None:
    if set(payload) != set(fresh):
        raise ValueError("matched-LR resume payload schema keys differ")
    mutable = {"results", "completed_work", "recoveries", "decision"}
    if any(
        payload.get(key) != fresh.get(key)
        for key in set(fresh) - mutable
    ):
        raise ValueError("matched-LR resume immutable binding mismatch")
    if payload.get("decision") != {"complete": False}:
        raise ValueError("only incomplete matched-LR payload can resume")
    completed = payload.get("completed_work")
    expected_order = work_order()
    if (
        not isinstance(completed, list)
        or completed != expected_order[:len(completed)]
        or not isinstance(payload.get("results"), dict)
        or len(payload["results"]) != len(completed)
        or set(payload["results"]) != set(completed)
    ):
        raise ValueError("matched-LR completed work is not an exact prefix")
    work = {result_key(arm, lr): (arm, lr) for arm, lr in NEW_ARMS}
    for key, result in payload.get("results", {}).items():
        arm, lr = work[key]
        if not arm_valid(result, args, arm, lr, prior_canonical, device):
            raise ValueError(f"invalid resumed matched-LR arm: {key}")
        journal = read_journal(arm, lr)
        if journal is None or journal["status"] not in {"started", "completed"}:
            raise ValueError(f"missing bound journal for completed arm: {key}")
    recoveries = payload.get("recoveries")
    if not isinstance(recoveries, list) or not finite(recoveries):
        raise ValueError("invalid matched-LR recovery journal")
    paths = []
    for recovery in recoveries:
        if not isinstance(recovery, dict) or set(recovery) != {
            "work_key", "arm", "learning_rate", "quarantined_path",
            "sha256", "reason",
        }:
            raise ValueError("invalid matched-LR recovery schema")
        key = recovery["work_key"]
        if key not in work or work[key] != (
            recovery["arm"], recovery["learning_rate"]
        ):
            raise ValueError(f"invalid matched-LR recovery binding: {recovery}")
        arm, lr = work[key]
        path = Path(recovery["quarantined_path"])
        if (
            arm != "backward_only_tve"
            or path.parent != QUARANTINE_DIR
            or not path.match(quarantine_glob(arm, lr))
            or recovery["reason"]
            != "bound export existed before arm payload checkpoint"
            or not path.exists()
            or sha256_file(path) != recovery["sha256"]
            or not replication._export_payload_valid(path, SEED, device)
        ):
            raise ValueError(f"invalid matched-LR recovery: {recovery}")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate matched-LR recovery paths")
    for arm, lr in NEW_ARMS:
        key = result_key(arm, lr)
        export = expected_export_path(args, arm, lr)
        completed_arm = key in payload.get("results", {})
        if arm == "canonical_baseline":
            if export.exists():
                raise ValueError(f"unexpected canonical export: {export}")
            continue
        if export.exists() != completed_arm:
            journal = read_journal(arm, lr)
            recoverable = (
                not completed_arm and export.exists()
                and journal is not None and journal["status"] == "started"
                and replication._export_payload_valid(export, SEED, device)
            )
            if not recoverable:
                raise ValueError(f"matched-LR resume export mismatch: {key}")


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
    mechanism_result = json.loads(mechanism.OUTPUT.read_text())
    prior_canonical = mechanism_result["arms"]["canonical_baseline"]
    train_file = mechanism.pilot.TokenFile(mechanism.TRAIN_FILE, args.sequence_length)
    validation_file = mechanism.pilot.TokenFile(
        mechanism.VALIDATION_FILE, args.sequence_length
    )
    device = torch.device("cuda")
    fresh_payload = {
        "schema": "gauge-ghost-gradient-matched-lr-v1",
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
        "completed_work": [],
        "recoveries": [],
        "decision": {"complete": False},
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text())
        validate_resume_payload(
            payload, fresh_payload, args, prior_canonical, device
        )
    else:
        if (
            any(JOURNAL_DIR.glob("*.json"))
            or any(QUARANTINE_DIR.glob("*.pt"))
            or any(args.export_dir.glob("**/*.pt"))
        ):
            raise ValueError("matched-LR artifacts exist without a bound payload")
        payload = fresh_payload
        mechanism.pilot.write_payload(args.output, payload)
    for arm, lr in args.new_arms:
        key = result_key(arm, lr)
        if key in payload["results"]:
            if not arm_valid(
                payload["results"][key], args, arm, lr, prior_canonical, device
            ):
                raise ValueError(f"invalid completed matched-LR arm: {key}")
            journal = read_journal(arm, lr)
            if journal is None or journal["status"] != "completed":
                write_journal(arm, lr, "completed")
            continue
        recoveries_before = len(payload["recoveries"])
        recover_bound_orphan(payload, arm, lr, args, device)
        if len(payload["recoveries"]) != recoveries_before:
            mechanism.pilot.write_payload(args.output, payload)
        write_journal(arm, lr, "started")
        run_args = argparse.Namespace(**vars(args))
        run_args.learning_rate = lr
        run_args.export_dir = args.export_dir / f"lr-{lr_key(lr)}"
        print(json.dumps({"starting_arm": arm, "learning_rate": lr}), flush=True)
        result = mechanism.train_arm(
            arm, run_args, train_file, validation_file, device
        )
        result["matched_lr_seed"] = args.seed
        result["matched_learning_rate"] = lr
        result["matched_preregistration_sha256"] = sha256_file(PREREGISTRATION)
        if not arm_valid(result, args, arm, lr, prior_canonical, device):
            raise RuntimeError(f"invalid matched-LR arm: {key}")
        payload["results"][key] = result
        payload["completed_work"].append(key)
        mechanism.pilot.write_payload(args.output, payload)
        write_journal(arm, lr, "completed")
    payload["decision"] = decide(
        payload["results"], args, prior_canonical, device
    )
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
