#!/usr/bin/env python3
"""Five-seed replication of training-only TVE gradients with ordinary serving."""

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

import experiments.triangular_value_encoding_forward_mechanism_v2 as mechanism


OUTPUT = Path("results/gauge-ghost-gradient-replication.json")
PREREGISTRATION = Path(
    "results/gauge-ghost-gradient-replication-preregistration.md"
)
MANIFEST = Path(
    "results/gauge-ghost-gradient-replication-integrity-manifest.json"
)
EXPORT_DIR = Path("results/gauge-ghost-gradient-replication-exports")
JOURNAL_DIR = Path("results/gauge-ghost-gradient-replication-journals")
QUARANTINE_DIR = Path("results/gauge-ghost-gradient-replication-orphans")
TRAIN_FILE = mechanism.TRAIN_FILE
VALIDATION_FILE = mechanism.VALIDATION_FILE
DATA_MANIFEST = mechanism.DATA_MANIFEST
SEEDS = (22003, 23003, 24007, 25013, 26003)
ARMS = ("canonical_baseline", "backward_only_tve")
ARM_ORDERS = {
    22003: ARMS,
    23003: tuple(reversed(ARMS)),
    24007: ARMS,
    25013: tuple(reversed(ARMS)),
    26003: ARMS,
}
EVAL_STEPS = (61, 152, 305)
STEPS = 305
T_CRITICAL_DF4 = 2.7764451051977987

DEPENDENCIES = {
    "replication_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "mechanism_source": Path(
        "experiments/triangular_value_encoding_forward_mechanism_v2.py"
    ),
    "mechanism_preregistration": Path(
        "results/triangular-value-encoding-forward-mechanism-v2-preregistration.md"
    ),
    "mechanism_manifest": Path(
        "results/triangular-value-encoding-forward-mechanism-v2-integrity-manifest.json"
    ),
    "mechanism_result": Path(
        "results/triangular-value-encoding-forward-mechanism-v2.json"
    ),
    "pilot_source": Path("experiments/triangular_value_encoding_lm_pilot.py"),
    "attention_source": Path("experiments/triangular_value_encoding_attention.py"),
    "attention_test": Path("tests/test_triangular_value_encoding_attention.py"),
    "data_manifest": DATA_MANIFEST,
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
    prior = json.loads(DEPENDENCIES["mechanism_result"].read_text())
    prior_decision = prior.get("decision", {})
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": (
            manifest.get("transformers_version") == transformers.__version__
        ),
        "triton_version": manifest.get("triton_version") == triton.__version__,
        "bound_mechanism_integrity_current": (
            mechanism.validate_integrity()["valid"] is True
        ),
        "bound_mechanism_screen_valid": prior_decision.get("screen_valid") is True,
        "bound_mechanism_classification_exact": (
            prior_decision.get("classification")
            == "backward_signal_captures_gain_this_seed"
        ),
        "bound_mechanism_seed_exact": prior.get("experiment_protocol", {})
        .get("actual", {}).get("seed") == 21017,
        "bound_mechanism_result_manifest_exact": (
            prior.get("formal_integrity", {}).get("manifest_sha256")
            == sha256_file(DEPENDENCIES["mechanism_manifest"])
        ),
    })
    if not all(checks.values()):
        raise ValueError(f"invalid ghost-gradient integrity: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = mechanism.pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "data_manifest": str(DATA_MANIFEST),
        "output": str(OUTPUT),
        "export_dir": str(EXPORT_DIR),
        "journal_dir": str(JOURNAL_DIR),
        "quarantine_dir": str(QUARANTINE_DIR),
        "seeds": list(SEEDS),
        "arm_orders": {str(k): list(v) for k, v in ARM_ORDERS.items()},
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
        "prediction_tokens_per_arm": 9_994_240,
        "formal": True,
        "hidden_size": 384,
        "layers": 12,
        "query_heads": 6,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 1024,
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "flash_sdp_enabled": False,
        "memory_efficient_sdp_enabled": False,
        "cudnn_sdp_enabled": False,
        "math_sdp_enabled": True,
        "fp16_bf16_reduction_math_sdp_allowed": False,
    }
    actual = {
        "device_contains": torch.cuda.get_device_name(0),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "data_manifest": str(args.data_manifest),
        "output": str(args.output),
        "export_dir": str(args.export_dir),
        "journal_dir": str(JOURNAL_DIR),
        "quarantine_dir": str(QUARANTINE_DIR),
        "seeds": args.seeds,
        "arm_orders": {str(k): list(v) for k, v in ARM_ORDERS.items()},
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
        "prediction_tokens_per_arm": (
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
        **mechanism.configure_deterministic_attention(),
    }
    checks = {
        key: (
            expected_value in actual[key]
            if key == "device_contains" else actual[key] == expected_value
        )
        for key, expected_value in expected.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid ghost-gradient protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(numbers_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numbers_finite(item) for item in value)
    return True


def _artifact_layer_valid(layer: Any) -> bool:
    expected = {
        "query_weight", "key_weight", "value_weight", "output_weight",
        "block_size", "tau", "kv_heads", "query_heads", "head_dim",
    }
    if not isinstance(layer, dict) or set(layer) != expected:
        return False
    if not all(
        isinstance(layer[name], torch.Tensor)
        and layer[name].dtype == torch.bfloat16
        and torch.isfinite(layer[name].float()).all().item()
        for name in ("query_weight", "key_weight", "value_weight", "output_weight")
    ):
        return False
    return (
        layer["block_size"] == 16 and layer["tau"] == 0.125
        and layer["kv_heads"] == 2 and layer["query_heads"] == 6
        and layer["head_dim"] == 64
        and tuple(layer["query_weight"].shape) == (384, 384)
        and tuple(layer["key_weight"].shape) == (128, 384)
        and tuple(layer["value_weight"].shape) == (128, 384)
        and tuple(layer["output_weight"].shape) == (384, 384)
    )


def _terminal_artifact_forward_identity(
    layers: list[dict[str, Any]], seed: int, device: torch.device
) -> bool:
    generator = torch.Generator(device=device).manual_seed(seed + 700_001)
    values = torch.randn(
        2, 2, 65, 64, device=device, dtype=torch.bfloat16,
        generator=generator,
    )
    for layer in layers:
        output = layer["output_weight"].to(device)
        pieces = []
        for start in range(0, 64, 16):
            block = values[..., start:start + 16]
            physical = torch.stack([
                output[
                    start:start + 16,
                    kv_head * 3 * 64 + start:kv_head * 3 * 64 + start + 16,
                ]
                for kv_head in range(2)
            ])
            coefficient = (
                torch.tril(physical, diagonal=-1).float() / 0.125
            ).to(torch.bfloat16)
            feature = (block.float() * block.float().abs()).to(torch.bfloat16)
            delta = torch.matmul(
                coefficient[None, :, None], feature.unsqueeze(-1)
            ).squeeze(-1)
            pieces.append((block.float() + delta.float()).to(torch.bfloat16))
        differentiable = torch.cat(pieces, dim=-1)
        ghost = values.detach() + (differentiable - differentiable.detach())
        if not torch.equal(ghost, values) or not torch.isfinite(differentiable).all():
            return False
    return True


def _export_payload_valid(path: Path, seed: int, device: torch.device) -> bool:
    exported = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(exported, dict) or set(exported) != {
        "seed", "arm", "forward_semantics", "layers",
    }:
        return False
    layers = exported["layers"]
    return (
        exported["seed"] == seed
        and exported["arm"] == "backward_only_tve"
        and exported["forward_semantics"] == "ordinary_canonical"
        and isinstance(layers, list) and len(layers) == 12
        and all(_artifact_layer_valid(layer) for layer in layers)
        and _terminal_artifact_forward_identity(layers, seed, device)
    )


def artifact_valid(
    result: dict[str, Any], seed: int, args: argparse.Namespace,
    device: torch.device,
) -> bool:
    artifact = result.get("terminal_artifact")
    if not isinstance(artifact, dict):
        return False
    path = Path(artifact.get("path", ""))
    expected = args.export_dir / f"seed-{seed}-backward_only_tve-bf16-attention.pt"
    basic = (
        path == expected and path.exists()
        and path.stat().st_size == artifact.get("bytes")
        and sha256_file(path) == artifact.get("sha256")
    )
    if not basic:
        return False
    return _export_payload_valid(path, seed, device)


def arm_accounting_valid(
    result: dict[str, Any], arm: str, seed: int,
    args: argparse.Namespace, device: torch.device,
) -> bool:
    expected_evaluations = {"0", *(str(step) for step in EVAL_STEPS)}
    if result.get("arm") != arm or set(result.get("evaluations", {})) != expected_evaluations:
        return False
    if not numbers_finite(result):
        return False
    for evaluation in result["evaluations"].values():
        per_batch = evaluation.get("per_batch_loss", [])
        if (
            len(per_batch) != args.eval_batches
            or evaluation.get("prediction_tokens")
            != args.eval_batches * args.eval_batch_size * args.sequence_length
            or not math.isclose(
                evaluation.get("loss", float("nan")),
                float(np.mean(per_batch)), rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            return False
    train = result.get("train", {})
    if (
        len(train.get("step_losses", [])) != args.steps
        or train.get("prediction_tokens")
        != args.steps * args.gradient_accumulation * args.micro_batch_size
        * args.sequence_length
        or train.get("nonfinite") is not False
        or train.get("tokens_per_second", 0) <= 0
        or train.get("elapsed_seconds", 0) <= 0
        or train.get("peak_allocated_bytes", 0) <= 0
    ):
        return False
    if arm == "canonical_baseline":
        return result.get("terminal_artifact") is None
    return artifact_valid(result, seed, args, device)


def seed_structural_valid(
    results: dict[str, dict[str, Any]], seed: int, args: argparse.Namespace,
    device: torch.device,
) -> bool:
    canonical = results["canonical_baseline"]
    backward = results["backward_only_tve"]
    equal_fields = (
        "total_parameters", "parameter_tensors", "model_buffer_values",
        "model_buffer_tensors", "state_dict_values", "state_dict_bytes",
        "optimizer_state_bytes", "metadata_bits",
    )
    diagnostics = backward["serving_diagnostics"]
    return (
        all(backward[field] == canonical[field] for field in equal_fields)
        and canonical["total_parameters"] == 37_758_336
        and canonical["initial_state_sha256"] == backward["initial_state_sha256"]
        and canonical["evaluations"]["0"]["per_batch_loss"]
        == backward["evaluations"]["0"]["per_batch_loss"]
        and canonical["metadata_bits"] == backward["metadata_bits"] == 0
        and backward["terminal_serving_bridge"] == {
            "forward_semantics": "ordinary_canonical",
            "tve_cache_encoding_required": False,
        }
        and diagnostics["coefficient_bf16_nonzero_fraction"] > 0
        and diagnostics["coefficient_bf16_abs_max"] <= 0.5
        and diagnostics["bf16_export_coefficient_bit_exact"]
        and not canonical["train"]["nonfinite"]
        and not backward["train"]["nonfinite"]
        and arm_accounting_valid(canonical, "canonical_baseline", seed, args, device)
        and arm_accounting_valid(backward, "backward_only_tve", seed, args, device)
    )


def seed_cluster(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    radius = T_CRITICAL_DF4 * standard_error
    return {
        "seed_means": values,
        "mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - radius,
        "upper_95": mean + radius,
        "t_critical_df4": T_CRITICAL_DF4,
    }


def paired_mean(candidate: dict[str, Any], control: dict[str, Any], step: str) -> float:
    left = np.asarray(candidate["evaluations"][step]["per_batch_loss"])
    right = np.asarray(control["evaluations"][step]["per_batch_loss"])
    return float((left - right).mean())


def decide(
    seed_results: dict[str, Any], probes: dict[str, Any], args: argparse.Namespace,
    integrity: dict[str, Any], protocol: dict[str, Any], ledger: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if set(seed_results) != {str(seed) for seed in SEEDS}:
        return {"complete": False, "completed_seeds": sorted(seed_results)}
    terminal = str(STEPS)
    trajectories = {}
    for step in EVAL_STEPS:
        key = str(step)
        trajectories[key] = seed_cluster([
            paired_mean(
                seed_results[str(seed)]["backward_only_tve"],
                seed_results[str(seed)]["canonical_baseline"],
                key,
            )
            for seed in SEEDS
        ])
    control_losses = [
        seed_results[str(seed)]["canonical_baseline"]["evaluations"][terminal]["loss"]
        for seed in SEEDS
    ]
    candidate_losses = [
        seed_results[str(seed)]["backward_only_tve"]["evaluations"][terminal]["loss"]
        for seed in SEEDS
    ]
    mean_control = float(np.mean(control_losses))
    mean_candidate = float(np.mean(candidate_losses))
    relative_improvement = (mean_control - mean_candidate) / mean_control
    terminal_cluster = trajectories[terminal]
    structural = {
        str(seed): seed_structural_valid(
            seed_results[str(seed)], seed, args, device
        )
        for seed in SEEDS
    }
    accounting = {
        str(seed): {
            arm: arm_accounting_valid(
                seed_results[str(seed)][arm], arm, seed, args, device
            )
            for arm in ARMS
        }
        for seed in SEEDS
    }
    probes_pass = {str(seed): probes[str(seed)]["pass"] for seed in SEEDS}
    throughput_ratios = [
        seed_results[str(seed)]["backward_only_tve"]["train"]["tokens_per_second"]
        / seed_results[str(seed)]["canonical_baseline"]["train"]["tokens_per_second"]
        for seed in SEEDS
    ]
    derived_numerics = {
        "control_losses": control_losses,
        "candidate_losses": candidate_losses,
        "mean_control": mean_control,
        "mean_candidate": mean_candidate,
        "relative_improvement": relative_improvement,
        "trajectories": trajectories,
        "throughput_ratios": throughput_ratios,
    }
    gates = {
        "integrity_protocol_and_data_valid": (
            integrity["valid"] and protocol["valid"] and ledger["valid"]
        ),
        "all_five_intervention_probes_pass": all(probes_pass.values()),
        "all_recorded_and_derived_numerics_finite": (
            numbers_finite(seed_results) and numbers_finite(probes)
            and numbers_finite(derived_numerics)
        ),
        "all_realized_arm_accounting_complete": all(
            passed for seed_checks in accounting.values()
            for passed in seed_checks.values()
        ),
        "all_five_structural_and_artifact_checks_pass": all(structural.values()),
        "all_five_terminal_seed_means_favor_ghost_gradient": all(
            value < 0 for value in terminal_cluster["seed_means"]
        ),
        "terminal_seed_cluster_clears_0_01_percent": (
            terminal_cluster["upper_95"] <= -0.0001 * mean_control
        ),
        "mean_relative_nll_improvement_at_least_0_05_percent": (
            relative_improvement >= 0.0005
        ),
    }
    return {
        "complete": True,
        "primary_endpoint": (
            "10M-token ordinary-serving ghost gradient versus canonical training"
        ),
        "mean_terminal_losses": {
            "canonical_baseline": mean_control,
            "backward_only_tve": mean_candidate,
        },
        "relative_terminal_nll_improvement": relative_improvement,
        "equivalent_perplexity_reduction": 1.0 - math.exp(
            mean_candidate - mean_control
        ),
        "terminal_seed_cluster": terminal_cluster,
        "trajectories": trajectories,
        "per_seed_structural": structural,
        "per_seed_arm_accounting": accounting,
        "per_seed_probe_pass": probes_pass,
        "candidate_over_control_training_throughput": {
            "per_seed": throughput_ratios,
            "mean": float(np.mean(throughput_ratios)),
            "minimum": min(throughput_ratios),
        },
        "gates": gates,
        "ghost_gradient_replication_pass": all(gates.values()),
    }


def _work_order(seeds: list[int]) -> list[list[Any]]:
    return [[seed, arm] for seed in seeds for arm in ARM_ORDERS[seed]]


def _journal_path(seed: int, arm: str) -> Path:
    return JOURNAL_DIR / f"seed-{seed}-{arm}.json"


def _journal_binding(seed: int, arm: str, status: str) -> dict[str, Any]:
    return {
        "schema": "gauge-ghost-gradient-arm-journal-v1",
        "replication_source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "integrity_manifest_sha256": sha256_file(MANIFEST),
        "seed": seed,
        "arm": arm,
        "status": status,
    }


def _write_journal(seed: int, arm: str, status: str) -> None:
    path = _journal_path(seed, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_journal_binding(seed, arm, status), indent=2, sort_keys=True)
        + "\n"
    )
    os.replace(temporary, path)


def _read_journal(seed: int, arm: str) -> dict[str, Any] | None:
    path = _journal_path(seed, arm)
    if not path.exists():
        return None
    journal = json.loads(path.read_text())
    if journal not in (
        _journal_binding(seed, arm, "started"),
        _journal_binding(seed, arm, "completed"),
    ):
        raise ValueError(f"invalid arm journal: {path}")
    return journal


def recover_bound_orphan(
    payload: dict[str, Any], seed: int, arm: str,
    args: argparse.Namespace, device: torch.device,
) -> None:
    journal = _read_journal(seed, arm)
    if journal is not None and journal["status"] == "completed":
        raise ValueError(f"completed journal without payload arm: {seed}/{arm}")
    if arm != "backward_only_tve":
        return
    export = args.export_dir / f"seed-{seed}-{arm}-bf16-attention.pt"
    recorded_paths = {
        recovery["quarantined_path"] for recovery in payload["recoveries"]
    }
    existing_quarantine = sorted(
        QUARANTINE_DIR.glob(f"seed-{seed}-{arm}-orphan-*.pt")
    )
    for path in existing_quarantine:
        if not _export_payload_valid(path, seed, device):
            raise ValueError(f"invalid quarantined export: {path}")
        if str(path) not in recorded_paths:
            payload["recoveries"].append({
                "seed": seed,
                "arm": arm,
                "quarantined_path": str(path),
                "sha256": sha256_file(path),
                "reason": "bound export existed before arm payload checkpoint",
            })
    if not export.exists():
        return
    if journal is None or journal["status"] != "started":
        raise ValueError(f"unbound orphan export: {export}")
    if not _export_payload_valid(export, seed, device):
        raise ValueError(f"invalid orphan export: {export}")
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = len(existing_quarantine)
    quarantined = QUARANTINE_DIR / f"seed-{seed}-{arm}-orphan-{suffix}.pt"
    if quarantined.exists():
        raise FileExistsError(quarantined)
    os.replace(export, quarantined)
    payload["recoveries"].append({
        "seed": seed,
        "arm": arm,
        "quarantined_path": str(quarantined),
        "sha256": sha256_file(quarantined),
        "reason": "bound export existed before arm payload checkpoint",
    })


def validate_resume_payload(
    payload: dict[str, Any], fresh: dict[str, Any],
    args: argparse.Namespace, device: torch.device,
) -> None:
    if set(payload) != set(fresh):
        raise ValueError("resume payload schema keys differ")
    immutable = set(fresh) - {
        "probes", "seeds", "decision", "completed_work", "recoveries",
    }
    mismatched = [key for key in immutable if payload.get(key) != fresh.get(key)]
    if mismatched:
        raise ValueError(f"resume payload binding mismatch: {mismatched}")
    if payload.get("decision") != {"complete": False}:
        raise ValueError("only an incomplete payload can resume")
    seed_keys = list(payload.get("seeds", {}))
    probe_keys = list(payload.get("probes", {}))
    if seed_keys != probe_keys or seed_keys != [
        str(seed) for seed in args.seeds[:len(seed_keys)]
    ]:
        raise ValueError("resume seeds/probes are not an ordered prefix")
    for index, seed_text in enumerate(seed_keys):
        seed = int(seed_text)
        probe = payload["probes"][seed_text]
        if probe.get("pass") is not True or not numbers_finite(probe):
            raise ValueError(f"invalid stored probe for seed {seed}")
        results = payload["seeds"][seed_text]
        expected_order = ARM_ORDERS[seed]
        if (
            len(results) > len(expected_order)
            or set(results) != set(expected_order[:len(results)])
        ):
            raise ValueError(f"invalid resume arm prefix at seed {seed}")
        if index < len(seed_keys) - 1 and set(results) != set(ARMS):
            raise ValueError(f"nonterminal resume seed {seed} is incomplete")
        for arm, result in results.items():
            if not arm_accounting_valid(result, arm, seed, args, device):
                raise ValueError(f"invalid stored arm {seed}/{arm}")
    expected_completed = _work_order(args.seeds)
    completed = payload.get("completed_work")
    if (
        not isinstance(completed, list)
        or completed != expected_completed[:len(completed)]
    ):
        raise ValueError("completed-work journal is not an exact ordered prefix")
    realized = [
        [seed, arm]
        for seed in args.seeds
        for arm in ARM_ORDERS[seed]
        if arm in payload.get("seeds", {}).get(str(seed), {})
    ]
    if realized != completed:
        raise ValueError("completed-work journal disagrees with stored arms")
    if not isinstance(payload.get("recoveries"), list) or not numbers_finite(
        payload["recoveries"]
    ):
        raise ValueError("invalid recovery journal")
    recovery_paths = []
    for recovery in payload["recoveries"]:
        if not isinstance(recovery, dict) or set(recovery) != {
            "seed", "arm", "quarantined_path", "sha256", "reason",
        }:
            raise ValueError("invalid recovery record schema")
        seed = recovery["seed"]
        path = Path(recovery["quarantined_path"])
        if (
            seed not in args.seeds or recovery["arm"] != "backward_only_tve"
            or path.parent != QUARANTINE_DIR
            or not path.name.startswith(
                f"seed-{seed}-backward_only_tve-orphan-"
            )
            or recovery["reason"]
            != "bound export existed before arm payload checkpoint"
            or not path.exists() or sha256_file(path) != recovery["sha256"]
            or not _export_payload_valid(path, seed, device)
        ):
            raise ValueError(f"invalid recovery record: {recovery}")
        recovery_paths.append(str(path))
    if len(recovery_paths) != len(set(recovery_paths)):
        raise ValueError("duplicate recovery paths")
    for seed in args.seeds:
        expected_export = (
            args.export_dir / f"seed-{seed}-backward_only_tve-bf16-attention.pt"
        )
        arm_completed = (
            "backward_only_tve" in payload.get("seeds", {}).get(str(seed), {})
        )
        if expected_export.exists() != arm_completed:
            journal = _read_journal(seed, "backward_only_tve")
            recoverable_orphan = (
                not arm_completed and expected_export.exists()
                and journal is not None and journal["status"] == "started"
                and _export_payload_valid(expected_export, seed, device)
            )
            if not recoverable_orphan:
                raise ValueError(f"resume export mismatch for seed {seed}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    determinism = mechanism.configure_deterministic_attention()
    integrity = validate_integrity()
    protocol = validate_protocol(args)
    ledger = mechanism.pilot.validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    train_file = mechanism.pilot.TokenFile(args.train_file, args.sequence_length)
    validation_file = mechanism.pilot.TokenFile(
        args.validation_file, args.sequence_length
    )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    fresh_payload = {
        "schema": "gauge-ghost-gradient-replication-v1",
        "scope": "five untouched initialization seeds with ordinary serving",
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
        "probes": {},
        "seeds": {},
        "completed_work": [],
        "recoveries": [],
        "decision": {"complete": False},
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text())
        validate_resume_payload(payload, fresh_payload, args, device)
    else:
        payload = fresh_payload
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        probe = mechanism.first_backward_probe(train_file, seed_args, device)
        if not probe["pass"]:
            raise RuntimeError(f"invalid intervention at seed {seed}: {probe['gates']}")
        if str(seed) in payload["probes"]:
            if payload["probes"][str(seed)] != probe:
                raise ValueError(f"recomputed probe mismatch at seed {seed}")
        else:
            payload["probes"][str(seed)] = probe
            payload["seeds"][str(seed)] = {}
        for arm in ARM_ORDERS[seed]:
            if arm in payload["seeds"][str(seed)]:
                if not arm_accounting_valid(
                    payload["seeds"][str(seed)][arm], arm, seed, args, device
                ):
                    raise ValueError(f"invalid completed arm {seed}/{arm}")
                journal = _read_journal(seed, arm)
                if journal is None or journal["status"] != "completed":
                    _write_journal(seed, arm, "completed")
                continue
            recoveries_before = len(payload["recoveries"])
            recover_bound_orphan(payload, seed, arm, args, device)
            if len(payload["recoveries"]) != recoveries_before:
                mechanism.pilot.write_payload(args.output, payload)
            _write_journal(seed, arm, "started")
            print(json.dumps({"seed": seed, "starting_ghost_arm": arm}), flush=True)
            payload["seeds"][str(seed)][arm] = mechanism.train_arm(
                arm, seed_args, train_file, validation_file, device
            )
            payload["completed_work"].append([seed, arm])
            mechanism.pilot.write_payload(args.output, payload)
            _write_journal(seed, arm, "completed")
    payload["decision"] = decide(
        payload["seeds"], payload["probes"], args,
        integrity, protocol, ledger, device,
    )
    mechanism.pilot.write_payload(args.output, payload)
    return payload


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
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
    payload = run(parser.parse_args())
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
