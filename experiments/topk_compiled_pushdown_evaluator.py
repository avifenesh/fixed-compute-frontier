#!/usr/bin/env python3
"""Post-checkpoint materializer for the sealed CPKV-TOPK-001 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    from experiments.topk_compiled_pushdown_reference import (
        _atomic_write_text,
        _build_split,
    )
except ModuleNotFoundError:  # Direct `python experiments/...py` execution.
    from topk_compiled_pushdown_reference import (  # type: ignore[no-redef]
        _atomic_write_text,
        _build_split,
    )


FROZEN_FAMILIES = ("D1", "A1", "A2")
FROZEN_ARMS = ("S32", "DIRECT3", "HARD3", "MLP", "RNN32", "LINK32")
FROZEN_SEEDS = tuple(range(1701, 1711))
REQUIRED_SOURCE_LABELS = {
    "candidate",
    "evaluator",
    "evaluator_tests",
    "locked_evaluation",
    "locked_evaluation_tests",
    "preregistration",
    "reference",
    "reference_tests",
    "training",
    "training_tests",
    "preflight",
    "preflight_tests",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _resolve_inside(project_root: Path, relative: str, *, label: str) -> Path:
    root = project_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes project root")
    return path


def _verify_frozen_sources(
    project_root: Path, source_hashes: object
) -> None:
    if not isinstance(source_hashes, dict):
        raise ValueError("manifest has no source hashes")
    if not REQUIRED_SOURCE_LABELS.issubset(source_hashes):
        missing = sorted(REQUIRED_SOURCE_LABELS - set(source_hashes))
        raise ValueError(f"manifest source hashes incomplete: {missing}")
    for label, record in source_hashes.items():
        if not isinstance(record, dict):
            raise ValueError(f"source record is not an object: {label}")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"source path/hash missing: {label}")
        path = _resolve_inside(project_root, relative, label=f"source {label}")
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"frozen source hash mismatch: {label}")


def _verify_runtime(manifest: Mapping[str, object]) -> None:
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("manifest environment missing")
    actual = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch_installed": importlib.metadata.version("torch"),
    }
    for name, value in actual.items():
        if environment.get(name) != value:
            raise ValueError(
                f"runtime mismatch for {name}: "
                f"manifest={environment.get(name)!r} current={value!r}"
            )
    if environment.get("torch_required") != actual["torch_installed"]:
        raise ValueError("current torch does not satisfy frozen requirement")


def _verify_oracle(project_root: Path, manifest: Mapping[str, object]) -> None:
    record = manifest.get("oracle_receipt")
    if not isinstance(record, dict):
        raise ValueError("manifest oracle receipt missing")
    relative = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError("manifest oracle path/hash missing")
    path = _resolve_inside(project_root, relative, label="oracle receipt")
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError("oracle receipt hash mismatch")
    oracle = _read_json(path)
    x4 = oracle.get("x4")
    expected_x4 = {
        "top3_history_mass": 0.75,
        "discarded_mass": 0.25,
        "read_error": 1.0 / math.sqrt(12.0),
        "bound": 0.5,
    }
    if (
        oracle.get("lineage") != "CPKV-TOPK-001"
        or oracle.get("status") != "pass"
        or oracle.get("max_length") != 6
        or oracle.get("checked_sequences") != 127
        or not isinstance(x4, dict)
        or any(
            isinstance(x4.get(name), bool)
            or not isinstance(x4.get(name), (int, float))
            or not math.isclose(
                float(x4[name]), value, rel_tol=0.0, abs_tol=1e-15
            )
            for name, value in expected_x4.items()
        )
    ):
        raise ValueError("oracle receipt contract mismatch")


def _verify_preflight(project_root: Path, manifest: Mapping[str, object]) -> None:
    record = manifest.get("preflight_receipt")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(record, dict) or not isinstance(source_hashes, dict):
        raise ValueError("manifest preflight receipt missing")
    relative = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError("manifest preflight path/hash missing")
    path = _resolve_inside(project_root, relative, label="preflight receipt")
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError("preflight receipt hash mismatch")
    preflight = _read_json(path)
    if (
        preflight.get("schema") != "cpkv-topk-001-preflight-v1"
        or preflight.get("lineage") != "CPKV-TOPK-001"
        or preflight.get("status") != "pass"
        or preflight.get("seed") != 1701
        or preflight.get("base_sample_count") != 512
        or preflight.get("projection_limit_hours") != 5.7
    ):
        raise ValueError("preflight receipt contract mismatch")
    families = preflight.get("families")
    if not isinstance(families, dict) or set(families) != set(FROZEN_FAMILIES):
        raise ValueError("preflight family ledger mismatch")
    for family, raw in families.items():
        if not isinstance(raw, dict):
            raise ValueError(f"invalid preflight family: {family}")
        hours = raw.get("projected_all_arm_cpu_hours")
        if (
            raw.get("status") != "pass"
            or isinstance(hours, bool)
            or not isinstance(hours, (int, float))
            or not math.isfinite(float(hours))
            or float(hours) > 5.7
            or raw.get("projection_limit_hours") != 5.7
            or raw.get("actual_freeze_limit_hours") != 6.0
        ):
            raise ValueError(f"preflight resource gate failed: {family}")
    preflight_hashes = preflight.get("source_hashes")
    required = {"preflight", "preflight_tests", "training", "locked_evaluation"}
    if not isinstance(preflight_hashes, dict) or set(preflight_hashes) != required:
        raise ValueError("preflight source seal mismatch")
    for label in required:
        if preflight_hashes[label] != source_hashes.get(label):
            raise ValueError(f"preflight source binding mismatch: {label}")


def _parameter_shapes(family: str, arm: str, vocabulary_size: int) -> dict[str, tuple[int, ...]]:
    common = {
        "E": (vocabulary_size, 64),
        "W_e": (64, 64),
        "W_h": (64, 64),
        "W_r": (64, 64),
        "b_h": (64,),
        "W_y": (vocabulary_size, 64),
        "U_y": (vocabulary_size, 64),
        "b_y": (vocabulary_size,),
    }
    if arm in {"S32", "DIRECT3", "HARD3", "LINK32"}:
        module = {
            "W_a": (15, 64),
            "T": (3, 4, 15),
            "W_v": (64, 64),
        }
    elif arm == "MLP":
        module = {
            "W1": (40, 64),
            "W2": (64, 40),
            "b1": (40,),
            "b2": (64,),
            "gains": (12,),
        }
    elif arm == "RNN32":
        module = {
            "A": (32, 64),
            "B": (32, 32),
            "C": (64, 32),
            "b_s": (32,),
            "b_r": (64,),
            "gains": (20,),
        }
    else:
        raise ValueError(f"unknown arm in checkpoint schema: {family}/{arm}")
    return {**common, **module}


def _earliest_minimum_epoch(values: list[float]) -> int:
    minimum = min(values)
    for index, value in enumerate(values, start=1):
        if value <= minimum + 1e-8:
            return index
    raise AssertionError("unreachable empty development-loss list")


def _validate_checkpoint_run(
    *,
    project_root: Path,
    manifest_sha256: str,
    manifest: Mapping[str, object],
    key: str,
    receipt_record: Mapping[str, object],
) -> Mapping[str, object]:
    family, arm, seed_text = key.split("/")
    seed = int(seed_text)
    relative = receipt_record.get("path")
    expected_hash = receipt_record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError(f"run receipt path/hash missing: {key}")
    run_path = _resolve_inside(project_root, relative, label=f"run receipt {key}")
    if not run_path.is_file() or _sha256(run_path) != expected_hash:
        raise ValueError(f"run receipt hash mismatch: {key}")
    run = _read_json(run_path)
    if run.get("schema") != "cpkv-topk-001-run-v1":
        raise ValueError(f"run schema mismatch: {key}")
    if run.get("lineage") != "CPKV-TOPK-001":
        raise ValueError(f"run lineage mismatch: {key}")
    if run.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"run manifest hash mismatch: {key}")
    if (run.get("family"), run.get("arm"), run.get("seed")) != (
        family,
        arm,
        seed,
    ):
        raise ValueError(f"run identity mismatch: {key}")
    if run.get("completed_epochs") != 20 or run.get("optimizer_steps") != 6_260:
        raise ValueError(f"run did not complete frozen training budget: {key}")
    if run.get("training_examples_seen") != 400_000:
        raise ValueError(f"run example count mismatch: {key}")

    token_maps = manifest.get("token_maps")
    if not isinstance(token_maps, dict) or not isinstance(token_maps.get(family), list):
        raise ValueError(f"token map missing: {key}")
    expected_shapes = _parameter_shapes(family, arm, len(token_maps[family]))

    epoch_checkpoints = run.get("epoch_checkpoints")
    if not isinstance(epoch_checkpoints, list) or len(epoch_checkpoints) != 20:
        raise ValueError(f"run needs 20 epoch checkpoints: {key}")
    epoch_paths: set[Path] = set()
    for epoch, raw_checkpoint in enumerate(epoch_checkpoints, start=1):
        if not isinstance(raw_checkpoint, dict):
            raise ValueError(f"invalid epoch checkpoint record: {key}/{epoch}")
        relative = raw_checkpoint.get("path")
        expected_hash = raw_checkpoint.get("sha256")
        checkpoint_nll = raw_checkpoint.get("development_nll")
        if (
            raw_checkpoint.get("epoch") != epoch
            or not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or isinstance(checkpoint_nll, bool)
            or not isinstance(checkpoint_nll, (int, float))
            or not math.isfinite(float(checkpoint_nll))
        ):
            raise ValueError(f"invalid epoch checkpoint metadata: {key}/{epoch}")
        checkpoint_path = _resolve_inside(
            project_root, relative, label=f"epoch checkpoint {key}/{epoch}"
        )
        if checkpoint_path in epoch_paths:
            raise ValueError(f"duplicate epoch checkpoint path: {key}/{epoch}")
        epoch_paths.add(checkpoint_path)
        if not checkpoint_path.is_file() or _sha256(checkpoint_path) != expected_hash:
            raise ValueError(f"epoch checkpoint hash mismatch: {key}/{epoch}")
        with np.load(checkpoint_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected_shapes):
                raise ValueError(f"epoch weight names mismatch: {key}/{epoch}")
            for name, shape in expected_shapes.items():
                array = arrays[name]
                if array.shape != shape or array.dtype != np.float32:
                    raise ValueError(
                        f"epoch weight shape/dtype mismatch: {key}/{epoch}/{name}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"nonfinite epoch weights: {key}/{epoch}/{name}")

    losses_raw = run.get("development_nlls")
    if not isinstance(losses_raw, list) or len(losses_raw) != 20:
        raise ValueError(f"run needs 20 development losses: {key}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in losses_raw
    ):
        raise ValueError(f"invalid development-loss history: {key}")
    losses = [float(value) for value in losses_raw]
    selected_epoch = _earliest_minimum_epoch(losses)
    if run.get("selected_epoch") != selected_epoch:
        raise ValueError(f"selected epoch is not earliest development minimum: {key}")
    selected_nll = run.get("development_nll")
    if (
        isinstance(selected_nll, bool)
        or not isinstance(selected_nll, (int, float))
        or not math.isclose(float(selected_nll), losses[selected_epoch - 1], abs_tol=1e-12)
    ):
        raise ValueError(f"selected development NLL mismatch: {key}")
    for epoch, (loss, checkpoint) in enumerate(
        zip(losses, epoch_checkpoints, strict=True), start=1
    ):
        if not math.isclose(
            float(checkpoint["development_nll"]), loss, abs_tol=1e-12
        ):
            raise ValueError(f"epoch checkpoint NLL mismatch: {key}/{epoch}")
    if receipt_record.get("selected_epoch") != selected_epoch or not math.isclose(
        float(receipt_record.get("development_nll", math.nan)),
        losses[selected_epoch - 1],
        abs_tol=1e-12,
    ):
        raise ValueError(f"outer receipt selection mismatch: {key}")

    fixed_splits = manifest.get("fixed_splits")
    if not isinstance(fixed_splits, dict):
        raise ValueError("manifest has no fixed split mapping")
    train_record = fixed_splits.get(f"{family}/train/{seed}")
    development_record = fixed_splits.get(f"{family}/development")
    if not isinstance(train_record, dict) or not isinstance(development_record, dict):
        raise ValueError(f"manifest data record missing: {key}")
    if run.get("training_data_sha256") != train_record.get("sha256"):
        raise ValueError(f"training data hash mismatch: {key}")
    if run.get("development_data_sha256") != development_record.get("sha256"):
        raise ValueError(f"development data hash mismatch: {key}")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("manifest environment missing")
    if run.get("torch_version") != environment.get("torch_installed"):
        raise ValueError(f"run torch version mismatch: {key}")
    if run.get("numpy_version") != environment.get("numpy"):
        raise ValueError(f"run NumPy version mismatch: {key}")

    weights_relative = run.get("weights_path")
    weights_hash = run.get("weights_sha256")
    if not isinstance(weights_relative, str) or not isinstance(weights_hash, str):
        raise ValueError(f"weights path/hash missing: {key}")
    weights_path = _resolve_inside(
        project_root, weights_relative, label=f"weights {key}"
    )
    if not weights_path.is_file() or _sha256(weights_path) != weights_hash:
        raise ValueError(f"weights hash mismatch: {key}")
    selected_record = epoch_checkpoints[selected_epoch - 1]
    if (
        weights_relative != selected_record.get("path")
        or weights_hash != selected_record.get("sha256")
    ):
        raise ValueError(f"selected weights are not selected epoch checkpoint: {key}")

    for metric in ("wall_seconds", "cpu_seconds", "peak_rss_kb"):
        value = run.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"invalid resource measurement {metric}: {key}")

    with np.load(weights_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected_shapes):
            raise ValueError(f"weight names mismatch: {key}")
        for name, shape in expected_shapes.items():
            array = arrays[name]
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f"weight shape/dtype mismatch: {key}/{name}")
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite weights: {key}/{name}")
    expected_count = 193 * len(token_maps[family]) + 17_588
    actual_count = sum(math.prod(shape) for shape in expected_shapes.values())
    if actual_count != expected_count:
        raise ValueError(f"checkpoint parameter count mismatch: {key}")
    return run


def _enforce_resource_limits(resource_cells: Mapping[str, Mapping[str, float]]) -> None:
    for key, cell in resource_cells.items():
        if cell["cpu_seconds"] > 6.0 * 60.0 * 60.0:
            raise ValueError(f"six CPU-hour acquisition budget exceeded: {key}")
        if cell["peak_rss_kb"] > 32.0 * 1024.0 * 1024.0:
            raise ValueError(f"32 GiB acquisition memory budget exceeded: {key}")


def _verify_recorded_resources(
    receipt: Mapping[str, object],
    resource_cells: Mapping[str, Mapping[str, float]],
) -> None:
    recorded = receipt.get("resource_cells")
    if not isinstance(recorded, dict) or set(recorded) != set(resource_cells):
        raise ValueError("checkpoint receipt resource-cell mapping mismatch")
    for key, actual in resource_cells.items():
        raw = recorded.get(key)
        if not isinstance(raw, dict):
            raise ValueError(f"invalid recorded resource cell: {key}")
        for metric in ("cpu_seconds", "peak_rss_kb"):
            value = raw.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(
                    float(value), actual[metric], rel_tol=0.0, abs_tol=1e-9
                )
            ):
                raise ValueError(f"recorded resource mismatch: {key}/{metric}")
    total = receipt.get("total_cpu_seconds")
    actual_total = math.fsum(
        cell["cpu_seconds"] for cell in resource_cells.values()
    )
    if (
        isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not math.isclose(float(total), actual_total, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("checkpoint receipt total CPU time mismatch")


def build_checkpoint_freeze_receipt(
    *, project_root: Path, manifest_path: Path, output_path: Path
) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    if manifest.get("lineage") != "CPKV-TOPK-001":
        raise ValueError("manifest lineage mismatch")
    _verify_frozen_sources(project_root, manifest.get("source_hashes"))
    _verify_runtime(manifest)
    _verify_oracle(project_root, manifest)
    _verify_preflight(project_root, manifest)
    token_maps = manifest.get("token_maps")
    protocol = manifest.get("training_protocol")
    if not isinstance(token_maps, dict) or not isinstance(protocol, dict):
        raise ValueError("manifest protocol is incomplete")
    families = tuple(token_maps)
    arms_raw = protocol.get("trained_arms")
    seeds_raw = protocol.get("model_seeds")
    if (
        set(families) != set(FROZEN_FAMILIES)
        or not isinstance(arms_raw, list)
        or set(arms_raw) != set(FROZEN_ARMS)
        or not isinstance(seeds_raw, list)
        or tuple(seeds_raw) != FROZEN_SEEDS
    ):
        raise ValueError("manifest family/arm/seed protocol mismatch")
    manifest_hash = _sha256(manifest_path)
    checkpoint_hashes: dict[str, object] = {}
    resource_cells: dict[str, dict[str, float]] = {}
    for family in families:
        for arm in arms_raw:
            for raw_seed in seeds_raw:
                seed = int(raw_seed)
                key = f"{family}/{arm}/{seed}"
                run_path = (
                    project_root
                    / "runtime"
                    / "cpkv-topk-001"
                    / family
                    / str(arm)
                    / str(seed)
                    / "run.json"
                )
                if not run_path.is_file():
                    raise ValueError(f"run receipt missing: {key}")
                run = _read_json(run_path)
                record = {
                    "path": run_path.relative_to(project_root).as_posix(),
                    "sha256": _sha256(run_path),
                    "selected_epoch": run.get("selected_epoch"),
                    "development_nll": run.get("development_nll"),
                }
                _validate_checkpoint_run(
                    project_root=project_root,
                    manifest_sha256=manifest_hash,
                    manifest=manifest,
                    key=key,
                    receipt_record=record,
                )
                checkpoint_hashes[key] = record
                cell = resource_cells.setdefault(
                    f"{family}/{seed}",
                    {"cpu_seconds": 0.0, "peak_rss_kb": 0.0},
                )
                cell["cpu_seconds"] += float(run["cpu_seconds"])
                cell["peak_rss_kb"] = max(
                    cell["peak_rss_kb"], float(run["peak_rss_kb"])
                )
    _enforce_resource_limits(resource_cells)
    receipt: dict[str, object] = {
        "schema": "cpkv-topk-001-checkpoint-freeze-v1",
        "lineage": "CPKV-TOPK-001",
        "status": "checkpoints-frozen",
        "manifest_sha256": manifest_hash,
        "checkpoint_hashes": checkpoint_hashes,
        "resource_cells": resource_cells,
        "total_cpu_seconds": math.fsum(
            cell["cpu_seconds"] for cell in resource_cells.values()
        ),
    }
    root = project_root.resolve()
    resolved_output = output_path.resolve()
    if not resolved_output.is_relative_to(root):
        raise ValueError("checkpoint freeze receipt output escapes project root")
    _atomic_write_text(
        resolved_output, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return {
        **receipt,
        "path": resolved_output.relative_to(root).as_posix(),
        "sha256": _sha256(resolved_output),
    }


def materialize(
    *, project_root: Path, manifest_path: Path, freeze_receipt_path: Path
) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    receipt = _read_json(freeze_receipt_path)
    if manifest.get("lineage") != "CPKV-TOPK-001":
        raise ValueError("manifest lineage mismatch")
    if receipt.get("lineage") != "CPKV-TOPK-001":
        raise ValueError("checkpoint receipt lineage mismatch")
    if receipt.get("schema") != "cpkv-topk-001-checkpoint-freeze-v1":
        raise ValueError("checkpoint receipt schema mismatch")
    if receipt.get("status") != "checkpoints-frozen":
        raise ValueError("checkpoint receipt is not frozen")
    if receipt.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("checkpoint receipt manifest hash mismatch")
    manifest_sha256 = _sha256(manifest_path)
    _verify_frozen_sources(project_root, manifest.get("source_hashes"))
    _verify_runtime(manifest)
    _verify_oracle(project_root, manifest)
    _verify_preflight(project_root, manifest)

    token_maps = manifest.get("token_maps")
    protocol = manifest.get("training_protocol")
    if not isinstance(token_maps, dict) or not isinstance(protocol, dict):
        raise ValueError("manifest protocol is incomplete")
    families = tuple(token_maps)
    arms_raw = protocol.get("trained_arms")
    seeds_raw = protocol.get("model_seeds")
    if (
        set(families) != set(FROZEN_FAMILIES)
        or not isinstance(arms_raw, list)
        or set(arms_raw) != set(FROZEN_ARMS)
        or not isinstance(seeds_raw, list)
        or tuple(seeds_raw) != FROZEN_SEEDS
    ):
        raise ValueError("manifest family/arm/seed protocol mismatch")
    arms = tuple(str(arm) for arm in arms_raw)
    seeds = tuple(int(seed) for seed in seeds_raw)
    checkpoint_hashes = receipt.get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("checkpoint receipt has no checkpoint mapping")

    expected_checkpoint_keys = {
        f"{family}/{arm}/{seed}"
        for family in families
        for arm in arms
        for seed in seeds
    }
    actual_checkpoint_keys = set(checkpoint_hashes)
    if actual_checkpoint_keys != expected_checkpoint_keys:
        missing = sorted(expected_checkpoint_keys - actual_checkpoint_keys)[:5]
        extra = sorted(actual_checkpoint_keys - expected_checkpoint_keys)[:5]
        raise ValueError(
            f"checkpoint keyset mismatch missing={missing} extra={extra}"
        )

    checkpoint_paths: set[Path] = set()
    resource_cells: dict[str, dict[str, float]] = {}
    for key in sorted(expected_checkpoint_keys):
        record = checkpoint_hashes[key]
        if not isinstance(record, dict):
            raise ValueError(f"checkpoint record is not an object: {key}")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        selected_epoch = record.get("selected_epoch")
        development_nll = record.get("development_nll")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"checkpoint path/hash missing: {key}")
        if (
            isinstance(selected_epoch, bool)
            or not isinstance(selected_epoch, int)
            or not 1 <= selected_epoch <= 20
        ):
            raise ValueError(f"invalid selected epoch: {key}")
        if (
            isinstance(development_nll, bool)
            or not isinstance(development_nll, (int, float))
            or not math.isfinite(float(development_nll))
        ):
            raise ValueError(f"invalid development NLL: {key}")
        checkpoint_path = _resolve_inside(
            project_root, relative, label=f"run receipt {key}"
        )
        if checkpoint_path in checkpoint_paths:
            raise ValueError(f"duplicate checkpoint path: {key}")
        checkpoint_paths.add(checkpoint_path)
        if not checkpoint_path.is_file():
            raise ValueError(f"checkpoint file missing: {key}")
        if _sha256(checkpoint_path) != expected_hash:
            raise ValueError(f"checkpoint hash mismatch: {key}")
        run = _validate_checkpoint_run(
            project_root=project_root,
            manifest_sha256=manifest_sha256,
            manifest=manifest,
            key=key,
            receipt_record=record,
        )
        family, _, seed_text = key.split("/")
        cell = resource_cells.setdefault(
            f"{family}/{seed_text}",
            {"cpu_seconds": 0.0, "peak_rss_kb": 0.0},
        )
        cell["cpu_seconds"] += float(run["cpu_seconds"])
        cell["peak_rss_kb"] = max(
            cell["peak_rss_kb"], float(run["peak_rss_kb"])
        )
    _enforce_resource_limits(resource_cells)
    _verify_recorded_resources(receipt, resource_cells)

    fixed_splits = manifest.get("fixed_splits")
    if not isinstance(fixed_splits, dict):
        raise ValueError("manifest has no fixed_splits mapping")

    materialized: dict[str, object] = {}
    for family in families:
        for split_name in ("E1", "E2"):
            key = f"{family}/{split_name}"
            record = fixed_splits.get(key)
            if not isinstance(record, dict):
                raise ValueError(f"missing manifest record {key}")
            relative = record.get("post_freeze_path")
            expected_hash = record.get("sha256_commitment")
            split_seed = record.get("seed")
            split_range = record.get("range")
            split_count = record.get("sequences")
            if (
                not isinstance(relative, str)
                or not isinstance(expected_hash, str)
                or not isinstance(split_seed, int)
                or not isinstance(split_range, list)
                or len(split_range) != 2
                or not all(isinstance(value, int) for value in split_range)
                or not isinstance(split_count, int)
            ):
                raise ValueError(f"incomplete manifest record {key}")
            path = project_root / relative
            if path.exists():
                actual_hash = _sha256(path)
                if actual_hash != expected_hash:
                    raise ValueError(f"existing sealed split hash mismatch: {key}")
                status = "already-materialized"
            else:
                actual_hash = _build_split(
                    family=family,
                    low=split_range[0],
                    high=split_range[1],
                    count=split_count,
                    seed=split_seed,
                    path=path,
                )
                if actual_hash != expected_hash:
                    path.unlink(missing_ok=True)
                    raise ValueError(f"new sealed split hash mismatch: {key}")
                status = "materialized-once"
            materialized[key] = {
                "path": relative,
                "sha256": actual_hash,
                "status": status,
            }
    return {
        "lineage": "CPKV-TOPK-001",
        "status": "evaluation-materialized",
        "checkpoint_freeze_receipt": freeze_receipt_path.as_posix(),
        "splits": materialized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--checkpoint-freeze-receipt", type=Path)
    operation.add_argument("--build-checkpoint-freeze-receipt", type=Path)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    if args.build_checkpoint_freeze_receipt is not None:
        output_path = args.build_checkpoint_freeze_receipt
        if not output_path.is_absolute():
            output_path = project_root / output_path
        result = build_checkpoint_freeze_receipt(
            project_root=project_root,
            manifest_path=manifest_path,
            output_path=output_path,
        )
    else:
        receipt_path = args.checkpoint_freeze_receipt
        assert receipt_path is not None
        if not receipt_path.is_absolute():
            receipt_path = project_root / receipt_path
        result = materialize(
            project_root=project_root,
            manifest_path=manifest_path,
            freeze_receipt_path=receipt_path,
        )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
