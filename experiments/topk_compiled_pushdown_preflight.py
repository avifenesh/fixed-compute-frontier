#!/usr/bin/env python3
"""Length-stratified CPU preflight for CPKV-TOPK-001."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from experiments.topk_compiled_pushdown_locked_eval import teacher_forced_trace
    from experiments.topk_compiled_pushdown_reference import _atomic_write_text
    from experiments.topk_compiled_pushdown_train import (
        ARMS,
        BATCH_SIZE,
        AcquisitionModel,
        evaluate_nll,
        load_sequences,
        optimizer_for,
        read_manifest,
        verify_runtime,
    )
except ModuleNotFoundError:  # Direct script execution.
    from topk_compiled_pushdown_locked_eval import (  # type: ignore[no-redef]
        teacher_forced_trace,
    )
    from topk_compiled_pushdown_reference import (  # type: ignore[no-redef]
        _atomic_write_text,
    )
    from topk_compiled_pushdown_train import (  # type: ignore[no-redef]
        ARMS,
        BATCH_SIZE,
        AcquisitionModel,
        evaluate_nll,
        load_sequences,
        optimizer_for,
        read_manifest,
        verify_runtime,
    )


LINEAGE = "CPKV-TOPK-001"
PREFLIGHT_SEED = 1701
BASE_SAMPLE_COUNT = 512
PROJECTION_LIMIT_HOURS = 5.7
PREFLIGHT_SOURCE_PATHS = {
    "preflight": "experiments/topk_compiled_pushdown_preflight.py",
    "preflight_tests": "tests/test_topk_compiled_pushdown_preflight.py",
    "training": "experiments/topk_compiled_pushdown_train.py",
    "locked_evaluation": "experiments/topk_compiled_pushdown_locked_eval.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_sample(
    sequences: Sequence[Sequence[int]], base_count: int = BASE_SAMPLE_COUNT
) -> list[list[int]]:
    if not sequences:
        raise ValueError("cannot sample an empty split")
    all_lengths = {len(sequence) - 2 for sequence in sequences}
    selected = [list(sequence) for sequence in sequences[:base_count]]
    selected_lengths = {len(sequence) - 2 for sequence in selected}
    for missing_length in sorted(all_lengths - selected_lengths):
        selected.append(
            list(
                next(
                    sequence
                    for sequence in sequences
                    if len(sequence) - 2 == missing_length
                )
            )
        )
    if {len(sequence) - 2 for sequence in selected} != all_lengths:
        raise AssertionError("stratified sample missed a content length")
    return selected


def training_pass_seconds(
    model: AcquisitionModel, sequences: Sequence[Sequence[int]]
) -> float:
    optimizer = optimizer_for(model)
    started = time.process_time()
    for start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start : start + BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        losses = [model.sequence_loss(tokens) for tokens in batch]
        targets = sum(len(tokens) - 1 for tokens in batch)
        loss = torch.stack(losses).sum() / targets
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return time.process_time() - started


def executed_state(
    model: AcquisitionModel,
    sequences: Sequence[Sequence[int]],
    arm: str,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for tokens in sequences:
        trace = teacher_forced_trace(
            model,
            tokens,
            evaluated_arm=arm,
        )
        for name, value in trace.stats.items():
            if name.startswith("maximum_") or name == "logical_peak_auxiliary_bytes":
                totals[name] = max(totals.get(name, 0), int(value))
            else:
                totals[name] = totals.get(name, 0) + int(value)
    return totals


def _split_sequences(
    *,
    project_root: Path,
    manifest: Mapping[str, object],
    family: str,
    split_key: str,
) -> list[list[int]]:
    token_maps = manifest["token_maps"]
    fixed_splits = manifest["fixed_splits"]
    token_to_id = {
        str(token): index for index, token in enumerate(token_maps[family])
    }
    return load_sequences(
        project_root=project_root,
        record=fixed_splits[split_key],
        token_to_id=token_to_id,
    )


def run_preflight(
    *, project_root: Path, manifest_path: Path
) -> dict[str, object]:
    manifest = read_manifest(manifest_path)
    verify_runtime(manifest)
    token_maps = manifest["token_maps"]
    families: dict[str, object] = {}
    overall_pass = True
    for family in ("D1", "A1", "A2"):
        training = _split_sequences(
            project_root=project_root,
            manifest=manifest,
            family=family,
            split_key=f"{family}/train/{PREFLIGHT_SEED}",
        )
        development = _split_sequences(
            project_root=project_root,
            manifest=manifest,
            family=family,
            split_key=f"{family}/development",
        )
        train_sample = stratified_sample(training)
        dev_sample = stratified_sample(development)
        train_targets = sum(len(sequence) - 1 for sequence in training)
        dev_targets = sum(len(sequence) - 1 for sequence in development)
        sample_train_targets = sum(len(sequence) - 1 for sequence in train_sample)
        sample_dev_targets = sum(len(sequence) - 1 for sequence in dev_sample)
        arm_records: dict[str, object] = {}
        projected_seconds = 0.0
        for arm in ARMS:
            model = AcquisitionModel(
                vocabulary_size=len(token_maps[family]),
                arm=arm,
                model_seed=PREFLIGHT_SEED,
            )
            # One unreported warmup batch removes import/allocator startup.
            training_pass_seconds(model, train_sample[:BATCH_SIZE])
            measured_training = max(
                training_pass_seconds(model, train_sample) for _ in range(2)
            )
            measured_development_runs: list[float] = []
            for _ in range(2):
                started = time.process_time()
                evaluate_nll(model, dev_sample)
                measured_development_runs.append(time.process_time() - started)
            measured_development = max(measured_development_runs)
            projected_training = (
                measured_training
                * train_targets
                * 20
                / sample_train_targets
            )
            projected_development = (
                measured_development
                * dev_targets
                * 20
                / sample_dev_targets
            )
            projected_seconds += projected_training + projected_development
            arm_records[arm] = {
                "training_sample_sequences": len(train_sample),
                "training_sample_targets": sample_train_targets,
                "training_process_seconds": measured_training,
                "development_sample_sequences": len(dev_sample),
                "development_sample_targets": sample_dev_targets,
                "development_process_seconds": measured_development,
                "projected_training_seconds": projected_training,
                "projected_development_seconds": projected_development,
                "content_lengths": sorted(
                    {len(sequence) - 2 for sequence in train_sample}
                ),
                "executed_state": executed_state(
                    model, train_sample[:64], arm
                ),
            }
        projected_hours = projected_seconds / 3600.0
        family_pass = projected_hours <= PROJECTION_LIMIT_HOURS
        overall_pass = overall_pass and family_pass
        families[family] = {
            "status": "pass" if family_pass else "fail",
            "projected_all_arm_cpu_hours": projected_hours,
            "projection_limit_hours": PROJECTION_LIMIT_HOURS,
            "actual_freeze_limit_hours": 6.0,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "arms": arm_records,
        }
    source_paths = {
        label: project_root / relative
        for label, relative in PREFLIGHT_SOURCE_PATHS.items()
    }
    return {
        "schema": "cpkv-topk-001-preflight-v1",
        "lineage": LINEAGE,
        "status": "pass" if overall_pass else "fail",
        "seed": PREFLIGHT_SEED,
        "base_sample_count": BASE_SAMPLE_COUNT,
        "projection_limit_hours": PROJECTION_LIMIT_HOURS,
        "families": families,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "source_hashes": {
            label: {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": _sha256(path),
            }
            for label, path in source_paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    project_root = args.project_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path = output_path.resolve()
    if not output_path.is_relative_to(project_root):
        raise ValueError("preflight output escapes project root")
    result = run_preflight(project_root=project_root, manifest_path=manifest_path)
    _atomic_write_text(
        output_path, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "path": output_path.relative_to(project_root).as_posix(),
                "sha256": _sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
