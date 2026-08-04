import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.topk_compiled_pushdown_evaluator import (
    _earliest_minimum_epoch,
    _enforce_resource_limits,
    _parameter_shapes,
    _verify_oracle,
    _verify_preflight,
    _verify_recorded_resources,
    build_checkpoint_freeze_receipt,
    materialize,
)


def test_evaluation_requires_frozen_checkpoint_hashes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "receipt.json"
    manifest.write_text(
        json.dumps({"lineage": "CPKV-TOPK-001", "fixed_splits": {}}),
        encoding="utf-8",
    )
    receipt.write_text(
        json.dumps(
            {
                "lineage": "CPKV-TOPK-001",
                "schema": "cpkv-topk-001-checkpoint-freeze-v1",
                "status": "checkpoints-frozen",
                "checkpoint_hashes": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        materialize(
            project_root=tmp_path,
            manifest_path=manifest,
            freeze_receipt_path=receipt,
        )


def test_nonempty_fake_receipt_does_not_unlock_evaluation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"lineage": "CPKV-TOPK-001", "fixed_splits": {}}),
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "lineage": "CPKV-TOPK-001",
                "schema": "cpkv-topk-001-checkpoint-freeze-v1",
                "status": "checkpoints-frozen",
                "manifest_sha256": manifest_hash,
                "checkpoint_hashes": {
                    "D1/S32/1701": {
                        "path": "fake.pt",
                        "sha256": "00",
                        "selected_epoch": 1,
                        "development_nll": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source hashes"):
        materialize(
            project_root=tmp_path,
            manifest_path=manifest,
            freeze_receipt_path=receipt,
        )


def test_checkpoint_schema_has_exact_frozen_parameter_count() -> None:
    for arm in ("S32", "DIRECT3", "HARD3", "MLP", "RNN32", "LINK32"):
        shapes = _parameter_shapes("A1", arm, 4)
        count = sum(math.prod(shape) for shape in shapes.values())
        assert count == 18_360


def test_checkpoint_selection_uses_earliest_tie_within_tolerance() -> None:
    losses = [2.0] * 20
    losses[4] = 1.0 + 5e-9
    losses[7] = 1.0
    assert _earliest_minimum_epoch(losses) == 5


def test_freeze_receipt_builder_rejects_incomplete_run_matrix(
    tmp_path: Path,
) -> None:
    labels = {
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
    source_hashes = {}
    for label in labels:
        path = tmp_path / f"{label}.txt"
        path.write_text(label, encoding="utf-8")
        source_hashes[label] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = tmp_path / "manifest.json"
    oracle = tmp_path / "oracle.json"
    oracle.write_text(
        json.dumps(
            {
                "lineage": "CPKV-TOPK-001",
                "status": "pass",
                "max_length": 6,
                "checked_sequences": 127,
                "x4": {
                    "top3_history_mass": 0.75,
                    "discarded_mass": 0.25,
                    "read_error": 1.0 / math.sqrt(12.0),
                    "bound": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": "cpkv-topk-001-preflight-v1",
                "lineage": "CPKV-TOPK-001",
                "status": "pass",
                "seed": 1701,
                "base_sample_count": 512,
                "projection_limit_hours": 5.7,
                "families": {
                    family: {
                        "status": "pass",
                        "projected_all_arm_cpu_hours": 5.0,
                        "projection_limit_hours": 5.7,
                        "actual_freeze_limit_hours": 6.0,
                    }
                    for family in ("D1", "A1", "A2")
                },
                "source_hashes": {
                    label: source_hashes[label]
                    for label in (
                        "preflight",
                        "preflight_tests",
                        "training",
                        "locked_evaluation",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "lineage": "CPKV-TOPK-001",
                "source_hashes": source_hashes,
                "oracle_receipt": {
                    "path": oracle.name,
                    "sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
                },
                "preflight_receipt": {
                    "path": preflight.name,
                    "sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
                },
                "environment": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "torch_installed": importlib.metadata.version("torch"),
                    "torch_required": importlib.metadata.version("torch"),
                },
                "token_maps": {"D1": [], "A1": [], "A2": []},
                "training_protocol": {
                    "trained_arms": [
                        "S32",
                        "DIRECT3",
                        "HARD3",
                        "MLP",
                        "RNN32",
                        "LINK32",
                    ],
                    "model_seeds": list(range(1701, 1711)),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run receipt missing"):
        build_checkpoint_freeze_receipt(
            project_root=tmp_path,
            manifest_path=manifest,
            output_path=tmp_path / "freeze.json",
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("discarded_mass", 0.2),
        ("read_error", 0.3),
        ("bound", 0.6),
    ),
)
def test_oracle_requires_complete_x4_contract(
    tmp_path: Path, field: str, invalid_value: float
) -> None:
    oracle_path = tmp_path / "oracle.json"
    oracle = {
        "lineage": "CPKV-TOPK-001",
        "status": "pass",
        "max_length": 6,
        "checked_sequences": 127,
        "x4": {
            "top3_history_mass": 0.75,
            "discarded_mass": 0.25,
            "read_error": math.sqrt(3.0) / 6.0,
            "bound": 0.5,
        },
    }

    def manifest_for_current_oracle() -> dict:
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        return {
            "oracle_receipt": {
                "path": oracle_path.name,
                "sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
            }
        }

    _verify_oracle(tmp_path, manifest_for_current_oracle())
    oracle["x4"][field] = invalid_value
    with pytest.raises(ValueError, match="oracle receipt contract mismatch"):
        _verify_oracle(tmp_path, manifest_for_current_oracle())


def test_preflight_receipt_fails_closed_on_projection_or_source_tamper(
    tmp_path: Path,
) -> None:
    source_hashes = {}
    for label in ("preflight", "preflight_tests", "training", "locked_evaluation"):
        path = tmp_path / f"{label}.py"
        path.write_text(label, encoding="utf-8")
        source_hashes[label] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt_path = tmp_path / "preflight.json"
    receipt = {
        "schema": "cpkv-topk-001-preflight-v1",
        "lineage": "CPKV-TOPK-001",
        "status": "pass",
        "seed": 1701,
        "base_sample_count": 512,
        "projection_limit_hours": 5.7,
        "families": {
            family: {
                "status": "pass",
                "projected_all_arm_cpu_hours": 5.0,
                "projection_limit_hours": 5.7,
                "actual_freeze_limit_hours": 6.0,
            }
            for family in ("D1", "A1", "A2")
        },
        "source_hashes": {
            label: dict(source_hashes[label])
            for label in ("preflight", "preflight_tests", "training", "locked_evaluation")
        },
    }

    def manifest_for_receipt() -> dict:
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return {
            "source_hashes": source_hashes,
            "preflight_receipt": {
                "path": receipt_path.name,
                "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            },
        }

    _verify_preflight(tmp_path, manifest_for_receipt())
    receipt["families"]["A2"]["projected_all_arm_cpu_hours"] = 5.8
    with pytest.raises(ValueError, match="preflight resource gate failed"):
        _verify_preflight(tmp_path, manifest_for_receipt())
    receipt["families"]["A2"]["projected_all_arm_cpu_hours"] = 5.0
    receipt["source_hashes"]["training"]["sha256"] = "00"
    with pytest.raises(ValueError, match="source binding mismatch"):
        _verify_preflight(tmp_path, manifest_for_receipt())


def test_resource_budget_is_recomputed_not_trusted() -> None:
    cells = {"A2/1701": {"cpu_seconds": 21_599.0, "peak_rss_kb": 1.0}}
    _enforce_resource_limits(cells)
    with pytest.raises(ValueError, match="CPU-hour"):
        _enforce_resource_limits(
            {"A2/1701": {"cpu_seconds": 21_601.0, "peak_rss_kb": 1.0}}
        )
    with pytest.raises(ValueError, match="resource mismatch"):
        _verify_recorded_resources(
            {
                "resource_cells": {
                    "A2/1701": {"cpu_seconds": 1.0, "peak_rss_kb": 1.0}
                },
                "total_cpu_seconds": 1.0,
            },
            cells,
        )
