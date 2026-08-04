"""Minimal G0 admission validator for fixed-compute experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EDGES = {"A", "B", "C", "D", "E"}
SCOPES = {"serve-only", "lifecycle"}
SLICES = {"P1", "P2", "P3", "P4", "P5"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDERS = {"TODO", "TBD", "UNSET", "TO_BE_CAPTURED"}


def _required(mapping: dict[str, Any], key: str, path: str,
              errors: list[str]) -> Any:
    if key not in mapping:
        errors.append(f"{path}.{key}: missing")
        return None
    return mapping[key]


def _mapping(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected object")
        return {}
    return value


def _exact_nonnegative(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{path}: expected nonnegative integer")


def _hash(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        errors.append(f"{path}: expected lowercase SHA-256")


def _frozen_string(value: Any, path: str, errors: list[str]) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().upper() in PLACEHOLDERS
        or "TO_BE_CAPTURED" in value.upper()
    ):
        errors.append(f"{path}: expected captured non-placeholder value")


def _validate_artifact(value: Any, path: str, errors: list[str]) -> None:
    artifact = _mapping(value, path, errors)
    _hash(_required(artifact, "checkpoint_sha256", path, errors),
          f"{path}.checkpoint_sha256", errors)
    _hash(_required(artifact, "tokenizer_sha256", path, errors),
          f"{path}.tokenizer_sha256", errors)
    learned = _mapping(
        _required(artifact, "learned_bytes", path, errors),
        f"{path}.learned_bytes", errors,
    )
    parts = ("core", "auxiliary", "metadata", "total")
    for part in parts:
        _exact_nonnegative(
            _required(learned, part, f"{path}.learned_bytes", errors),
            f"{path}.learned_bytes.{part}", errors,
        )
    if all(isinstance(learned.get(part), int) for part in parts):
        expected = learned["core"] + learned["auxiliary"] + learned["metadata"]
        if learned["total"] != expected:
            errors.append(
                f"{path}.learned_bytes.total: expected {expected} from components"
            )
        if learned["total"] == 0:
            errors.append(f"{path}.learned_bytes.total: must be greater than zero")


def _validate_cells(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("workload.cells: expected nonempty array")
        return
    names: set[str] = set()
    for index, raw_cell in enumerate(value):
        path = f"workload.cells[{index}]"
        cell = _mapping(raw_cell, path, errors)
        name = _required(cell, "name", path, errors)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{path}.name: expected nonempty string")
        elif name in names:
            errors.append(f"{path}.name: duplicate {name!r}")
        else:
            names.add(name)
        for key in ("input_tokens", "output_tokens", "concurrency"):
            value = _required(cell, key, path, errors)
            _exact_nonnegative(value, f"{path}.{key}", errors)
            if value == 0:
                errors.append(f"{path}.{key}: must be greater than zero")


def validate_manifest(manifest: Any) -> list[str]:
    """Return all admission failures. An empty list means G0 admission passes."""
    errors: list[str] = []
    root = _mapping(manifest, "$", errors)
    if root.get("schema_version") != 1:
        errors.append("schema_version: expected 1")

    experiment_id = _required(root, "experiment_id", "$", errors)
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        errors.append("experiment_id: expected nonempty string")

    claim = _mapping(_required(root, "claim", "$", errors), "claim", errors)
    scope = _required(claim, "scope", "claim", errors)
    if scope not in SCOPES:
        errors.append(f"claim.scope: expected one of {sorted(SCOPES)}")
    primary = _required(claim, "primary_edge", "claim", errors)
    edges = _required(claim, "edges", "claim", errors)
    if not isinstance(edges, list) or not edges:
        errors.append("claim.edges: expected nonempty array")
        edges = []
    unknown = set(edges) - EDGES
    if unknown:
        errors.append(f"claim.edges: unknown edges {sorted(unknown)}")
    if primary not in EDGES:
        errors.append(f"claim.primary_edge: expected one of {sorted(EDGES)}")
    elif primary not in edges:
        errors.append("claim.primary_edge: must also appear in claim.edges")

    artifacts = _mapping(
        _required(root, "artifacts", "$", errors), "artifacts", errors
    )
    for side in ("baseline", "candidate"):
        _validate_artifact(
            _required(artifacts, side, "artifacts", errors),
            f"artifacts.{side}", errors,
        )

    hardware = _mapping(
        _required(root, "hardware", "$", errors), "hardware", errors
    )
    for key in ("gpu_stratum", "gpu_pci_id", "container_digest", "server_commit"):
        value = _required(hardware, key, "hardware", errors)
        _frozen_string(value, f"hardware.{key}", errors)

    workload = _mapping(
        _required(root, "workload", "$", errors), "workload", errors
    )
    _hash(_required(workload, "trace_sha256", "workload", errors),
          "workload.trace_sha256", errors)
    if workload.get("frozen") is not True:
        errors.append("workload.frozen: must be true")
    _validate_cells(_required(workload, "cells", "workload", errors), errors)

    quality = _mapping(
        _required(root, "quality", "$", errors), "quality", errors
    )
    slices = quality.get("protected_slices")
    if not isinstance(slices, list) or set(slices) != SLICES:
        errors.append(f"quality.protected_slices: expected exactly {sorted(SLICES)}")
    margins = _mapping(quality.get("noninferiority_margins"),
                       "quality.noninferiority_margins", errors)
    for slice_name in SLICES:
        margin = _required(margins, slice_name,
                           "quality.noninferiority_margins", errors)
        if not isinstance(margin, (int, float)) or isinstance(margin, bool) or margin < 0:
            errors.append(
                f"quality.noninferiority_margins.{slice_name}: "
                "expected nonnegative number"
            )

    planned_seeds = root.get("planned_training_seeds")
    if not isinstance(planned_seeds, list) or len(set(planned_seeds)) < 3:
        errors.append("planned_training_seeds: require at least 3 unique seeds")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    errors = validate_manifest(manifest)
    if errors:
        print("G0: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("G0: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
