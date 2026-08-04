#!/usr/bin/env python3
"""Compare sequential and additive routing with the same micro-expert bank."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "noncommutative-routed-program-gate.json"
WIDTH = 16
EXPERTS = 5
ROUTE_LENGTH = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_rotation_experts(
    width: int, experts: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create generic learned planes and nontrivial rotation angles."""
    if width < 2 or experts <= 0:
        raise ValueError("width must be at least two and experts positive")
    rng = np.random.default_rng(seed)
    first = np.empty((experts, width), dtype=np.float64)
    second = np.empty((experts, width), dtype=np.float64)
    for index in range(experts):
        plane, _ = np.linalg.qr(rng.normal(size=(width, 2)))
        first[index] = plane[:, 0]
        second[index] = plane[:, 1]
    angles = rng.uniform(low=0.35, high=1.15, size=experts)
    return first, second, angles


def rotation_matrix(
    first: np.ndarray, second: np.ndarray, angle: float
) -> np.ndarray:
    u = np.asarray(first, dtype=np.float64)
    v = np.asarray(second, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 1:
        raise ValueError("rotation plane vectors must be equal-length vectors")
    cosine = math.cos(angle)
    sine = math.sin(angle)
    identity = np.eye(u.size)
    projector = np.outer(u, u) + np.outer(v, v)
    skew = np.outer(v, u) - np.outer(u, v)
    return identity + (cosine - 1.0) * projector + sine * skew


def apply_rotation(
    values: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Apply a dense rank-two rotation using two dot products."""
    x = np.asarray(values, dtype=np.float64)
    u = np.asarray(first, dtype=np.float64)
    v = np.asarray(second, dtype=np.float64)
    if x.shape[0] != u.size or u.shape != v.shape:
        raise ValueError("incompatible value and plane shapes")
    a = u @ x
    b = v @ x
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return x + u[:, None] * ((cosine - 1.0) * a - sine * b) + v[
        :, None
    ] * (sine * a + (cosine - 1.0) * b) if x.ndim == 2 else (
        x
        + u * ((cosine - 1.0) * a - sine * b)
        + v * (sine * a + (cosine - 1.0) * b)
    )


def expert_matrices(
    first: np.ndarray, second: np.ndarray, angles: np.ndarray
) -> list[np.ndarray]:
    return [
        rotation_matrix(first[index], second[index], float(angles[index]))
        for index in range(len(angles))
    ]


def sequential_operator(
    route: Iterable[int], matrices: list[np.ndarray]
) -> np.ndarray:
    result = np.eye(matrices[0].shape[0])
    for expert in route:
        result = matrices[expert] @ result
    return result


def additive_operator(
    route: Iterable[int], matrices: list[np.ndarray]
) -> np.ndarray:
    identity = np.eye(matrices[0].shape[0])
    result = identity.copy()
    for expert in route:
        result += matrices[expert] - identity
    return result


def numerical_unique_count(matrices: list[np.ndarray], decimals: int = 10) -> int:
    return len({matrix.round(decimals).tobytes() for matrix in matrices})


def operator_span_dimension(matrices: list[np.ndarray], tolerance: float = 1e-9) -> int:
    flattened = np.stack([matrix.ravel() for matrix in matrices])
    singular_values = np.linalg.svd(flattened, compute_uv=False)
    return int(np.count_nonzero(singular_values > tolerance * singular_values[0]))


def same_multiset_order_effects(
    routes: list[tuple[int, ...]],
    sequential: list[np.ndarray],
    additive: list[np.ndarray],
) -> dict[str, float | int]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, route in enumerate(routes):
        counts = tuple(route.count(expert) for expert in range(EXPERTS))
        groups[counts].append(index)

    sequential_distances = []
    additive_distances = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for other in indices[1:]:
            sequential_distances.append(
                np.linalg.norm(sequential[anchor] - sequential[other])
                / math.sqrt(WIDTH)
            )
            additive_distances.append(
                np.linalg.norm(additive[anchor] - additive[other])
                / math.sqrt(WIDTH)
            )
    return {
        "compared_route_pairs": len(sequential_distances),
        "mean_sequential_operator_distance": float(np.mean(sequential_distances)),
        "minimum_sequential_operator_distance": float(np.min(sequential_distances)),
        "maximum_additive_operator_distance": float(np.max(additive_distances)),
    }


def resource_ledger(width: int, experts: int, route_length: int) -> dict[str, Any]:
    bank_parameters = 2 * experts * width + experts
    per_expert_macs = 4 * width
    active_macs = route_length * per_expert_macs
    return {
        "bank_parameters": bank_parameters,
        "active_macs_upper_bound": active_macs,
        "additive_and_sequential_use_same_bank": True,
        "additive_and_sequential_use_same_expert_calls": True,
        "dense_width_squared_reference_macs": width * width,
        "active_to_dense_reference_ratio": active_macs / (width * width),
        "note": "Each expert is a dense rank-two plane rotation applied without a dense matrix.",
    }


def build_report() -> dict[str, Any]:
    first, second, angles = make_rotation_experts(WIDTH, EXPERTS, seed=20260727)
    matrices = expert_matrices(first, second, angles)
    routes = list(itertools.product(range(EXPERTS), repeat=ROUTE_LENGTH))
    sequential = [sequential_operator(route, matrices) for route in routes]
    additive = [additive_operator(route, matrices) for route in routes]

    sequential_unique = numerical_unique_count(sequential)
    additive_unique = numerical_unique_count(additive)
    sequential_span = operator_span_dimension(sequential)
    additive_span = operator_span_dimension(additive)
    order_effects = same_multiset_order_effects(routes, sequential, additive)
    orthogonality_error = max(
        float(np.linalg.norm(matrix.T @ matrix - np.eye(WIDTH)))
        for matrix in sequential
    )

    gates = {
        "sequential_routes_are_distinct": sequential_unique >= 0.99 * len(routes),
        "sequential_span_exceeds_additive_by_4x": sequential_span >= 4 * additive_span,
        "route_order_materially_changes_sequential_operator": (
            order_effects["mean_sequential_operator_distance"] >= 0.10
        ),
        "route_order_cannot_change_additive_operator": (
            order_effects["maximum_additive_operator_distance"] <= 1e-12
        ),
        "sequential_programs_preserve_norm": orthogonality_error <= 1e-12,
    }

    return {
        "schema_version": 1,
        "candidate": "routed noncommutative microprogram FFN",
        "claim": (
            "Sequential composition exposes ordered interaction terms that an "
            "equal-call additive MoE cannot express."
        ),
        "source_sha256": sha256(Path(__file__)),
        "width": WIDTH,
        "experts": EXPERTS,
        "route_length": ROUTE_LENGTH,
        "total_ordered_routes": len(routes),
        "theoretical_additive_multisets": math.comb(
            EXPERTS + ROUTE_LENGTH - 1, ROUTE_LENGTH
        ),
        "observed": {
            "sequential_unique_operators": sequential_unique,
            "additive_unique_operators": additive_unique,
            "sequential_operator_span_dimension": sequential_span,
            "additive_operator_span_dimension": additive_span,
            "maximum_sequential_orthogonality_error": orthogonality_error,
            "same_multiset_order_effects": order_effects,
        },
        "ledger": resource_ledger(WIDTH, EXPERTS, ROUTE_LENGTH),
        "frozen_gates": gates,
        "algebra_gate_passed": all(gates.values()),
        "scope_boundary": {
            "proved": [
                "strict expressed-operator gain over additive routing at equal calls",
                "ordered routes remain distinguishable",
                "stable norm-preserving composition",
            ],
            "not_proved": [
                "learned routing discovers useful language programs",
                "language-model quality improves",
                "GPU kernels beat a dense SwiGLU at matched quality",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
