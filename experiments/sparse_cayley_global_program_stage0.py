#!/usr/bin/env python3
"""Algebra gate for sparse Cayley global expert programs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "sparse-cayley-global-program-stage0.json"
WIDTH = 16
EXPERTS = 5
DEGREE = 3
ROUTE_LENGTH = 4
ALPHA = 0.25
NEUMANN_ORDER = 4
SEED = 20260730


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sparse_skew_generators(
    width: int,
    experts: int,
    degree: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    if width <= 0 or width % 2 or experts <= 0 or degree <= 0:
        raise ValueError("width must be positive and even; experts/degree positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    generators = []
    for _ in range(experts):
        generator = np.zeros((width, width), dtype=np.float64)
        for _ in range(degree):
            permutation = rng.permutation(width)
            for offset in range(0, width, 2):
                first, second = permutation[offset : offset + 2]
                weight = rng.normal()
                generator[first, second] += weight
                generator[second, first] -= weight
        norm = np.linalg.svd(generator, compute_uv=False)[0]
        if norm == 0.0:
            raise RuntimeError("degenerate random generator")
        generators.append(generator * (alpha / norm))
    return generators


def cayley(generator: np.ndarray) -> np.ndarray:
    identity = np.eye(generator.shape[0], dtype=generator.dtype)
    return (identity - generator) @ np.linalg.inv(identity + generator)


def neumann_cayley_apply(
    generator: np.ndarray, values: np.ndarray, order: int
) -> np.ndarray:
    term = np.asarray(values, dtype=np.float64).copy()
    inverse_action = term.copy()
    for _ in range(order):
        term = -generator @ term
        inverse_action += term
    return 2.0 * inverse_action - values


def numerical_unique_count(matrices: list[np.ndarray], decimals: int = 10) -> int:
    return len({matrix.round(decimals).tobytes() for matrix in matrices})


def operator_span_dimension(
    matrices: list[np.ndarray], relative_tolerance: float = 1e-9
) -> tuple[int, float]:
    singular_values = np.linalg.svd(
        np.stack([matrix.ravel() for matrix in matrices]), compute_uv=False
    )
    rank = int(
        np.count_nonzero(singular_values > relative_tolerance * singular_values[0])
    )
    return rank, float(singular_values[-1] / singular_values[0])


def compose(route: tuple[int, ...], experts: list[np.ndarray]) -> np.ndarray:
    result = np.eye(experts[0].shape[0])
    for expert in route:
        result = experts[expert] @ result
    return result


def additive(route: tuple[int, ...], experts: list[np.ndarray]) -> np.ndarray:
    identity = np.eye(experts[0].shape[0])
    result = identity.copy()
    for expert in route:
        result += experts[expert] - identity
    return result


def scaling_ledger(width: int, degree: int, order: int, route_length: int) -> dict[str, Any]:
    # One undirected skew edge updates two coordinates.  We conservatively count
    # both directed scalar visits for each degree at every Neumann multiplication.
    active_scalar_edge_visits = route_length * order * degree * width
    dense_macs = width * width
    return {
        "width": width,
        "degree": degree,
        "neumann_sparse_multiplies_per_expert": order,
        "route_length": route_length,
        "active_scalar_edge_visits_upper_bound": active_scalar_edge_visits,
        "dense_matvec_macs": dense_macs,
        "ideal_active_to_dense_ratio": active_scalar_edge_visits / dense_macs,
        "excluded": [
            "route selection",
            "index and edge-weight reads",
            "physical gather/scatter efficiency",
            "kernel synchronization and launch overhead",
            "training backward work",
        ],
    }


def build_report() -> dict[str, Any]:
    generators = make_sparse_skew_generators(
        WIDTH, EXPERTS, DEGREE, ALPHA, SEED
    )
    experts = [cayley(generator) for generator in generators]
    rng = np.random.default_rng(SEED + 1)
    vectors = rng.normal(size=(WIDTH, 64))
    bound = 2.0 * ALPHA ** (NEUMANN_ORDER + 1) / (1.0 - ALPHA)

    expert_rows = []
    for generator, expert in zip(generators, experts, strict=True):
        approximate = neumann_cayley_apply(generator, vectors, NEUMANN_ORDER)
        exact = expert @ vectors
        expert_rows.append(
            {
                "generator_skew_error": float(
                    np.linalg.norm(generator + generator.T)
                ),
                "generator_spectral_norm": float(
                    np.linalg.svd(generator, compute_uv=False)[0]
                ),
                "generator_rank": int(np.linalg.matrix_rank(generator)),
                "cayley_orthogonality_error": float(
                    np.linalg.norm(expert.T @ expert - np.eye(WIDTH))
                ),
                "cayley_numerical_density": float(np.mean(np.abs(expert) > 1e-12)),
                "cayley_displacement_rank": int(
                    np.linalg.matrix_rank(expert - np.eye(WIDTH))
                ),
                "truncated_relative_error": float(
                    np.linalg.norm(approximate - exact) / np.linalg.norm(exact)
                ),
            }
        )

    routes = list(itertools.product(range(EXPERTS), repeat=ROUTE_LENGTH))
    sequential = [compose(route, experts) for route in routes]
    additive_control = [additive(route, experts) for route in routes]
    sequential_span, sequential_condition = operator_span_dimension(sequential)
    additive_span, additive_condition = operator_span_dimension(additive_control)
    sequential_unique = numerical_unique_count(sequential)
    additive_unique = numerical_unique_count(additive_control)
    maximum_route_orthogonality_error = max(
        float(np.linalg.norm(matrix.T @ matrix - np.eye(WIDTH)))
        for matrix in sequential
    )
    ledger = scaling_ledger(4096, DEGREE, NEUMANN_ORDER, ROUTE_LENGTH)

    gates = {
        "generators_are_full_rank_sparse_skew_contractions": all(
            row["generator_skew_error"] <= 1e-12
            and abs(row["generator_spectral_norm"] - ALPHA) <= 1e-12
            and row["generator_rank"] == WIDTH
            for row in expert_rows
        ),
        "cayley_experts_are_dense_full_displacement_orthogonal": all(
            row["cayley_orthogonality_error"] <= 1e-12
            and row["cayley_numerical_density"] >= 0.90
            and row["cayley_displacement_rank"] == WIDTH
            for row in expert_rows
        ),
        "truncated_apply_obeys_bound_and_target": all(
            row["truncated_relative_error"] <= bound
            and row["truncated_relative_error"] <= 0.003
            for row in expert_rows
        ),
        "all_ordered_routes_are_distinct": sequential_unique == EXPERTS ** ROUTE_LENGTH,
        "sequential_routes_span_full_matrix_space": sequential_span == WIDTH * WIDTH,
        "additive_control_collapses_to_multisets_and_small_span": (
            additive_unique == math.comb(EXPERTS + ROUTE_LENGTH - 1, ROUTE_LENGTH)
            and additive_span <= EXPERTS + 1
        ),
        "all_sequential_routes_remain_orthogonal": maximum_route_orthogonality_error <= 1e-11,
        "scaling_ledger_is_below_three_percent_dense": ledger["ideal_active_to_dense_ratio"] < 0.03,
    }

    return {
        "schema_version": 1,
        "candidate": "sparse Cayley global expert programs",
        "source_sha256": sha256(Path(__file__)),
        "configuration": {
            "width": WIDTH,
            "experts": EXPERTS,
            "degree": DEGREE,
            "route_length": ROUTE_LENGTH,
            "alpha": ALPHA,
            "neumann_order": NEUMANN_ORDER,
            "seed": SEED,
        },
        "analytic_truncation_bound": bound,
        "experts": expert_rows,
        "programs": {
            "total_routes": len(routes),
            "sequential_unique": sequential_unique,
            "sequential_span_dimension": sequential_span,
            "sequential_smallest_to_largest_singular_ratio": sequential_condition,
            "maximum_route_orthogonality_error": maximum_route_orthogonality_error,
            "additive_unique": additive_unique,
            "additive_span_dimension": additive_span,
            "additive_smallest_to_largest_singular_ratio": additive_condition,
        },
        "scaling_ledger": ledger,
        "frozen_gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": {
            "proved_if_pass": [
                "each linearly sized sparse generator has a dense norm-preserving global response",
                "each expert changes all feature directions",
                "short ordered programs span strictly more operator directions than additive aggregation",
                "truncated application has a width-independent error bound and near-linear logical work",
            ],
            "not_proved": [
                "useful routes are learnable",
                "composed routes are independent stored knowledge",
                "language-model quality",
                "GPU speed or memory-traffic superiority",
                "novelty beyond the complete sparse-global-program combination",
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
