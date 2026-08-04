#!/usr/bin/env python3
"""Gate dense, non-low-rank mixing implemented by a sparse precision graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "sparse-precision-dense-mixer-gate.json"
WIDTHS = (64, 128, 256, 512)
DEGREE = 3
RHO = 0.5
RELATIVE_TOLERANCE = 0.01


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_permutations(width: int, degree: int, seed: int) -> list[np.ndarray]:
    if width <= 1 or degree <= 0:
        raise ValueError("width must exceed one and degree must be positive")
    rng = np.random.default_rng(seed)
    return [rng.permutation(width) for _ in range(degree)]


def dense_operator(permutations: list[np.ndarray]) -> np.ndarray:
    """Return S = mean_j (Pi_j + Pi_j^T) / 2."""
    if not permutations:
        raise ValueError("at least one permutation is required")
    width = len(permutations[0])
    operator = np.zeros((width, width), dtype=np.float64)
    rows = np.arange(width)
    scale = 1.0 / (2 * len(permutations))
    for permutation in permutations:
        if permutation.shape != (width,):
            raise ValueError("all permutations must have the same width")
        operator[rows, permutation] += scale
        operator[permutation, rows] += scale
    return operator


def apply_operator(
    permutations: list[np.ndarray], values: np.ndarray
) -> np.ndarray:
    """Apply S without materializing S."""
    vector = np.asarray(values, dtype=np.float64)
    width = len(permutations[0])
    if vector.shape[0] != width:
        raise ValueError("leading value dimension must equal permutation width")
    result = np.zeros_like(vector)
    scale = 1.0 / (2 * len(permutations))
    for permutation in permutations:
        inverse = np.argsort(permutation)
        result += scale * vector[permutation]
        result += scale * vector[inverse]
    return result


def is_connected(permutations: list[np.ndarray]) -> bool:
    width = len(permutations[0])
    adjacency: list[list[int]] = [[] for _ in range(width)]
    for permutation in permutations:
        for source, target in enumerate(permutation):
            adjacency[source].append(int(target))
            adjacency[int(target)].append(source)
    visited = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for target in adjacency[source]:
            if target not in visited:
                visited.add(target)
                frontier.append(target)
    return len(visited) == width


def required_neumann_order(rho: float, relative_tolerance: float) -> int:
    """Smallest T for the uniform relative bound through degree T."""
    if not 0 < rho < 1 or not 0 < relative_tolerance < 1:
        raise ValueError("rho and relative tolerance must lie in (0, 1)")
    order = 0
    while neumann_relative_bound(rho, order) > relative_tolerance:
        order += 1
    return order


def neumann_relative_bound(rho: float, order: int) -> float:
    if not 0 < rho < 1 or order < 0:
        raise ValueError("rho must lie in (0, 1) and order be nonnegative")
    return (1 + rho) * rho ** (order + 1) / (1 - rho)


def neumann_apply(
    permutations: list[np.ndarray],
    values: np.ndarray,
    rho: float,
    order: int,
) -> np.ndarray:
    """Apply sum_{k=0}^T rho^k S^k using sparse permutation gathers."""
    if not 0 < rho < 1 or order < 0:
        raise ValueError("rho must lie in (0, 1) and order be nonnegative")
    term = np.asarray(values, dtype=np.float64).copy()
    result = term.copy()
    for _ in range(order):
        term = rho * apply_operator(permutations, term)
        result += term
    return result


def required_rank_fraction(
    singular_values: np.ndarray, relative_tolerance: float
) -> tuple[int, float]:
    values = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("singular values must be nonempty and nonnegative")
    allowed_tail = relative_tolerance**2 * float(np.square(values).sum())
    tail = np.concatenate(
        (np.cumsum(np.square(values)[::-1])[::-1], np.zeros(1))
    )
    valid = np.flatnonzero(tail <= allowed_tail + 1e-15)
    rank = int(valid[0]) if valid.size else values.size
    return rank, rank / values.size


def best_global_sparse_error(matrix: np.ndarray, nonzero_budget: int) -> float:
    """Relative Frobenius error after retaining the largest entries globally."""
    values = np.asarray(matrix, dtype=np.float64)
    if not 0 <= nonzero_budget <= values.size:
        raise ValueError("invalid nonzero budget")
    energy = np.square(values.ravel())
    total = float(energy.sum())
    if total == 0:
        return 0.0
    if nonzero_budget == values.size:
        return 0.0
    if nonzero_budget == 0:
        return 1.0
    retained = np.partition(energy, -nonzero_budget)[-nonzero_budget:].sum()
    return math.sqrt(max(0.0, 1.0 - float(retained) / total))


def haar_orthogonal(width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    orthogonal, triangular = np.linalg.qr(rng.normal(size=(width, width)))
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1
    return orthogonal * signs


def theoretical_rank_fraction_lower_bound(
    rho: float, relative_tolerance: float
) -> float:
    """Width-independent lower bound using only ||S||_2 <= 1."""
    condition_ratio = (1 + rho) / (1 - rho)
    return max(0.0, 1.0 - (relative_tolerance * condition_ratio) ** 2)


def logical_cost_ledger(
    width: int, degree: int, order: int
) -> dict[str, float | int]:
    dense_macs = width * width
    sparse_scalar_visits = (1 + 2 * degree * order) * width
    return {
        "dense_matvec_macs": dense_macs,
        "sparse_scalar_visits_upper_bound": sparse_scalar_visits,
        "ideal_sparse_to_dense_ratio": sparse_scalar_visits / dense_macs,
        "permutation_gathers_per_iteration": 2 * degree,
        "neumann_iterations": order,
        "excludes": "physical kernel launch, memory traffic, and backward-pass cost",
    }


def run_width(
    width: int,
    *,
    degree: int,
    rho: float,
    order: int,
    seed: int,
) -> dict[str, Any]:
    permutations = make_permutations(width, degree, seed)
    operator = dense_operator(permutations)
    precision = np.eye(width) - rho * operator
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    mixer_eigenvalues = 1.0 / (1.0 - rho * eigenvalues)
    mixer = (eigenvectors * mixer_eigenvalues) @ eigenvectors.T

    rng = np.random.default_rng(seed + 1)
    inputs = rng.normal(size=(width, 8))
    exact = np.linalg.solve(precision, inputs)
    approximate = neumann_apply(permutations, inputs, rho, order)
    measured_relative_error = float(
        np.linalg.norm(exact - approximate) / np.linalg.norm(exact)
    )

    singular_values = np.sort(np.abs(mixer_eigenvalues))[::-1]
    rank_rows = {}
    for tolerance in (0.10, 0.05, 0.01):
        rank, fraction = required_rank_fraction(singular_values, tolerance)
        rank_rows[str(tolerance)] = {
            "required_rank": rank,
            "required_rank_fraction": fraction,
            "theoretical_fraction_lower_bound": (
                theoretical_rank_fraction_lower_bound(rho, tolerance)
            ),
        }

    interaction = np.eye(width) - precision
    nonzero_budget = int(np.count_nonzero(np.abs(interaction) > 1e-15))
    orthogonal = haar_orthogonal(width, seed + 2)
    rotated_precision = orthogonal @ precision @ orthogonal.T
    rotated_interaction = np.eye(width) - rotated_precision

    return {
        "width": width,
        "degree": degree,
        "rho": rho,
        "connected": is_connected(permutations),
        "operator_symmetry_error": float(np.linalg.norm(operator - operator.T)),
        "operator_spectral_norm": float(np.max(np.abs(eigenvalues))),
        "precision_minimum_eigenvalue": float(
            np.min(1.0 - rho * eigenvalues)
        ),
        "mixer_minimum_eigenvalue": float(np.min(mixer_eigenvalues)),
        "mixer_maximum_eigenvalue": float(np.max(mixer_eigenvalues)),
        "mixer_numerical_density_at_1e-12": float(
            np.mean(np.abs(mixer) > 1e-12)
        ),
        "neumann": {
            "order": order,
            "uniform_relative_error_bound": neumann_relative_bound(rho, order),
            "measured_relative_error": measured_relative_error,
        },
        "low_rank": rank_rows,
        "same_spectrum_control": {
            "mixer_spectrum_max_abs_difference": float(
                np.max(
                    np.abs(
                        np.sort(np.linalg.eigvalsh(orthogonal @ mixer @ orthogonal.T))
                        - np.sort(mixer_eigenvalues)
                    )
                )
            ),
            "precision_interaction_nonzero_budget": nonzero_budget,
            "structured_best_sparse_relative_error": best_global_sparse_error(
                interaction, nonzero_budget
            ),
            "haar_rotated_best_sparse_relative_error": best_global_sparse_error(
                rotated_interaction, nonzero_budget
            ),
        },
        "ledger": logical_cost_ledger(width, degree, order),
    }


def build_report() -> dict[str, Any]:
    order = required_neumann_order(RHO, RELATIVE_TOLERANCE)
    rows = [
        run_width(
            width,
            degree=DEGREE,
            rho=RHO,
            order=order,
            seed=20260727 + width,
        )
        for width in WIDTHS
    ]
    largest = rows[-1]
    gates = {
        "dense_mixer": largest["mixer_numerical_density_at_1e-12"] >= 0.90,
        "not_low_rank_at_10_percent_error": (
            largest["low_rank"]["0.1"]["required_rank_fraction"] >= 0.85
        ),
        "one_percent_solver_accuracy": (
            largest["neumann"]["measured_relative_error"]
            <= RELATIVE_TOLERANCE
        ),
        "sub_15_percent_logical_work_at_D512": (
            largest["ledger"]["ideal_sparse_to_dense_ratio"] <= 0.15
        ),
        "same_spectrum_control_loses_sparse_precision": (
            largest["same_spectrum_control"][
                "haar_rotated_best_sparse_relative_error"
            ]
            >= 0.75
        ),
    }
    return {
        "schema_version": 1,
        "candidate": "feature-axis sparse precision graph with implicit dense response",
        "claim": (
            "Dense full-rank and low-rank-resistant maps can have near-linear "
            "matvec circuits when represented as the inverse of a stable sparse operator."
        ),
        "source_sha256": sha256(Path(__file__)),
        "construction": "W=(I-rho*S)^-1; S is an average of permutation matrices and transposes",
        "widths": list(WIDTHS),
        "degree": DEGREE,
        "rho": RHO,
        "target_relative_tolerance": RELATIVE_TOLERANCE,
        "neumann_order": order,
        "rows": rows,
        "frozen_gates": gates,
        "algebra_gate_passed": all(gates.values()),
        "scope_boundary": {
            "proved": [
                "dense and full-rank response",
                "bad low-rank approximability",
                "near-linear logical application cost at fixed epsilon",
                "same spectrum does not imply the same sparse-precision circuit",
            ],
            "not_proved": [
                "language-model quality",
                "GPU wall-clock speed",
                "replacement of nonlinear FFN capacity",
                "novelty of the algebra outside feature-axis LLM use",
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
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "source_sha256": report["source_sha256"],
                "neumann_order": report["neumann_order"],
                "largest_width": report["rows"][-1],
                "frozen_gates": report["frozen_gates"],
                "algebra_gate_passed": report["algebra_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
