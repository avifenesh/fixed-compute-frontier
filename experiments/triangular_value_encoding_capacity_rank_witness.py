#!/usr/bin/env python3
"""Numerical witness for the TVE capacity-rank theorem.

This is not the proof. It detects indexing, orientation, and independence errors
in the theorem on small generic systems.
"""

from __future__ import annotations

import json
import math

import numpy as np


def skew_basis(dimension: int) -> list[np.ndarray]:
    basis = []
    for row in range(1, dimension):
        for column in range(row):
            value = np.zeros((dimension, dimension), dtype=np.float64)
            value[row, column] = 1.0
            value[column, row] = -1.0
            basis.append(value)
    return basis


def strict_lower_vector(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.tril_indices(matrix.shape[0], k=-1)]


def matrix_exponential_skew(generator: np.ndarray, scale: float) -> np.ndarray:
    # A real skew-symmetric matrix is normal. The complex eigendecomposition is
    # stable enough for this small numerical witness.
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    inverse = np.linalg.inv(eigenvectors)
    result = eigenvectors @ np.diag(np.exp(scale * eigenvalues)) @ inverse
    return np.real_if_close(result, tol=1000).real


def run_case(dimension: int, seed: int) -> dict[str, float | int | bool]:
    generator = np.random.default_rng(seed)
    upper = np.triu(generator.normal(size=(dimension, dimension)))
    diagonal = generator.uniform(0.75, 1.25, size=dimension)
    upper[np.diag_indices(dimension)] = diagonal
    value = generator.normal(size=(dimension, dimension))
    while abs(np.linalg.det(value)) < 1e-6:
        value = generator.normal(size=(dimension, dimension))

    basis = skew_basis(dimension)
    gauge_map = np.stack(
        [strict_lower_vector(-(upper @ direction)) for direction in basis],
        axis=1,
    )
    gauge_rank = int(np.linalg.matrix_rank(gauge_map))

    sample_count = max(dimension, math.ceil(2 * len(basis) / dimension))
    inputs = generator.normal(size=(sample_count, dimension))
    transformed = inputs @ value.T
    features = transformed * np.abs(transformed)
    functional_columns = []
    for direction in basis:
        coefficient_direction = np.tril(-(upper @ direction), k=-1)
        outputs = features @ coefficient_direction.T @ upper.T
        functional_columns.append(outputs.reshape(-1))
    functional_matrix = np.stack(functional_columns, axis=1)
    functional_rank = int(np.linalg.matrix_rank(functional_matrix))

    direction = sum((index + 1) * item for index, item in enumerate(basis))
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    gauge = matrix_exponential_skew(direction, epsilon)
    moved_value = gauge @ value
    moved_output = upper @ gauge.T
    standard_before = inputs @ value.T @ upper.T
    standard_after = inputs @ moved_value.T @ moved_output.T
    standard_gauge_max_abs = float(np.max(np.abs(standard_after - standard_before)))

    coefficient = np.tril(moved_output, k=-1)
    encoded = moved_value @ inputs.T
    encoded = encoded + coefficient @ (encoded * np.abs(encoded))
    tve_after = (moved_output @ encoded).T
    finite_difference = (tve_after - standard_before) / epsilon
    coefficient_derivative = np.tril(-(upper @ direction), k=-1)
    analytic = (upper @ coefficient_derivative @ features.T).T
    derivative_relative_error = float(
        np.linalg.norm(finite_difference - analytic)
        / np.linalg.norm(analytic)
    )

    expected = len(basis)
    return {
        "dimension": dimension,
        "gauge_directions": expected,
        "gauge_map_rank": gauge_rank,
        "sampled_function_rank": functional_rank,
        "sample_count": sample_count,
        "standard_gauge_max_abs": standard_gauge_max_abs,
        "tve_derivative_relative_error": derivative_relative_error,
        "pass": (
            gauge_rank == expected
            and functional_rank == expected
            and standard_gauge_max_abs <= 1e-10
            and derivative_relative_error <= 1e-4
        ),
    }


def main() -> None:
    cases = [run_case(dimension, 1901 + dimension) for dimension in (4, 8, 16)]
    print(json.dumps({"cases": cases, "all_pass": all(case["pass"] for case in cases)}, indent=2))


if __name__ == "__main__":
    main()
