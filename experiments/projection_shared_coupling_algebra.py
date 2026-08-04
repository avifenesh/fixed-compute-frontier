#!/usr/bin/env python3
"""Executable algebra checks for projection-shared coupling."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def coupling(a: np.ndarray, b: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    b_prime = b + alpha * np.abs(a)
    a_prime = a + beta * np.abs(b_prime)
    return a_prime, b_prime


def inverse_coupling(
    a_prime: np.ndarray,
    b_prime: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
):
    a = a_prime - beta * np.abs(b_prime)
    b = b_prime - alpha * np.abs(a)
    return a, b


def one_way_jacobian(signs: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    p = alpha.size
    result = np.eye(2 * p)
    result[p:, :p] = np.diag(alpha * signs)
    return result


def main() -> None:
    rng = np.random.default_rng(7)
    p = 4
    a = rng.normal(size=p)
    b = rng.normal(size=p)
    alpha = rng.normal(size=p)
    beta = rng.normal(size=p)

    a_prime, b_prime = coupling(a, b, alpha, beta)
    a_roundtrip, b_roundtrip = inverse_coupling(
        a_prime, b_prime, alpha, beta
    )

    zero_a, zero_b = coupling(a, b, np.zeros(p), np.zeros(p))
    jacobians = [
        one_way_jacobian(np.asarray(signs), alpha)
        for signs in itertools.product((-1.0, 1.0), repeat=p)
    ]
    distinct = len({matrix.tobytes() for matrix in jacobians})

    gains = rng.normal(size=2 * p)
    normalized = rng.normal(size=(5, 2 * p))
    consumers = [rng.normal(size=(2 * p, width)) for width in (3, 7, 11)]
    gain_fold_errors = []
    for weight in consumers:
        baseline = (normalized * gains) @ weight
        folded = normalized @ (gains[:, None] * weight)
        gain_fold_errors.append(float(np.max(np.abs(baseline - folded))))

    payload = {
        "schema": "projection-shared-coupling-algebra-v1",
        "pairs": p,
        "identity_max_error": float(
            max(np.max(np.abs(zero_a - a)), np.max(np.abs(zero_b - b)))
        ),
        "inverse_max_error": float(
            max(
                np.max(np.abs(a_roundtrip - a)),
                np.max(np.abs(b_roundtrip - b)),
            )
        ),
        "one_way_jacobian_determinants": [
            float(np.linalg.det(matrix)) for matrix in jacobians
        ],
        "distinct_one_way_jacobians": distinct,
        "expected_distinct_jacobians": 2**p,
        "multiple_consumer_gain_fold_max_errors": gain_fold_errors,
        "parameter_chart": {
            "baseline_rmsnorm_gain": 2 * p,
            "candidate_alpha_plus_beta": 2 * p,
        },
    }
    output = Path("results/projection-shared-coupling-algebra.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

