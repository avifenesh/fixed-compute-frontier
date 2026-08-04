#!/usr/bin/env python3
"""Frozen algebra gate for the in-place midpoint hinge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OUTPUT = Path("results/inplace-hinge-reduction-algebra.json")


def dense(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def midpoint_hinge(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a + b + alpha * np.abs(a)


def final_hinge(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    summed = a + b
    return summed + alpha * np.abs(summed)


def silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def run(seed: int = 20260728) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    a = generator.integers(-128, 129, size=4096, dtype=np.int64)
    b = generator.integers(-128, 129, size=4096, dtype=np.int64)
    endpoint_error = int(np.max(np.abs(midpoint_hinge(a, b, 0.0) - dense(a, b))))

    collision_a = np.array([1.0, 2.0])
    collision_b = -collision_a
    collision_dense = dense(collision_a, collision_b)
    collision_candidate = midpoint_hinge(collision_a, collision_b, 1.0)

    alpha = 0.75
    midpoint_side_gradients = np.array(
        [[1.0 + alpha, 1.0], [1.0 - alpha, 1.0]], dtype=np.float64
    )
    final_side_gradients = np.array(
        [[1.0 + alpha, 1.0 + alpha], [1.0 - alpha, 1.0 - alpha]],
        dtype=np.float64,
    )

    # A single SwiGLU neuron's up/down sign orbit is function-null and can
    # encode a binary layer mode without adding a stored parameter.
    gate = generator.normal(size=128)
    up = generator.normal(size=128)
    down = generator.normal(size=64)
    inputs = generator.normal(size=(31, 128))
    ordinary = np.outer(silu(inputs @ gate) * (inputs @ up), down)
    gauge_flipped = np.outer(silu(inputs @ gate) * (inputs @ (-up)), -down)

    return {
        "schema": "inplace-hinge-reduction-algebra-v1",
        "seed": seed,
        "baseline_endpoint": {
            "maximum_absolute_error_at_alpha_zero": endpoint_error,
            "exact": endpoint_error == 0,
        },
        "strict_information_witness": {
            "partial_a": collision_a.tolist(),
            "partial_b": collision_b.tolist(),
            "dense_outputs": collision_dense.tolist(),
            "midpoint_hinge_outputs": collision_candidate.tolist(),
            "same_dense_different_candidate": bool(
                collision_dense[0] == collision_dense[1]
                and collision_candidate[0] != collision_candidate[1]
            ),
        },
        "side_gradient_span": {
            "alpha": alpha,
            "midpoint_gradients": midpoint_side_gradients.tolist(),
            "final_only_gradients": final_side_gradients.tolist(),
            "midpoint_rank": int(np.linalg.matrix_rank(midpoint_side_gradients)),
            "final_only_rank": int(np.linalg.matrix_rank(final_side_gradients)),
            "midpoint_determinant": float(np.linalg.det(midpoint_side_gradients)),
            "gradient_jump": [2.0 * alpha, 0.0],
        },
        "mode_bit_funding": {
            "mechanism": "paired sign flip of one up row and matching down column",
            "maximum_function_error": float(np.max(np.abs(ordinary - gauge_flipped))),
            "exact_to_float_tolerance": bool(
                np.max(np.abs(ordinary - gauge_flipped)) < 1e-12
            ),
            "extra_stored_parameters": 0,
        },
        "interpretation": {
            "alpha_one": "B + 2*ReLU(A)",
            "gain": "two nonparallel slope regimes from one projection row",
            "control": "final-only hinge has only one projection direction",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    arguments = parser.parse_args()
    payload = run(arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
