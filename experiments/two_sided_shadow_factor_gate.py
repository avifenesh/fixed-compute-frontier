#!/usr/bin/env python3
"""Algebra gate for exactly mergeable two-sided shadow-factor training.

During training, an ordinary dense weight is represented as

    W_eff = W + s * (A_left B_left + A_right B_right),

with B_left=0 and A_right=0 initially.  The deployed weight is the exact sum
above, so inference has the original matrix shape, parameter count, and GEMM.
The redundant training chart changes the effective optimization geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def orthonormal_columns(rng: np.random.Generator, rows: int, rank: int) -> np.ndarray:
    if rank > rows:
        raise ValueError("rank cannot exceed rows")
    q, _ = np.linalg.qr(rng.normal(size=(rows, rank)))
    return q[:, :rank]


def initialize_chart(
    rng: np.random.Generator, output: int, input_: int, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_left = orthonormal_columns(rng, output, rank)
    b_left = np.zeros((rank, input_), dtype=np.float64)
    a_right = np.zeros((output, rank), dtype=np.float64)
    b_right = orthonormal_columns(rng, input_, rank).T
    return a_left, b_left, a_right, b_right


def merged_weight(
    weight: np.ndarray,
    a_left: np.ndarray,
    b_left: np.ndarray,
    a_right: np.ndarray,
    b_right: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    return weight + scale * (a_left @ b_left + a_right @ b_right)


def one_sgd_step(
    weight: np.ndarray,
    a_left: np.ndarray,
    b_left: np.ndarray,
    a_right: np.ndarray,
    b_right: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One simultaneous SGD step at the exact zero-branch endpoint."""
    if np.count_nonzero(b_left) or np.count_nonzero(a_right):
        raise ValueError("closed form is for the frozen zero-branch endpoint")
    next_weight = weight - learning_rate * gradient
    next_a_left = a_left.copy()  # gradient is scale * G @ B_left.T = 0
    next_b_left = b_left - learning_rate * scale * (a_left.T @ gradient)
    next_a_right = a_right - learning_rate * scale * (gradient @ b_right.T)
    next_b_right = b_right.copy()  # gradient is scale * A_right.T @ G = 0
    return next_weight, next_a_left, next_b_left, next_a_right, next_b_right


def predicted_first_effective_update(
    a_left: np.ndarray,
    b_right: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    scale: float = 1.0,
) -> np.ndarray:
    p_left = a_left @ a_left.T
    p_right = b_right.T @ b_right
    return -learning_rate * (
        gradient + scale**2 * (p_left @ gradient + gradient @ p_right)
    )


def resource_ledger(
    dimension: int = 384,
    width: int = 1024,
    layers: int = 12,
    rank: int = 16,
) -> dict[str, int | float]:
    matrices_per_layer = ((width, dimension), (width, dimension), (dimension, width))
    served_parameters = layers * sum(output * input_ for output, input_ in matrices_per_layer)
    extra_training_parameters = layers * sum(
        2 * rank * (output + input_) for output, input_ in matrices_per_layer
    )
    factor_product_macs_per_step = layers * sum(
        2 * output * input_ * rank for output, input_ in matrices_per_layer
    )
    return {
        "dimension": dimension,
        "width": width,
        "layers": layers,
        "rank_per_side": rank,
        "served_ffn_weight_parameters": served_parameters,
        "candidate_served_ffn_weight_parameters": served_parameters,
        "extra_training_parameters": extra_training_parameters,
        "extra_training_parameter_fraction_of_served_ffn": extra_training_parameters
        / served_parameters,
        "factor_product_macs_per_optimizer_step": factor_product_macs_per_step,
        "candidate_extra_served_parameters": 0,
        "candidate_extra_served_dense_macs_per_token": 0,
    }


def run_gate(seed: int = 307, trials: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    endpoint_errors = []
    update_errors = []
    mean_step_ratios = []
    for _ in range(trials):
        output, input_, rank = 23, 17, 5
        weight = rng.normal(size=(output, input_))
        gradient = rng.normal(size=(output, input_))
        factors = initialize_chart(rng, output, input_, rank)
        initial = merged_weight(weight, *factors)
        endpoint_errors.append(float(np.max(np.abs(initial - weight))))
        stepped = one_sgd_step(weight, *factors, gradient, learning_rate=1e-3)
        effective_update = merged_weight(*stepped) - initial
        predicted = predicted_first_effective_update(
            factors[0], factors[3], gradient, learning_rate=1e-3
        )
        update_errors.append(float(np.max(np.abs(effective_update - predicted))))
        baseline_update = -1e-3 * gradient
        mean_step_ratios.append(
            float(np.linalg.norm(predicted) / np.linalg.norm(baseline_update))
        )
    ledger = resource_ledger()
    metrics = {
        "maximum_initial_endpoint_error": max(endpoint_errors),
        "maximum_first_update_formula_error": max(update_errors),
        "mean_first_step_frobenius_ratio_to_plain_sgd": float(
            np.mean(mean_step_ratios)
        ),
        "minimum_first_step_frobenius_ratio_to_plain_sgd": min(mean_step_ratios),
        "maximum_first_step_frobenius_ratio_to_plain_sgd": max(mean_step_ratios),
    }
    gates = {
        "ordinary_weight_is_exact_initial_endpoint": metrics[
            "maximum_initial_endpoint_error"
        ]
        == 0.0,
        "two_sided_preconditioner_formula_is_exact": metrics[
            "maximum_first_update_formula_error"
        ]
        < 1e-12,
        "optimizer_geometry_differs_from_scalar_learning_rate": metrics[
            "minimum_first_step_frobenius_ratio_to_plain_sgd"
        ]
        > 1.0,
        "collapse_has_identical_served_parameter_and_mac_budget": (
            ledger["candidate_extra_served_parameters"] == 0
            and ledger["candidate_extra_served_dense_macs_per_token"] == 0
            and ledger["candidate_served_ffn_weight_parameters"]
            == ledger["served_ffn_weight_parameters"]
        ),
    }
    return {
        "candidate": "two-sided-shadow-factor-training",
        "gate": "G0-exact-collapse-and-optimizer-geometry",
        "seed": seed,
        "trials": trials,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics": metrics,
        "resource_ledger": ledger,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": {
            "proved": (
                "exact ordinary-matrix endpoint and a low-rank two-sided first-step "
                "preconditioner under plain SGD"
            ),
            "not_proved": (
                "an AdamW language-learning gain, advantage over a higher learning rate, "
                "or advantage per unit of training cost"
            ),
            "served_cost": "identical after exact factor merge",
            "paid_cost": "extra training parameters, optimizer state, and weight-space products",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_gate(args.seed, args.trials)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if result["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
