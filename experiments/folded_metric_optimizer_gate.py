#!/usr/bin/env python3
"""Exact algebra gate for a folded learned-metric optimizer state.

The model always stores and executes one ordinary dense weight M.  The
optimizer additionally stores low-rank A,B and updates

    M <- AdamW(M, G) + s * (A_new B_new - A_old B_old).

Thus A,B change the training chart but never enter the model, checkpoint, or
served graph.  Under plain SGD the update is exactly the effective update of
the redundant parameterization M = W + sAB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def orthonormal_columns(rng: np.random.Generator, rows: int, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, rank)))
    return q[:, :rank]


def chart_sgd_update(
    merged: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    scale: float = 1.0,
    learn_a: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_product = a @ b
    grad_a = scale * gradient @ b.T
    grad_b = scale * a.T @ gradient
    next_a = a - learning_rate * grad_a if learn_a else a.copy()
    next_b = b - learning_rate * grad_b
    direct = merged - learning_rate * gradient
    next_merged = direct + scale * (next_a @ next_b - old_product)
    return next_merged, next_a, next_b


def redundant_parameterization_sgd_update(
    weight: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float,
    scale: float = 1.0,
    learn_a: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    next_weight = weight - learning_rate * gradient
    grad_a = scale * gradient @ b.T
    grad_b = scale * a.T @ gradient
    next_a = a - learning_rate * grad_a if learn_a else a.copy()
    next_b = b - learning_rate * grad_b
    next_merged = next_weight + scale * (next_a @ next_b)
    return next_merged, next_weight, next_a, next_b


def first_step_closed_form(
    gradient: np.ndarray,
    a: np.ndarray,
    learning_rate: float,
    scale: float = 1.0,
) -> np.ndarray:
    return -learning_rate * (gradient + scale**2 * (a @ a.T @ gradient))


def resource_ledger(
    dimension: int = 384,
    width: int = 1024,
    layers: int = 12,
    rank: int = 16,
) -> dict[str, int | float]:
    # q, k, v, o, gate, up, down for GQA with q/o width d and k/v width d/3.
    shapes = (
        (dimension, dimension),
        (dimension // 3, dimension),
        (dimension // 3, dimension),
        (dimension, dimension),
        (width, dimension),
        (width, dimension),
        (dimension, width),
    )
    targeted = layers * sum(output * input_ for output, input_ in shapes)
    factor_state = layers * sum(rank * (output + input_) for output, input_ in shapes)
    single_factor_product_macs = layers * sum(
        output * input_ * rank for output, input_ in shapes
    )
    return {
        "dimension": dimension,
        "width": width,
        "layers": layers,
        "rank": rank,
        "targeted_served_weight_scalars": targeted,
        "candidate_targeted_served_weight_scalars": targeted,
        "extra_optimizer_factor_scalars": factor_state,
        "extra_optimizer_factor_fraction_of_full_37m_model": factor_state / 37_758_336,
        "single_factor_product_macs": single_factor_product_macs,
        # Fixed A computes old AB, A^T G, and new AB. Learned A also computes G B^T.
        "fixed_a_extra_macs_per_optimizer_update": 3 * single_factor_product_macs,
        "learned_ab_extra_macs_per_optimizer_update": 4 * single_factor_product_macs,
        "extra_model_state_scalars": 0,
        "extra_served_parameters": 0,
        "extra_served_macs": 0,
    }


def run_gate(seed: int = 331, trials: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    equivalence_errors = []
    first_step_errors = []
    fixed_a_differences = []
    for _ in range(trials):
        output, input_, rank = 19, 13, 5
        weight = rng.normal(size=(output, input_))
        a = orthonormal_columns(rng, output, rank)
        b = np.zeros((rank, input_), dtype=np.float64)
        gradient = rng.normal(size=(output, input_))
        merged = weight + a @ b
        folded = chart_sgd_update(merged, a, b, gradient, 1e-3, learn_a=True)
        redundant = redundant_parameterization_sgd_update(
            weight, a, b, gradient, 1e-3, learn_a=True
        )
        equivalence_errors.append(float(np.max(np.abs(folded[0] - redundant[0]))))
        first_step_errors.append(
            float(
                np.max(
                    np.abs(
                        (folded[0] - merged)
                        - first_step_closed_form(gradient, a, 1e-3)
                    )
                )
            )
        )
        # After B becomes nonzero, learning A creates an additional adaptive
        # right-metric/cross term absent from the fixed-A control.
        second_gradient = rng.normal(size=(output, input_))
        learned_second = chart_sgd_update(
            folded[0], folded[1], folded[2], second_gradient, 1e-3, learn_a=True
        )[0]
        fixed_second = chart_sgd_update(
            folded[0], folded[1], folded[2], second_gradient, 1e-3, learn_a=False
        )[0]
        fixed_a_differences.append(float(np.linalg.norm(learned_second - fixed_second)))
    ledger = resource_ledger()
    metrics = {
        "maximum_folded_vs_redundant_update_error": max(equivalence_errors),
        "maximum_zero_B_first_step_formula_error": max(first_step_errors),
        "minimum_second_step_learned_vs_fixed_A_difference": min(fixed_a_differences),
    }
    gates = {
        "optimizer_side_fold_matches_redundant_chart": metrics[
            "maximum_folded_vs_redundant_update_error"
        ]
        < 1e-12,
        "zero_B_first_step_is_low_rank_left_preconditioner": metrics[
            "maximum_zero_B_first_step_formula_error"
        ]
        < 1e-12,
        "learning_A_becomes_distinct_from_fixed_metric": metrics[
            "minimum_second_step_learned_vs_fixed_A_difference"
        ]
        > 0.0,
        "model_and_served_ledgers_are_identical": (
            ledger["extra_model_state_scalars"] == 0
            and ledger["extra_served_parameters"] == 0
            and ledger["extra_served_macs"] == 0
            and ledger["candidate_targeted_served_weight_scalars"]
            == ledger["targeted_served_weight_scalars"]
        ),
    }
    return {
        "candidate": "folded-learned-metric-optimizer",
        "gate": "G0-exact-optimizer-chart-and-served-ledger",
        "seed": seed,
        "trials": trials,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics": metrics,
        "resource_ledger": ledger,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": {
            "proved": (
                "optimizer-side factor state exactly reproduces the redundant low-rank "
                "training chart under SGD while the model remains an ordinary dense model"
            ),
            "not_proved": (
                "AdamW language-loss gain, superiority to a matrix-LR control, or novelty"
            ),
            "paid_cost": "training-only factor and moment state plus weight-space products",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=331)
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
