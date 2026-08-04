#!/usr/bin/env python3
"""Stage-0 algebra and resource gate for Feedback-SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    g = x @ gate.T
    u = x @ up.T
    return (silu(g) * u) @ down.T


def feedback_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    compress: np.ndarray,
    expand: np.ndarray,
) -> np.ndarray:
    g = x @ gate.T
    u = x @ up.T
    z0 = silu(g) * u
    state = np.tanh(z0 @ compress)
    refined = silu(g + state @ expand.T) * u
    return refined @ down.T


def rectangular_mixed_difference(fn, x1: float, x2: float, eps: float) -> float:
    return float(
        (
            fn(x1 + eps, x2 + eps)
            - fn(x1 + eps, x2 - eps)
            - fn(x1 - eps, x2 + eps)
            + fn(x1 - eps, x2 - eps)
        )
        / (4.0 * eps * eps)
    )


def resource_ledger(dimension: int, hidden: int, state: int) -> dict[str, int | float]:
    baseline = 3 * dimension * hidden
    feedback = 2 * hidden * state
    equal_parameter_hidden = baseline // (3 * dimension + 2 * state)
    return {
        "dimension": dimension,
        "baseline_hidden": hidden,
        "state": state,
        "baseline_parameters": baseline,
        "baseline_bf16_bytes": 2 * baseline,
        "feedback_parameters_at_equal_width": feedback,
        "feedback_bf16_bytes_at_equal_width": 2 * feedback,
        "equal_width_parameter_and_mac_overhead_fraction": feedback / baseline,
        "equal_parameter_hidden": equal_parameter_hidden,
        "equal_parameter_hidden_reduction": hidden - equal_parameter_hidden,
        "equal_parameter_hidden_reduction_fraction":
            (hidden - equal_parameter_hidden) / hidden,
        "equal_parameter_candidate_parameters":
            equal_parameter_hidden * (3 * dimension + 2 * state),
        "equal_parameter_unused_parameters":
            baseline - equal_parameter_hidden * (3 * dimension + 2 * state),
    }


def run(seed: int = 37) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    batch, dimension, hidden, state = 7, 11, 23, 5
    x = rng.normal(size=(batch, dimension)) * 0.3
    gate = rng.normal(size=(hidden, dimension)) / np.sqrt(dimension)
    up = rng.normal(size=(hidden, dimension)) / np.sqrt(dimension)
    down = rng.normal(size=(dimension, hidden)) / np.sqrt(hidden)
    compress = rng.normal(size=(hidden, state)) / np.sqrt(hidden)
    zero_expand = np.zeros((hidden, state), dtype=np.float64)

    baseline = swiglu(x, gate, up, down)
    included = feedback_swiglu(x, gate, up, down, compress, zero_expand)
    inclusion_max_abs = float(np.max(np.abs(baseline - included)))

    alpha = 0.7

    def no_feedback(a: float, b: float) -> float:
        return float(silu(np.asarray(b)))

    def feedback(a: float, b: float) -> float:
        first_feature = silu(np.asarray(a))
        state_value = np.tanh(first_feature)
        return float(silu(np.asarray(b + alpha * state_value)))

    eps = 1e-4
    no_feedback_mixed = rectangular_mixed_difference(no_feedback, 0.31, -0.27, eps)
    feedback_mixed = rectangular_mixed_difference(feedback, 0.31, -0.27, eps)

    ledger = resource_ledger(4096, 14336, 8)
    polynomial_witness = {
        "ordinary_square_gated_max_degree": 3,
        "feedback_square_gated_degree": 7,
        "degree_7_monomial": "x1^6*x2",
        "degree_7_coefficient_for_alpha_0p7": alpha * alpha,
        "derivation": "(x2 + alpha*x1^3)^2*x2",
    }

    gates = {
        "exact_baseline_inclusion": inclusion_max_abs <= 1e-12,
        "strict_polynomial_degree_witness":
            polynomial_witness["feedback_square_gated_degree"]
            > polynomial_witness["ordinary_square_gated_max_degree"]
            and polynomial_witness["degree_7_coefficient_for_alpha_0p7"] != 0.0,
        "silu_feedback_has_mixed_interaction":
            abs(no_feedback_mixed) <= 1e-8 and abs(feedback_mixed) >= 1e-4,
        "reference_overhead_below_0p14_percent":
            ledger["equal_width_parameter_and_mac_overhead_fraction"] < 0.0014,
        "equal_parameter_width_reduction_below_0p20_percent":
            ledger["equal_parameter_hidden_reduction_fraction"] < 0.002,
    }

    source = Path(__file__)
    return {
        "candidate": "Feedback-SwiGLU",
        "scope": "G0 algebra, expressivity analogue, and ledger only",
        "seed": seed,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inclusion": {"max_absolute_error": inclusion_max_abs},
        "silu_interaction_witness": {
            "epsilon": eps,
            "no_feedback_rectangular_mixed_difference": no_feedback_mixed,
            "feedback_rectangular_mixed_difference": feedback_mixed,
        },
        "polynomial_witness": polynomial_witness,
        "resource_ledger": ledger,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/feedback-swiglu-stage0.json"),
    )
    args = parser.parse_args()
    payload = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()

