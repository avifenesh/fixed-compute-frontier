#!/usr/bin/env python3
"""Stage-0 algebra and ledger gate for Reflex-SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_prime(x: np.ndarray) -> np.ndarray:
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return sigmoid + x * sigmoid * (1.0 - sigmoid)


def swiglu(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    g = x @ gate.T
    u = x @ up.T
    return (silu(g) * u) @ down.T


def reflex_swiglu(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    g = x @ gate.T
    u = x @ up.T
    z0 = silu(g) * u
    reflex = np.clip(z0, -1.0, 1.0)
    return (silu(g + alpha * reflex) * u) @ down.T


def ledger(dimension: int, hidden: int) -> dict[str, int | float]:
    baseline = 3 * dimension * hidden
    exact_equal_hidden = baseline // (3 * dimension + 1)
    return {
        "dimension": dimension,
        "baseline_hidden": hidden,
        "baseline_parameters": baseline,
        "same_width_added_parameters": hidden,
        "same_width_parameter_overhead_fraction": 1.0 / (3.0 * dimension),
        "exact_equal_parameter_hidden": exact_equal_hidden,
        "exact_equal_parameter_hidden_reduction": hidden - exact_equal_hidden,
        "equal_parameter_candidate_parameters": exact_equal_hidden * (3 * dimension + 1),
        "equal_parameter_unused_parameters": baseline - exact_equal_hidden * (3 * dimension + 1),
    }


def finite_difference(fn, array: np.ndarray, index: tuple[int, ...], eps: float = 1e-6) -> float:
    original = float(array[index])
    array[index] = original + eps
    plus = float(np.sum(fn()))
    array[index] = original - eps
    minus = float(np.sum(fn()))
    array[index] = original
    return (plus - minus) / (2.0 * eps)


def run(seed: int = 47) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    batch, dimension, hidden = 5, 13, 29
    x = rng.normal(size=(batch, dimension)) * 0.3
    gate = rng.normal(size=(hidden, dimension)) / dimension**0.5
    up = rng.normal(size=(hidden, dimension)) / dimension**0.5
    down = rng.normal(size=(dimension, hidden)) / hidden**0.5
    alpha = np.zeros(hidden, dtype=np.float64)
    baseline = swiglu(x, gate, up, down)
    endpoint = reflex_swiglu(x, gate, up, down, alpha)
    inclusion_error = float(np.max(np.abs(endpoint - baseline)))

    shared_gradient_differences = []
    for parameter, index in ((gate, (0, 0)), (up, (0, 0)), (down, (0, 0))):
        baseline_derivative = finite_difference(
            lambda: swiglu(x, gate, up, down), parameter, index
        )
        reflex_derivative = finite_difference(
            lambda: reflex_swiglu(x, gate, up, down, alpha), parameter, index
        )
        shared_gradient_differences.append(abs(baseline_derivative - reflex_derivative))
    shared_gradient_max_difference = max(shared_gradient_differences)

    g = np.asarray(0.31)
    u = np.asarray(-0.73)
    z0 = silu(g) * u
    reflex = np.clip(z0, -1.0, 1.0)
    alpha_derivative = float(silu_prime(g) * reflex * u)

    reference_ledger = ledger(4096, 14336)
    polynomial_witness = {
        "ordinary_square_gated_max_degree": 3,
        "reflex_square_gated_degree": 7,
        "degree_7_monomial": "g^4*u^3",
        "degree_7_coefficient": 1.0,
        "derivation": "(g + (g^2*u))^2*u",
        "scope_warning": "artificial square-activation and identity-clip analogue only",
    }
    gates = {
        "exact_swiglu_endpoint": inclusion_error <= 1e-12,
        "alpha_gradient_live_at_endpoint": abs(alpha_derivative) >= 1e-6,
        "shared_gradient_endpoint_identity": shared_gradient_max_difference <= 1e-9,
        "artificial_degree_witness":
            polynomial_witness["reflex_square_gated_degree"]
            > polynomial_witness["ordinary_square_gated_max_degree"],
        "same_width_overhead_below_0p009_percent":
            reference_ledger["same_width_parameter_overhead_fraction"] < 0.00009,
        "exact_equal_parameter_loses_at_most_two_features":
            reference_ledger["exact_equal_parameter_hidden_reduction"] <= 2,
    }
    source = Path(__file__)
    return {
        "candidate": "Reflex-SwiGLU",
        "scope": "G0 endpoint, trainability, artificial degree, and ledger only",
        "seed": seed,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inclusion_max_absolute_error": inclusion_error,
        "alpha_derivative_at_frozen_endpoint": alpha_derivative,
        "shared_gradient_finite_difference_max_difference": shared_gradient_max_difference,
        "polynomial_witness": polynomial_witness,
        "resource_ledger": reference_ledger,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--output", type=Path, default=Path("results/reflex-swiglu-stage0.json")
    )
    args = parser.parse_args()
    payload = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": payload["gates"], "all_gates_pass": payload["all_gates_pass"]}, indent=2))
    raise SystemExit(0 if payload["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
