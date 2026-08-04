#!/usr/bin/env python3
"""Algebra and resource gate for block-algebra SwiGLU.

Ordinary SwiGLU multiplies matching projected scalars.  This candidate groups
two gate and two value coordinates and uses real complex multiplication:

    (a0, a1) * (u0, u1)
      = (a0*u0 - a1*u1, a0*u1 + a1*u0).

The three dense G/U/V matrices and hidden width are unchanged.  The real
multiplication tensor has rank three, whereas two independent coordinatewise
products have rank two.  This file records the exact rank witness, the barrier
that invalidates a tiny zero-init retrofit, and the target-scale arithmetic
ledger before any language-model experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def diagonal_core() -> np.ndarray:
    """Return C[k,i,j] for two coordinatewise products."""
    core = np.zeros((2, 2, 2), dtype=np.float64)
    core[0, 0, 0] = 1.0
    core[1, 1, 1] = 1.0
    return core


def complex_core(scale: float = 1.0) -> np.ndarray:
    """Return C[k,i,j] for scaled real complex multiplication."""
    core = np.zeros((2, 2, 2), dtype=np.float64)
    core[0, 0, 0] = scale
    core[0, 1, 1] = -scale
    core[1, 0, 1] = scale
    core[1, 1, 0] = scale
    return core


def apply_core(a: np.ndarray, u: np.ndarray, core: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...j,kij->...k", a, u, core)


def complex_direct(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.stack(
        (a[..., 0] * u[..., 0] - a[..., 1] * u[..., 1],
         a[..., 0] * u[..., 1] + a[..., 1] * u[..., 0]),
        axis=-1,
    )


def complex_gauss_three_multiply(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Gauss's exact rank-three real multiplication algorithm."""
    p0 = a[..., 0] * u[..., 0]
    p1 = a[..., 1] * u[..., 1]
    p2 = (a[..., 0] + a[..., 1]) * (u[..., 0] + u[..., 1])
    return np.stack((p0 - p1, p2 - p0 - p1), axis=-1)


def determinant_pencil_coefficients(core: np.ndarray) -> tuple[float, float, float]:
    """Return A,B,C for det(s*C[0] + t*C[1]) = A*s^2+B*s*t+C*t^2."""
    first, second = core
    coefficient_s2 = float(np.linalg.det(first))
    coefficient_t2 = float(np.linalg.det(second))
    # Evaluate at s=t=1 to recover the mixed coefficient exactly enough for
    # these small integer cores.
    coefficient_st = float(
        np.linalg.det(first + second) - coefficient_s2 - coefficient_t2
    )
    return coefficient_s2, coefficient_st, coefficient_t2


def pencil_discriminant(core: np.ndarray) -> float:
    a, b, c = determinant_pencil_coefficients(core)
    return b * b - 4.0 * a * c


def interpolated_core(alpha: float) -> np.ndarray:
    return (1.0 - alpha) * diagonal_core() + alpha * complex_core()


def interpolation_discriminant(alpha: float) -> float:
    """Closed form for the diagonal-to-complex tensor pencil."""
    return (1.0 - alpha) ** 2 - 4.0 * alpha**3


def rank_three_boundary() -> float:
    """Unique root in (0,1) after which the real tensor is nonsplit/rank 3."""
    lower, upper = 0.0, 1.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if interpolation_discriminant(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def resource_ledger(dimension: int = 4096, width: int = 14336) -> dict[str, int | float]:
    if width % 2:
        raise ValueError("width must be divisible by two")
    dense_macs = 3 * dimension * width
    baseline_products = width
    complex_direct_products = 2 * width
    complex_gauss_products = 3 * width // 2
    return {
        "dimension": dimension,
        "width": width,
        "dense_projection_macs_per_token": dense_macs,
        "baseline_pointwise_products_per_token": baseline_products,
        "complex_direct_products_per_token": complex_direct_products,
        "complex_gauss_products_per_token": complex_gauss_products,
        "gauss_extra_products_over_baseline": complex_gauss_products - baseline_products,
        "gauss_extra_product_fraction_of_dense_macs": (
            complex_gauss_products - baseline_products
        ) / dense_macs,
        "dense_weight_parameters": dense_macs,
        "candidate_dense_weight_parameters": dense_macs,
        "candidate_extra_parameters_fixed_core": 0,
        "activation_output_scalars": width,
    }


def run_gate(seed: int = 211, trials: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    maximum_core_error = 0.0
    maximum_gauss_error = 0.0
    for _ in range(trials):
        a = rng.normal(size=(37, 2))
        u = rng.normal(size=(37, 2))
        expected = complex_direct(a, u)
        maximum_core_error = max(
            maximum_core_error,
            float(np.max(np.abs(apply_core(a, u, complex_core()) - expected))),
        )
        maximum_gauss_error = max(
            maximum_gauss_error,
            float(np.max(np.abs(complex_gauss_three_multiply(a, u) - expected))),
        )

    diagonal_coefficients = determinant_pencil_coefficients(diagonal_core())
    complex_coefficients = determinant_pencil_coefficients(complex_core())
    boundary = rank_three_boundary()
    ledger = resource_ledger()
    metrics = {
        "maximum_core_implementation_error": maximum_core_error,
        "maximum_gauss_implementation_error": maximum_gauss_error,
        "diagonal_pencil_coefficients": diagonal_coefficients,
        "diagonal_pencil_discriminant": pencil_discriminant(diagonal_core()),
        "complex_pencil_coefficients": complex_coefficients,
        "complex_pencil_discriminant": pencil_discriminant(complex_core()),
        "diagonal_real_tensor_rank": 2,
        "complex_real_tensor_rank": 3,
        "interpolation_rank_three_boundary": boundary,
        "discriminant_just_below_boundary": interpolation_discriminant(boundary - 1e-6),
        "discriminant_just_above_boundary": interpolation_discriminant(boundary + 1e-6),
    }
    gates = {
        "core_matches_complex_product": maximum_core_error < 1e-12,
        "gauss_rank_three_algorithm_is_exact": maximum_gauss_error < 1e-12,
        "diagonal_pencil_has_two_real_roots": pencil_discriminant(diagonal_core()) > 0.0,
        "complex_pencil_has_no_real_roots": pencil_discriminant(complex_core()) < 0.0,
        "real_change_of_basis_cannot_diagonalize_complex_core": (
            pencil_discriminant(diagonal_core()) * pencil_discriminant(complex_core()) < 0.0
        ),
        "zero_init_neighborhood_stays_rank_two": boundary > 0.4,
        "fixed_core_adds_no_parameters_or_dense_macs": (
            ledger["candidate_extra_parameters_fixed_core"] == 0
            and ledger["candidate_dense_weight_parameters"] == ledger["dense_weight_parameters"]
        ),
        "gauss_extra_products_below_0p01_percent_of_dense_macs": (
            ledger["gauss_extra_product_fraction_of_dense_macs"] < 1e-4
        ),
    }
    return {
        "candidate": "rank-three-block-algebra-swiglu",
        "gate": "G0-real-tensor-rank-and-resource-ledger",
        "seed": seed,
        "trials": trials,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics": metrics,
        "resource_ledger": ledger,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "next_valid_test": (
            "matched from-scratch LM screen initialized in the rank-three region; "
            "a tiny zero-initialized checkpoint sidecar is invalid"
        ),
        "claim_boundary": {
            "proved": (
                "1.5x real bilinear tensor rank per two hidden coordinates at identical "
                "G/U/V dimensions and fixed-core parameter count"
            ),
            "not_proved": "better language loss or deployable fused-kernel latency",
            "prior_boundary": (
                "hypercomplex neural layers exist; this gate only tests the activation-core "
                "placement inside a dense real SwiGLU FFN"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_gate(seed=args.seed, trials=args.trials)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if result["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
