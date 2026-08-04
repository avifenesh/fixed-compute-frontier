#!/usr/bin/env python3
"""Minimal matched separation: coupling versus linear and self-hinge maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def design_matrices(samples: np.ndarray) -> dict[str, np.ndarray]:
    a = samples[:, 0]
    b = samples[:, 1]
    return {
        "linear": np.column_stack((a, b)),
        "coupling": np.column_stack((a, b, np.abs(a))),
    }


def least_squares_error(design: np.ndarray, target: np.ndarray) -> float:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    return float(np.mean((prediction - target) ** 2))


def run(points: int = 4096, q: float = 0.7, coefficient_cap: float = 1.0) -> dict:
    angles = (np.arange(points) + 0.5) * (2 * np.pi / points)
    samples = np.column_stack((np.cos(angles), np.sin(angles)))
    target = samples[:, 1] + q * np.abs(samples[:, 0])
    designs = design_matrices(samples)
    errors = {
        name: least_squares_error(design, target)
        for name, design in designs.items()
    }
    # For a projected self-hinge, the a-dependent part is
    # u*a + u*gamma*|a|.  On a uniform circle a and |a| are orthogonal.  Under
    # the invertibility bound |gamma|<=c, minimizing over u and gamma gives
    # 0.5*q^2/(1+c^2), attained at |gamma|=c.
    bounded_self_hinge_mse = 0.5 * q * q / (1.0 + coefficient_cap**2)
    return {
        "schema": "projection-shared-coupling-separation-v1",
        "points": points,
        "q": q,
        "coefficient_cap": coefficient_cap,
        "mse": errors,
        "bounded_self_hinge_mse": bounded_self_hinge_mse,
        "linear_gap": errors["linear"],
        "coupling_exact": errors["coupling"] < 1e-25,
        "coupling_separates_from_bounded_self_hinge": (
            errors["coupling"] < 1e-25 and bounded_self_hinge_mse > 0.1
        ),
    }


def main() -> None:
    payload = run()
    output = Path("results/projection-shared-coupling-separation.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
