#!/usr/bin/env python3
"""Deterministic algebra gate for capped dense-exception scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "dense-exception-scaling-law.json"
WIDTHS = (256, 512, 1024, 2048, 4096, 8192)
SPECTRAL_EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
TOTAL_TOLERANCES = (0.10, 0.05, 0.01)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isotropic_error(
    width: int,
    rank: int,
    *,
    bulk_weight: float = 0.75,
    exception_weight: float = 0.25,
) -> float:
    """Relative error after the best rank-r correction to an isotropic exception."""
    if not 0 <= rank <= width:
        raise ValueError("rank must lie in [0, width]")
    total_energy = bulk_weight**2 + exception_weight**2
    residual_energy = exception_weight**2 * (1.0 - rank / width)
    return math.sqrt(residual_energy / total_energy)


def isotropic_required_rank(
    width: int,
    tolerance: float,
    *,
    bulk_weight: float = 0.75,
    exception_weight: float = 0.25,
) -> int:
    if width <= 0 or tolerance < 0 or exception_weight <= 0:
        raise ValueError("invalid width, tolerance, or exception weight")
    total_energy = bulk_weight**2 + exception_weight**2
    retained_fraction = 1.0 - tolerance**2 * total_energy / exception_weight**2
    return min(width, max(0, math.ceil(width * retained_fraction - 1e-12)))


def spectrum(width: int, exponent: float) -> np.ndarray:
    if width <= 0 or exponent < 0:
        raise ValueError("width must be positive and exponent nonnegative")
    indices = np.arange(1, width + 1, dtype=np.float64)
    return indices ** (-exponent)


def required_rank_from_spectrum(
    singular_values: np.ndarray,
    total_tolerance: float,
    *,
    bulk_weight: float = 0.75,
    exception_weight: float = 0.25,
) -> int:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("singular values must be a nonempty nonnegative vector")
    if total_tolerance < 0 or exception_weight <= 0:
        raise ValueError("invalid tolerance or exception weight")
    energy = np.square(values)
    exception_tail_fraction = (
        total_tolerance**2
        * (bulk_weight**2 + exception_weight**2)
        / exception_weight**2
    )
    allowed_tail = exception_tail_fraction * float(energy.sum())
    tail_after_rank = np.concatenate(
        (np.cumsum(energy[::-1])[::-1], np.zeros(1, dtype=np.float64))
    )
    valid = np.flatnonzero(tail_after_rank <= allowed_tail + 1e-15)
    return int(valid[0]) if valid.size else values.size


def feature_exception_cost_ratio(width: int, rank: int) -> dict[str, float | int]:
    """Ideal logical ratio for D log D bulk plus a rank-r dense correction."""
    if width <= 1 or not 0 <= rank <= width:
        raise ValueError("invalid width or rank")
    dense_macs = width * width
    bulk_macs = width * math.log2(width)
    exception_macs = 2 * width * rank
    return {
        "width": width,
        "rank": rank,
        "rank_fraction": rank / width,
        "dense_macs": dense_macs,
        "ideal_bulk_macs": bulk_macs,
        "rank_correction_macs": exception_macs,
        "total_to_dense_ratio": (bulk_macs + exception_macs) / dense_macs,
    }


def near_linear_fallback_rate(width: int) -> float:
    """Maximum full-dense fallback rate compatible with an O(D log D) ledger."""
    if width <= 1:
        raise ValueError("width must exceed one")
    return math.log2(width) / width


def random_orthogonal_witness(width: int = 64, seed: int = 260726) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    orthogonal, triangular = np.linalg.qr(rng.normal(size=(width, width)))
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    orthogonal = orthogonal * signs
    singular_values = np.linalg.svd(0.25 * orthogonal, compute_uv=False)
    errors = {}
    for rank in (0, width // 10, width // 2, width - 1, width):
        residual = singular_values[rank:]
        errors[str(rank)] = float(
            np.sqrt(np.square(residual).sum() / np.square(singular_values).sum())
        )
    return {
        "width": width,
        "minimum_singular_value": float(singular_values.min()),
        "maximum_singular_value": float(singular_values.max()),
        "exception_relative_errors": errors,
        "all_singular_values_equal": bool(
            np.allclose(singular_values, singular_values[0], atol=1e-12, rtol=1e-12)
        ),
    }


def build_report() -> dict[str, Any]:
    isotropic_rows = []
    for width in WIDTHS:
        for tolerance in TOTAL_TOLERANCES:
            rank = isotropic_required_rank(width, tolerance)
            isotropic_rows.append(
                {
                    "width": width,
                    "tolerance": tolerance,
                    "required_rank": rank,
                    "achieved_error": isotropic_error(width, rank),
                    "cost": feature_exception_cost_ratio(width, rank),
                }
            )

    spectral_rows = []
    for width in WIDTHS:
        for exponent in SPECTRAL_EXPONENTS:
            values = spectrum(width, exponent)
            for tolerance in TOTAL_TOLERANCES:
                rank = required_rank_from_spectrum(values, tolerance)
                spectral_rows.append(
                    {
                        "width": width,
                        "spectral_exponent": exponent,
                        "tolerance": tolerance,
                        "required_rank": rank,
                        "rank_fraction": rank / width,
                        "cost_ratio": feature_exception_cost_ratio(width, rank)[
                            "total_to_dense_ratio"
                        ],
                    }
                )

    geometry = {
        "feature_rank": {
            "logical_cost": "C_bulk(D) + 2 D r(D)",
            "breaks_quadratic_only_if": "r(D) = o(D)",
            "near_linear_if": "r(D) = O(log D)",
        },
        "local_dense_block": {
            "logical_cost": "D b(D) plus cross-block communication",
            "breaks_quadratic_only_if": "b(D) = o(D)",
        },
        "token_fallback": {
            "logical_cost": "C_bulk(D) + p(D) D^2",
            "breaks_quadratic_only_if": "p(D) = o(1)",
            "near_linear_if": "p(D) = O(log D / D)",
        },
        "layer_fallback": {
            "logical_cost": "C_bulk(D) + q(D) D^2",
            "breaks_quadratic_only_if": "q(D) = o(1)",
            "near_linear_if": "q(D) = O(log D / D)",
        },
        "resident_full_fallback": {
            "static_bytes": "Theta(D^2) even when execution is skipped",
            "cannot_claim": "smaller-model edge",
        },
    }

    return {
        "schema_version": 1,
        "claim": "a capped dense exception must shrink proportionally with scale",
        "source_sha256": sha256(Path(__file__)),
        "mixture": {"bulk_weight": 0.75, "exception_weight": 0.25},
        "widths": list(WIDTHS),
        "tolerances": list(TOTAL_TOLERANCES),
        "random_orthogonal_witness": random_orthogonal_witness(),
        "isotropic_exception": isotropic_rows,
        "power_law_exception": spectral_rows,
        "near_linear_full_fallback_rate": {
            str(width): near_linear_fallback_rate(width) for width in WIDTHS
        },
        "exception_geometries": geometry,
        "decision": {
            "broad_key_shift_is_a_candidate": False,
            "single_width_cap_is_evidence": False,
            "required_next_evidence": (
                "At three or more widths, protected quality must hold while the "
                "measured exception fraction decreases with width."
            ),
            "gpu_rental_justified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "output": str(arguments.output),
        "source_sha256": report["source_sha256"],
        "isotropic_D4096": [
            row
            for row in report["isotropic_exception"]
            if row["width"] == 4096
        ],
        "near_linear_fallback_rate_D4096": report[
            "near_linear_full_fallback_rate"
        ]["4096"],
        "decision": report["decision"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
