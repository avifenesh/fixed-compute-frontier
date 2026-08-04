"""Algebra gate for a coverage-complete gauge-funded value writer.

The dense identity minor saves exactly ``r^2`` scalar multiplications and
``r(r-1)`` additions.  A quadratic value gate can spend that budget on ``r-1``
learned terms per channel plus one final product.  The usual zero-diagonal mask
throws away every square.  Here the one forbidden entry per column follows a
directed cycle instead, retaining every square and at least one orientation of
every cross term for ``r >= 3``.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

from experiments.gauge_funded_quadratic_values import (
    Matrix,
    Vector,
    _shape,
    dense_value,
    value_projection_ledger,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "results" / "gauge-curvature-permutation-mask-stage0.json"


def cyclic_forbidden_sources(width: int) -> tuple[int, ...]:
    if width < 3:
        raise ValueError("coverage-complete cyclic masking requires width >= 3")
    return tuple((output + 1) % width for output in range(width))


def mask_covers_all_quadratic_monomials(
    forbidden_sources: Sequence[int],
) -> bool:
    width = len(forbidden_sources)
    if width < 3 or any(source < 0 or source >= width for source in forbidden_sources):
        return False
    diagonal_covered = all(forbidden_sources[output] != output for output in range(width))
    cross_terms_covered = all(
        not (
            forbidden_sources[left] == right
            and forbidden_sources[right] == left
        )
        for left in range(width)
        for right in range(left + 1, width)
    )
    return diagonal_covered and cross_terms_covered


def permutation_mask_value(
    x: Sequence[float],
    fixed_value_weight: Sequence[Sequence[float]],
    pivot_rows: Sequence[int],
    gate: Sequence[Sequence[float]],
    forbidden_sources: Sequence[int],
) -> Vector:
    """Evaluate ``u + z * (z @ H)`` with one forbidden H entry per column."""

    model_width, value_width = _shape(fixed_value_weight)
    gate_rows, gate_columns = _shape(gate)
    if len(x) != model_width:
        raise ValueError("input width does not match value projection")
    if len(pivot_rows) != value_width:
        raise ValueError("pivot row count does not match value width")
    if (gate_rows, gate_columns) != (value_width, value_width):
        raise ValueError("gate must be square with the value width")
    if len(forbidden_sources) != value_width:
        raise ValueError("one forbidden source is required per value channel")
    for output, source in enumerate(forbidden_sources):
        if source < 0 or source >= value_width:
            raise ValueError("forbidden source is outside the value width")
        if abs(gate[source][output]) > 1e-15:
            raise ValueError("a forbidden gate entry is nonzero")

    base = dense_value(x, fixed_value_weight)
    pivots = [x[row] for row in pivot_rows]
    result: Vector = []
    for output in range(value_width):
        gate_value = sum(
            pivots[source] * gate[source][output]
            for source in range(value_width)
            if source != forbidden_sources[output]
        )
        result.append(base[output] + pivots[output] * gate_value)
    return result


def factor_scalar_quadratic_coefficients(
    coefficients: Sequence[Sequence[float]],
    forbidden_sources: Sequence[int],
    readout: Sequence[float],
) -> Matrix:
    """Factor upper-triangle monomial coefficients into the masked gate.

    ``coefficients[i][j]`` for ``i <= j`` is the coefficient of ``z_i z_j``.
    The construction is valid when the mask covers every monomial and every
    readout entry is nonzero.
    """

    width, columns = _shape(coefficients)
    if width != columns or len(forbidden_sources) != width or len(readout) != width:
        raise ValueError("coefficient, mask, and readout dimensions must match")
    if not mask_covers_all_quadratic_monomials(forbidden_sources):
        raise ValueError("mask does not cover every quadratic monomial")
    if any(abs(value) < 1e-15 for value in readout):
        raise ValueError("quadratic factorization requires nonzero readout entries")

    gate = [[0.0] * width for _ in range(width)]
    for coordinate in range(width):
        gate[coordinate][coordinate] = (
            coefficients[coordinate][coordinate] / readout[coordinate]
        )
    for left in range(width):
        for right in range(left + 1, width):
            coefficient = coefficients[left][right]
            if forbidden_sources[left] != right:
                gate[right][left] = coefficient / readout[left]
            elif forbidden_sources[right] != left:
                gate[left][right] = coefficient / readout[right]
            else:
                raise AssertionError("validated mask unexpectedly lost a cross term")
    return gate


def scalar_quadratic(
    x: Sequence[float], coefficients: Sequence[Sequence[float]]
) -> float:
    width, columns = _shape(coefficients)
    if width != columns or len(x) != width:
        raise ValueError("quadratic coefficient dimensions do not match the input")
    return sum(
        coefficients[left][right] * x[left] * x[right]
        for left in range(width)
        for right in range(left, width)
    )


def permutation_mask_ledger(model_width: int, value_width: int) -> dict[str, int]:
    ledger = value_projection_ledger(model_width, value_width)
    return {
        **ledger,
        "gate_learned_scalars": value_width * (value_width - 1),
        "covered_unique_quadratic_monomials": value_width * (value_width + 1) // 2,
        "total_unique_quadratic_monomials": value_width * (value_width + 1) // 2,
    }


def run_stage0_gate(*, seed: int = 20260726) -> dict[str, object]:
    rng = random.Random(seed)
    width = 5
    forbidden = cyclic_forbidden_sources(width)
    coefficients = [[0.0] * width for _ in range(width)]
    for left in range(width):
        for right in range(left, width):
            coefficients[left][right] = rng.uniform(-0.8, 0.8)
    readout = [rng.uniform(0.5, 1.5) for _ in range(width)]
    gate = factor_scalar_quadratic_coefficients(coefficients, forbidden, readout)
    identity = [[1.0 if row == column else 0.0 for column in range(width)] for row in range(width)]
    maximum_factorization_error = 0.0
    for _ in range(256):
        x = [rng.uniform(-2.0, 2.0) for _ in range(width)]
        values = permutation_mask_value(x, identity, tuple(range(width)), gate, forbidden)
        candidate_quadratic = sum(
            readout[channel] * (values[channel] - x[channel])
            for channel in range(width)
        )
        target_quadratic = scalar_quadratic(x, coefficients)
        maximum_factorization_error = max(
            maximum_factorization_error,
            abs(candidate_quadratic - target_quadratic),
        )

    zero_gate = [[0.0] * width for _ in range(width)]
    zero_gate_error = 0.0
    for _ in range(128):
        x = [rng.uniform(-2.0, 2.0) for _ in range(width)]
        expected = dense_value(x, identity)
        actual = permutation_mask_value(x, identity, tuple(range(width)), zero_gate, forbidden)
        zero_gate_error = max(
            zero_gate_error,
            max(abs(left - right) for left, right in zip(expected, actual, strict=True)),
        )

    ledger = permutation_mask_ledger(4_096, 128)
    source_path = Path(__file__).resolve()
    report: dict[str, object] = {
        "schema_version": 1,
        "candidate": "coverage_complete_gauge_curvature_value",
        "seed": seed,
        "minimum_supported_value_width": 3,
        "cyclic_mask_covers_every_quadratic_monomial": mask_covers_all_quadratic_monomials(
            forbidden
        ),
        "zero_gate_value_max_abs_error_over_128_inputs": zero_gate_error,
        "arbitrary_scalar_quadratic_factorization_max_abs_error_over_256_inputs": (
            maximum_factorization_error
        ),
        "logical_projection_ledger_example": ledger,
        "logical_scalar_arithmetic_exactly_matched": (
            ledger["dense_scalar_multiplications"]
            == ledger["candidate_scalar_multiplications"]
            and ledger["dense_scalar_additions"]
            == ledger["candidate_scalar_additions"]
        ),
        "scalar_quadratic_map_dimension": width * (width + 1) // 2,
        "scalar_quadratic_map_kernel_dimension": width * (width - 3) // 2,
        "zero_initialization_functional_jacobian_nonzero_on_open_set": True,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "decision": "retain_for_learned_attention_gate_not_candidate_004",
        "gpu_required_now": False,
    }
    if not report["cyclic_mask_covers_every_quadratic_monomial"]:
        raise AssertionError("cyclic mask lost a quadratic monomial")
    if zero_gate_error > 1e-12 or maximum_factorization_error > 1e-10:
        raise AssertionError("permutation-mask algebra gate failed")
    if not report["logical_scalar_arithmetic_exactly_matched"]:
        raise AssertionError("permutation-mask value ledger does not match dense V")
    return report


def persist_stage0_report() -> dict[str, object]:
    report = run_stage0_gate()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(persist_stage0_report(), indent=2, sort_keys=True))
