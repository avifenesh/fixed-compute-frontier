"""CPU gate for gauge-funded quadratic value projections.

The construction starts from the exact GL(r) redundancy of a value/output
factorization.  It fixes an identity minor in the value projection and spends
the scalar multiplications and additions that minor used to consume on an
off-diagonal quadratic value map.  The gate checks three separate claims:

1. gauge fixing preserves the linear value/output circuit;
2. the quadratic map contains the gauge-fixed linear map at G=0;
3. its scalar multiply/add ledger exactly matches a dense value projection.

This is a logical arithmetic gate, not a GPU latency claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "results" / "gauge-funded-quadratic-values-stage0.json"

Vector = list[float]
Matrix = list[list[float]]


def _shape(matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrices must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal length")
    return len(matrix), width


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    rows, columns = _shape(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(columns)]


def matmul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> Matrix:
    left_rows, shared = _shape(left)
    right_rows, right_columns = _shape(right)
    if shared != right_rows:
        raise ValueError("matrix dimensions do not align")
    return [
        [
            sum(left[i][k] * right[k][j] for k in range(shared))
            for j in range(right_columns)
        ]
        for i in range(left_rows)
    ]


def row_times_matrix(
    row: Sequence[float], matrix: Sequence[Sequence[float]]
) -> Vector:
    matrix_rows, columns = _shape(matrix)
    if len(row) != matrix_rows:
        raise ValueError("row and matrix dimensions do not align")
    return [
        sum(row[i] * matrix[i][j] for i in range(matrix_rows))
        for j in range(columns)
    ]


def inverse(matrix: Sequence[Sequence[float]]) -> Matrix:
    size, columns = _shape(matrix)
    if size != columns:
        raise ValueError("only square matrices are invertible")
    augmented = [
        [float(value) for value in row]
        + [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda i: abs(augmented[i][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [row[size:] for row in augmented]


def max_abs_difference(
    left: Sequence[float], right: Sequence[float]
) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal length")
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def gauge_fix_value_output(
    value_weight: Sequence[Sequence[float]],
    output_weight: Sequence[Sequence[float]],
    pivot_rows: Sequence[int],
) -> tuple[Matrix, Matrix]:
    """Set a full-rank row minor of W_V to identity without changing W_V W_O."""

    model_width, value_width = _shape(value_weight)
    output_rows, _ = _shape(output_weight)
    if output_rows != value_width:
        raise ValueError("value and output dimensions do not align")
    if len(pivot_rows) != value_width or len(set(pivot_rows)) != value_width:
        raise ValueError("pivot_rows must contain one distinct row per value channel")
    if any(row < 0 or row >= model_width for row in pivot_rows):
        raise ValueError("pivot row is outside the model width")

    pivot_minor = [list(value_weight[row]) for row in pivot_rows]
    right_gauge = inverse(pivot_minor)
    fixed_value = matmul(value_weight, right_gauge)
    fixed_output = matmul(pivot_minor, output_weight)
    return fixed_value, fixed_output


def dense_value(
    x: Sequence[float], value_weight: Sequence[Sequence[float]]
) -> Vector:
    return row_times_matrix(x, value_weight)


def structured_quadratic_value(
    x: Sequence[float],
    fixed_value_weight: Sequence[Sequence[float]],
    pivot_rows: Sequence[int],
    off_diagonal_gate: Sequence[Sequence[float]],
) -> Vector:
    """Evaluate the gauge-fixed value map plus funded quadratic interactions.

    For value channel j,

        v_j = x[p_j] + sum_{i not in P} x_i A_ij
              + x[p_j] * sum_{k != j} x[p_k] G_kj.

    G must have a zero diagonal.  At G=0 this is exactly the gauge-fixed
    linear projection.
    """

    model_width, value_width = _shape(fixed_value_weight)
    gate_rows, gate_columns = _shape(off_diagonal_gate)
    if len(x) != model_width:
        raise ValueError("input width does not match the value projection")
    if len(pivot_rows) != value_width:
        raise ValueError("pivot row count does not match the value width")
    if (gate_rows, gate_columns) != (value_width, value_width):
        raise ValueError("gate must be square with the value width")
    if any(abs(off_diagonal_gate[j][j]) > 1e-15 for j in range(value_width)):
        raise ValueError("the quadratic gate diagonal must be zero")

    pivot_set = set(pivot_rows)
    nonpivot_rows = [i for i in range(model_width) if i not in pivot_set]
    result: Vector = []
    for output in range(value_width):
        pivot_value = x[pivot_rows[output]]
        linear = pivot_value + sum(
            x[row] * fixed_value_weight[row][output] for row in nonpivot_rows
        )
        interaction = sum(
            x[pivot_rows[source]] * off_diagonal_gate[source][output]
            for source in range(value_width)
            if source != output
        )
        result.append(linear + pivot_value * interaction)
    return result


def key_coupled_quadratic_value(
    x: Sequence[float],
    fixed_value_weight: Sequence[Sequence[float]],
    pivot_rows: Sequence[int],
    key: Sequence[float],
    gate_scale: Sequence[float],
) -> Vector:
    """Reuse an already-computed pre-RoPE key as the quadratic value gate.

    For value channel j,

        v_j = x[p_j] + sum_{i not in P} x_i A_ij
              + x[p_j] * gate_scale[j] * key[j].

    gate_scale=0 exactly recovers the gauge-fixed linear value projection.
    """

    model_width, value_width = _shape(fixed_value_weight)
    if len(x) != model_width:
        raise ValueError("input width does not match the value projection")
    if len(pivot_rows) != value_width:
        raise ValueError("pivot row count does not match the value width")
    if len(key) != value_width or len(gate_scale) != value_width:
        raise ValueError("key and gate_scale must match the value width")

    pivot_set = set(pivot_rows)
    nonpivot_rows = [i for i in range(model_width) if i not in pivot_set]
    result: Vector = []
    for output in range(value_width):
        pivot_value = x[pivot_rows[output]]
        linear = pivot_value + sum(
            x[row] * fixed_value_weight[row][output] for row in nonpivot_rows
        )
        result.append(
            linear + pivot_value * gate_scale[output] * key[output]
        )
    return result


def value_projection_ledger(model_width: int, value_width: int) -> dict[str, int]:
    """Exact scalar arithmetic/weight ledger for r >= 2 and D >= r."""

    if value_width < 2 or model_width < value_width:
        raise ValueError("ledger requires model_width >= value_width >= 2")

    dense_multiplications = model_width * value_width
    dense_additions = (model_width - 1) * value_width
    base_multiplications = (model_width - value_width) * value_width
    base_additions = (model_width - value_width) * value_width
    gate_multiplications = value_width * (value_width - 1)
    gate_additions = value_width * (value_width - 2)
    interaction_multiplications = value_width
    interaction_additions = value_width
    candidate_multiplications = (
        base_multiplications + gate_multiplications + interaction_multiplications
    )
    candidate_additions = base_additions + gate_additions + interaction_additions
    dense_weights = model_width * value_width
    candidate_weights = base_multiplications + gate_multiplications
    return {
        "model_width": model_width,
        "value_width": value_width,
        "dense_value_weights": dense_weights,
        "candidate_value_weights": candidate_weights,
        "candidate_weight_saving": dense_weights - candidate_weights,
        "dense_scalar_multiplications": dense_multiplications,
        "candidate_scalar_multiplications": candidate_multiplications,
        "dense_scalar_additions": dense_additions,
        "candidate_scalar_additions": candidate_additions,
        "value_cache_scalars_per_token_both": value_width,
    }


def key_coupled_kv_ledger(model_width: int, value_width: int) -> dict[str, int]:
    """Compare dense K+V projections with the key-coupled value variant."""

    if value_width < 2 or model_width < value_width:
        raise ValueError("ledger requires model_width >= value_width >= 2")
    dense_weights = 2 * model_width * value_width
    candidate_weights = (
        model_width * value_width
        + (model_width - value_width) * value_width
        + value_width
    )
    dense_multiplications = dense_weights
    candidate_multiplications = (
        model_width * value_width
        + (model_width - value_width) * value_width
        + 2 * value_width
    )
    dense_additions = 2 * (model_width - 1) * value_width
    candidate_additions = (
        (model_width - 1) * value_width
        + (model_width - value_width) * value_width
        + value_width
    )
    return {
        "model_width": model_width,
        "value_width": value_width,
        "dense_kv_weights": dense_weights,
        "candidate_kv_weights": candidate_weights,
        "candidate_weight_saving": dense_weights - candidate_weights,
        "dense_scalar_multiplications": dense_multiplications,
        "candidate_scalar_multiplications": candidate_multiplications,
        "candidate_multiplication_saving": (
            dense_multiplications - candidate_multiplications
        ),
        "dense_scalar_additions": dense_additions,
        "candidate_scalar_additions": candidate_additions,
        "candidate_addition_saving": dense_additions - candidate_additions,
        "kv_cache_scalars_per_token_both": 2 * value_width,
    }


def mixed_finite_difference(
    function, x00: Sequence[float], x10: Sequence[float], x01: Sequence[float], x11: Sequence[float]
) -> float:
    return function(x11) - function(x10) - function(x01) + function(x00)


def _random_matrix(rows: int, columns: int, rng: random.Random) -> Matrix:
    return [
        [rng.uniform(-0.75, 0.75) for _ in range(columns)]
        for _ in range(rows)
    ]


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage0_gate(*, seed: int = 20260726) -> dict[str, object]:
    rng = random.Random(seed)
    model_width = 8
    value_width = 3
    pivot_rows = (0, 1, 2)

    pivot_minor = [
        [1.25, 0.20, -0.15],
        [0.10, 1.10, 0.25],
        [-0.20, 0.15, 0.95],
    ]
    value_weight = pivot_minor + _random_matrix(
        model_width - value_width, value_width, rng
    )
    output_weight = _random_matrix(value_width, 5, rng)
    fixed_value, fixed_output = gauge_fix_value_output(
        value_weight, output_weight, pivot_rows
    )

    identity_error = max(
        abs(fixed_value[pivot_rows[i]][j] - (1.0 if i == j else 0.0))
        for i in range(value_width)
        for j in range(value_width)
    )
    product_error = max(
        abs(a - b)
        for row_a, row_b in zip(
            matmul(value_weight, output_weight),
            matmul(fixed_value, fixed_output),
            strict=True,
        )
        for a, b in zip(row_a, row_b, strict=True)
    )

    zero_gate = [[0.0] * value_width for _ in range(value_width)]
    maximum_output_error = 0.0
    for _ in range(128):
        x = [rng.uniform(-2.0, 2.0) for _ in range(model_width)]
        original_output = row_times_matrix(dense_value(x, value_weight), output_weight)
        candidate_value = structured_quadratic_value(
            x, fixed_value, pivot_rows, zero_gate
        )
        candidate_output = row_times_matrix(candidate_value, fixed_output)
        maximum_output_error = max(
            maximum_output_error,
            max_abs_difference(original_output, candidate_output),
        )

    strict_fixed_value = [[1.0, 0.0], [0.0, 1.0]]
    strict_gate = [[0.0, 0.0], [1.0, 0.0]]

    def strict_scalar(x: Sequence[float]) -> float:
        return structured_quadratic_value(
            x, strict_fixed_value, (0, 1), strict_gate
        )[0]

    strict_mixed_difference = mixed_finite_difference(
        strict_scalar,
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    )

    key_coupled_zero_gate_error = 0.0
    for _ in range(128):
        x = [rng.uniform(-2.0, 2.0) for _ in range(model_width)]
        key = [rng.uniform(-2.0, 2.0) for _ in range(value_width)]
        expected = dense_value(x, fixed_value)
        actual = key_coupled_quadratic_value(
            x,
            fixed_value,
            pivot_rows,
            key,
            [0.0] * value_width,
        )
        key_coupled_zero_gate_error = max(
            key_coupled_zero_gate_error,
            max_abs_difference(expected, actual),
        )

    def key_coupled_strict_scalar(x: Sequence[float]) -> float:
        key = list(x)
        return key_coupled_quadratic_value(
            x,
            strict_fixed_value,
            (0, 1),
            key,
            (1.0, 0.0),
        )[0]

    key_coupled_second_difference = (
        key_coupled_strict_scalar((1.0, 0.0))
        - 2.0 * key_coupled_strict_scalar((0.0, 0.0))
        + key_coupled_strict_scalar((-1.0, 0.0))
    )

    ledger = value_projection_ledger(4_096, 128)
    key_coupled_ledger = key_coupled_kv_ledger(4_096, 128)
    arithmetic_matched = (
        ledger["dense_scalar_multiplications"]
        == ledger["candidate_scalar_multiplications"]
        and ledger["dense_scalar_additions"]
        == ledger["candidate_scalar_additions"]
    )
    source_path = Path(__file__).resolve()
    report: dict[str, object] = {
        "schema_version": 1,
        "candidate": "gauge_funded_quadratic_values",
        "seed": seed,
        "algebra_gate": {
            "pivot_identity_max_abs_error": identity_error,
            "value_output_product_max_abs_error": product_error,
            "zero_gate_output_max_abs_error_over_128_inputs": maximum_output_error,
        },
        "strict_expressivity_witness": {
            "two_variable_mixed_finite_difference": strict_mixed_difference,
            "affine_value_map_mixed_finite_difference": 0.0,
        },
        "logical_projection_ledger_example": ledger,
        "logical_scalar_arithmetic_exactly_matched": arithmetic_matched,
        "key_coupled_variant": {
            "zero_gate_value_max_abs_error_over_128_inputs": (
                key_coupled_zero_gate_error
            ),
            "one_variable_second_finite_difference": (
                key_coupled_second_difference
            ),
            "affine_value_map_second_finite_difference": 0.0,
            "logical_kv_projection_ledger_example": key_coupled_ledger,
        },
        "kv_cache_shape_unchanged": True,
        "attention_scan_unchanged": True,
        "source_sha256": _source_sha256(source_path),
        "decision": "retain_as_pre_candidate_gpu_kernel_and_lm_quality_unproven",
        "gpu_required_now": False,
    }
    if identity_error > 1e-11 or product_error > 1e-11:
        raise AssertionError("value/output gauge fixing failed")
    if maximum_output_error > 1e-10:
        raise AssertionError("G=0 did not recover the original value/output map")
    if abs(strict_mixed_difference - 1.0) > 1e-12:
        raise AssertionError("quadratic witness did not separate from affine values")
    if key_coupled_zero_gate_error > 1e-12:
        raise AssertionError("key-coupled zero gate did not recover linear values")
    if abs(key_coupled_second_difference - 2.0) > 1e-12:
        raise AssertionError("key-coupled quadratic witness failed")
    if not arithmetic_matched:
        raise AssertionError("candidate and dense value arithmetic ledgers differ")
    return report


def persist_stage0_report() -> dict[str, object]:
    report = run_stage0_gate()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(persist_stage0_report(), indent=2, sort_keys=True))
