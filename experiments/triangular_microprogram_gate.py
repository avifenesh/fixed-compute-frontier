#!/usr/bin/env python3
"""Algebra gate for a triangular microprogram layer (TML).

TML retrieves L shared input/output feature vectors and a tiny program-specific
strict-lower-triangular scalar circuit:

    a_i = <u_i, x> + b_i
    c_i = s_i phi(a_i + sum_{j<i} T_ij c_j)
    y   = x + sum_i c_i v_i

T=0 is exactly an additive PEER-style singleton-expert layer.  Setting
T_ij=<u_i,v_j> exactly compiles sequential rank-one residual experts.  Learned
T therefore keeps both endpoints while spending parameter bytes on feature
composition rather than more D-vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-x))


def additive_layer(x, u, v, bias, scale, activation=silu):
    coefficients = scale * activation(u @ x + bias)
    return x + coefficients @ v


def triangular_layer(x, u, v, bias, scale, coupling, activation=silu):
    base = u @ x + bias
    coefficients = np.empty(u.shape[0], dtype=np.float64)
    for i in range(u.shape[0]):
        preactivation = base[i] + coupling[i, :i] @ coefficients[:i]
        coefficients[i] = scale[i] * activation(preactivation)
    return x + coefficients @ v, coefficients


def direct_rank_one_program(x, u, v, bias, scale, activation=silu):
    y = np.array(x, dtype=np.float64, copy=True)
    for i in range(u.shape[0]):
        coefficient = scale[i] * activation(u[i] @ y + bias[i])
        y = y + coefficient * v[i]
    return y


def effective_polynomial_degree(length: int) -> tuple[int, int]:
    """Return additive and triangular degrees for phi(z)=z^2 in one dimension."""
    x = np.array([0.0, 1.0])
    triangular_coefficients = []
    for i in range(length):
        preactivation = x.copy()
        if i:
            previous = triangular_coefficients[-1]
            if previous.size > preactivation.size:
                preactivation = np.pad(
                    preactivation, (0, previous.size - preactivation.size)
                )
            preactivation[: previous.size] += previous
        triangular_coefficients.append(
            np.polynomial.polynomial.polymul(preactivation, preactivation)
        )

    triangular_output = np.array([0.0])
    for coefficient in triangular_coefficients:
        if coefficient.size > triangular_output.size:
            triangular_output = np.pad(
                triangular_output, (0, coefficient.size - triangular_output.size)
            )
        triangular_output[: coefficient.size] += coefficient

    # Any additive sum of squared affine functions of scalar x is degree <= 2.
    additive_degree = 2
    triangular_degree = int(np.flatnonzero(np.abs(triangular_output) > 1e-12)[-1])
    return additive_degree, triangular_degree


def resource_ledger(
    dimension=4096,
    bank_size=16384,
    programs=1_000_000,
    length=8,
    scalar_bytes=2,
):
    triangle = length * (length - 1) // 2
    id_bytes = 2 if bank_size <= 2**16 else 4
    expert_bytes = (2 * dimension + 2) * scalar_bytes  # u, v, bias, scale
    bank_bytes = bank_size * expert_bytes
    program_record_bytes = length * id_bytes + triangle * scalar_bytes
    program_bytes = programs * program_record_bytes
    total = bank_bytes + program_bytes
    equal_byte_peer_bank = total // expert_bytes
    return {
        "dimension": dimension,
        "bank_size": bank_size,
        "programs": programs,
        "length": length,
        "triangle_scalars_per_program": triangle,
        "program_record_bytes": program_record_bytes,
        "expert_bank_bytes": bank_bytes,
        "program_table_bytes": program_bytes,
        "total_bytes_excluding_router_and_alignment": total,
        "equal_byte_additive_peer_bank_size": equal_byte_peer_bank,
        "active_expert_bytes": length * expert_bytes,
        "vector_macs": 2 * length * dimension,
        "triangular_scalar_macs": triangle,
        "scalar_overhead_fraction": triangle / (2 * length * dimension),
    }


def run_gate(seed=31, trials=64, dimension=48, length=8):
    rng = np.random.default_rng(seed)
    zero_coupling_errors = []
    compiled_sequential_errors = []
    learned_coupling_distances = []

    for _ in range(trials):
        x = rng.normal(size=dimension)
        u = rng.normal(size=(length, dimension)) / math.sqrt(dimension)
        v = rng.normal(size=(length, dimension)) / math.sqrt(dimension)
        bias = rng.normal(scale=0.2, size=length)
        scale = rng.uniform(0.2, 0.8, size=length)

        additive = additive_layer(x, u, v, bias, scale)
        zero, _ = triangular_layer(
            x, u, v, bias, scale, np.zeros((length, length))
        )
        zero_coupling_errors.append(float(np.linalg.norm(additive - zero)))

        coupling = np.tril(u @ v.T, k=-1)
        compiled, _ = triangular_layer(x, u, v, bias, scale, coupling)
        direct = direct_rank_one_program(x, u, v, bias, scale)
        compiled_sequential_errors.append(
            float(np.linalg.norm(compiled - direct) / max(np.linalg.norm(direct), 1e-12))
        )

        learned_coupling = np.tril(rng.normal(scale=0.5, size=(length, length)), -1)
        learned, _ = triangular_layer(x, u, v, bias, scale, learned_coupling)
        learned_coupling_distances.append(float(np.linalg.norm(learned - additive)))

    additive_degree, triangular_degree = effective_polynomial_degree(length)
    ledger = resource_ledger(length=length)
    metrics = {
        "max_zero_coupling_absolute_error": max(zero_coupling_errors),
        "max_compiled_sequential_relative_error": max(compiled_sequential_errors),
        "median_learned_coupling_distance_from_additive": float(
            np.median(learned_coupling_distances)
        ),
        "square_activation_additive_degree": additive_degree,
        "square_activation_triangular_degree": triangular_degree,
    }
    gates = {
        "contains_additive_endpoint_exactly": metrics[
            "max_zero_coupling_absolute_error"
        ]
        < 1e-12,
        "contains_sequential_rank_one_endpoint": metrics[
            "max_compiled_sequential_relative_error"
        ]
        < 1e-12,
        "learned_coupling_changes_function": metrics[
            "median_learned_coupling_distance_from_additive"
        ]
        > 1e-3,
        "degree_grows_exponentially_for_square_activation": triangular_degree
        == 2**length,
        "scalar_runtime_overhead_below_0p1_percent": ledger[
            "scalar_overhead_fraction"
        ]
        < 0.001,
    }
    return {
        "candidate": "triangular-microprogram-layer",
        "gate": "G0-algebra-and-ledger-only",
        "seed": seed,
        "trials": trials,
        "dimension": dimension,
        "length": length,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics": metrics,
        "resource_ledger": ledger,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "capability_status": "untested",
        "claim_boundary": {
            "strict_gain": "program-specific nonlinear feature composition with additive PEER as T=0",
            "not_free": "lower-triangular scalars are learned bytes and their recurrence is serial",
            "structural_prior": "many useful functions reuse a smaller global bank of input/output feature vectors",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--dimension", type=int, default=48)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_gate(args.seed, args.trials, args.dimension, args.length)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if result["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
