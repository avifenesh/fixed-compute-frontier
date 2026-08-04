#!/usr/bin/env python3
"""Algebra and exact resource gate for a Cayley program tree."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "cayley-program-tree-stage0.json"
PREREGISTRATION = ROOT / "results" / "cayley-program-tree-stage0-preregistration.md"
WIDTH = 8
DEPTH = 6
BASES = 3
DEGREE = 3
ALPHA = 0.25
SEED = 31
TOPOLOGY_SEEDS = (21, 124, 233, 93, 160, 191, 18, 31, 38)
AFFINE_MULTIPLIERS = (61, 359, 169, 373, 131, 115, 113, 19, 29)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_basis(width: int, degree: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    generator = np.zeros((width, width), dtype=np.float64)
    for _ in range(degree):
        permutation = rng.permutation(width)
        for offset in range(0, width, 2):
            first, second = permutation[offset : offset + 2]
            weight = rng.normal()
            generator[first, second] += weight
            generator[second, first] -= weight
    generator *= alpha / np.linalg.svd(generator, compute_uv=False)[0]
    identity = np.eye(width)
    return (identity - generator) @ np.linalg.inv(identity + generator)


def make_implicit_basis(
    width: int,
    degree: int,
    alpha: float,
    topology_offset: int,
    rng: np.random.Generator,
) -> np.ndarray:
    generator = make_implicit_generator(
        width, degree, alpha, topology_offset, rng
    )
    identity = np.eye(width)
    return (identity - generator) @ np.linalg.inv(identity + generator)


def make_implicit_generator(
    width: int,
    degree: int,
    alpha: float,
    topology_offset: int,
    rng: np.random.Generator,
) -> np.ndarray:
    topology_indices = range(topology_offset, topology_offset + degree)
    if topology_offset + degree > len(TOPOLOGY_SEEDS):
        raise ValueError("invalid implicit permutation topology")
    generator = np.zeros((width, width), dtype=np.float64)
    for topology_index in topology_indices:
        permutation = implicit_permutation(width, topology_index)
        for offset in range(0, width, 2):
            first, second = permutation[offset : offset + 2]
            weight = (alpha / degree) * np.tanh(rng.normal())
            generator[first, second] += weight
            generator[second, first] -= weight
    return generator


def truncated_cayley(generator: np.ndarray, order: int = 4) -> np.ndarray:
    identity = np.eye(generator.shape[0])
    term = identity.copy()
    inverse_action = identity.copy()
    for _ in range(order):
        term = -generator @ term
        inverse_action += term
    return 2.0 * inverse_action - identity


def implicit_permutation(width: int, topology_index: int) -> np.ndarray:
    seed = TOPOLOGY_SEEDS[topology_index]
    multiplier = AFFINE_MULTIPLIERS[topology_index]
    if math.gcd(multiplier, width) != 1:
        raise ValueError("implicit permutation multiplier must be coprime")
    positions = np.arange(width, dtype=np.int64)
    left_quadratic = 6 * (2 * seed + 1)
    left_offset = 97 * seed + 17
    right_quadratic = 6 * (2 * ((53 * seed + 7) % 257) + 1)
    right_offset = 193 * seed + 29
    permutation = (positions + left_quadratic * positions**2) % width
    permutation = (multiplier * permutation + left_offset) % width
    return (
        permutation + right_quadratic * permutation**2 + right_offset
    ) % width


def topology_diameter(width: int, topology_offset: int) -> int:
    neighbors = [set() for _ in range(width)]
    edge_sets = []
    for topology_index in range(topology_offset, topology_offset + DEGREE):
        permutation = implicit_permutation(width, topology_index)
        if np.unique(permutation).size != width:
            raise AssertionError("implicit formula is not a permutation")
        edges = {
            tuple(sorted((int(permutation[index]), int(permutation[index + 1]))))
            for index in range(0, width, 2)
        }
        edge_sets.append(edges)
        for left, right in edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
    if any(
        left & right
        for index, left in enumerate(edge_sets)
        for right in edge_sets[index + 1 :]
    ):
        raise AssertionError("implicit matchings overlap")
    if {len(row) for row in neighbors} != {DEGREE}:
        raise AssertionError("implicit graph is not regular")
    diameter = 0
    for source in range(width):
        distances = [-1] * width
        distances[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        if min(distances) < 0:
            raise AssertionError("implicit graph is disconnected")
        diameter = max(diameter, max(distances))
    return diameter


def silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


def silu_derivative(values: np.ndarray) -> np.ndarray:
    sigmoid = 1.0 / (1.0 + np.exp(-values))
    return sigmoid + values * sigmoid * (1.0 - sigmoid)


def apply_edge(
    hidden: np.ndarray,
    basis: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    projected = basis @ hidden
    gate_argument = gate * projected
    activation = silu(gate_argument) * (up * projected)
    scale = 1.0 / math.sqrt(depth)
    output = hidden + scale * basis.T @ (down * activation)
    diagonal = (
        scale
        * down
        * up
        * (
            silu(gate_argument)
            + projected * gate * silu_derivative(gate_argument)
        )
    )
    jacobian = np.eye(hidden.size) + basis.T @ np.diag(diagonal) @ basis
    return output, jacobian


def execute_path(
    initial: np.ndarray,
    bits: tuple[int, ...],
    bases: list[np.ndarray],
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = initial.copy()
    jacobian = np.eye(initial.size)
    node = 0
    for depth_index, bit in enumerate(bits):
        hidden, step_jacobian = apply_edge(
            hidden,
            bases[depth_index % len(bases)],
            gate[node, bit],
            up[node, bit],
            down[node, bit],
            len(bits),
        )
        jacobian = step_jacobian @ jacobian
        node = 2 * node + 1 + bit
    return hidden, jacobian


def finite_difference_jacobian(function, values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    columns = []
    for index in range(values.size):
        displacement = np.zeros_like(values)
        displacement[index] = epsilon
        columns.append(
            (function(values + displacement) - function(values - displacement))
            / (2.0 * epsilon)
        )
    return np.stack(columns, axis=1)


def width_384_ledger() -> dict[str, Any]:
    width = 384
    intermediate = 1024
    depth = 9
    nodes = 2**depth - 1
    bases = 3
    degree = 3
    neumann_order = 4
    payload_parameters = nodes * 2 * 3 * width
    generator_parameters = bases * degree * width // 2
    threshold_parameters = nodes
    candidate = payload_parameters + generator_parameters + threshold_parameters
    baseline = 3 * width * intermediate
    active = depth * (
        2 * neumann_order * degree * width + 4 * width
    )
    return {
        "width": width,
        "intermediate_width": intermediate,
        "depth": depth,
        "internal_nodes": nodes,
        "path_payload_parameters": payload_parameters,
        "generator_parameters": generator_parameters,
        "threshold_parameters": threshold_parameters,
        "candidate_ffn_parameters": candidate,
        "baseline_swiglu_parameters_and_macs": baseline,
        "candidate_minus_baseline_parameters": candidate - baseline,
        "hard_path_payload_scalars_read": depth * 3 * width,
        "candidate_multiply_like_active_upper_bound": active,
        "candidate_to_dense_active_ratio": active / baseline,
        "excluded": [
            "routing and index arithmetic",
            "additions and SiLU",
            "gather efficiency and batch divergence",
            "serial dependencies and kernel overhead",
            "training backward and soft-routing work",
        ],
    }


def build_report() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    generators = [
        make_implicit_generator(
            WIDTH, DEGREE, ALPHA, basis_index * DEGREE, rng
        )
        for basis_index in range(BASES)
    ]
    identity = np.eye(WIDTH)
    exact_bases = [
        (identity - generator) @ np.linalg.inv(identity + generator)
        for generator in generators
    ]
    bases = [truncated_cayley(generator) for generator in generators]
    nodes = 2**DEPTH - 1
    gate = 1.0 + 0.3 * rng.normal(size=(nodes, 2, WIDTH))
    up = 1.0 + 0.3 * rng.normal(size=(nodes, 2, WIDTH))
    down = 0.15 * rng.normal(size=(nodes, 2, WIDTH))
    initial = rng.normal(size=WIDTH)
    paths = list(itertools.product((0, 1), repeat=DEPTH))
    outputs = []
    jacobians = []
    for bits in paths:
        output, jacobian = execute_path(initial, bits, bases, gate, up, down)
        outputs.append(output)
        jacobians.append(jacobian)

    flattened = np.stack([jacobian.ravel() for jacobian in jacobians])
    singular_values = np.linalg.svd(flattened, compute_uv=False)
    span = int(np.count_nonzero(singular_values > 1e-9 * singular_values[0]))
    probe_path = paths[37]
    analytic = jacobians[37]
    finite = finite_difference_jacobian(
        lambda values: execute_path(values, probe_path, bases, gate, up, down)[0],
        initial,
    )
    finite_relative_error = float(
        np.linalg.norm(analytic - finite) / np.linalg.norm(analytic)
    )
    basis_rows = [
        {
            "orthogonality_error": float(
                np.linalg.norm(basis.T @ basis - np.eye(WIDTH))
            ),
            "numerical_density": float(np.mean(np.abs(basis) > 1e-12)),
        }
        for basis in exact_bases
    ]
    neumann_error_bound = 2 * ALPHA**5 / (1 - ALPHA)
    runtime_basis_rows = [
        {
            "operator_error_from_exact": float(
                np.linalg.norm(runtime - exact, ord=2)
            ),
            "minimum_singular_value": float(
                np.linalg.svd(runtime, compute_uv=False)[-1]
            ),
            "orthogonality_error": float(
                np.linalg.norm(runtime.T @ runtime - np.eye(WIDTH))
            ),
            "numerical_density": float(np.mean(np.abs(runtime) > 1e-12)),
        }
        for runtime, exact in zip(bases, exact_bases, strict=True)
    ]
    ledger = width_384_ledger()
    unique_outputs = len({output.round(10).tobytes() for output in outputs})
    minimum_jacobian_rank = int(
        min(np.linalg.matrix_rank(jacobian) for jacobian in jacobians)
    )
    minimum_jacobian_density = min(
        float(np.mean(np.abs(jacobian) > 1e-12)) for jacobian in jacobians
    )
    topology_rows = [
        {
            "width": width,
            "diameters": [
                topology_diameter(width, basis_index * DEGREE)
                for basis_index in range(BASES)
            ],
        }
        for width in (WIDTH, 384, 4096)
    ]
    gates = {
        "bases_are_dense_and_orthogonal": all(
            row["orthogonality_error"] <= 1e-12
            and row["numerical_density"] >= 0.90
            for row in basis_rows
        ),
        "runtime_neumann_bases_are_faithful_and_invertible": all(
            row["operator_error_from_exact"] <= neumann_error_bound
            and row["minimum_singular_value"] >= 1.0 - neumann_error_bound
            and row["numerical_density"] >= 0.90
            for row in runtime_basis_rows
        ),
        "all_forced_paths_have_distinct_outputs": unique_outputs == 2**DEPTH,
        "all_path_jacobians_are_dense_and_full_rank": (
            minimum_jacobian_rank == WIDTH and minimum_jacobian_density >= 0.90
        ),
        "path_jacobians_span_full_matrix_space": span == WIDTH * WIDTH,
        "analytic_jacobian_matches_finite_difference": finite_relative_error <= 1e-5,
        "candidate_parameters_do_not_exceed_swiglu": (
            ledger["candidate_ffn_parameters"]
            <= ledger["baseline_swiglu_parameters_and_macs"]
        ),
        "ideal_active_ratio_below_nine_percent": (
            ledger["candidate_to_dense_active_ratio"] < 0.09
        ),
        "implicit_topologies_are_connected_low_diameter": (
            topology_rows[0]["diameters"] == [3, 3, 3]
            and topology_rows[1]["diameters"] == [10, 10, 10]
            and topology_rows[2]["diameters"] == [16, 15, 15]
        ),
    }
    return {
        "schema_version": 1,
        "candidate": "Cayley program tree",
        "source_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "toy_configuration": {
            "width": WIDTH,
            "depth": DEPTH,
            "bases": BASES,
            "degree": DEGREE,
            "alpha": ALPHA,
            "seed": SEED,
        },
        "basis_rows": basis_rows,
        "runtime_neumann_basis_rows": runtime_basis_rows,
        "neumann_operator_error_upper_bound": neumann_error_bound,
        "implicit_topology": {
            "formula": "reversible-quadratic-affine-quadratic",
            "seeds": list(TOPOLOGY_SEEDS),
            "affine_multipliers": list(AFFINE_MULTIPLIERS),
            "width_rows": topology_rows,
        },
        "paths": {
            "count": len(paths),
            "unique_outputs": unique_outputs,
            "minimum_jacobian_rank": minimum_jacobian_rank,
            "minimum_jacobian_density": minimum_jacobian_density,
            "jacobian_span_dimension": span,
            "smallest_to_largest_span_singular_ratio": float(
                singular_values[-1] / singular_values[0]
            ),
            "finite_difference_relative_error": finite_relative_error,
        },
        "same_basis_warning": (
            "If every depth uses one Q, then in z=Qh coordinates all node "
            "payload transforms are coordinate-wise; cycling independent bases "
            "is required for cross-coordinate composition."
        ),
        "width_384_ledger": ledger,
        "frozen_gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": {
            "proved_if_pass": [
                "same-parameter path storage with sub-9-percent ideal active work",
                "dense full-rank nonlinear path Jacobians",
                "full toy matrix-space span across forced paths",
            ],
            "not_proved": [
                "route learnability",
                "language-model capability",
                "independent information per path",
                "GPU speed",
                "novelty over broad conditional-computation families",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
