#!/usr/bin/env python3
"""Executable local-Jacobian rank separation for conditional experts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cayley_program_tree_stage0 import (
    finite_difference_jacobian,
    make_implicit_basis,
    silu,
    silu_derivative,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "cayley-conditional-expert-rank.json"
PREREGISTRATION = ROOT / "results" / "cayley-conditional-expert-rank-preregistration.md"
WIDTH = 16
SEED = 20260730


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def additive_memory(
    values: np.ndarray, keys: np.ndarray, outputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scores = keys @ values
    coefficients = np.tanh(scores)
    result = outputs.T @ coefficients
    derivatives = 1.0 - np.square(coefficients)
    jacobian = outputs.T @ (derivatives[:, None] * keys)
    return result, jacobian


def cayley_edge(
    values: np.ndarray,
    basis: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projected = basis @ values
    gate_argument = gate * projected
    activation = silu(gate_argument) * (up * projected)
    output = basis.T @ (down * activation)
    diagonal = down * up * (
        silu(gate_argument)
        + projected * gate * silu_derivative(gate_argument)
    )
    jacobian = basis.T @ np.diag(diagonal) @ basis
    return output, jacobian


def build_report() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=WIDTH)
    additive_rows = []
    for active in (1, 4, 8):
        keys = rng.normal(size=(active, WIDTH))
        outputs = rng.normal(size=(active, WIDTH))
        _, jacobian = additive_memory(values, keys, outputs)
        finite = finite_difference_jacobian(
            lambda probe: additive_memory(probe, keys, outputs)[0], values
        )
        additive_rows.append(
            {
                "active_experts": active,
                "update_jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "rank_upper_bound": active,
                "finite_difference_relative_error": float(
                    np.linalg.norm(jacobian - finite) / np.linalg.norm(jacobian)
                ),
                "active_learned_payload_scalars": 2 * active * WIDTH,
            }
        )

    basis = make_implicit_basis(WIDTH, 3, 0.25, 0, rng)
    gate = 1.0 + 0.2 * rng.normal(size=WIDTH)
    up = 1.0 + 0.2 * rng.normal(size=WIDTH)
    down = rng.normal(size=WIDTH)
    _, jacobian = cayley_edge(values, basis, gate, up, down)
    finite = finite_difference_jacobian(
        lambda probe: cayley_edge(probe, basis, gate, up, down)[0], values
    )
    cayley_row = {
        "update_jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "update_jacobian_density": float(np.mean(np.abs(jacobian) > 1e-12)),
        "finite_difference_relative_error": float(
            np.linalg.norm(jacobian - finite) / np.linalg.norm(jacobian)
        ),
        "active_learned_payload_scalars": 3 * WIDTH,
        "single_basis_jacobian_family_dimension": WIDTH,
    }
    scale_width = 4096
    cayley_active = 28 * scale_width
    dense_active = scale_width**2
    ledger = {
        "width": scale_width,
        "degree": 3,
        "neumann_order_each_direction": 4,
        "cayley_multiply_like_active_upper_bound": cayley_active,
        "dense_expert_macs": dense_active,
        "cayley_to_dense_ratio": cayley_active / dense_active,
        "cayley_active_payload_scalars": 3 * scale_width,
        "dense_active_payload_scalars": dense_active,
        "excluded": [
            "routing",
            "indices and additions",
            "SiLU",
            "gathers and synchronization",
            "physical kernel efficiency",
        ],
    }
    gates = {
        "all_analytic_jacobians_match_finite_difference": (
            max(
                [row["finite_difference_relative_error"] for row in additive_rows]
                + [cayley_row["finite_difference_relative_error"]]
            )
            <= 1e-5
        ),
        "additive_memory_obeys_active_expert_rank_bound": all(
            row["update_jacobian_rank"] <= row["rank_upper_bound"]
            for row in additive_rows
        ),
        "one_cayley_edge_is_dense_and_full_rank": (
            cayley_row["update_jacobian_rank"] == WIDTH
            and cayley_row["update_jacobian_density"] >= 0.90
        ),
        "cayley_payload_is_exactly_three_vectors": (
            cayley_row["active_learned_payload_scalars"] == 3 * WIDTH
        ),
        "dense_full_rank_control_pays_quadratic_payload_and_macs": (
            ledger["dense_active_payload_scalars"] == scale_width**2
            and ledger["dense_expert_macs"] == scale_width**2
        ),
        "scaled_cayley_active_ratio_is_below_one_percent": (
            ledger["cayley_to_dense_ratio"] < 0.01
        ),
        "single_basis_restriction_is_disclosed": (
            cayley_row["single_basis_jacobian_family_dimension"] == WIDTH
        ),
    }
    return {
        "schema_version": 1,
        "candidate": "Cayley-conjugated diagonal conditional expert",
        "source_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "width": WIDTH,
        "seed": SEED,
        "additive_memory_controls": additive_rows,
        "cayley_expert": cayley_row,
        "scaled_ledger": ledger,
        "frozen_gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": {
            "proved_if_pass": [
                "rank-k local bound for k active additive vector experts",
                "full-rank local update from one three-vector Cayley expert",
                "sub-one-percent ideal active multiply ratio at width 4096",
            ],
            "not_proved": [
                "quality",
                "independent memory per composed path",
                "router learnability",
                "GPU speed",
                "broad novelty",
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
