#!/usr/bin/env python3
"""Deterministic ledger for the full-rank conditional-tree scaling law."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


THEOREM = Path("results/full-rank-conditional-tree-scaling-law.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ledger(
    width: int,
    intermediate_width: int,
    *,
    bases: int = 3,
    degree: int = 3,
    neumann_order: int = 4,
    alpha: float = 0.25,
) -> dict[str, int | float]:
    dense = 3 * width * intermediate_width
    generators = bases * degree * width // 2
    parameters_per_node = 6 * width + 1
    internal_nodes = (dense - generators) // parameters_per_node
    depth = math.ceil(math.log2(internal_nodes + 1))
    candidate = internal_nodes * parameters_per_node + generators
    active = depth * (
        2 * neumann_order * degree * width + 4 * width
    )
    optimistic_moe_width = active // (3 * width)
    neumann_error = 2 * alpha ** (neumann_order + 1) / (1 - alpha)
    return {
        "width": width,
        "intermediate_width": intermediate_width,
        "dense_parameters_and_macs": dense,
        "internal_nodes": internal_nodes,
        "maximum_balanced_depth": depth,
        "candidate_parameters": candidate,
        "parameter_slack": dense - candidate,
        "hard_path_payload_scalars": 3 * width * depth,
        "active_multiply_like_upper_bound": active,
        "active_ratio": active / dense,
        "ideal_reduction": dense / active,
        "neumann_operator_error_upper_bound": neumann_error,
        "truncated_basis_sigma_min_lower_bound": 1 - neumann_error,
        "tree_local_update_rank": width,
        "optimistic_equal_active_moe_width_and_rank_upper_bound": (
            optimistic_moe_width
        ),
        "tree_to_moe_local_rank_lower_ratio": width / optimistic_moe_width,
    }


def build_payload() -> dict[str, object]:
    rows = [
        ledger(384, 1024),
        ledger(4096, 14336),
        ledger(8192, 28672),
    ]
    gates = {
        "all_candidates_fit_dense_parameter_budget": all(
            row["candidate_parameters"] <= row["dense_parameters_and_macs"]
            for row in rows
        ),
        "active_ratio_strictly_decreases_with_width": all(
            left["active_ratio"] > right["active_ratio"]
            for left, right in zip(rows, rows[1:])
        ),
        "truncated_basis_provably_invertible": all(
            row["truncated_basis_sigma_min_lower_bound"] > 0 for row in rows
        ),
        "tree_rank_exceeds_optimistic_equal_active_moe_rank": all(
            row["tree_local_update_rank"]
            > row["optimistic_equal_active_moe_width_and_rank_upper_bound"]
            for row in rows
        ),
    }
    return {
        "schema": "full-rank-conditional-tree-scaling-v1",
        "source_sha256": sha256(Path(__file__)),
        "theorem_sha256": sha256(THEOREM),
        "rows": rows,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "boundary": (
            "The MoE width bound gives the expert every active multiply and "
            "charges zero routing work, so it is optimistic for the MoE. "
            "The ledger proves a local rank/resource separation, not language "
            "quality or GPU speed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/full-rank-conditional-tree-scaling-law.json"),
    )
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
