#!/usr/bin/env python3
"""Frozen algebra/rank gate for nonlinear-reduction SwiGLU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


OUTPUT = Path("results/nonlinear-reduction-swiglu-stage0.json")
PREREGISTRATION = Path("results/nonlinear-reduction-swiglu-stage0-preregistration.md")
SEED = 43
D = 4
M = 3
O = 4
BLOCKS = ((0, 2), (2, 4))
SELECTED = (0, 2)
ALPHA_MAX = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cursor = 0
    norm = theta[cursor:cursor + D]
    cursor += D
    gate = theta[cursor:cursor + M * D].reshape(M, D)
    cursor += M * D
    up = theta[cursor:cursor + M * D].reshape(M, D)
    cursor += M * D
    down = theta[cursor:cursor + O * M].reshape(O, M)
    cursor += O * M
    if cursor != theta.numel():
        raise ValueError("parameter ledger drift")
    return norm, gate, up, down


def baseline(theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    norm, gate, up, down = unpack(theta)
    hidden = inputs * norm
    g = hidden @ gate.T
    u = hidden @ up.T
    return (F.silu(g) * u) @ down.T


def candidate(theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    norm, gate, up, down = unpack(theta)
    hidden = inputs * norm
    p = torch.zeros((inputs.shape[0], M), dtype=inputs.dtype)
    q = torch.zeros_like(p)
    for block_index, (start, stop) in enumerate(BLOCKS):
        p = p + hidden[:, start:stop] @ gate[:, start:stop].T
        q = q + hidden[:, start:stop] @ up[:, start:stop].T
        alpha = ALPHA_MAX * torch.tanh(norm[SELECTED[block_index]] - 1.0)
        q = q * (1.0 + alpha * torch.clamp(p, -2.0, 2.0))
    return (F.silu(p) * q) @ down.T


def parameter_vector(generator: torch.Generator, norm: torch.Tensor) -> torch.Tensor:
    tensors = [
        norm,
        torch.randn(M, D, generator=generator, dtype=torch.float64) / math.sqrt(D),
        torch.randn(M, D, generator=generator, dtype=torch.float64) / math.sqrt(D),
        torch.randn(O, M, generator=generator, dtype=torch.float64) / math.sqrt(M),
    ]
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def gauge_fix(theta: torch.Tensor) -> torch.Tensor:
    norm, gate, up, down = (tensor.clone() for tensor in unpack(theta))
    for coordinate in SELECTED:
        scale = norm[coordinate].clone()
        gate[:, coordinate] *= scale
        up[:, coordinate] *= scale
        norm[coordinate] = 1.0
    return torch.cat([norm, gate.reshape(-1), up.reshape(-1), down.reshape(-1)])


def jacobian(function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.functional.jacobian(
        lambda value: function(value, inputs).reshape(-1),
        theta,
        vectorize=True,
    )


def rank_record(matrix: torch.Tensor) -> dict[str, Any]:
    singular_values = torch.linalg.svdvals(matrix)
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * singular_values[0]
    rank = int((singular_values > tolerance).sum())
    return {
        "shape": list(matrix.shape),
        "rank": rank,
        "nullity": matrix.shape[1] - rank,
        "tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular_values],
    }


def gauge_direction(theta: torch.Tensor, coordinate: int) -> torch.Tensor:
    norm, gate, up, _ = unpack(theta)
    direction = torch.zeros_like(theta)
    direction[coordinate] = norm[coordinate]
    gate_offset = D
    up_offset = D + M * D
    for row in range(M):
        direction[gate_offset + row * D + coordinate] = -gate[row, coordinate]
        direction[up_offset + row * D + coordinate] = -up[row, coordinate]
    return direction


def derivative_record(function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      theta: torch.Tensor, inputs: torch.Tensor,
                      direction: torch.Tensor) -> dict[str, float]:
    _, analytic = torch.autograd.functional.jvp(
        lambda value: function(value, inputs).reshape(-1),
        theta, direction, create_graph=False,
    )
    epsilon = 1e-6
    finite_difference = (
        function(theta + epsilon * direction, inputs).reshape(-1)
        - function(theta - epsilon * direction, inputs).reshape(-1)
    ) / (2.0 * epsilon)
    difference = torch.linalg.vector_norm(analytic - finite_difference)
    denominator = torch.linalg.vector_norm(finite_difference).clamp_min(1e-30)
    return {
        "analytic_norm": float(torch.linalg.vector_norm(analytic)),
        "finite_difference_norm": float(torch.linalg.vector_norm(finite_difference)),
        "relative_error": float(difference / denominator),
        "max_absolute_error": float((analytic - finite_difference).abs().max()),
    }


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def run() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(64, D, generator=generator, dtype=torch.float64) * 0.25
    theta = parameter_vector(generator, torch.ones(D, dtype=torch.float64)).requires_grad_()
    base_output = baseline(theta, inputs)
    candidate_output = candidate(theta, inputs)
    endpoint_max_abs = float((candidate_output - base_output).abs().max())

    nonunit_norm = torch.tensor([0.7, 1.3, 1.6, 0.9], dtype=torch.float64)
    source = parameter_vector(generator, nonunit_norm).requires_grad_()
    converted = gauge_fix(source.detach()).requires_grad_()
    source_output = baseline(source, inputs)
    converted_output = candidate(converted, inputs)
    gauge_endpoint_max_abs = float((converted_output - source_output).abs().max())
    converted_norm = unpack(converted)[0]
    converted_alphas = [
        float(ALPHA_MAX * torch.tanh(converted_norm[index] - 1.0))
        for index in SELECTED
    ]

    base_jacobian = jacobian(baseline, theta, inputs)
    candidate_jacobian = jacobian(candidate, theta, inputs)
    ranks = {
        "baseline": rank_record(base_jacobian),
        "candidate": rank_record(candidate_jacobian),
    }
    derivatives = {}
    for coordinate in SELECTED:
        direction = gauge_direction(theta.detach(), coordinate)
        derivatives[str(coordinate)] = {
            "baseline": derivative_record(baseline, theta, inputs, direction),
            "candidate": derivative_record(candidate, theta, inputs, direction),
        }

    raw_parameters = D + 3 * D * M
    ledger = {
        "raw_parameters_each": raw_parameters,
        "dense_projection_parameters_each": 3 * D * M,
        "dense_projection_macs_per_token_each": 3 * D * M,
        "stored_alpha_parameters": 0,
        "packed_input_accumulators": 2,
        "down_input_width": M,
        "recurrence_boundaries": len(BLOCKS),
    }
    derivative_gate = all(
        record["baseline"]["analytic_norm"] <= 1e-8
        and record["baseline"]["finite_difference_norm"] <= 1e-8
        and record["baseline"]["max_absolute_error"] <= 1e-8
        and record["candidate"]["analytic_norm"] >= 1e-6
        and record["candidate"]["relative_error"] <= 1e-6
        for record in derivatives.values()
    )
    evidence = {
        "endpoint_max_absolute_error": endpoint_max_abs,
        "gauge_endpoint_max_absolute_error": gauge_endpoint_max_abs,
        "converted_alphas": converted_alphas,
        "ranks": ranks,
        "gauge_direction_derivatives": derivatives,
        "ledger": ledger,
    }
    gates = {
        "alpha_zero_endpoint_exact": endpoint_max_abs <= 1e-12,
        "gauge_conversion_endpoint_exact": (
            gauge_endpoint_max_abs <= 1e-12
            and max(abs(value) for value in converted_alphas) <= 1e-15
        ),
        "raw_parameter_dense_mac_and_state_ledger_exact": ledger == {
            "raw_parameters_each": 40,
            "dense_projection_parameters_each": 36,
            "dense_projection_macs_per_token_each": 36,
            "stored_alpha_parameters": 0,
            "packed_input_accumulators": 2,
            "down_input_width": 3,
            "recurrence_boundaries": 2,
        },
        "candidate_functional_rank_strictly_higher": (
            ranks["candidate"]["rank"] > ranks["baseline"]["rank"]
        ),
        "broken_gauge_derivatives_live_and_verified": derivative_gate,
        "all_evidence_finite": all_finite(evidence),
    }
    return {
        "schema": "nonlinear-reduction-swiglu-stage0-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "seed": SEED,
        "dimensions": {"input": D, "hidden": M, "output": O, "blocks": len(BLOCKS)},
        **evidence,
        "gates": gates,
        "stage0_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": result["gates"], "ranks": result["ranks"],
                      "stage0_pass": result["stage0_pass"]}, indent=2, sort_keys=True))
    if not result["stage0_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
