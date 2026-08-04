#!/usr/bin/env python3
"""Algebra, gauge, rank, and stability gate for associative nonlinear reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


OUTPUT = Path("results/associative-nonlinear-reduction-stage0.json")
PREREGISTRATION = Path("results/associative-nonlinear-reduction-stage0-preregistration.md")
SEED = 47
D, M, O, K = 8, 4, 8, 4
SELECTED = 0
ALPHA_MAX = 0.25
BLOCK_WIDTH = D // K


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cursor = 0
    norm = theta[cursor:cursor + D]; cursor += D
    gate = theta[cursor:cursor + M * D].reshape(M, D); cursor += M * D
    up = theta[cursor:cursor + M * D].reshape(M, D); cursor += M * D
    down = theta[cursor:cursor + O * M].reshape(O, M); cursor += O * M
    if cursor != theta.numel():
        raise ValueError("parameter ledger drift")
    return norm, gate, up, down


def clean_hidden(norm: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(norm)
    mask[SELECTED] = 0.0
    effective_norm = norm * mask + (1.0 - mask)
    return inputs * effective_norm


def alpha_from_carrier(norm: torch.Tensor) -> torch.Tensor:
    return ALPHA_MAX * torch.tanh(norm[SELECTED] - 1.0)


def parameter_vector(generator: torch.Generator, norm: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        norm,
        (torch.randn(M, D, generator=generator, dtype=torch.float64) / math.sqrt(D)).reshape(-1),
        (torch.randn(M, D, generator=generator, dtype=torch.float64) / math.sqrt(D)).reshape(-1),
        (torch.randn(O, M, generator=generator, dtype=torch.float64) / math.sqrt(M)).reshape(-1),
    ])


def partials(theta: torch.Tensor, inputs: torch.Tensor):
    norm, gate, up, down = unpack(theta)
    hidden = clean_hidden(norm, inputs)
    pairs = []
    for block in range(K):
        start, stop = block * BLOCK_WIDTH, (block + 1) * BLOCK_WIDTH
        pairs.append((
            hidden[:, start:stop] @ gate[:, start:stop].T,
            hidden[:, start:stop] @ up[:, start:stop].T,
        ))
    return norm, down, pairs


def baseline(theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    _, down, pairs = partials(theta, inputs)
    p = sum(pair[0] for pair in pairs)
    q = sum(pair[1] for pair in pairs)
    return (F.silu(p) * q) @ down.T


def serial(theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    norm, down, pairs = partials(theta, inputs)
    alpha = alpha_from_carrier(norm)
    p = torch.zeros_like(pairs[0][0]); q = torch.zeros_like(p)
    for gate, up in pairs:
        p = p + gate
        q = (q + up) * (1.0 + alpha * torch.clamp(p, -2.0, 2.0))
    return (F.silu(p) * q) @ down.T


def dual_combine(left, right, alpha: torch.Tensor | float):
    p, q = left; gate, up = right
    return (
        p + gate + alpha * p * gate,
        q + up + alpha * (p * up + q * gate),
    )


def associative(theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    norm, down, pairs = partials(theta, inputs)
    alpha = alpha_from_carrier(norm)
    state = (torch.zeros_like(pairs[0][0]), torch.zeros_like(pairs[0][1]))
    for pair in pairs:
        state = dual_combine(state, pair, alpha)
    return (F.silu(state[0]) * state[1]) @ down.T


def gauge_convert(theta: torch.Tensor) -> torch.Tensor:
    norm, gate, up, down = (value.clone() for value in unpack(theta))
    scale = norm[SELECTED].clone()
    gate[:, SELECTED] *= scale
    up[:, SELECTED] *= scale
    norm[SELECTED] = 1.0
    return torch.cat([norm, gate.reshape(-1), up.reshape(-1), down.reshape(-1)])


def ordinary_with_norm(theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    norm, gate, up, down = unpack(theta)
    hidden = inputs * norm
    return (F.silu(hidden @ gate.T) * (hidden @ up.T)) @ down.T


def jacobian(function: Callable, theta: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.functional.jacobian(
        lambda value: function(value, inputs).reshape(-1), theta, vectorize=True,
    )


def rank_record(matrix: torch.Tensor) -> dict[str, Any]:
    singular = torch.linalg.svdvals(matrix)
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * singular[0]
    rank = int((singular > tolerance).sum())
    return {
        "shape": list(matrix.shape), "rank": rank,
        "nullity": matrix.shape[1] - rank, "tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular],
        "smallest_counted_over_tolerance": float(singular[rank - 1] / tolerance),
    }


def derivative_record(function: Callable, theta: torch.Tensor, inputs: torch.Tensor):
    direction = torch.zeros_like(theta); direction[SELECTED] = 1.0
    _, analytic = torch.autograd.functional.jvp(
        lambda value: function(value, inputs).reshape(-1), theta, direction,
    )
    epsilon = 1e-6
    finite_difference = (
        function(theta + epsilon * direction, inputs).reshape(-1)
        - function(theta - epsilon * direction, inputs).reshape(-1)
    ) / (2.0 * epsilon)
    difference = analytic - finite_difference
    return {
        "analytic_norm": float(torch.linalg.vector_norm(analytic)),
        "finite_difference_norm": float(torch.linalg.vector_norm(finite_difference)),
        "relative_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(finite_difference).clamp_min(1e-30)
        ),
        "max_absolute_error": float(difference.abs().max()),
    }


def associativity_probe(generator: torch.Generator) -> dict[str, float]:
    pairs = [
        (torch.randn(7, M, generator=generator, dtype=torch.float64),
         torch.randn(7, M, generator=generator, dtype=torch.float64))
        for _ in range(K)
    ]
    alpha = 0.17
    zero = (torch.zeros_like(pairs[0][0]), torch.zeros_like(pairs[0][1]))
    forward = zero
    for pair in pairs: forward = dual_combine(forward, pair, alpha)
    reverse = zero
    for pair in reversed(pairs): reverse = dual_combine(reverse, pair, alpha)
    left = dual_combine(dual_combine(pairs[0], pairs[1], alpha),
                        dual_combine(pairs[2], pairs[3], alpha), alpha)
    right = dual_combine(pairs[0], dual_combine(pairs[1],
                         dual_combine(pairs[2], pairs[3], alpha), alpha), alpha)
    gates = torch.stack([pair[0] for pair in pairs])
    ups = torch.stack([pair[1] for pair in pairs])
    product = torch.prod(1.0 + alpha * gates, dim=0)
    closed_p = (product - 1.0) / alpha
    closed_q = sum(
        ups[index] * torch.prod(
            torch.cat((1.0 + alpha * gates[:index],
                       1.0 + alpha * gates[index + 1:])), dim=0,
        )
        for index in range(K)
    )
    def error(left_pair, right_pair):
        return max(float((left_pair[0] - right_pair[0]).abs().max()),
                   float((left_pair[1] - right_pair[1]).abs().max()))
    return {
        "forward_reverse_max_abs": error(forward, reverse),
        "forward_balanced_max_abs": error(forward, left),
        "forward_right_associated_max_abs": error(forward, right),
        "forward_closed_form_max_abs": error(forward, (closed_p, closed_q)),
    }


def stability_probe() -> dict[str, Any]:
    generator = np.random.default_rng(SEED)
    gates = generator.normal(0.0, 1.0 / math.sqrt(K), size=(500_000, K))
    alpha = 0.25
    prefix = np.cumsum(gates, axis=1)
    serial_factors = 1.0 + alpha * np.clip(prefix, -2.0, 2.0)
    dual_factors = 1.0 + alpha * gates
    serial_paths = np.stack([
        np.prod(serial_factors[:, start:], axis=1) for start in range(K)
    ], axis=1)
    dual_paths = np.stack([
        np.prod(np.concatenate((dual_factors[:, :skip], dual_factors[:, skip + 1:]), axis=1), axis=1)
        for skip in range(K)
    ], axis=1)
    serial_gain = np.max(np.abs(serial_paths), axis=1)
    dual_gain = np.max(np.abs(dual_paths), axis=1)
    def record(values, paths):
        absolute = np.abs(values)
        return {
            "p01": float(np.quantile(absolute, 0.01)),
            "median": float(np.quantile(absolute, 0.5)),
            "p99": float(np.quantile(absolute, 0.99)),
            "p999": float(np.quantile(absolute, 0.999)),
            "rms": float(np.sqrt(np.mean(values * values))),
            "any_path_sign_flip_fraction": float(np.mean(
                np.any(paths < 0.0, axis=1)
            )),
        }
    return {"samples": 500_000, "serial": record(serial_gain, serial_paths),
            "associative": record(dual_gain, dual_paths)}


def finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str): return True
    if isinstance(value, (int, float)): return math.isfinite(float(value))
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return True


def run() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(100, D, generator=generator, dtype=torch.float64) * 0.25
    theta = parameter_vector(generator, torch.ones(D, dtype=torch.float64)).requires_grad_()
    endpoints = {
        "serial_vs_baseline": float((serial(theta, inputs) - baseline(theta, inputs)).abs().max()),
        "associative_vs_baseline": float((associative(theta, inputs) - baseline(theta, inputs)).abs().max()),
    }
    source_norm = torch.tensor([0.7, 1.1, 0.9, 1.3, 0.8, 1.2, 1.4, 0.6], dtype=torch.float64)
    source = parameter_vector(generator, source_norm).requires_grad_()
    converted = gauge_convert(source.detach()).requires_grad_()
    source_output = ordinary_with_norm(source, inputs)
    gauge_endpoints = {
        "baseline": float((source_output - baseline(converted, inputs)).abs().max()),
        "serial": float((source_output - serial(converted, inputs)).abs().max()),
        "associative": float((source_output - associative(converted, inputs)).abs().max()),
    }
    converted_alpha = float(alpha_from_carrier(unpack(converted)[0]))
    ranks = {
        name: rank_record(jacobian(function, theta, inputs))
        for name, function in (("baseline", baseline), ("serial", serial),
                               ("associative", associative))
    }
    derivatives = {
        name: derivative_record(function, theta, inputs)
        for name, function in (("baseline", baseline), ("serial", serial),
                               ("associative", associative))
    }
    algebra = associativity_probe(generator)
    stability = stability_probe()
    ledger = {
        "raw_parameters_each": D + 3 * D * M,
        "dense_weights_each": 3 * D * M,
        "dense_macs_per_token_each": 3 * D * M,
        "stored_alpha_parameters": 0,
        "mathematical_state_widths": 2,
        "down_input_width": M,
    }
    gates = {
        "zero_alpha_endpoints_exact": max(endpoints.values()) <= 1e-12,
        "carrier_null_gauge_conversion_exact": (
            max(gauge_endpoints.values()) <= 1e-12 and abs(converted_alpha) <= 1e-15
        ),
        "dual_operation_commutative_associative_and_closed": max(algebra.values()) <= 1e-12,
        "both_nonlinear_ranks_exceed_baseline": (
            ranks["serial"]["rank"] > ranks["baseline"]["rank"]
            and ranks["associative"]["rank"] > ranks["baseline"]["rank"]
            and ranks["associative"]["smallest_counted_over_tolerance"] >= 1e6
        ),
        "carrier_derivatives_verified": (
            derivatives["baseline"]["analytic_norm"] <= 1e-8
            and derivatives["baseline"]["finite_difference_norm"] <= 1e-8
            and derivatives["baseline"]["max_absolute_error"] <= 1e-8
            and all(derivatives[name]["analytic_norm"] >= 1e-6
                    and derivatives[name]["relative_error"] <= 1e-6
                    for name in ("serial", "associative"))
        ),
        "associative_stability_dominates_serial": (
            stability["associative"]["p99"] < 2.0
            and stability["associative"]["p99"] < 0.5 * stability["serial"]["p99"]
            and stability["associative"]["any_path_sign_flip_fraction"] == 0.0
        ),
        "exact_parameter_mac_state_ledger": ledger == {
            "raw_parameters_each": 104, "dense_weights_each": 96,
            "dense_macs_per_token_each": 96, "stored_alpha_parameters": 0,
            "mathematical_state_widths": 2, "down_input_width": 4,
        },
    }
    evidence = {"endpoints": endpoints, "gauge_endpoints": gauge_endpoints,
                "converted_alpha": converted_alpha,
                "algebra": algebra, "ranks": ranks, "derivatives": derivatives,
                "stability": stability, "ledger": ledger}
    gates["all_evidence_finite"] = finite(evidence)
    return {
        "schema": "associative-nonlinear-reduction-stage0-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "seed": SEED, "dimensions": {"input": D, "hidden": M, "output": O, "blocks": K},
        **evidence, "gates": gates, "stage0_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = run(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": result["gates"], "ranks": result["ranks"],
                      "stage0_pass": result["stage0_pass"]}, indent=2, sort_keys=True))
    if not result["stage0_pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
