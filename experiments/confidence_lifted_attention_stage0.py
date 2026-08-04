#!/usr/bin/env python3
"""Algebra and local-capacity gate for confidence-lifted attention."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, NamedTuple

import torch


OUTPUT = Path("results/confidence-lifted-attention-stage0.json")
PREREGISTRATION = Path("results/confidence-lifted-attention-stage0-preregistration.md")
SEED = 20260728
PROBES, LENGTH, D, QD, VD, OD, CHUNK = 48, 8, 4, 2, 3, 4, 2
RECENCY_SCALE, CONFIDENCE_SCALE = 2.0, 8.0


class State(NamedTuple):
    maximum: torch.Tensor
    mass: torch.Tensor
    output: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cursor = 0
    norm = theta[cursor:cursor + D]; cursor += D
    query = theta[cursor:cursor + D * QD].reshape(D, QD); cursor += D * QD
    key = theta[cursor:cursor + D * QD].reshape(D, QD); cursor += D * QD
    value = theta[cursor:cursor + D * VD].reshape(D, VD); cursor += D * VD
    out = theta[cursor:cursor + VD * OD].reshape(VD, OD); cursor += VD * OD
    if cursor != theta.numel():
        raise ValueError("parameter ledger drift")
    return norm, query, key, value, out


def pack(parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([part.reshape(-1) for part in parts])


def coefficients(norm: torch.Tensor, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
    zero = norm.new_zeros(())
    if arm == "ordinary":
        return zero, zero
    recency = RECENCY_SCALE * torch.tanh(norm[0] - 1.0)
    lift = zero if arm == "recency" else CONFIDENCE_SCALE * torch.tanh(norm[1] - 1.0)
    return recency, lift


def score_value(theta: torch.Tensor, inputs: torch.Tensor, arm: str):
    norm, query, key, value, out = unpack(theta)
    hidden = inputs * norm if arm == "ordinary" else inputs
    q = hidden[:, -1] @ query
    k = hidden @ key
    v = hidden @ value
    scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(QD)
    return norm, scores, v, out


def local_state(scores: torch.Tensor, values: torch.Tensor) -> tuple[State, torch.Tensor]:
    maximum = scores.max(dim=-1).values
    shifted = torch.exp(scores - maximum[:, None])
    mass = shifted.sum(dim=-1)
    count = scores.shape[-1]
    concentration = torch.zeros_like(mass) if count == 1 else (count - mass) / (count - 1)
    output = torch.einsum("bn,bnv->bv", shifted, values)
    return State(maximum, mass, output), concentration


def chunk_position(index: int, count: int) -> float:
    return 0.0 if count == 1 else index / (count - 1)


def effective_logits(scores: torch.Tensor, recency: torch.Tensor,
                     lift: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    chunks = math.ceil(scores.shape[-1] / CHUNK)
    effective, concentrations = [], []
    for block, start in enumerate(range(0, scores.shape[-1], CHUNK)):
        local = scores[:, start:start + CHUNK]
        dummy = torch.zeros((*local.shape, 1), dtype=local.dtype, device=local.device)
        _, concentration = local_state(local, dummy)
        concentrations.append(concentration)
        offset = recency * chunk_position(block, chunks) + lift * concentration
        effective.append(local + offset[:, None])
    return torch.cat(effective, dim=-1), concentrations


def forward(theta: torch.Tensor, inputs: torch.Tensor, arm: str) -> torch.Tensor:
    norm, scores, values, out = score_value(theta, inputs, arm)
    recency, lift = coefficients(norm, arm)
    logits, _ = effective_logits(scores, recency, lift)
    weights = torch.softmax(logits, dim=-1)
    mixed = torch.einsum("bn,bnv->bv", weights, values)
    return mixed @ out


def gauge_convert(theta: torch.Tensor) -> torch.Tensor:
    norm, query, key, value, out = (part.clone() for part in unpack(theta))
    query *= norm[:, None]
    key *= norm[:, None]
    value *= norm[:, None]
    norm[:] = 1.0
    return pack((norm, query, key, value, out))


def lifted_states(scores: torch.Tensor, values: torch.Tensor,
                  recency: torch.Tensor, lift: torch.Tensor) -> list[State]:
    chunks = math.ceil(scores.shape[-1] / CHUNK)
    states = []
    for block, start in enumerate(range(0, scores.shape[-1], CHUNK)):
        state, concentration = local_state(
            scores[:, start:start + CHUNK], values[:, start:start + CHUNK]
        )
        offset = recency * chunk_position(block, chunks) + lift * concentration
        states.append(State(state.maximum + offset, state.mass, state.output))
    return states


def compose(left: State, right: State) -> State:
    maximum = torch.maximum(left.maximum, right.maximum)
    left_scale = torch.exp(left.maximum - maximum)
    right_scale = torch.exp(right.maximum - maximum)
    return State(
        maximum,
        left_scale * left.mass + right_scale * right.mass,
        left_scale[:, None] * left.output + right_scale[:, None] * right.output,
    )


def reduce_left(states: list[State]) -> State:
    state = states[0]
    for item in states[1:]: state = compose(state, item)
    return state


def reduce_right(states: list[State]) -> State:
    state = states[-1]
    for item in reversed(states[:-1]): state = compose(item, state)
    return state


def reduce_balanced(states: list[State]) -> State:
    if len(states) == 1: return states[0]
    midpoint = len(states) // 2
    return compose(reduce_balanced(states[:midpoint]), reduce_balanced(states[midpoint:]))


def normalized(state: State) -> torch.Tensor:
    return state.output / state.mass[:, None]


def state_error(left: State, right: State) -> float:
    return max(
        float((left.maximum - right.maximum).abs().max()),
        float((left.mass - right.mass).abs().max()),
        float((left.output - right.output).abs().max()),
    )


def rank_record(matrix: torch.Tensor) -> dict[str, Any]:
    singular = torch.linalg.svdvals(matrix)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular[0]
    rank = int((singular > tolerance).sum())
    return {
        "shape": list(matrix.shape), "rank": rank,
        "nullity": matrix.shape[1] - rank, "tolerance": float(tolerance),
        "singular_values": [float(value) for value in singular],
        "smallest_counted_over_tolerance": float(singular[rank - 1] / tolerance),
    }


def jacobian(function: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
    return torch.autograd.functional.jacobian(
        lambda value: function(value).reshape(-1), theta, vectorize=True,
    )


def derivative_record(function: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor,
                      index: int, reference_jacobian: torch.Tensor) -> dict[str, float]:
    direction = torch.zeros_like(theta); direction[index] = 1.0
    _, analytic = torch.autograd.functional.jvp(
        lambda value: function(value).reshape(-1), theta, direction,
    )
    epsilon = 1e-6
    finite_difference = (
        function(theta + epsilon * direction).reshape(-1)
        - function(theta - epsilon * direction).reshape(-1)
    ) / (2.0 * epsilon)
    difference = analytic - finite_difference
    left, singular, _ = torch.linalg.svd(reference_jacobian, full_matrices=False)
    tolerance = max(reference_jacobian.shape) * torch.finfo(reference_jacobian.dtype).eps * singular[0]
    basis = left[:, singular > tolerance]
    projection = basis @ (basis.T @ analytic)
    residual = analytic - projection
    return {
        "analytic_norm": float(torch.linalg.vector_norm(analytic)),
        "finite_difference_norm": float(torch.linalg.vector_norm(finite_difference)),
        "relative_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(finite_difference).clamp_min(1e-30)
        ),
        "max_absolute_error": float(difference.abs().max()),
        "outside_reference_span_norm": float(torch.linalg.vector_norm(residual)),
        "outside_reference_span_fraction": float(
            torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(analytic).clamp_min(1e-30)
        ),
    }


def algebra_probe(generator: torch.Generator) -> dict[str, Any]:
    scores = torch.randn(23, LENGTH, generator=generator, dtype=torch.float64) * 3.0
    values = torch.randn(23, LENGTH, 5, generator=generator, dtype=torch.float64)
    recency, lift = scores.new_tensor(0.31), scores.new_tensor(-1.7)
    states = lifted_states(scores, values, recency, lift)
    left, right, balanced = reduce_left(states), reduce_right(states), reduce_balanced(states)
    permutation = [2, 0, 3, 1]
    permuted = reduce_left([states[index] for index in permutation])
    logits, _ = effective_logits(scores, recency, lift)
    weights = torch.softmax(logits, dim=-1)
    direct = torch.einsum("bn,bnv->bv", weights, values)

    shifted_scores = scores + torch.linspace(-1000.0, 1000.0, scores.shape[0])[:, None]
    shifted = normalized(reduce_balanced(lifted_states(shifted_scores, values, recency, lift)))
    return {
        "left_vs_right_state_max_abs": state_error(left, right),
        "left_vs_balanced_state_max_abs": state_error(left, balanced),
        "left_vs_permuted_state_max_abs": state_error(left, permuted),
        "left_vs_direct_output_max_abs": float((normalized(left) - direct).abs().max()),
        "common_shift_output_max_abs": float((shifted - direct).abs().max()),
        "weight_minimum": float(weights.min()),
        "weight_sum_max_abs_error": float((weights.sum(dim=-1) - 1.0).abs().max()),
    }


def matched_mass_probe() -> dict[str, float]:
    scores = torch.tensor([[0.0, -40.0, -math.log(2.0), -math.log(2.0)]], dtype=torch.float64)
    zero = scores.new_zeros(())
    ordinary_logits, _ = effective_logits(scores, zero, zero)
    lifted_logits, concentrations = effective_logits(scores, zero, scores.new_tensor(1.0))
    ordinary_weights = torch.softmax(ordinary_logits, dim=-1)
    lifted_weights = torch.softmax(lifted_logits, dim=-1)
    ordinary_shares = torch.stack((ordinary_weights[:, :2].sum(), ordinary_weights[:, 2:].sum()))
    lifted_shares = torch.stack((lifted_weights[:, :2].sum(), lifted_weights[:, 2:].sum()))
    return {
        "decisive_concentration": float(concentrations[0]),
        "diffuse_concentration": float(concentrations[1]),
        "ordinary_chunk_share_gap": float((ordinary_shares[0] - ordinary_shares[1]).abs()),
        "lifted_chunk_share_gap": float((lifted_shares[0] - lifted_shares[1]).abs()),
        "lifted_decisive_share": float(lifted_shares[0]),
    }


def iia_probe() -> dict[str, float]:
    base = torch.tensor([0.4, -0.2, 0.1, -0.5, 0.2, -0.4, 0.7, 0.0], dtype=torch.float64)
    distractors = torch.linspace(-4.0, 4.0, 257, dtype=torch.float64)
    ordinary, candidate = [], []
    for distractor in distractors:
        scores = base.clone(); scores[7] = distractor
        ordinary_weights = torch.softmax(scores, dim=-1)
        logits, _ = effective_logits(scores[None, :], scores.new_zeros(()), scores.new_tensor(1.3))
        candidate_weights = torch.softmax(logits[0], dim=-1)
        ordinary.append(torch.log(ordinary_weights[0] / ordinary_weights[6]))
        candidate.append(torch.log(candidate_weights[0] / candidate_weights[6]))
    ordinary_values = torch.stack(ordinary); candidate_values = torch.stack(candidate)
    return {
        "ordinary_log_ratio_range": float(ordinary_values.max() - ordinary_values.min()),
        "confidence_log_ratio_range": float(candidate_values.max() - candidate_values.min()),
    }


def concentration_distribution(generator: torch.Generator) -> dict[str, Any]:
    records = {}
    for width in (64, 128):
        scores = torch.randn(50_000, width, generator=generator, dtype=torch.float64)
        dummy = torch.zeros(50_000, width, 1, dtype=torch.float64)
        _, concentration = local_state(scores, dummy)
        records[str(width)] = {
            "samples": 50_000,
            "mean": float(concentration.mean()),
            "std": float(concentration.std()),
            "p01": float(torch.quantile(concentration, 0.01)),
            "median": float(torch.quantile(concentration, 0.5)),
            "p99": float(torch.quantile(concentration, 0.99)),
        }
    return records


def stress_probe(generator: torch.Generator) -> dict[str, Any]:
    rows, blocks, width, value_width = 31, 256, CHUNK, 7
    scores = torch.randn(rows, blocks * width, generator=generator, dtype=torch.float64) * 30.0
    offsets = torch.linspace(-10_000.0, 10_000.0, rows, dtype=torch.float64)
    scores = scores + offsets[:, None]
    values = torch.randn(rows, blocks * width, value_width, generator=generator, dtype=torch.float64)
    states = lifted_states(scores, values, scores.new_tensor(-1.7), scores.new_tensor(7.9))
    left, balanced = reduce_left(states), reduce_balanced(states)
    left_output, balanced_output = normalized(left), normalized(balanced)
    return {
        "rows": rows, "blocks": blocks,
        "score_offset_min": float(offsets.min()), "score_offset_max": float(offsets.max()),
        "all_finite": bool(torch.isfinite(left_output).all() and torch.isfinite(balanced_output).all()),
        "left_vs_balanced_output_max_abs": float((left_output - balanced_output).abs().max()),
    }


def finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str): return True
    if isinstance(value, (int, float)): return math.isfinite(float(value))
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return True


def run() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(SEED)
    inputs = torch.randn(PROBES, LENGTH, D, generator=generator, dtype=torch.float64)
    inputs = inputs / torch.sqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-6)
    parameter_count = D + D * QD + D * QD + D * VD + VD * OD
    theta = torch.randn(parameter_count, generator=generator, dtype=torch.float64) / 3.0
    theta[:D] = 1.0
    theta = theta.requires_grad_()

    outputs = {arm: forward(theta, inputs, arm) for arm in ("ordinary", "recency", "confidence")}
    endpoints = {
        "ordinary_vs_recency": float((outputs["ordinary"] - outputs["recency"]).abs().max()),
        "ordinary_vs_confidence": float((outputs["ordinary"] - outputs["confidence"]).abs().max()),
    }

    source_parts = list(unpack(theta.detach().clone()))
    source_parts[0] = torch.tensor([0.7, 1.1, 0.9, 1.3], dtype=torch.float64)
    source = pack(tuple(source_parts)).requires_grad_()
    converted = gauge_convert(source.detach()).requires_grad_()
    source_output = forward(source, inputs, "ordinary")
    gauge = {
        "recency": float((source_output - forward(converted, inputs, "recency")).abs().max()),
        "confidence": float((source_output - forward(converted, inputs, "confidence")).abs().max()),
    }

    jacobians = {
        arm: jacobian(lambda value, selected=arm: forward(value, inputs, selected), theta)
        for arm in ("ordinary", "recency", "confidence")
    }
    ranks = {arm: rank_record(matrix) for arm, matrix in jacobians.items()}
    derivatives = {
        "recency_carrier": derivative_record(
            lambda value: forward(value, inputs, "recency"), theta, 0, jacobians["ordinary"]
        ),
        "confidence_carrier": derivative_record(
            lambda value: forward(value, inputs, "confidence"), theta, 1, jacobians["recency"]
        ),
    }

    result: dict[str, Any] = {
        "schema": "confidence-lifted-attention-stage0-v1",
        "seed": SEED, "device": str(theta.device), "dtype": str(theta.dtype),
        "torch_version": torch.__version__,
        "shape": {"probes": PROBES, "length": LENGTH, "input": D, "qk": QD,
                  "value": VD, "output": OD, "chunk": CHUNK},
        "parameter_ledger": {
            "all_arms_stored_scalars": parameter_count,
            "qkvo_mac_change": 0,
            "per_token_cache_fields_added": 0,
            "parallel_partial_scalars_added": 0,
            "possible_scalar_exp_rescales_per_query_head_chunk_added": 1,
            "score_tile_reductions_added": 0,
        },
        "endpoints": endpoints, "gauge_conversion": gauge,
        "algebra": algebra_probe(generator),
        "matched_mass": matched_mass_probe(),
        "iia": iia_probe(), "rank": ranks, "derivatives": derivatives,
        "concentration_distribution": concentration_distribution(generator),
        "stress": stress_probe(generator),
        "source_sha256": sha256_file(Path(__file__)),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
    }

    algebra, matched = result["algebra"], result["matched_mass"]
    rank_margin_ok = all(
        ranks[arm]["smallest_counted_over_tolerance"] >= 1e4
        for arm in ("recency", "confidence")
    )
    result["gates"] = {
        "zero_endpoint": max(endpoints.values()) <= 1e-12,
        "gauge_conversion": max(gauge.values()) <= 1e-11,
        "associative_commutative_direct": max(
            algebra["left_vs_right_state_max_abs"],
            algebra["left_vs_balanced_state_max_abs"],
            algebra["left_vs_permuted_state_max_abs"],
            algebra["left_vs_direct_output_max_abs"],
        ) <= 1e-11,
        "common_shift_invariant": algebra["common_shift_output_max_abs"] <= 1e-11,
        "matched_mass_separated": (
            matched["ordinary_chunk_share_gap"] <= 1e-12
            and matched["lifted_chunk_share_gap"] >= 0.20
        ),
        "iia_separation": (
            result["iia"]["ordinary_log_ratio_range"] <= 1e-12
            and result["iia"]["confidence_log_ratio_range"] >= 1e-3
        ),
        "positive_unit_weights": (
            algebra["weight_minimum"] > 0.0
            and algebra["weight_sum_max_abs_error"] <= 1e-12
        ),
        "rank_strictly_increases": (
            ranks["ordinary"]["rank"] < ranks["recency"]["rank"]
            < ranks["confidence"]["rank"] and rank_margin_ok
        ),
        "derivatives_match": all(
            record["relative_error"] <= 1e-6 for record in derivatives.values()
        ),
        "confidence_outside_recency_span": (
            derivatives["confidence_carrier"]["outside_reference_span_norm"] > 1e-8
        ),
        "stress_finite": result["stress"]["all_finite"],
        "ledger_exact": parameter_count == 44,
    }
    result["stage0_pass"] = all(result["gates"].values()) and finite(result)
    return result


def main() -> None:
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage0_pass": result["stage0_pass"], "gates": result["gates"],
        "rank": {key: value["rank"] for key, value in result["rank"].items()},
        "matched_mass": result["matched_mass"], "iia": result["iia"],
        "algebra": result["algebra"], "concentration_distribution": result["concentration_distribution"],
        "stress": result["stress"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
