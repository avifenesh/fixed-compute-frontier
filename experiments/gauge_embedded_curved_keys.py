#!/usr/bin/env python3
"""Stage-0 algebra gate for full-gauge bi-curved keys (GECK-G2)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results" / "gauge-embedded-curved-keys-preregistration.md"
TEST_SOURCE = ROOT / "tests" / "test_gauge_embedded_curved_keys.py"
INTEGRITY_MANIFEST = ROOT / "results" / "gauge-embedded-curved-keys-integrity-manifest.json"
OUTPUT = ROOT / "results" / "gauge-embedded-curved-keys-stage0.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def choose_pivot(key_weight: np.ndarray, tau: float) -> int:
    radii = np.linalg.norm(key_weight, axis=0)
    nonzero = np.flatnonzero(radii > 0.0)
    if nonzero.size == 0:
        return 0
    log_distance = np.abs(np.log(radii[nonzero] / tau))
    return int(nonzero[int(np.argmin(log_distance))])


def curvature_coefficients(
    pivot_even: float, pivot_odd: float, tau: float
) -> tuple[float, float]:
    radius_squared = pivot_even * pivot_even + pivot_odd * pivot_odd
    rotation_coefficient = (
        0.0 if radius_squared == 0.0 else pivot_even / math.sqrt(radius_squared)
    )
    scale_coefficient = (radius_squared - tau * tau) / (radius_squared + tau * tau)
    return rotation_coefficient, scale_coefficient


def polar_pivot(phi: float, log_scale: float, tau: float) -> np.ndarray:
    radius = tau * math.exp(log_scale)
    return np.array([radius * math.sin(phi), radius * math.cos(phi)])


def gauge_fix_pair(
    query_weights: np.ndarray, key_weight: np.ndarray, pivot: int, tau: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use rotation and reciprocal scale to map one K column to ``(0, tau)``."""

    coefficient_pair = key_weight[:, pivot]
    radius = float(np.linalg.norm(coefficient_pair))
    if radius == 0.0:
        gauge = np.eye(2, dtype=np.float64)
    else:
        even, odd = coefficient_pair
        rotate = np.array([[odd / radius, -even / radius], [even / radius, odd / radius]])
        gauge = (tau / radius) * rotate
    inverse_transpose = np.linalg.inv(gauge).T
    return np.einsum("ab,hbd->had", inverse_transpose, query_weights), gauge @ key_weight, gauge


def curved_key(
    key_weight: np.ndarray, hidden: np.ndarray, pivot: int, tau: float
) -> np.ndarray:
    key = key_weight @ hidden
    even, odd = key
    pivot_even = key_weight[0, pivot]
    pivot_odd = key_weight[1, pivot]
    rotation_coefficient, scale_coefficient = curvature_coefficients(
        pivot_even, pivot_odd, tau
    )
    return np.array([
        even + rotation_coefficient * odd * odd,
        odd + scale_coefficient * even * even,
    ])


def pair_score(
    query_weight: np.ndarray,
    key_weight: np.ndarray,
    query_hidden: np.ndarray,
    key_hidden: np.ndarray,
    relative_angle: float,
    *,
    pivot: int,
    tau: float,
    curved: bool,
) -> float:
    query = query_weight @ query_hidden
    key = curved_key(key_weight, key_hidden, pivot, tau) if curved else key_weight @ key_hidden
    return float(query @ rotation(relative_angle) @ key)


def gauge_containment_gate(seed: int = 20260727) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    hidden_width = 7
    query_heads = 6
    kv_heads = 2
    queries_per_kv = query_heads // kv_heads
    pairs = 4
    tau = 1.0 / math.sqrt(hidden_width)
    maximum_score_error = 0.0
    maximum_curved_baseline_error = 0.0
    maximum_commutator_error = 0.0
    maximum_forbidden_coefficient = 0.0
    gauge_scale_factors = []
    for kv_head in range(kv_heads):
        key_weights = rng.normal(scale=tau, size=(pairs, 2, hidden_width))
        query_weights = rng.normal(scale=tau, size=(queries_per_kv, pairs, 2, hidden_width))
        fixed_keys = []
        fixed_queries = np.empty_like(query_weights)
        gauges = []
        pivots = []
        for pair in range(pairs):
            pivot = choose_pivot(key_weights[pair], tau)
            pivots.append(pivot)
            original_radius = float(np.linalg.norm(key_weights[pair, :, pivot]))
            fixed_query, fixed_key, gauge = gauge_fix_pair(
                query_weights[:, pair], key_weights[pair], pivot, tau
            )
            if original_radius > 0.0:
                gauge_scale_factors.append(tau / original_radius)
            fixed_queries[:, pair] = fixed_query
            fixed_keys.append(fixed_key)
            gauges.append(gauge)
            maximum_forbidden_coefficient = max(
                maximum_forbidden_coefficient,
                abs(float(fixed_key[0, pivot])),
                abs(float(fixed_key[1, pivot] - tau)),
            )
        fixed_keys = np.stack(fixed_keys)
        for _ in range(32):
            query_hidden = rng.normal(size=hidden_width)
            key_hidden = rng.normal(size=hidden_width)
            angles = rng.uniform(-20.0, 20.0, size=pairs)
            for local_query in range(queries_per_kv):
                original_score = 0.0
                fixed_score = 0.0
                curved_baseline_score = 0.0
                for pair in range(pairs):
                    pivot = pivots[pair]
                    original_score += pair_score(
                        query_weights[local_query, pair], key_weights[pair],
                        query_hidden, key_hidden, angles[pair], pivot=pivot, tau=tau, curved=False
                    )
                    fixed_score += pair_score(
                        fixed_queries[local_query, pair], fixed_keys[pair],
                        query_hidden, key_hidden, angles[pair], pivot=pivot, tau=tau, curved=False
                    )
                    curved_baseline_score += pair_score(
                        fixed_queries[local_query, pair], fixed_keys[pair],
                        query_hidden, key_hidden, angles[pair], pivot=pivot, tau=tau, curved=True
                    )
                    rope = rotation(angles[pair])
                    maximum_commutator_error = max(
                        maximum_commutator_error,
                        float(np.max(np.abs(gauges[pair] @ rope - rope @ gauges[pair]))),
                    )
                maximum_score_error = max(maximum_score_error, abs(original_score - fixed_score))
                maximum_curved_baseline_error = max(
                    maximum_curved_baseline_error, abs(original_score - curved_baseline_score)
                )
    checks = {
        "gqa_rope_score_preserved": maximum_score_error <= 1e-10,
        "curved_map_contains_gauge_fixed_baseline": maximum_curved_baseline_error <= 1e-10,
        "gauges_commute_with_rope": maximum_commutator_error <= 1e-12,
        "both_aliased_coefficients_are_zero_on_canonical_baseline_slice": maximum_forbidden_coefficient <= 1e-12,
        "closest_radius_atlas_avoids_extreme_initial_scale": min(gauge_scale_factors) >= 0.5 and max(gauge_scale_factors) <= 2.0,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "maximum_score_error": maximum_score_error,
        "maximum_curved_baseline_error": maximum_curved_baseline_error,
        "maximum_commutator_error": maximum_commutator_error,
        "maximum_forbidden_coefficient": maximum_forbidden_coefficient,
        "minimum_gauge_scale_factor": min(gauge_scale_factors),
        "maximum_gauge_scale_factor": max(gauge_scale_factors),
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "rope_pairs_per_head": pairs,
    }


def strict_extension_witness() -> dict[str, object]:
    query_weight = np.array([[1.0], [0.0]])
    key_weight = np.array([[1.0], [1.0]])
    query_hidden = np.array([1.0])
    scores = [
        pair_score(
            query_weight, key_weight, query_hidden, np.array([value]), 0.0,
            pivot=0, tau=1.0, curved=True
        )
        for value in (0.0, 1.0, 2.0)
    ]
    ordinary_scores = [
        pair_score(
            query_weight, key_weight, query_hidden, np.array([value]), 0.0,
            pivot=0, tau=1.0, curved=False
        )
        for value in (0.0, 1.0, 2.0)
    ]
    second_difference = scores[2] - 2.0 * scores[1] + scores[0]
    ordinary_second_difference = (
        ordinary_scores[2] - 2.0 * ordinary_scores[1] + ordinary_scores[0]
    )
    return {
        "scores_at_0_1_2": scores,
        "ordinary_scores_at_0_1_2": ordinary_scores,
        "second_difference": second_difference,
        "ordinary_bilinear_second_difference": ordinary_second_difference,
        "strict_extension": abs(second_difference) > 1e-12 and abs(ordinary_second_difference) <= 1e-12,
    }


def atlas_and_polar_gate() -> dict[str, object]:
    tau = 0.25
    key_with_zero_column = np.array([
        [0.0, 3.0 * tau, 0.0, tau],
        [0.0, 4.0 * tau, 2.0 * tau, 0.0],
    ])
    chosen = choose_pivot(key_with_zero_column, tau)
    query = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    fixed_query, fixed_key, _ = gauge_fix_pair(
        query, key_with_zero_column[:, :3], 2, tau
    )
    zero_key = np.zeros((2, 4), dtype=np.float64)
    zero_hidden = np.array([1.0, -2.0, 3.0, -4.0])
    zero_curved = curved_key(
        zero_key, zero_hidden, choose_pivot(zero_key, tau), tau
    )
    polar_errors = []
    for phi, log_scale in ((0.0, 0.0), (0.2, -0.3), (-0.7, 0.5), (1.1, -0.8)):
        pivot = polar_pivot(phi, log_scale, tau)
        actual = curvature_coefficients(float(pivot[0]), float(pivot[1]), tau)
        expected = (math.sin(phi), math.tanh(log_scale))
        polar_errors.append(
            max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1]))
        )
    neutral = polar_pivot(0.0, 0.0, tau)
    neutral_coefficients = curvature_coefficients(
        float(neutral[0]), float(neutral[1]), tau
    )
    checks = {
        "atlas_skips_zero_column_and_selects_radius_tau": chosen == 3,
        "nonzero_alternative_column_canonicalizes": bool(
            abs(fixed_key[0, 2]) <= 1e-12
            and abs(fixed_key[1, 2] - tau) <= 1e-12
        ),
        "all_zero_key_pair_remains_zero": bool(
            np.array_equal(zero_curved, np.zeros(2))
        ),
        "polar_coefficients_equal_sine_and_tanh": max(polar_errors) <= 1e-12,
        "polar_origin_is_nonlinear_off_decay_fixed_point": neutral_coefficients == (0.0, 0.0),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "chosen_pivot_with_zero_column": chosen,
        "maximum_polar_identity_error": max(polar_errors),
        "neutral_physical_pivot": neutral.tolist(),
        "neutral_coefficients": list(neutral_coefficients),
        "fixed_query_finite": bool(np.isfinite(fixed_query).all()),
    }


def sampled_function(
    parameters: np.ndarray, samples: np.ndarray, *, curved: bool, tau: float
) -> np.ndarray:
    query_weight = parameters[:6].reshape(2, 3)
    key_weight = parameters[6:].reshape(2, 3)
    return np.array([
        pair_score(query_weight, key_weight, row[:3], row[3:6], row[6],
                   pivot=0, tau=tau, curved=curved)
        - pair_score(query_weight, key_weight, row[:3], row[7:10], row[10],
                     pivot=0, tau=tau, curved=curved)
        for row in samples
    ])


def numerical_jacobian(
    parameters: np.ndarray, samples: np.ndarray, *, curved: bool, tau: float,
    epsilon: float = 1e-6
) -> np.ndarray:
    columns = []
    for index in range(parameters.size):
        delta = np.zeros_like(parameters)
        delta[index] = epsilon
        columns.append(
            (sampled_function(parameters + delta, samples, curved=curved, tau=tau)
             - sampled_function(parameters - delta, samples, curved=curved, tau=tau))
            / (2.0 * epsilon)
        )
    return np.stack(columns, axis=1)


def matrix_rank(matrix: np.ndarray) -> tuple[int, list[float]]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular_values[0] * 1e-7
    return int(np.sum(singular_values > tolerance)), singular_values.tolist()


def functional_rank_gate(seed: int = 314159) -> dict[str, object]:
    tau = 0.4
    worlds = []
    for world in range(5):
        rng = np.random.default_rng(seed + world)
        samples = rng.normal(size=(64, 11))
        samples[:, 6] = rng.uniform(-math.pi, math.pi, size=64)
        samples[:, 10] = rng.uniform(-math.pi, math.pi, size=64)
        parameters = rng.normal(scale=0.4, size=12)
        gauge_slice = parameters.copy()
        gauge_slice[6] = 0.0
        gauge_slice[9] = tau
        generic_curved = gauge_slice.copy()
        generic_curved[6] = 0.37
        generic_curved[9] = 0.61
        baseline_rank, baseline_singular = matrix_rank(
            numerical_jacobian(parameters, samples, curved=False, tau=tau)
        )
        slice_rank, slice_singular = matrix_rank(
            numerical_jacobian(gauge_slice, samples, curved=True, tau=tau)
        )
        curved_rank, curved_singular = matrix_rank(
            numerical_jacobian(generic_curved, samples, curved=True, tau=tau)
        )
        worlds.append({
            "seed": seed + world,
            "ordinary_rank": baseline_rank,
            "curved_baseline_slice_rank": slice_rank,
            "generic_curved_rank": curved_rank,
            "smallest_singular_values": {
                "ordinary": baseline_singular[-3:],
                "curved_baseline_slice": slice_singular[-3:],
                "generic_curved": curved_singular[-3:],
            },
        })
    checks = {
        "ordinary_rope_pair_rank_is_10_in_all_worlds": all(world["ordinary_rank"] == 10 for world in worlds),
        "curved_rank_on_baseline_slice_is_full_12_in_all_worlds": all(world["curved_baseline_slice_rank"] == 12 for world in worlds),
        "generic_curved_rank_is_full_12_in_all_worlds": all(world["generic_curved_rank"] == 12 for world in worlds),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "ordinary_rank": worlds[0]["ordinary_rank"],
        "curved_baseline_slice_rank": worlds[0]["curved_baseline_slice_rank"],
        "generic_curved_rank": worlds[0]["generic_curved_rank"],
        "parameter_count_both": 12,
        "worlds": worlds,
    }


def resource_ledger(hidden_width: int, head_dim: int, kv_heads: int) -> dict[str, object]:
    if head_dim % 2:
        raise ValueError("GECK requires an even RoPE head dimension")
    pairs = head_dim // 2
    return {
        "dense_key_weights_both": hidden_width * head_dim * kv_heads,
        "dense_key_macs_per_token_both": hidden_width * head_dim * kv_heads,
        "key_cache_scalars_per_token_both": head_dim * kv_heads,
        "extra_learned_scalars": 0,
        "extra_scalar_multiplications_per_token": 4 * pairs * kv_heads,
        "extra_scalar_additions_per_token": 2 * pairs * kv_heads,
        "extra_persistent_scalars": 0,
        "existing_weight_coefficients_read_by_epilogue": 2 * pairs * kv_heads,
        "compile_time_pivot_indices": pairs * kv_heads,
        "compile_time_pivot_index_bits": math.ceil(math.log2(hidden_width)) * pairs * kv_heads,
        "weight_only_coefficient_multiplications": 2 * pairs * kv_heads,
        "weight_only_coefficient_add_subtracts": 3 * pairs * kv_heads,
        "weight_only_coefficient_square_roots": pairs * kv_heads,
        "weight_only_coefficient_divisions": 2 * pairs * kv_heads,
        "coefficient_cost_scope": "once per pair per forward/kernel unless derived coefficients are cached; no cache is credited here",
    }


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "test": TEST_SOURCE,
    }
    checks = {
        name: manifest.get(f"{name}_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid GECK integrity manifest: {checks}")
    return checks


def run() -> dict[str, object]:
    integrity_checks = validate_integrity()
    containment = gauge_containment_gate()
    atlas_and_polar = atlas_and_polar_gate()
    extension = strict_extension_witness()
    ranks = functional_rank_gate()
    ledger = resource_ledger(hidden_width=640, head_dim=64, kv_heads=2)
    decision = {
        "stage0_pass": containment["pass"] and atlas_and_polar["pass"] and extension["strict_extension"] and ranks["pass"],
        "h100_required": False,
        "next_gate": "matched learned addressed-retrieval screen",
    }
    return {
        "schema": "gauge-embedded-curved-keys-stage0-v1",
        "candidate": "GECK-G2-full-gauge-bi-curved-keys",
        "integrity_checks": integrity_checks,
        "hashes": {
            "source": sha256_file(Path(__file__)),
            "preregistration": sha256_file(PREREGISTRATION),
            "test": sha256_file(TEST_SOURCE),
        },
        "construction": "canonicalize one K column to (0,tau), then u += (a/r)v^2 and v += ((r^2-tau^2)/(r^2+tau^2))u^2",
        "containment": containment,
        "atlas_and_polar": atlas_and_polar,
        "strict_extension": extension,
        "functional_rank": ranks,
        "resource_ledger_d640_h64_kv2": ledger,
        "scope_limits": {
            "supported": "ordinary norm-free RoPE attention on the generic nonzero-pivot chart",
            "unsupported": "head-wide or learned coordinatewise QK normalization; per-pair reciprocal scaling does not commute with it",
        },
        "decision": decision,
    }


def main() -> None:
    payload = run()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
