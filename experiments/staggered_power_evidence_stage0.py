#!/usr/bin/env python3
"""Stage-0 algebra and score-locked capability gate for PEA2."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "results" / "staggered-power-evidence-stage0-preregistration.md"
OUTPUT = ROOT / "results" / "staggered-power-evidence-stage0.json"
PRIOR_CAUSAL = ROOT / "results" / "peak-lifted-attention-causal-control.json"
PRIOR_CAUSAL_SHA256 = "8091612564cc9e2e373e0d02021c7005864284becc7adf7adf95ac59a9238b78"
SEED = 20260801
N = 128
C = 16
DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks(length: int, width: int, origin: int) -> list[np.ndarray]:
    positions = np.arange(length)
    labels = (positions + origin) // width
    return [positions[labels == label] for label in np.unique(labels)]


def normalize_logweights(logweights: np.ndarray) -> np.ndarray:
    centered = logweights - np.max(logweights, axis=-1, keepdims=True)
    weights = np.exp(centered)
    return weights / weights.sum(axis=-1, keepdims=True)


def effective_weights(scores: np.ndarray, groups: list[np.ndarray], arm: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if arm == "ordinary":
        return normalize_logweights(scores)
    if arm == "temperature2":
        return normalize_logweights(2.0 * scores)
    logweights = scores.copy()
    for ids in groups:
        block = scores[..., ids]
        maximum = np.max(block, axis=-1, keepdims=True)
        shifted = np.exp(block - maximum)
        l1 = shifted.sum(axis=-1)
        if arm == "pea1":
            lift = np.zeros_like(l1)
        elif arm == "pea2":
            l2 = np.square(shifted).sum(axis=-1)
            lift = maximum[..., 0] + np.log(l2 / l1)
        elif arm == "peak":
            lift = maximum[..., 0]
        elif arm == "confidence":
            n = len(ids)
            lift = (n - l1) / (n - 1) if n > 1 else np.zeros_like(l1)
        else:
            raise ValueError(arm)
        logweights[..., ids] += lift[..., None]
    return normalize_logweights(logweights)


def state_output(scores: np.ndarray, values: np.ndarray, groups: list[np.ndarray], p: int) -> np.ndarray:
    router_logits = []
    local_means = []
    for ids in groups:
        block = scores[ids]
        maximum = np.max(block)
        shifted = np.exp(block - maximum)
        l1 = shifted.sum()
        lp = np.power(shifted, p).sum()
        router_logits.append(p * maximum + np.log(lp))
        local_means.append((shifted[:, None] * values[ids]).sum(axis=0) / l1)
    router = normalize_logweights(np.asarray(router_logits))
    return (router[:, None] * np.asarray(local_means)).sum(axis=0)


def algebra_probe(rng: np.random.Generator) -> dict:
    scores = rng.normal(size=37)
    values = rng.normal(size=(37, 5))
    groups = chunks(37, 11, 4)
    direct2 = effective_weights(scores, groups, "pea2") @ values
    state2 = state_output(scores, values, groups, 2)
    baseline = normalize_logweights(scores) @ values
    pea1_errors = []
    for origin in (0, 8):
        group = chunks(37, C, origin)
        pea1_errors.append(float(np.max(np.abs(effective_weights(scores, group, "pea1") - normalize_logweights(scores)))))

    local = np.array([-0.4, 0.2, 1.1])
    rest = np.array([-0.1, 0.3, -0.7])
    witness_scores = np.concatenate((local, rest))
    witness_groups = [np.arange(3), np.arange(3, 6)]
    ordinary = effective_weights(witness_scores, witness_groups, "ordinary")
    pea = effective_weights(witness_scores, witness_groups, "pea2")
    temperature = effective_weights(witness_scores, witness_groups, "temperature2")
    ordinary_local = ordinary[:3] / ordinary[:3].sum()
    pea_local = pea[:3] / pea[:3].sum()
    temperature_local = temperature[:3] / temperature[:3].sum()

    iia_ranges = {}
    for arm in ("ordinary", "pea2"):
        ratios = []
        for third in np.linspace(-2.0, 2.0, 17):
            s = np.array([0.2, third, -0.4, 0.1])
            w = effective_weights(s, [np.arange(2), np.arange(2, 4)], arm)
            ratios.append(np.log(w[0] / w[2]))
        iia_ranges[arm] = float(np.ptp(ratios))

    stress_errors = []
    for shift in (-500.0, 500.0):
        shifted_scores = scores + shift
        direct = effective_weights(shifted_scores, groups, "pea2") @ values
        state = state_output(shifted_scores, values, groups, 2)
        stress_errors.append(float(np.max(np.abs(direct - state))))

    first = np.array([1.0, 0.6, 0.2])
    second = np.array([1.0, 0.5, 0.3])
    missing = {
        "maximum_difference": float(abs(first.max() - second.max())),
        "z1_difference": float(abs(first.sum() - second.sum())),
        "z2_difference": float(abs(np.square(first).sum() - np.square(second).sum())),
    }
    return {
        "direct_vs_state_max_abs": float(np.max(np.abs(direct2 - state2))),
        "pea1_vs_ordinary_weight_max_abs": max(pea1_errors),
        "pea1_state_vs_ordinary_output_max_abs": float(np.max(np.abs(state_output(scores, values, groups, 1) - baseline))),
        "pea2_local_vs_ordinary_max_abs": float(np.max(np.abs(pea_local - ordinary_local))),
        "temperature2_local_vs_ordinary_max_abs": float(np.max(np.abs(temperature_local - ordinary_local))),
        "iia_log_ratio_ranges": iia_ranges,
        "stress_direct_vs_state_max_abs": max(stress_errors),
        "stress_all_finite": bool(np.isfinite(stress_errors).all()),
        "missing_statistic_witness": missing,
    }


def permutation_cardinality_probe() -> dict:
    multisets = (
        np.array([-1.7, -0.8, -0.2, 0.0, 0.3, 0.9, 1.4, 2.2]),
        np.array([-2.0, -2.0, -0.5, -0.5, 0.5, 0.5, 2.0, 2.0]),
        np.zeros(8),
    )
    groups = [np.arange(1), np.arange(1, 4), np.arange(4, 8)]
    expected_router = np.array([1.0, 3.0, 4.0]) / 8.0
    records = []
    for multiset in multisets:
        router_sum = np.zeros(3)
        token_sum = np.zeros(8)
        permutations = 0
        for order in itertools.permutations(range(8)):
            scores = multiset[np.asarray(order)]
            log_router = []
            for ids in groups:
                block = scores[ids]
                maximum = block.max()
                log_router.append(2.0 * maximum + np.log(np.square(np.exp(block - maximum)).sum()))
            router = normalize_logweights(np.asarray(log_router))
            weights = effective_weights(scores, groups, "pea2")
            router_sum += router
            token_sum += weights
            permutations += 1
        router_mean = router_sum / permutations
        token_mean = token_sum / permutations
        records.append({
            "scores": multiset.tolist(),
            "permutations": permutations,
            "router_mean": router_mean.tolist(),
            "token_mean": token_mean.tolist(),
            "router_max_abs_error": float(np.max(np.abs(router_mean - expected_router))),
            "token_max_abs_error": float(np.max(np.abs(token_mean - 1.0 / 8.0))),
        })
    return {
        "chunk_sizes": [1, 3, 4],
        "records": records,
        "maximum_router_error": max(row["router_max_abs_error"] for row in records),
        "maximum_token_error": max(row["token_max_abs_error"] for row in records),
    }


def exact_label_accuracy(coefficients: np.ndarray, target: int) -> float:
    correct = 0.0
    total = 0
    for labels in itertools.product((-1.0, 1.0), repeat=len(coefficients)):
        labels_array = np.asarray(labels)
        output = float(coefficients @ labels_array)
        if abs(output) <= 1e-14:
            correct += 0.5
        elif np.sign(output) == labels_array[target]:
            correct += 1.0
        total += 1
    return correct / total


def prior_control_probe() -> dict:
    digest = sha256(PRIOR_CAUSAL)
    result = json.loads(PRIOR_CAUSAL.read_text())
    gates = result["gates"]
    all_six_false = len(gates) == 6 and not any(gates.values())
    return {
        "path": str(PRIOR_CAUSAL.relative_to(ROOT)),
        "sha256": digest,
        "expected_sha256": PRIOR_CAUSAL_SHA256,
        "hash_matches": digest == PRIOR_CAUSAL_SHA256,
        "all_six_gates_false": all_six_false,
        "live_variant": result["live_variant"],
        "valid_rejection": digest == PRIOR_CAUSAL_SHA256 and all_six_false and result["live_variant"] is None,
    }


def anchor_probe() -> dict:
    same_payloads = np.array([4, 20, 36, 52, 68, 84, 100, 116])
    split_payloads = np.array([4, 20, 32, 52, 68, 84, 100, 116])
    target = 2
    arms = ("ordinary", "temperature2", "pea2", "peak", "confidence")
    records = []
    for delta in DELTAS:
        same_scores = np.zeros(N)
        same_scores[34] = delta
        split_scores = np.zeros(N)
        split_scores[31] = delta
        row = {"delta": delta, "same": {}, "split_origin0": {}}
        for arm in arms:
            same_weights = effective_weights(same_scores, chunks(N, C, 0), arm)
            split_weights = effective_weights(split_scores, chunks(N, C, 0), arm)
            row["same"][arm] = {
                "target_probability": float(same_weights[same_payloads[target]]),
                "label_accuracy": exact_label_accuracy(same_weights[same_payloads], target),
            }
            row["split_origin0"][arm] = {
                "target_probability": float(split_weights[split_payloads[target]]),
                "label_accuracy": exact_label_accuracy(split_weights[split_payloads], target),
            }
        split0 = effective_weights(split_scores, chunks(N, C, 0), "pea2")
        split8 = effective_weights(split_scores, chunks(N, C, 8), "pea2")
        staggered = 0.5 * (split0 + split8)
        row["staggered_pea2"] = {
            "target_probability": float(staggered[split_payloads[target]]),
            "label_accuracy": exact_label_accuracy(staggered[split_payloads], target),
        }
        ordinary_probability = row["same"]["ordinary"]["target_probability"]
        row["same_pea2_probability_ratio"] = row["same"]["pea2"]["target_probability"] / ordinary_probability
        row["staggered_pea2_probability_ratio"] = row["staggered_pea2"]["target_probability"] / ordinary_probability
        records.append(row)
    return {"records": records}


def run() -> dict:
    rng = np.random.default_rng(SEED)
    algebra = algebra_probe(rng)
    cardinality = permutation_cardinality_probe()
    anchor = anchor_probe()
    prior_control = prior_control_probe()
    rows = anchor["records"]
    gates = {
        "direct_state_exact": algebra["direct_vs_state_max_abs"] <= 1e-12,
        "p1_baseline_exact": max(algebra["pea1_vs_ordinary_weight_max_abs"], algebra["pea1_state_vs_ordinary_output_max_abs"]) <= 1e-12,
        "select_sharp_read_soft_separation": algebra["pea2_local_vs_ordinary_max_abs"] <= 1e-12 and algebra["temperature2_local_vs_ordinary_max_abs"] >= 1e-3,
        "cross_chunk_iia_broken": algebra["iia_log_ratio_ranges"]["ordinary"] <= 1e-12 and algebra["iia_log_ratio_ranges"]["pea2"] >= 0.1,
        "shifted_execution_stable": algebra["stress_all_finite"] and algebra["stress_direct_vs_state_max_abs"] <= 1e-12,
        "ordinary_state_cannot_reconstruct_z2": algebra["missing_statistic_witness"]["maximum_difference"] <= 1e-15 and algebra["missing_statistic_witness"]["z1_difference"] <= 1e-15 and algebra["missing_statistic_witness"]["z2_difference"] >= 1e-2,
        "exact_iid_cardinality_neutral": cardinality["maximum_router_error"] <= 1e-12 and cardinality["maximum_token_error"] <= 1e-12,
        "same_chunk_probability_gain": all(row["same_pea2_probability_ratio"] >= 1.05 and row["same"]["pea2"]["target_probability"] > row["same"]["temperature2"]["target_probability"] for row in rows),
        "staggered_boundary_probability_noninferior": all(row["staggered_pea2_probability_ratio"] >= 1.01 for row in rows),
        "same_chunk_label_gain_over_temperature": all(row["same"]["pea2"]["label_accuracy"] > row["same"]["temperature2"]["label_accuracy"] for row in rows),
        "staggered_label_gain_over_ordinary": all(row["staggered_pea2"]["label_accuracy"] > row["same"]["ordinary"]["label_accuracy"] for row in rows),
        "prior_peak_confidence_pareto_control_bound": prior_control["valid_rejection"],
    }
    return {
        "schema": "staggered-power-evidence-stage0-v1",
        "seed": SEED,
        "source_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREG),
        "algebra": algebra,
        "cardinality": cardinality,
        "anchor": anchor,
        "prior_causal_control": prior_control,
        "resource_ledger": {
            "qk_matmul_change": 0,
            "pv_matmul_change": 0,
            "qkv_ffn_matmul_change": 0,
            "kv_cache_fields_added": 0,
            "persistent_activation_fields_added": 0,
            "additional_exponentials_per_score": 0,
            "p2_score_multiplications_added": 1,
            "p2_scalar_reduction_accumulators_added": 1,
            "p2_scalar_divisions_per_semantic_chunk_added": 1,
            "p2_value_vector_rescales_per_semantic_chunk_added": 1,
            "p2_value_vector_rescale_width": "d_v",
            "scalar_merge_exponential_change_if_chunk_is_native_pv_unit": 0,
            "unaligned_chunk_extra_merge_work_possible": True,
            "prefill_extra_live_value_accumulator_possible": True,
        },
        "gates": gates,
        "stage0_pass": all(gates.values()),
    }


def main() -> None:
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": result["gates"], "stage0_pass": result["stage0_pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
