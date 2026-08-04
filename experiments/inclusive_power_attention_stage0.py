#!/usr/bin/env python3
"""Frozen Stage-0 comparison of IPA2 and PEA2."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "results" / "inclusive-power-attention-stage0-preregistration.md"
OUTPUT = ROOT / "results" / "inclusive-power-attention-stage0.json"
SEED = 20260802
N = 128
C = 16
DELTAS = (0.5, 1.0, 2.0, 3.0, 4.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks(length: int, width: int, origin: int) -> list[np.ndarray]:
    positions = np.arange(length)
    labels = (positions + origin) // width
    return [positions[labels == label] for label in np.unique(labels)]


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(centered)
    return weights / weights.sum(axis=-1, keepdims=True)


def effective_weights(scores: np.ndarray, groups: list[np.ndarray], arm: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if arm == "ordinary":
        return softmax(scores)
    if arm == "temperature2":
        return softmax(2.0 * scores)
    logits = scores.copy()
    for ids in groups:
        block = scores[..., ids]
        maximum = np.max(block, axis=-1, keepdims=True)
        shifted = np.exp(block - maximum)
        l1 = shifted.sum(axis=-1)
        if arm == "ipa1":
            lift = np.zeros_like(l1)
        elif arm == "ipa2":
            lift = maximum[..., 0] + np.log(l1) - np.log(len(ids))
        elif arm == "pea2":
            l2 = np.square(shifted).sum(axis=-1)
            lift = maximum[..., 0] + np.log(l2 / l1)
        else:
            raise ValueError(arm)
        logits[..., ids] += lift[..., None]
    return softmax(logits)


def state_output(scores: np.ndarray, values: np.ndarray, groups: list[np.ndarray], arm: str) -> np.ndarray:
    router_logits = []
    local_means = []
    for ids in groups:
        block = scores[ids]
        maximum = block.max()
        shifted = np.exp(block - maximum)
        l1 = shifted.sum()
        if arm == "ipa2":
            router_logits.append(2.0 * maximum + 2.0 * np.log(l1) - np.log(len(ids)))
        elif arm == "pea2":
            router_logits.append(2.0 * maximum + np.log(np.square(shifted).sum()))
        else:
            raise ValueError(arm)
        local_means.append((shifted[:, None] * values[ids]).sum(axis=0) / l1)
    router = softmax(np.asarray(router_logits))
    return (router[:, None] * np.asarray(local_means)).sum(axis=0)


def algebra_probe(rng: np.random.Generator) -> dict:
    scores = rng.normal(size=37)
    values = rng.normal(size=(37, 5))
    groups = chunks(37, 11, 4)
    direct = effective_weights(scores, groups, "ipa2") @ values
    state = state_output(scores, values, groups, "ipa2")
    baseline = softmax(scores)
    ipa1_errors = []
    for origin in (0, 8):
        partition = chunks(37, C, origin)
        ipa1_errors.append(float(np.max(np.abs(effective_weights(scores, partition, "ipa1") - baseline))))

    local = np.array([-0.4, 0.2, 1.1])
    rest = np.array([-0.1, 0.3, -0.7])
    witness = np.concatenate((local, rest))
    witness_groups = [np.arange(3), np.arange(3, 6)]
    ordinary = effective_weights(witness, witness_groups, "ordinary")
    ipa = effective_weights(witness, witness_groups, "ipa2")
    temperature = effective_weights(witness, witness_groups, "temperature2")
    ordinary_local = ordinary[:3] / ordinary[:3].sum()
    ipa_local = ipa[:3] / ipa[:3].sum()
    temperature_local = temperature[:3] / temperature[:3].sum()

    iia_ranges = {}
    for arm in ("ordinary", "ipa2"):
        ratios = []
        for third in np.linspace(-2.0, 2.0, 17):
            row = np.array([0.2, third, -0.4, 0.1])
            weights = effective_weights(row, [np.arange(2), np.arange(2, 4)], arm)
            ratios.append(np.log(weights[0] / weights[2]))
        iia_ranges[arm] = float(np.ptp(ratios))

    shift_probability_errors = []
    stress_state_errors = []
    for shift in (-500.0, 500.0):
        shifted = scores + shift
        shift_probability_errors.append(float(np.max(np.abs(
            effective_weights(shifted, groups, "ipa2") - effective_weights(scores, groups, "ipa2")
        ))))
        stress_state_errors.append(float(np.max(np.abs(
            effective_weights(shifted, groups, "ipa2") @ values - state_output(shifted, values, groups, "ipa2")
        ))))

    constant_errors = []
    for n in (1, 3, 16):
        for constant in (-2.0, 0.0, 1.7):
            x = np.full(n, np.exp(constant))
            ipa_mass = x.sum() ** 2 / n
            pea_mass = np.square(x).sum()
            expected = n * np.exp(2.0 * constant)
            constant_errors.extend((abs(ipa_mass - expected), abs(pea_mass - expected)))

    identity_records = []
    for x in (np.array([1.0, 0.6, 0.2]), np.array([1.0, 0.5, 0.3]), np.ones(5)):
        pea_mass = np.square(x).sum()
        ipa_mass = x.sum() ** 2 / len(x)
        cv2 = x.var() / x.mean() ** 2
        identity_records.append({
            "x": x.tolist(),
            "pea_over_ipa": float(pea_mass / ipa_mass),
            "one_plus_cv2": float(1.0 + cv2),
            "absolute_error": float(abs(pea_mass / ipa_mass - (1.0 + cv2))),
        })

    return {
        "direct_vs_state_max_abs": float(np.max(np.abs(direct - state))),
        "ipa1_vs_ordinary_max_abs": max(ipa1_errors),
        "ipa_local_vs_ordinary_max_abs": float(np.max(np.abs(ipa_local - ordinary_local))),
        "temperature_local_vs_ordinary_max_abs": float(np.max(np.abs(temperature_local - ordinary_local))),
        "iia_log_ratio_ranges": iia_ranges,
        "row_shift_probability_max_abs": max(shift_probability_errors),
        "stress_state_max_abs": max(stress_state_errors),
        "stress_all_finite": bool(np.isfinite(stress_state_errors).all()),
        "constant_mass_max_abs": float(max(constant_errors)),
        "pea_ipa_dispersion_identity": identity_records,
        "identity_max_abs": max(row["absolute_error"] for row in identity_records),
        "constant_identity_abs": identity_records[-1]["absolute_error"],
        "nonconstant_identity_gap": min(row["pea_over_ipa"] - 1.0 for row in identity_records[:-1]),
    }


def iid_cardinality_probe() -> dict:
    x_values = np.array([0.5, 2.0])
    groups = [np.arange(1), np.arange(1, 4), np.arange(4, 8)]
    sizes = np.array([1.0, 3.0, 4.0])
    accumulators = {name: np.zeros(3) for name in ("pea_raw_per_token", "ipa_raw_per_token", "pea_share", "ipa_share")}
    assignments = 0
    for bits in itertools.product((0, 1), repeat=8):
        x = x_values[np.asarray(bits)]
        pea_mass = np.asarray([np.square(x[ids]).sum() for ids in groups])
        ipa_mass = np.asarray([x[ids].sum() ** 2 / len(ids) for ids in groups])
        accumulators["pea_raw_per_token"] += pea_mass / sizes
        accumulators["ipa_raw_per_token"] += ipa_mass / sizes
        accumulators["pea_share"] += pea_mass / pea_mass.sum()
        accumulators["ipa_share"] += ipa_mass / ipa_mass.sum()
        assignments += 1
    means = {name: values / assignments for name, values in accumulators.items()}
    expected_share = sizes / sizes.sum()
    mu = x_values.mean()
    variance = x_values.var()
    expected_pea_raw = np.full(3, np.square(x_values).mean())
    expected_ipa_raw = mu * mu + variance / sizes
    return {
        "x_values": x_values.tolist(),
        "chunk_sizes": sizes.astype(int).tolist(),
        "assignments": assignments,
        "pea_raw_per_token": means["pea_raw_per_token"].tolist(),
        "ipa_raw_per_token": means["ipa_raw_per_token"].tolist(),
        "expected_pea_raw_per_token": expected_pea_raw.tolist(),
        "expected_ipa_raw_per_token": expected_ipa_raw.tolist(),
        "pea_router_share": means["pea_share"].tolist(),
        "ipa_router_share": means["ipa_share"].tolist(),
        "expected_cardinality_share": expected_share.tolist(),
        "pea_raw_max_abs_error": float(np.max(np.abs(means["pea_raw_per_token"] - expected_pea_raw))),
        "ipa_raw_max_abs_error": float(np.max(np.abs(means["ipa_raw_per_token"] - expected_ipa_raw))),
        "pea_share_max_abs_error": float(np.max(np.abs(means["pea_share"] - expected_share))),
        "ipa_share_max_abs_deviation": float(np.max(np.abs(means["ipa_share"] - expected_share))),
    }


def anchor_probe() -> dict:
    payloads = np.array([4, 20, 36, 52, 68, 84, 100, 116])
    target = 2
    groups = chunks(N, C, 0)
    records = []
    for delta in DELTAS:
        scores = np.zeros(N)
        scores[34] = delta
        weights = {arm: effective_weights(scores, groups, arm) for arm in ("ordinary", "ipa2", "pea2")}
        target_group = next(ids for ids in groups if 34 in ids)
        x = np.exp(scores[target_group])
        identity = np.square(x).sum() / (x.sum() ** 2 / len(x))
        records.append({
            "delta": delta,
            "target_payload_probability": {arm: float(row[payloads[target]]) for arm, row in weights.items()},
            "target_chunk_pea_over_ipa_mass": float(identity),
            "target_chunk_one_plus_cv2": float(1.0 + x.var() / x.mean() ** 2),
            "identity_abs_error": float(abs(identity - (1.0 + x.var() / x.mean() ** 2))),
        })
    return {"records": records}


def consensus_spike_probe() -> dict:
    spike = np.zeros(C)
    spike[0] = 3.0
    consensus = np.zeros(C)
    consensus[:4] = 1.8
    x_spike = np.exp(spike)
    x_consensus = np.exp(consensus)
    masses = {
        "ipa_spike": float(x_spike.sum() ** 2 / C),
        "ipa_consensus": float(x_consensus.sum() ** 2 / C),
        "pea_spike": float(np.square(x_spike).sum()),
        "pea_consensus": float(np.square(x_consensus).sum()),
    }
    return masses | {
        "ipa_prefers_consensus": masses["ipa_consensus"] > masses["ipa_spike"],
        "pea_prefers_spike": masses["pea_spike"] > masses["pea_consensus"],
    }


def run() -> dict:
    algebra = algebra_probe(np.random.default_rng(SEED))
    iid = iid_cardinality_probe()
    anchor = anchor_probe()
    consensus = consensus_spike_probe()
    anchor_rows = anchor["records"]
    gates = {
        "ipa_direct_state_exact": algebra["direct_vs_state_max_abs"] <= 1e-12,
        "ipa1_baseline_exact": algebra["ipa1_vs_ordinary_max_abs"] <= 1e-12,
        "select_sharp_read_soft_separation": algebra["ipa_local_vs_ordinary_max_abs"] <= 1e-12 and algebra["temperature_local_vs_ordinary_max_abs"] >= 1e-3,
        "cross_chunk_iia_broken": algebra["iia_log_ratio_ranges"]["ordinary"] <= 1e-12 and algebra["iia_log_ratio_ranges"]["ipa2"] >= 0.1,
        "row_shift_invariant_and_stable": algebra["row_shift_probability_max_abs"] <= 1e-12 and algebra["stress_state_max_abs"] <= 1e-12 and algebra["stress_all_finite"],
        "constant_score_cardinality_exact": algebra["constant_mass_max_abs"] <= 1e-12,
        "pea_equals_ipa_times_dispersion": algebra["identity_max_abs"] <= 1e-12 and algebra["constant_identity_abs"] <= 1e-12 and algebra["nonconstant_identity_gap"] >= 1e-2,
        "pea_iid_cardinality_neutral": iid["pea_raw_max_abs_error"] <= 1e-12 and iid["pea_share_max_abs_error"] <= 1e-12,
        "ipa_iid_variance_premium_exact": iid["ipa_raw_max_abs_error"] <= 1e-12 and iid["ipa_share_max_abs_deviation"] >= 0.02,
        "pea_anchor_gain_over_ipa": all(row["target_payload_probability"]["pea2"] > row["target_payload_probability"]["ipa2"] for row in anchor_rows),
        "ipa_anchor_gain_over_ordinary": all(row["target_payload_probability"]["ipa2"] > row["target_payload_probability"]["ordinary"] for row in anchor_rows),
        "anchor_dispersion_identity_exact": max(row["identity_abs_error"] for row in anchor_rows) <= 1e-12,
        "consensus_spike_preferences_separate": consensus["ipa_prefers_consensus"] and consensus["pea_prefers_spike"],
    }
    return {
        "schema": "inclusive-power-attention-stage0-v1",
        "seed": SEED,
        "source_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREG),
        "algebra": algebra,
        "iid_cardinality": iid,
        "anchor": anchor,
        "consensus_spike": consensus,
        "resource_ledger": {
            "qk_matmul_change": 0,
            "pv_matmul_change": 0,
            "qkv_ffn_matmul_change": 0,
            "kv_cache_fields_added": 0,
            "learned_parameters_added": 0,
            "score_multiplications_added": 0,
            "score_reduction_accumulators_added": 0,
            "additional_exponentials_per_score": 0,
            "native_chunk_scalar_transforms": ["l/n", "l*l/n"],
            "unaligned_chunk_extra_merge_work_possible": True,
            "prefill_extra_live_value_accumulator_possible": True,
        },
        "gates": gates,
        "comparison_pass": all(gates.values()),
    }


def main() -> None:
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": result["gates"], "comparison_pass": result["comparison_pass"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
