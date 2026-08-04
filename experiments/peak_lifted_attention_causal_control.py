#!/usr/bin/env python3
"""Causal partial-chunk and anchor-payload controls for peak lift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "results" / "peak-lifted-attention-causal-control-preregistration.md"
OUTPUT = ROOT / "results" / "peak-lifted-attention-causal-control.json"
SEED = 20260731
C = 64
ALPHA = 0.5
TRIALS = 4096


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks(length: int, origin: int = 0, query_relative: bool = False) -> list[np.ndarray]:
    positions = np.arange(length)
    labels = ((length - 1 - positions) // C if query_relative else (positions + origin) // C)
    return [positions[labels == label] for label in np.unique(labels)]


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def lifted_logits(
    scores: np.ndarray,
    groups: list[np.ndarray],
    arm: str,
    kappa: np.ndarray,
) -> np.ndarray:
    out = scores.copy()
    if arm == "ordinary":
        return out
    if arm == "global_temperature":
        return (1.0 + ALPHA) * out
    for ids in groups:
        block = scores[:, ids]
        maximum = np.max(block, axis=1)
        n = len(ids)
        if arm == "raw_peak":
            statistic = maximum
        elif arm == "normal_corrected_peak":
            statistic = maximum - kappa[n]
        elif arm == "confidence":
            if n == 1:
                statistic = np.zeros(scores.shape[0])
            else:
                mass = np.exp(block - maximum[:, None]).sum(axis=1)
                statistic = (n - mass) / (n - 1)
        else:
            raise ValueError(arm)
        out[:, ids] += ALPHA * statistic[:, None]
    return out


def sample_scores(rng: np.random.Generator, family: str, shape: tuple[int, int]) -> np.ndarray:
    if family == "normal":
        return rng.normal(size=shape)
    if family == "laplace":
        return rng.laplace(scale=1.0 / np.sqrt(2.0), size=shape)
    if family == "student_t5":
        return rng.standard_t(5, size=shape) * np.sqrt(3.0 / 5.0)
    if family == "constant":
        return np.zeros(shape)
    raise ValueError(family)


def calibrate_kappa(rng: np.random.Generator, draws: int = 131072) -> np.ndarray:
    result = np.zeros(C + 1, dtype=np.float64)
    batch = rng.normal(size=(draws, C))
    running = np.full(draws, -np.inf)
    for n in range(1, C + 1):
        running = np.maximum(running, batch[:, n - 1])
        result[n] = np.mean(running)
    return result


def fairness_probe(rng: np.random.Generator, kappa: np.ndarray) -> dict:
    arms = ("raw_peak", "normal_corrected_peak", "confidence")
    families = ("normal", "laplace", "student_t5", "constant")
    layouts = (("fixed_0", 0, False), ("fixed_32", C // 2, False), ("query_relative", 0, True))
    records: dict[str, dict] = {}
    for family in families:
        records[family] = {}
        for layout_name, origin, relative in layouts:
            per_arm = {arm: [] for arm in arms}
            for remainder in range(1, C + 1):
                length = 3 * C + remainder
                scores = sample_scores(rng, family, (TRIALS, length))
                groups = chunks(length, origin, relative)
                ordinary_weights = softmax(scores)
                for arm in arms:
                    weights = softmax(lifted_logits(scores, groups, arm, kappa))
                    ratios = []
                    bounds = []
                    chunk_records = []
                    for chunk_index, ids in enumerate(groups):
                        fair_share = len(ids) / length
                        candidate_mass = weights[:, ids].sum(axis=1)
                        ordinary_mass = ordinary_weights[:, ids].sum(axis=1)
                        paired_relative = (candidate_mass - ordinary_mass) / fair_share
                        relative_mean = float(paired_relative.mean())
                        relative_se = float(paired_relative.std(ddof=1) / np.sqrt(TRIALS))
                        bound = abs(relative_mean) + 5.0 * relative_se
                        ratios.append(1.0 + relative_mean)
                        bounds.append(bound)
                        chunk_records.append({
                            "chunk_index": chunk_index,
                            "chunk_size": len(ids),
                            "signed_relative_paired_mean": relative_mean,
                            "paired_standard_error": relative_se,
                            "five_se_abs_bound": bound,
                            "is_latest_chunk": bool(length - 1 in ids),
                        })
                    latest_group = next(i for i, ids in enumerate(groups) if length - 1 in ids)
                    per_arm[arm].append({
                        "remainder": remainder,
                        "maximum_abs_chunk_share_deviation": float(np.max(np.abs(np.asarray(ratios) - 1.0))),
                        "maximum_five_se_abs_chunk_share_bound": float(np.max(bounds)),
                        "latest_chunk_share_ratio": ratios[latest_group],
                        "chunks": chunk_records,
                    })
            records[family][layout_name] = {}
            for arm, rows in per_arm.items():
                maxima = np.array([row["maximum_abs_chunk_share_deviation"] for row in rows])
                bounds = np.array([row["maximum_five_se_abs_chunk_share_bound"] for row in rows])
                latest = np.array([row["latest_chunk_share_ratio"] for row in rows])
                records[family][layout_name][arm] = {
                    "maximum_abs_chunk_share_deviation": float(maxima.max()),
                    "maximum_five_se_abs_chunk_share_bound": float(bounds.max()),
                    "latest_chunk_share_ratio_min": float(latest.min()),
                    "latest_chunk_share_ratio_max": float(latest.max()),
                    "latest_chunk_share_ratio_range": float(latest.max() - latest.min()),
                    "worst_remainder": int(rows[int(np.argmax(bounds))]["remainder"]),
                    "by_remainder": rows,
                }
    return records


def anchor_probe(rng: np.random.Generator, kappa: np.ndarray) -> dict:
    trials = 16384
    length = 4 * C
    payload_distance = 8
    arms = ("ordinary", "global_temperature", "raw_peak", "normal_corrected_peak", "confidence")
    by_origin: dict[str, dict] = {}
    for origin in (0, C // 2):
        origin_rng = np.random.default_rng(SEED + 2)
        ratios = {arm: [] for arm in arms if arm != "ordinary"}
        same_chunk = []
        groups = chunks(length, origin, False)
        label = np.empty(length, dtype=np.int64)
        for i, ids in enumerate(groups):
            label[ids] = i
        # Reuse the same noise across arms at each position.
        for anchor_position in range(C):
            payload_position = (anchor_position + payload_distance) % length
            distractor_position = 2 * C + 17
            scores = origin_rng.normal(size=(trials, length))
            scores[:, anchor_position] = 4.0
            scores[:, payload_position] = -0.5
            scores[:, distractor_position] = 3.8
            ordinary = softmax(scores)[:, payload_position].mean()
            for arm in ratios:
                probability = softmax(lifted_logits(scores, groups, arm, kappa))[:, payload_position].mean()
                ratios[arm].append(float(probability / ordinary))
            same_chunk.append(bool(label[anchor_position] == label[payload_position]))
        by_origin[str(origin)] = {}
        same_chunk_array = np.asarray(same_chunk, dtype=bool)
        for arm, values in ratios.items():
            value_array = np.asarray(values)
            by_origin[str(origin)][arm] = {
                "mean_payload_probability_ratio": float(np.mean(values)),
                "minimum_payload_probability_ratio": float(np.min(values)),
                "maximum_payload_probability_ratio": float(np.max(values)),
                "same_chunk_mean_ratio": float(value_array[same_chunk_array].mean()),
                "same_chunk_minimum_ratio": float(value_array[same_chunk_array].min()),
                "crossing_boundary_mean_ratio": float(value_array[~same_chunk_array].mean()),
                "crossing_boundary_minimum_ratio": float(value_array[~same_chunk_array].min()),
            }
        by_origin[str(origin)]["anchor_payload_same_chunk_fraction"] = float(np.mean(same_chunk))
    return by_origin


def run() -> dict:
    calibration_rng = np.random.default_rng(SEED)
    kappa = calibrate_kappa(calibration_rng)
    rng = np.random.default_rng(SEED + 1)
    fairness = fairness_probe(rng, kappa)
    anchor = anchor_probe(rng, kappa)

    fairness_gates = {}
    for arm in ("raw_peak", "normal_corrected_peak", "confidence"):
        threshold_by_family = {"normal": 0.05, "laplace": 0.05, "student_t5": 0.05, "constant": 0.01}
        fairness_gates[arm] = all(
            fairness[family][layout][arm]["maximum_five_se_abs_chunk_share_bound"] <= threshold_by_family[family]
            for family in threshold_by_family
            for layout in fairness[family]
        )
    anchor_gates = {}
    for arm in ("raw_peak", "normal_corrected_peak", "confidence"):
        best = max(anchor.values(), key=lambda x: x[arm]["mean_payload_probability_ratio"])
        anchor_gates[arm] = (
            best[arm]["mean_payload_probability_ratio"] >= 1.10
            and best[arm]["mean_payload_probability_ratio"] > best["global_temperature"]["mean_payload_probability_ratio"]
            and best[arm]["crossing_boundary_minimum_ratio"] >= 1.0
        )
    gates = {
        "raw_peak_iid_fair": fairness_gates["raw_peak"],
        "normal_corrected_peak_distribution_robust": fairness_gates["normal_corrected_peak"],
        "confidence_iid_fair": fairness_gates["confidence"],
        "raw_peak_anchor_signal": anchor_gates["raw_peak"],
        "normal_corrected_peak_anchor_signal": anchor_gates["normal_corrected_peak"],
        "confidence_anchor_signal": anchor_gates["confidence"],
    }
    live_variant = "confidence" if gates["confidence_iid_fair"] and gates["confidence_anchor_signal"] else None
    return {
        "schema": "peak-lifted-attention-causal-control-v1",
        "seed": SEED,
        "chunk_width": C,
        "alpha": ALPHA,
        "trials_per_iid_cell": TRIALS,
        "source_sha256": sha256(Path(__file__)),
        "preregistration_sha256": sha256(PREREG),
        "normal_expected_maximum": kappa.tolist(),
        "fairness": fairness,
        "anchor": anchor,
        "gates": gates,
        "live_variant": live_variant,
    }


def main() -> None:
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": result["gates"], "live_variant": result["live_variant"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
