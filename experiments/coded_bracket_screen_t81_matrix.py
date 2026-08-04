#!/usr/bin/env python3
"""Matrix-only falsifier for T81 star-coded sparse bracket screening."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def somp(a: np.ndarray, y: np.ndarray, sparsity: int) -> np.ndarray:
    """Simultaneous OMP with a known row sparsity and least-squares debiasing."""
    n_features = a.shape[1]
    width = y.shape[1]
    if sparsity <= 0:
        return np.zeros((n_features, width), dtype=np.float64)
    target = min(sparsity, a.shape[0], n_features)
    residual = y.copy()
    support: list[int] = []
    available = np.ones(n_features, dtype=bool)
    column_norm = np.linalg.norm(a, axis=0)
    column_norm = np.maximum(column_norm, 1e-12)
    coefficients = np.empty((0, width), dtype=np.float64)
    for _ in range(target):
        scores = np.linalg.norm(a.T @ residual, axis=1) / column_norm
        scores[~available] = -np.inf
        chosen = int(np.argmax(scores))
        if not np.isfinite(scores[chosen]):
            break
        support.append(chosen)
        available[chosen] = False
        coefficients, *_ = np.linalg.lstsq(a[:, support], y, rcond=None)
        residual = y - a[:, support] @ coefficients
    estimate = np.zeros((n_features, width), dtype=np.float64)
    if support:
        estimate[np.asarray(support)] = coefficients
    return estimate


def generate_world(k: int, degree: int, width: int, rng: np.random.Generator) -> list[np.ndarray]:
    stars: list[np.ndarray] = []
    for p in range(k - 1):
        n_p = k - p - 1
        d_p = min(degree, n_p)
        q = np.zeros((n_p, width), dtype=np.float64)
        support = rng.choice(n_p, size=d_p, replace=False)
        directions = rng.normal(size=(d_p, width))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        magnitudes = rng.uniform(1.0, 1.5, size=(d_p, 1))
        q[support] = directions * magnitudes
        stars.append(q)
    return stars


def flatten_stars(stars: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(stars, axis=0)


def support_scores(truth: np.ndarray, estimate: np.ndarray) -> tuple[bool, float, float, float, float]:
    true_support = np.linalg.norm(truth, axis=1) > 1e-12
    estimated_support = np.linalg.norm(estimate, axis=1) > 1e-12
    tp = int(np.count_nonzero(true_support & estimated_support))
    fp = int(np.count_nonzero(~true_support & estimated_support))
    fn = int(np.count_nonzero(true_support & ~estimated_support))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = float(np.sum(truth * truth))
    nmse = float(np.sum((estimate - truth) ** 2) / denom) if denom else 0.0
    return bool(np.array_equal(true_support, estimated_support)), precision, recall, f1, nmse


def exhaustive_pairs(stars: list[np.ndarray], sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
    truth = flatten_stars(stars)
    observations = truth + rng.normal(scale=sigma, size=truth.shape)
    sparsity = int(np.count_nonzero(np.linalg.norm(truth, axis=1) > 0.0))
    chosen = np.argpartition(np.linalg.norm(observations, axis=1), -sparsity)[-sparsity:]
    estimate = np.zeros_like(truth)
    estimate[chosen] = observations[chosen]
    loops = float(truth.shape[0])
    return estimate, loops, 2.0 * loops, 2.0 * loops


def adaptive_pairs(
    stars: list[np.ndarray], sigma: float, beta_threshold: float, rng: np.random.Generator
) -> tuple[np.ndarray, float, float, float]:
    estimates: list[np.ndarray] = []
    loops = 0.0
    for q in stars:
        n_p, width = q.shape
        d_p = int(np.count_nonzero(np.linalg.norm(q, axis=1) > 0.0))
        estimate = np.zeros_like(q)
        detections = 0
        for j in rng.permutation(n_p):
            observation = q[j] + rng.normal(scale=sigma, size=width)
            loops += 1.0
            if np.linalg.norm(observation) >= beta_threshold:
                estimate[j] = observation
                detections += 1
                if detections >= d_p:
                    break
        estimates.append(estimate)
    return flatten_stars(estimates), loops, 2.0 * loops, 2.0 * loops


def star_code(
    stars: list[np.ndarray],
    requested_r: int,
    sigma: float,
    equal_energy: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, float]:
    estimates: list[np.ndarray] = []
    loops = 0.0
    active_components = 0.0
    energy = 0.0
    for q in stars:
        n_p, width = q.shape
        d_p = int(np.count_nonzero(np.linalg.norm(q, axis=1) > 0.0))
        r_p = min(requested_r, n_p)
        if r_p >= n_p:
            a = np.eye(n_p, dtype=np.float64)
            active_components += 2.0 * n_p
            energy += 2.0 * n_p
        else:
            a = rng.choice(np.array([-1.0, 1.0]), size=(r_p, n_p))
            if equal_energy:
                a /= math.sqrt(n_p)
                energy += 2.0 * r_p
            else:
                energy += (1.0 + n_p) * r_p
            active_components += (1.0 + n_p) * r_p
        y = a @ q + rng.normal(scale=sigma, size=(r_p, width))
        estimates.append(somp(a, y, d_p))
        loops += r_p
    return flatten_stars(estimates), loops, active_components, energy


def new_accumulator() -> dict[str, float]:
    return {
        "trials": 0,
        "exact": 0,
        "precision_sum": 0.0,
        "recall_sum": 0.0,
        "f1_sum": 0.0,
        "nmse_sum": 0.0,
        "loops_sum": 0.0,
        "active_sum": 0.0,
        "energy_sum": 0.0,
    }


def record(acc: dict[str, float], truth: np.ndarray, result: tuple[np.ndarray, float, float, float]) -> None:
    estimate, loops, active, energy = result
    exact, precision, recall, f1, nmse = support_scores(truth, estimate)
    acc["trials"] += 1
    acc["exact"] += int(exact)
    acc["precision_sum"] += precision
    acc["recall_sum"] += recall
    acc["f1_sum"] += f1
    acc["nmse_sum"] += nmse
    acc["loops_sum"] += loops
    acc["active_sum"] += active
    acc["energy_sum"] += energy


def finish(acc: dict[str, float]) -> dict[str, float | int | list[float]]:
    trials = int(acc["trials"])
    exact = int(acc["exact"])
    lower, upper = wilson_interval(exact, trials)
    return {
        "trials": trials,
        "exact_successes": exact,
        "exact_rate": exact / trials,
        "exact_wilson95": [lower, upper],
        "precision": acc["precision_sum"] / trials,
        "recall": acc["recall_sum"] / trials,
        "f1": acc["f1_sum"] / trials,
        "nmse": acc["nmse_sum"] / trials,
        "identity_loops": acc["loops_sum"] / trials,
        "active_components": acc["active_sum"] / trials,
        "energy": acc["energy_sum"] / trials,
    }


def passes_phase(result: dict[str, float | int | list[float]], k: int) -> bool:
    lower_required = 0.90 if k == 32 else 0.85
    interval = result["exact_wilson95"]
    assert isinstance(interval, list)
    return float(result["exact_rate"]) >= 0.95 and float(interval[0]) >= lower_required


def run_star_stage(k: int, seeds: int, base_seed: int) -> dict[str, object]:
    degrees = [1, 2, 4]
    sigmas = [0.0, 0.05, 0.10]
    r_values = list(range(2, 25))
    cells: list[dict[str, object]] = []

    for degree in degrees:
        for sigma in sigmas:
            accumulators: dict[str, dict[str, float]] = {
                "exhaustive_pairs": new_accumulator(),
                "adaptive_pairs": new_accumulator(),
            }
            for r in r_values:
                accumulators[f"star_unnormalized_r{r}"] = new_accumulator()
                accumulators[f"star_equal_energy_r{r}"] = new_accumulator()

            for seed_index in range(seeds):
                world_rng = np.random.default_rng(np.random.SeedSequence([base_seed, k, degree, int(1000 * sigma), seed_index]))
                stars = generate_world(k, degree, 4, world_rng)
                truth = flatten_stars(stars)

                record(
                    accumulators["exhaustive_pairs"],
                    truth,
                    exhaustive_pairs(stars, sigma, np.random.default_rng(np.random.SeedSequence([base_seed, 11, k, degree, int(1000 * sigma), seed_index]))),
                )
                record(
                    accumulators["adaptive_pairs"],
                    truth,
                    adaptive_pairs(
                        stars,
                        sigma,
                        0.5,
                        np.random.default_rng(np.random.SeedSequence([base_seed, 13, k, degree, int(1000 * sigma), seed_index])),
                    ),
                )
                for r in r_values:
                    record(
                        accumulators[f"star_unnormalized_r{r}"],
                        truth,
                        star_code(
                            stars,
                            r,
                            sigma,
                            False,
                            np.random.default_rng(np.random.SeedSequence([base_seed, 17, k, degree, int(1000 * sigma), r, seed_index])),
                        ),
                    )
                    record(
                        accumulators[f"star_equal_energy_r{r}"],
                        truth,
                        star_code(
                            stars,
                            r,
                            sigma,
                            True,
                            np.random.default_rng(np.random.SeedSequence([base_seed, 19, k, degree, int(1000 * sigma), r, seed_index])),
                        ),
                    )

            methods = {name: finish(acc) for name, acc in accumulators.items()}
            selected: dict[str, object | None] = {}
            for prefix in ("star_unnormalized", "star_equal_energy"):
                passing = [
                    (r, methods[f"{prefix}_r{r}"])
                    for r in r_values
                    if passes_phase(methods[f"{prefix}_r{r}"], k)
                ]
                selected[prefix] = {"r": passing[0][0], **passing[0][1]} if passing else None

            pair_candidates = [
                methods[name]
                for name in ("adaptive_pairs", "exhaustive_pairs")
                if passes_phase(methods[name], k)
            ]
            selected_pair = min(pair_candidates, key=lambda item: float(item["identity_loops"])) if pair_candidates else None
            selected["best_pairwise"] = selected_pair

            ratio = None
            star_selected = selected["star_unnormalized"]
            if selected_pair is not None and isinstance(star_selected, dict):
                ratio = float(selected_pair["identity_loops"]) / float(star_selected["identity_loops"])

            cells.append(
                {
                    "k": k,
                    "degree": degree,
                    "sigma": sigma,
                    "methods": methods,
                    "selected": selected,
                    "pair_to_star_identity_ratio": ratio,
                }
            )

    ratios = [float(cell["pair_to_star_identity_ratio"]) for cell in cells if cell["pair_to_star_identity_ratio"] is not None]
    at_least_two = sum(ratio >= 2.0 for ratio in ratios)
    below_noninferiority = sum(ratio < 1.0 / 1.2 for ratio in ratios)
    median_ratio = float(np.median(ratios)) if ratios else None
    development_pass = bool(
        ratios
        and median_ratio is not None
        and median_ratio >= 1.2
        and at_least_two / len(ratios) >= 0.75
        and below_noninferiority == 0
    )
    return {
        "stage": "star_vs_pairwise",
        "k": k,
        "seeds": seeds,
        "cells": cells,
        "summary": {
            "cells_with_comparable_phase": len(ratios),
            "median_pair_to_star_identity_ratio": median_ratio,
            "fraction_cells_at_least_2x": at_least_two / len(ratios) if ratios else 0.0,
            "cells_star_loses_over_20pct": below_noninferiority,
            "development_pass_for_dense_stage": development_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=32, choices=(32, 64))
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=81032026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else (64 if args.k == 32 else 32)
    started = time.perf_counter()
    result = run_star_stage(args.k, seeds, args.base_seed)
    result["manifest"] = {
        "base_seed": args.base_seed,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "manifest": result["manifest"]}, indent=2))


if __name__ == "__main__":
    main()
