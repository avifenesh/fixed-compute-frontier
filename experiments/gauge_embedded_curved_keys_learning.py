#!/usr/bin/env python3
"""Matched learned two-key retrieval gate for GECK-G2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results" / "gauge-embedded-curved-keys-learning-preregistration.md"
TEST_SOURCE = ROOT / "tests" / "test_gauge_embedded_curved_keys_learning.py"
INTEGRITY_MANIFEST = ROOT / "results" / "gauge-embedded-curved-keys-learning-integrity-manifest.json"
OUTPUT = ROOT / "results" / "gauge-embedded-curved-keys-learning.json"
STAGE0_SOURCE = ROOT / "experiments" / "gauge_embedded_curved_keys.py"
STAGE0_PREREGISTRATION = ROOT / "results" / "gauge-embedded-curved-keys-preregistration.md"
STAGE0_TEST = ROOT / "tests" / "test_gauge_embedded_curved_keys.py"
STAGE0_INTEGRITY = ROOT / "results" / "gauge-embedded-curved-keys-integrity-manifest.json"
STAGE0_RESULT = ROOT / "results" / "gauge-embedded-curved-keys-stage0.json"

ARMS = (
    "bilinear",
    "released_pivot_linear_only",
    "linear_epilogue_functional_control",
    "constant_only",
    "one_curvature",
    "centered_full",
    "full_geck_g2",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    return (
        parameters[:4].reshape(2, 2),
        parameters[4:8].reshape(2, 2),
        float(parameters[8]),
        float(parameters[9]),
    )


def model_logits(
    parameters: np.ndarray,
    arm: str,
    query_types: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    query_weight, key_weight, phi, log_scale = unpack(parameters)
    query = query_weight[:, query_types].T
    raw_key = np.einsum("ab,nkb->nka", key_weight, keys, optimize=True)
    even = raw_key[:, :, 0]
    odd = raw_key[:, :, 1]
    rotation_coefficient = np.sin(phi)
    scale_coefficient = np.tanh(log_scale)
    if arm == "full_geck_g2":
        transformed = np.stack((
            even + rotation_coefficient * odd * odd,
            odd + scale_coefficient * even * even,
        ), axis=-1)
    elif arm == "centered_full":
        transformed = np.stack((
            even + rotation_coefficient * (odd * odd - 1.0),
            odd + scale_coefficient * (even * even - 1.0),
        ), axis=-1)
    elif arm == "one_curvature":
        transformed = np.stack((
            even + rotation_coefficient * odd * odd,
            odd,
        ), axis=-1)
    elif arm == "linear_epilogue_functional_control":
        transformed = np.stack((
            even + rotation_coefficient * odd,
            odd + scale_coefficient * even,
        ), axis=-1)
    elif arm == "constant_only":
        transformed = np.stack((
            even + rotation_coefficient,
            odd + scale_coefficient,
        ), axis=-1)
    elif arm in ("bilinear", "released_pivot_linear_only"):
        transformed = raw_key
    else:
        raise ValueError(f"unknown arm: {arm}")
    scores = np.einsum("na,nka->nk", query, transformed, optimize=True)
    return scores[:, 0] - scores[:, 1]


def binary_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    positive = logits >= 0.0
    result = np.empty_like(logits)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def initial_parameters() -> np.ndarray:
    return np.concatenate((np.eye(2).reshape(-1), np.eye(2).reshape(-1), [0.0, 0.0]))


def nonlinear_teacher_parameters() -> np.ndarray:
    return np.concatenate((
        np.eye(2).reshape(-1),
        np.eye(2).reshape(-1),
        [np.arcsin(0.8), np.arctanh(0.75)],
    ))


def bilinear_teacher_parameters() -> np.ndarray:
    return np.concatenate((
        np.array([[1.0, 0.2], [-0.3, 0.8]]).reshape(-1),
        np.array([[0.7, -0.4], [0.2, 1.1]]).reshape(-1),
        [0.0, 0.0],
    ))


def make_world(seed: int, samples: int = 5_000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    query_types = rng.integers(0, 2, size=samples)
    keys = rng.normal(size=(samples, 2, 2))
    order = rng.permutation(samples)
    train = order[:3_500]
    test = order[3_500:]
    nonlinear_logits = model_logits(
        nonlinear_teacher_parameters(), "full_geck_g2", query_types, keys
    )
    bilinear_logits = model_logits(
        bilinear_teacher_parameters(), "full_geck_g2", query_types, keys
    )
    return {
        "query_types": query_types,
        "keys": keys,
        "train": train,
        "test": test,
        "nonlinear_teacher_logits": nonlinear_logits,
        "nonlinear_labels": sigmoid(nonlinear_logits),
        "bilinear_teacher_logits": bilinear_logits,
        "bilinear_labels": sigmoid(bilinear_logits),
    }


def make_restart_starts(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = np.stack([
        initial_parameters() + rng.normal(scale=0.01, size=10)
        for _ in range(3)
    ])
    starts[:, 8:] = 0.0
    return starts


def fit_arm(
    world: dict[str, np.ndarray],
    arm: str,
    label_key: str,
    restart_starts: np.ndarray,
) -> dict[str, object]:
    query_types = world["query_types"]
    keys = world["keys"]
    labels = world[label_key]
    train = world["train"]
    test = world["test"]

    def objective(parameters: np.ndarray) -> float:
        return binary_nll(
            model_logits(parameters, arm, query_types[train], keys[train]),
            labels[train],
        )

    best = None
    for start in restart_starts:
        optimization = minimize(
            objective,
            start,
            method="L-BFGS-B",
            options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        test_logits = model_logits(
            optimization.x, arm, query_types[test], keys[test]
        )
        result = {
            "train_nll": float(optimization.fun),
            "test_nll": binary_nll(test_logits, labels[test]),
            "optimization_success": bool(optimization.success),
            "iterations": int(optimization.nit),
            "parameters": optimization.x.tolist(),
            "phi": float(optimization.x[8]),
            "log_scale": float(optimization.x[9]),
            "hard_teacher_agreement": float(
                np.mean((test_logits >= 0.0) == (world[label_key.replace("labels", "teacher_logits")][test] >= 0.0))
            ),
        }
        result["finite"] = bool(np.isfinite([
            result["train_nll"],
            result["test_nll"],
            result["phi"],
            result["log_scale"],
            result["hard_teacher_agreement"],
            *result["parameters"],
        ]).all())
        if best is None or result["train_nll"] < best["train_nll"]:
            best = result
    if best is None:
        raise AssertionError("optimizer produced no result")
    return best


def run_world(seed: int) -> dict[str, object]:
    world = make_world(seed)
    test = world["test"]
    nonlinear_teacher_nll = binary_nll(
        world["nonlinear_teacher_logits"][test], world["nonlinear_labels"][test]
    )
    bilinear_teacher_nll = binary_nll(
        world["bilinear_teacher_logits"][test], world["bilinear_labels"][test]
    )
    nonlinear_starts = make_restart_starts(seed * 100)
    bilinear_starts = make_restart_starts(seed * 1_000)
    nonlinear = {
        arm: fit_arm(world, arm, "nonlinear_labels", nonlinear_starts)
        for arm in ARMS
    }
    bilinear = {
        arm: fit_arm(world, arm, "bilinear_labels", bilinear_starts)
        for arm in ("bilinear", "one_curvature", "full_geck_g2")
    }
    selected_fits = (*nonlinear.values(), *bilinear.values())
    gates = {
        "all_selected_fits_successful_and_finite": all(
            fit["optimization_success"] and fit["finite"]
            for fit in selected_fits
        ),
        "full_reaches_nonlinear_teacher_nll": nonlinear["full_geck_g2"]["test_nll"] <= nonlinear_teacher_nll + 1e-5,
        "full_hard_teacher_agreement_at_least_99_9_percent": nonlinear["full_geck_g2"]["hard_teacher_agreement"] >= 0.999,
        "full_beats_one_curvature_by_0_04_nll": nonlinear["full_geck_g2"]["test_nll"] <= nonlinear["one_curvature"]["test_nll"] - 0.04,
        "full_beats_bilinear_by_0_08_nll": nonlinear["full_geck_g2"]["test_nll"] <= nonlinear["bilinear"]["test_nll"] - 0.08,
        "centered_square_matches_full": abs(nonlinear["centered_full"]["test_nll"] - nonlinear["full_geck_g2"]["test_nll"]) <= 1e-7,
        "released_pivot_linear_only_matches_bilinear": abs(nonlinear["released_pivot_linear_only"]["test_nll"] - nonlinear["bilinear"]["test_nll"]) <= 1e-5,
        "linear_epilogue_does_not_beat_bilinear": nonlinear["linear_epilogue_functional_control"]["test_nll"] >= nonlinear["bilinear"]["test_nll"] - 1e-5,
        "constant_only_does_not_beat_bilinear": nonlinear["constant_only"]["test_nll"] >= nonlinear["bilinear"]["test_nll"] - 1e-5,
        "bilinear_teacher_full_matches_optimum": bilinear["full_geck_g2"]["test_nll"] <= bilinear_teacher_nll + 1e-5,
        "bilinear_teacher_no_geck_regression": bilinear["full_geck_g2"]["test_nll"] <= bilinear["bilinear"]["test_nll"] + 1e-5,
        "bilinear_teacher_returns_curvature_near_zero": abs(bilinear["full_geck_g2"]["phi"]) <= 1e-3 and abs(bilinear["full_geck_g2"]["log_scale"]) <= 1e-3,
    }
    return {
        "seed": seed,
        "raw_parameter_count_all_arms": 10,
        "key_pivot_input_is_identically_zero": True,
        "nonlinear_teacher_test_nll": nonlinear_teacher_nll,
        "bilinear_teacher_test_nll": bilinear_teacher_nll,
        "nonlinear_teacher": nonlinear,
        "bilinear_teacher": bilinear,
        "gates": gates,
        "pass": all(gates.values()),
    }


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "test": TEST_SOURCE,
        "stage0_source": STAGE0_SOURCE,
        "stage0_preregistration": STAGE0_PREREGISTRATION,
        "stage0_test": STAGE0_TEST,
        "stage0_integrity": STAGE0_INTEGRITY,
        "stage0_result": STAGE0_RESULT,
    }
    checks = {
        name: manifest.get(f"{name}_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid GECK learning integrity: {checks}")
    stage0 = json.loads(STAGE0_RESULT.read_text())
    if not stage0["decision"]["stage0_pass"]:
        raise ValueError("GECK Stage 0 did not pass")
    return checks


def run() -> dict[str, object]:
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    expected_runtime = {"python": "3.14.4", "numpy": "2.3.5", "scipy": "1.18.0"}
    if runtime != expected_runtime:
        raise RuntimeError(f"frozen GECK learning runtime mismatch: {runtime}")
    integrity = validate_integrity()
    worlds = [run_world(seed) for seed in (41, 73, 101, 137, 179)]
    decision = {
        "learning_gate_pass": all(world["pass"] for world in worlds),
        "worlds_passed": sum(world["pass"] for world in worlds),
        "worlds_total": len(worlds),
        "next_gate": "natural RoPE addressed-bag screen with positional controls",
        "h100_admitted": False,
    }
    return {
        "schema": "gauge-embedded-curved-keys-learning-v1",
        "candidate": "GECK-G2-full-gauge-bi-curved-keys",
        "integrity_checks": integrity,
        "runtime": runtime,
        "hashes": {
            "source": sha256_file(Path(__file__)),
            "preregistration": sha256_file(PREREGISTRATION),
            "test": sha256_file(TEST_SOURCE),
            "stage0_result": sha256_file(STAGE0_RESULT),
        },
        "protocol": {
            "world_seeds": [41, 73, 101, 137, 179],
            "samples_per_world": 5_000,
            "train_samples": 3_500,
            "test_samples": 1_500,
            "optimizer": "scipy L-BFGS-B",
            "restarts": 3,
            "identical_restart_starts_within_each_teacher_world": True,
            "soft_teacher_labels": True,
        },
        "worlds": worlds,
        "decision": decision,
    }


def main() -> None:
    payload = run()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
