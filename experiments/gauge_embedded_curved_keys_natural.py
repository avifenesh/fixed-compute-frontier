#!/usr/bin/env python3
"""Natural RoPE addressed-bag screen for GECK-G2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results" / "gauge-embedded-curved-keys-natural-preregistration.md"
TEST_SOURCE = ROOT / "tests" / "test_gauge_embedded_curved_keys_natural.py"
INTEGRITY_MANIFEST = ROOT / "results" / "gauge-embedded-curved-keys-natural-integrity-manifest.json"
OUTPUT = ROOT / "results" / "gauge-embedded-curved-keys-natural.json"
LEARNING_SOURCE = ROOT / "experiments" / "gauge_embedded_curved_keys_learning.py"
LEARNING_PREREGISTRATION = ROOT / "results" / "gauge-embedded-curved-keys-learning-preregistration.md"
LEARNING_TEST = ROOT / "tests" / "test_gauge_embedded_curved_keys_learning.py"
LEARNING_INTEGRITY = ROOT / "results" / "gauge-embedded-curved-keys-learning-integrity-manifest.json"
LEARNING_RESULT = ROOT / "results" / "gauge-embedded-curved-keys-learning.json"

ARMS = (
    "bilinear",
    "released_pivot_linear_only",
    "linear_epilogue_functional_control",
    "position_only_constant_control",
    "one_curvature",
    "centered_full",
    "full_geck_g2",
)
NEGATIVE_CONTROL_ARMS = ("bilinear", "one_curvature", "full_geck_g2")
WORLD_SEEDS = (53, 89, 127, 167, 211)
ITEMS = 8
SAMPLES = 3_000
TRAIN_SAMPLES = 2_000
ROPE_BASE = 10_000.0
ROPE_HEAD_DIMENSION = 64
ROPE_PAIR_INDEX = 16
ROPE_FREQUENCY = ROPE_BASE ** (-2.0 * ROPE_PAIR_INDEX / ROPE_HEAD_DIMENSION)
QUERY_POSITION = 128
TARGET_TEMPERATURE = 1.25
WEIGHT_DECAY = 1e-4


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_parameters() -> np.ndarray:
    return np.concatenate((
        np.eye(2).reshape(-1),
        np.eye(2).reshape(-1),
        [0.0, 0.0],
    ))


def make_restart_starts(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = np.stack([
        initial_parameters() + rng.normal(scale=0.01, size=10)
        for _ in range(3)
    ])
    starts[:, 8:] = 0.0
    return starts


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    exponential = np.exp(centered)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    return float(np.mean(
        np.log(np.sum(np.exp(centered), axis=-1))
        - np.sum(labels * centered, axis=-1)
    ))


def make_positions(rng: np.random.Generator, samples: int) -> np.ndarray:
    return np.stack([
        rng.choice(QUERY_POSITION, size=ITEMS, replace=False)
        for _ in range(samples)
    ])


def make_world(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    query_types = rng.integers(0, 2, size=SAMPLES)
    keys = rng.normal(size=(SAMPLES, ITEMS, 2))
    positions = make_positions(rng, SAMPLES)
    resampled_positions = make_positions(
        np.random.default_rng(seed * 10_000 + 7), SAMPLES
    )
    angles = (positions - QUERY_POSITION) * ROPE_FREQUENCY
    resampled_angles = (
        (resampled_positions - QUERY_POSITION) * ROPE_FREQUENCY
    )
    selected_feature = np.where(
        query_types[:, None] == 0,
        keys[:, :, 1],
        keys[:, :, 0],
    )
    magnitude_labels = softmax(TARGET_TEMPERATURE * np.abs(selected_feature))
    linear_labels = softmax(TARGET_TEMPERATURE * selected_feature)
    order = rng.permutation(SAMPLES)
    return {
        "query_types": query_types,
        "keys": keys,
        "angles": angles,
        "resampled_angles": resampled_angles,
        "magnitude_labels": magnitude_labels,
        "linear_labels": linear_labels,
        "magnitude_targets": np.argmax(np.abs(selected_feature), axis=-1),
        "linear_targets": np.argmax(selected_feature, axis=-1),
        "train": order[:TRAIN_SAMPLES],
        "test": order[TRAIN_SAMPLES:],
    }


def forward_with_cache(
    parameters: np.ndarray,
    arm: str,
    query_types: np.ndarray,
    keys: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    query_weight = parameters[:4].reshape(2, 2)
    key_weight = parameters[4:8].reshape(2, 2)
    phi = float(parameters[8])
    log_scale = float(parameters[9])
    query = query_weight[:, query_types].T
    raw_key = np.einsum("ab,nkb->nka", key_weight, keys, optimize=True)
    even = raw_key[:, :, 0]
    odd = raw_key[:, :, 1]
    rotation_coefficient = np.sin(phi)
    scale_coefficient = np.tanh(log_scale)

    if arm == "full_geck_g2":
        transformed_even = even + rotation_coefficient * odd * odd
        transformed_odd = odd + scale_coefficient * even * even
    elif arm == "centered_full":
        transformed_even = even + rotation_coefficient * (odd * odd - 1.0)
        transformed_odd = odd + scale_coefficient * (even * even - 1.0)
    elif arm == "one_curvature":
        transformed_even = even + rotation_coefficient * odd * odd
        transformed_odd = odd
    elif arm == "linear_epilogue_functional_control":
        transformed_even = even + rotation_coefficient * odd
        transformed_odd = odd + scale_coefficient * even
    elif arm == "position_only_constant_control":
        transformed_even = even + rotation_coefficient
        transformed_odd = odd + scale_coefficient
    elif arm in ("bilinear", "released_pivot_linear_only"):
        transformed_even = even
        transformed_odd = odd
    else:
        raise ValueError(f"unknown arm: {arm}")

    cosine = np.cos(angles)
    sine = np.sin(angles)
    rotated_even = cosine * transformed_even - sine * transformed_odd
    rotated_odd = sine * transformed_even + cosine * transformed_odd
    logits = (
        query[:, None, 0] * rotated_even
        + query[:, None, 1] * rotated_odd
    )
    cache: dict[str, np.ndarray | float] = {
        "query": query,
        "keys": keys,
        "even": even,
        "odd": odd,
        "rotation_coefficient": rotation_coefficient,
        "scale_coefficient": scale_coefficient,
        "cosine": cosine,
        "sine": sine,
        "transformed_even": transformed_even,
        "transformed_odd": transformed_odd,
    }
    return logits, cache


def model_logits(
    parameters: np.ndarray,
    arm: str,
    query_types: np.ndarray,
    keys: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    return forward_with_cache(parameters, arm, query_types, keys, angles)[0]


def objective_and_gradient(
    parameters: np.ndarray,
    arm: str,
    query_types: np.ndarray,
    keys: np.ndarray,
    angles: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    logits, cache = forward_with_cache(
        parameters, arm, query_types, keys, angles
    )
    samples = logits.shape[0]
    logit_gradient = (softmax(logits) - labels) / samples
    query = cache["query"]
    even = cache["even"]
    odd = cache["odd"]
    cosine = cache["cosine"]
    sine = cache["sine"]
    rotation_coefficient = float(cache["rotation_coefficient"])
    scale_coefficient = float(cache["scale_coefficient"])
    transformed_even = cache["transformed_even"]
    transformed_odd = cache["transformed_odd"]

    rotated_even = cosine * transformed_even - sine * transformed_odd
    rotated_odd = sine * transformed_even + cosine * transformed_odd
    query_weight_gradient = np.zeros((2, 2), dtype=np.float64)
    for query_type in (0, 1):
        mask = query_types == query_type
        query_weight_gradient[0, query_type] = np.sum(
            logit_gradient[mask] * rotated_even[mask]
        )
        query_weight_gradient[1, query_type] = np.sum(
            logit_gradient[mask] * rotated_odd[mask]
        )

    transformed_even_gradient = logit_gradient * (
        query[:, None, 0] * cosine + query[:, None, 1] * sine
    )
    transformed_odd_gradient = logit_gradient * (
        -query[:, None, 0] * sine + query[:, None, 1] * cosine
    )
    phi_gradient = 0.0
    log_scale_gradient = 0.0

    if arm == "full_geck_g2":
        even_gradient = (
            transformed_even_gradient
            + transformed_odd_gradient * 2.0 * scale_coefficient * even
        )
        odd_gradient = (
            transformed_odd_gradient
            + transformed_even_gradient * 2.0 * rotation_coefficient * odd
        )
        phi_gradient = np.sum(
            transformed_even_gradient * np.cos(parameters[8]) * odd * odd
        )
        log_scale_gradient = np.sum(
            transformed_odd_gradient
            * (1.0 - scale_coefficient * scale_coefficient)
            * even
            * even
        )
    elif arm == "centered_full":
        even_gradient = (
            transformed_even_gradient
            + transformed_odd_gradient * 2.0 * scale_coefficient * even
        )
        odd_gradient = (
            transformed_odd_gradient
            + transformed_even_gradient * 2.0 * rotation_coefficient * odd
        )
        phi_gradient = np.sum(
            transformed_even_gradient
            * np.cos(parameters[8])
            * (odd * odd - 1.0)
        )
        log_scale_gradient = np.sum(
            transformed_odd_gradient
            * (1.0 - scale_coefficient * scale_coefficient)
            * (even * even - 1.0)
        )
    elif arm == "one_curvature":
        even_gradient = transformed_even_gradient
        odd_gradient = (
            transformed_odd_gradient
            + transformed_even_gradient * 2.0 * rotation_coefficient * odd
        )
        phi_gradient = np.sum(
            transformed_even_gradient * np.cos(parameters[8]) * odd * odd
        )
    elif arm == "linear_epilogue_functional_control":
        even_gradient = (
            transformed_even_gradient
            + transformed_odd_gradient * scale_coefficient
        )
        odd_gradient = (
            transformed_odd_gradient
            + transformed_even_gradient * rotation_coefficient
        )
        phi_gradient = np.sum(
            transformed_even_gradient * np.cos(parameters[8]) * odd
        )
        log_scale_gradient = np.sum(
            transformed_odd_gradient
            * (1.0 - scale_coefficient * scale_coefficient)
            * even
        )
    elif arm == "position_only_constant_control":
        even_gradient = transformed_even_gradient
        odd_gradient = transformed_odd_gradient
        phi_gradient = np.sum(
            transformed_even_gradient * np.cos(parameters[8])
        )
        log_scale_gradient = np.sum(
            transformed_odd_gradient
            * (1.0 - scale_coefficient * scale_coefficient)
        )
    else:
        even_gradient = transformed_even_gradient
        odd_gradient = transformed_odd_gradient

    key_weight_gradient = np.stack((
        np.einsum("nk,nkb->b", even_gradient, keys, optimize=True),
        np.einsum("nk,nkb->b", odd_gradient, keys, optimize=True),
    ))
    gradient = np.concatenate((
        query_weight_gradient.reshape(-1),
        key_weight_gradient.reshape(-1),
        [phi_gradient, log_scale_gradient],
    ))
    objective = cross_entropy(logits, labels) + WEIGHT_DECAY * float(
        parameters @ parameters
    )
    gradient += 2.0 * WEIGHT_DECAY * parameters
    return objective, gradient


def fit_arm(
    world: dict[str, np.ndarray],
    arm: str,
    label_key: str,
    target_key: str,
    restart_starts: np.ndarray,
) -> dict[str, object]:
    train = world["train"]
    test = world["test"]
    query_types = world["query_types"]
    keys = world["keys"]
    angles = world["angles"]
    labels = world[label_key]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return objective_and_gradient(
            parameters,
            arm,
            query_types[train],
            keys[train],
            angles[train],
            labels[train],
        )

    best = None
    for start in restart_starts:
        optimization = minimize(
            objective,
            np.array(start, copy=True),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        test_logits = model_logits(
            optimization.x,
            arm,
            query_types[test],
            keys[test],
            angles[test],
        )
        resampled_logits = model_logits(
            optimization.x,
            arm,
            query_types[test],
            keys[test],
            world["resampled_angles"][test],
        )
        result = {
            "train_regularized_objective": float(optimization.fun),
            "test_nll": cross_entropy(test_logits, labels[test]),
            "resampled_position_test_nll": cross_entropy(
                resampled_logits, labels[test]
            ),
            "hard_retrieval_accuracy": float(np.mean(
                np.argmax(test_logits, axis=-1) == world[target_key][test]
            )),
            "resampled_position_hard_retrieval_accuracy": float(np.mean(
                np.argmax(resampled_logits, axis=-1) == world[target_key][test]
            )),
            "optimization_success": bool(optimization.success),
            "iterations": int(optimization.nit),
            "parameters": optimization.x.tolist(),
            "phi": float(optimization.x[8]),
            "log_scale": float(optimization.x[9]),
        }
        result["finite"] = bool(np.isfinite([
            result["train_regularized_objective"],
            result["test_nll"],
            result["resampled_position_test_nll"],
            result["hard_retrieval_accuracy"],
            result["resampled_position_hard_retrieval_accuracy"],
            result["phi"],
            result["log_scale"],
            *result["parameters"],
        ]).all())
        if (
            best is None
            or result["train_regularized_objective"]
            < best["train_regularized_objective"]
        ):
            best = result
    if best is None:
        raise AssertionError("optimizer produced no result")
    return best


def run_world(seed: int) -> dict[str, object]:
    world = make_world(seed)
    magnitude_starts = make_restart_starts(seed * 100)
    linear_starts = make_restart_starts(seed * 1_000)
    magnitude = {
        arm: fit_arm(
            world,
            arm,
            "magnitude_labels",
            "magnitude_targets",
            magnitude_starts,
        )
        for arm in ARMS
    }
    linear = {
        arm: fit_arm(
            world,
            arm,
            "linear_labels",
            "linear_targets",
            linear_starts,
        )
        for arm in NEGATIVE_CONTROL_ARMS
    }
    selected_fits = (*magnitude.values(), *linear.values())
    gates = {
        # Thresholds are frozen only after development and independent review.
        "all_selected_fits_successful_and_finite": all(
            fit["optimization_success"] and fit["finite"]
            for fit in selected_fits
        ),
    }
    return {
        "seed": seed,
        "raw_parameter_count_all_arms": 10,
        "key_pivot_input_is_identically_zero": True,
        "magnitude_target": magnitude,
        "linear_target_negative_control": linear,
        "gates": gates,
        "pass": all(gates.values()),
    }


def validate_integrity() -> dict[str, bool]:
    manifest = json.loads(INTEGRITY_MANIFEST.read_text())
    paths = {
        "source": Path(__file__),
        "preregistration": PREREGISTRATION,
        "test": TEST_SOURCE,
        "learning_source": LEARNING_SOURCE,
        "learning_preregistration": LEARNING_PREREGISTRATION,
        "learning_test": LEARNING_TEST,
        "learning_integrity": LEARNING_INTEGRITY,
        "learning_result": LEARNING_RESULT,
    }
    checks = {
        name: manifest.get(f"{name}_sha256") == sha256_file(path)
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid GECK natural-screen integrity: {checks}")
    learning = json.loads(LEARNING_RESULT.read_text())
    if not learning["decision"]["learning_gate_pass"]:
        raise ValueError("GECK learning gate did not pass")
    return checks


def run() -> dict[str, object]:
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    expected_runtime = {
        "python": "3.14.4",
        "numpy": "2.3.5",
        "scipy": "1.18.0",
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"frozen GECK natural runtime mismatch: {runtime}")
    integrity = validate_integrity()
    worlds = [run_world(seed) for seed in WORLD_SEEDS]
    decision = {
        "natural_screen_pass": all(world["pass"] for world in worlds),
        "worlds_passed": sum(world["pass"] for world in worlds),
        "worlds_total": len(worlds),
        "next_gate": "frozen only after preregistered thresholds",
        "h100_admitted": False,
    }
    return {
        "schema": "gauge-embedded-curved-keys-natural-v1",
        "candidate": "GECK-G2-full-gauge-bi-curved-keys",
        "integrity_checks": integrity,
        "runtime": runtime,
        "hashes": {
            "source": sha256_file(Path(__file__)),
            "preregistration": sha256_file(PREREGISTRATION),
            "test": sha256_file(TEST_SOURCE),
            "learning_result": sha256_file(LEARNING_RESULT),
        },
        "protocol": {
            "world_seeds": list(WORLD_SEEDS),
            "samples_per_world": SAMPLES,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": SAMPLES - TRAIN_SAMPLES,
            "items_per_bag": ITEMS,
            "rope_base": ROPE_BASE,
            "rope_head_dimension": ROPE_HEAD_DIMENSION,
            "rope_pair_index": ROPE_PAIR_INDEX,
            "rope_frequency": ROPE_FREQUENCY,
            "query_position": QUERY_POSITION,
            "target_temperature": TARGET_TEMPERATURE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "scipy L-BFGS-B with analytic gradient",
            "restarts": 3,
            "identical_restart_starts_within_each_target_world": True,
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
