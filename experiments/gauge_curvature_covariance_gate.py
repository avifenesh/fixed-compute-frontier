"""Oracle-routing CPU gate for gauge-funded nonlinear value writers.

Every example is a bag of paired ``z, -z`` tokens.  Its attended linear mean is
exactly zero; only second-order token statistics identify the class.  The gate
compares the off-diagonal GFQV statistic with an exact-ledger square writer, a
generic low-rank bilinear writer, and linear/full-covariance controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT_DIR / "results" / "gauge-curvature-covariance-gate.json"

Array = np.ndarray


def orthogonal(rng: np.random.Generator, width: int) -> Array:
    matrix, upper = np.linalg.qr(rng.normal(size=(width, width)))
    signs = np.sign(np.diag(upper))
    signs[signs == 0.0] = 1.0
    return matrix * signs


def balanced_labels(rng: np.random.Generator, samples: int) -> Array:
    if samples % 2:
        raise ValueError("sample count must be even")
    labels = np.concatenate((np.zeros(samples // 2), np.ones(samples // 2)))
    rng.shuffle(labels)
    return labels


def covariance_examples(
    rng: np.random.Generator,
    rotation: Array,
    samples: int,
    *,
    independent_draws: int = 8,
    delta: float = 0.15,
    mean_shift: bool = False,
) -> tuple[Array, Array, Array]:
    """Return exact means, empirical second moments, and balanced labels."""

    width = rotation.shape[0]
    labels = balanced_labels(rng, samples)
    signs = 2.0 * labels - 1.0
    spectrum_sign = np.concatenate((np.ones(width // 2), -np.ones(width // 2)))
    if mean_shift:
        eigenvalues = np.ones((samples, width))
    else:
        eigenvalues = 1.0 + delta * signs[:, None] * spectrum_sign[None, :]
    standard = rng.normal(size=(samples, independent_draws, width))
    eigenbasis_samples = standard * np.sqrt(eigenvalues)[:, None, :]
    draws = eigenbasis_samples @ rotation.T
    covariance = np.einsum("nki,nkj->nij", draws, draws, optimize=True)
    covariance /= independent_draws
    if mean_shift:
        direction = rotation[:, 0]
        mean = 0.35 * signs[:, None] * direction[None, :]
        covariance += np.einsum("ni,nj->nij", mean, mean, optimize=True)
    else:
        mean = np.zeros((samples, width))
    return mean, covariance, labels


def binary_nll(logits: Array, labels: Array) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def binary_accuracy(logits: Array, labels: Array) -> float:
    return float(np.mean((logits >= 0.0) == (labels >= 0.5)))


def fit_logistic(
    train_features: Array,
    train_labels: Array,
    validation_features: Array,
    validation_labels: Array,
    test_features: Array,
    test_labels: Array,
) -> dict[str, object]:
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale < 1e-10] = 1.0
    standardized_train = (train_features - feature_mean) / feature_scale
    standardized_validation = (validation_features - feature_mean) / feature_scale
    standardized_test = (test_features - feature_mean) / feature_scale
    classifier = None
    best_validation_nll = math.inf
    best_regularization = None
    for inverse_regularization in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        current = LogisticRegression(
            C=inverse_regularization,
            fit_intercept=True,
            max_iter=5_000,
            solver="lbfgs",
            tol=1e-10,
        )
        current.fit(standardized_train, train_labels)
        validation_logits = current.decision_function(standardized_validation)
        validation_nll = binary_nll(validation_logits, validation_labels)
        if validation_nll < best_validation_nll:
            classifier = current
            best_validation_nll = validation_nll
            best_regularization = inverse_regularization
    if classifier is None:
        raise AssertionError("regularization selection produced no classifier")
    logits = classifier.decision_function(standardized_test)
    raw_weight = classifier.coef_[0] / feature_scale
    raw_bias = float(classifier.intercept_[0] - raw_weight @ feature_mean)
    return {
        "accuracy": binary_accuracy(logits, test_labels),
        "nll": binary_nll(logits, test_labels),
        "raw_weight": raw_weight,
        "raw_bias": raw_bias,
        "logits": logits,
        "validation_nll": best_validation_nll,
        "inverse_regularization": best_regularization,
    }


def covariance_features(covariance: Array, *, diagonal: bool) -> Array:
    offset = 0 if diagonal else 1
    rows, columns = np.triu_indices(covariance.shape[-1], k=offset)
    return covariance[:, rows, columns]


def sigmoid(logits: Array) -> Array:
    positive = logits >= 0.0
    result = np.empty_like(logits)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


class Adam:
    def __init__(self, params: dict[str, Array], learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.step = 0
        self.first = {name: np.zeros_like(value) for name, value in params.items()}
        self.second = {name: np.zeros_like(value) for name, value in params.items()}

    def update(
        self,
        params: dict[str, Array],
        gradients: dict[str, Array],
        masks: dict[str, Array] | None = None,
    ) -> None:
        self.step += 1
        norm = math.sqrt(
            max(sum(float(np.sum(gradient * gradient)) for gradient in gradients.values()), 1e-30)
        )
        scale = min(1.0, 5.0 / norm)
        for name, value in params.items():
            gradient = gradients[name] * scale
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradient
            self.second[name] = 0.999 * self.second[name] + 0.001 * gradient * gradient
            first = self.first[name] / (1.0 - 0.9**self.step)
            second = self.second[name] / (1.0 - 0.999**self.step)
            value -= self.learning_rate * first / (np.sqrt(second) + 1e-8)
            if masks and name in masks:
                value *= masks[name]


def square_logits(mean: Array, covariance: Array, params: dict[str, Array]) -> Array:
    projection = params["projection"]
    square_features = np.einsum(
        "nij,ia,ja->na", covariance, projection, projection, optimize=True
    )
    return mean @ params["linear"] + square_features @ params["readout"] + params["bias"][0]


def square_gradients(
    mean: Array, covariance: Array, labels: Array, params: dict[str, Array]
) -> dict[str, Array]:
    projection = params["projection"]
    square_features = np.einsum(
        "nij,ia,ja->na", covariance, projection, projection, optimize=True
    )
    logits = mean @ params["linear"] + square_features @ params["readout"] + params["bias"][0]
    derivative = (sigmoid(logits) - labels) / labels.size
    weighted_covariance = np.einsum("n,nij->ij", derivative, covariance, optimize=True)
    return {
        "projection": 2.0 * (weighted_covariance @ projection) * params["readout"][None, :],
        "readout": square_features.T @ derivative,
        "linear": mean.T @ derivative,
        "bias": np.array([derivative.sum()]),
    }


def bilinear_logits(mean: Array, covariance: Array, params: dict[str, Array]) -> Array:
    features = np.einsum(
        "ia,nij,ja->na", params["left"], covariance, params["right"], optimize=True
    )
    return mean @ params["linear"] + features @ params["readout"] + params["bias"][0]


def bilinear_gradients(
    mean: Array, covariance: Array, labels: Array, params: dict[str, Array]
) -> dict[str, Array]:
    left = params["left"]
    right = params["right"]
    readout = params["readout"]
    features = np.einsum("ia,nij,ja->na", left, covariance, right, optimize=True)
    logits = mean @ params["linear"] + features @ readout + params["bias"][0]
    derivative = (sigmoid(logits) - labels) / labels.size
    weighted = derivative[:, None] * readout[None, :]
    return {
        "left": np.einsum("na,nij,ja->ia", weighted, covariance, right, optimize=True),
        "right": np.einsum("na,nij,ia->ja", weighted, covariance, left, optimize=True),
        "readout": features.T @ derivative,
        "linear": mean.T @ derivative,
        "bias": np.array([derivative.sum()]),
    }


def train_nonlinear_writer(
    kind: str,
    train: tuple[Array, Array, Array],
    validation: tuple[Array, Array, Array],
    test: tuple[Array, Array, Array],
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    initialization_scale: float,
) -> dict[str, object]:
    train_mean, train_covariance, train_labels = train
    validation_mean, validation_covariance, validation_labels = validation
    test_mean, test_covariance, test_labels = test
    width = train_mean.shape[-1]
    rng = np.random.default_rng(seed)
    linear_baseline = fit_logistic(
        train_mean,
        train_labels,
        validation_mean,
        validation_labels,
        validation_mean,
        validation_labels,
    )
    initial_linear = np.asarray(linear_baseline["raw_weight"]).copy()
    initial_bias = np.array([float(linear_baseline["raw_bias"])])
    if kind == "square_writer":
        mask = np.ones((width, width)) - np.eye(width)
        params = {
            "projection": rng.normal(
                scale=initialization_scale / math.sqrt(width), size=(width, width)
            )
            * mask,
            "readout": np.zeros(width),
            "linear": initial_linear,
            "bias": initial_bias.copy(),
        }
        forward = square_logits
        gradient = square_gradients
        masks = {"projection": mask}
    elif kind == "bilinear_writer":
        rank = 5
        params = {
            "left": rng.normal(scale=1.0 / math.sqrt(width), size=(width, rank)),
            "right": rng.normal(scale=1.0 / math.sqrt(width), size=(width, rank)),
            "readout": np.zeros(rank),
            "linear": initial_linear,
            "bias": initial_bias.copy(),
        }
        forward = bilinear_logits
        gradient = bilinear_gradients
        masks = None
    else:
        raise ValueError(f"unknown writer: {kind}")

    optimizer = Adam(params, learning_rate)
    best_params = {name: value.copy() for name, value in params.items()}
    best_validation = binary_nll(
        forward(validation_mean, validation_covariance, params), validation_labels
    )
    for step in range(1, steps + 1):
        indices = rng.integers(0, train_labels.size, size=batch_size)
        gradients = gradient(
            train_mean[indices], train_covariance[indices], train_labels[indices], params
        )
        optimizer.update(params, gradients, masks)
        if step % 25 == 0 or step == steps:
            validation_logits = forward(validation_mean, validation_covariance, params)
            validation_nll = binary_nll(validation_logits, validation_labels)
            if validation_nll < best_validation:
                best_validation = validation_nll
                best_params = {name: value.copy() for name, value in params.items()}
    test_logits = forward(test_mean, test_covariance, best_params)
    result: dict[str, object] = {
        "accuracy": binary_accuracy(test_logits, test_labels),
        "nll": binary_nll(test_logits, test_labels),
        "validation_nll": best_validation,
        "initialization_scale": initialization_scale,
        "seed": seed,
    }
    if kind == "square_writer":
        projection = best_params["projection"]
        quadratic_mean = np.einsum(
            "nij,ia,ja->na", test_covariance, projection, projection, optimize=True
        )
        centered = quadratic_mean - quadratic_mean.mean(axis=0, keepdims=True)
        variance = float(np.mean(centered * centered))
        fourth = float(np.mean(centered**4))
        result["cached_square_feature_rms"] = float(np.sqrt(np.mean(quadratic_mean**2)))
        result["cached_square_feature_kurtosis"] = fourth / max(variance * variance, 1e-15)
    return result


def evaluate_world(
    world: int,
    *,
    axis_aligned: bool,
    mean_shift: bool,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    width = 16
    world_rng = np.random.default_rng(50_000 + world)
    rotation = np.eye(width) if axis_aligned else orthogonal(world_rng, width)
    train = covariance_examples(
        np.random.default_rng(60_000 + world), rotation, train_samples, mean_shift=mean_shift
    )
    validation = covariance_examples(
        np.random.default_rng(70_000 + world),
        rotation,
        validation_samples,
        mean_shift=mean_shift,
    )
    test = covariance_examples(
        np.random.default_rng(80_000 + world), rotation, test_samples, mean_shift=mean_shift
    )
    train_mean, train_covariance, train_labels = train
    validation_mean, validation_covariance, validation_labels = validation
    _, test_covariance, test_labels = test
    off_diagonal_train = np.concatenate(
        (train_mean, covariance_features(train_covariance, diagonal=False)), axis=1
    )
    off_diagonal_validation = np.concatenate(
        (
            validation_mean,
            covariance_features(validation_covariance, diagonal=False),
        ),
        axis=1,
    )
    off_diagonal_test = np.concatenate(
        (test[0], covariance_features(test_covariance, diagonal=False)), axis=1
    )
    diagonal_train = np.concatenate(
        (train_mean, np.diagonal(train_covariance, axis1=1, axis2=2)), axis=1
    )
    diagonal_validation = np.concatenate(
        (
            validation_mean,
            np.diagonal(validation_covariance, axis1=1, axis2=2),
        ),
        axis=1,
    )
    diagonal_test = np.concatenate(
        (test[0], np.diagonal(test_covariance, axis1=1, axis2=2)), axis=1
    )
    full_train = np.concatenate(
        (train_mean, covariance_features(train_covariance, diagonal=True)), axis=1
    )
    full_validation = np.concatenate(
        (validation_mean, covariance_features(validation_covariance, diagonal=True)),
        axis=1,
    )
    full_test = np.concatenate(
        (test[0], covariance_features(test_covariance, diagonal=True)), axis=1
    )
    post_train = np.concatenate((train_mean, train_mean * train_mean), axis=1)
    post_validation = np.concatenate(
        (validation_mean, validation_mean * validation_mean), axis=1
    )
    post_test = np.concatenate((test[0], test[0] * test[0]), axis=1)

    linear = fit_logistic(
        train_mean,
        train_labels,
        validation_mean,
        validation_labels,
        test[0],
        test_labels,
    )
    post = fit_logistic(
        post_train,
        train_labels,
        post_validation,
        validation_labels,
        post_test,
        test_labels,
    )
    off_diagonal = fit_logistic(
        off_diagonal_train,
        train_labels,
        off_diagonal_validation,
        validation_labels,
        off_diagonal_test,
        test_labels,
    )
    diagonal = fit_logistic(
        diagonal_train,
        train_labels,
        diagonal_validation,
        validation_labels,
        diagonal_test,
        test_labels,
    )
    oracle = fit_logistic(
        full_train,
        train_labels,
        full_validation,
        validation_labels,
        full_test,
        test_labels,
    )
    models: dict[str, object] = {
        "linear_mean": {"accuracy": linear["accuracy"], "nll": linear["nll"]},
        "post_mean_degree2": {"accuracy": post["accuracy"], "nll": post["nll"]},
        "gfqv_offdiag": {
            "accuracy": off_diagonal["accuracy"],
            "nll": off_diagonal["nll"],
        },
        "diagonal_moments": {
            "accuracy": diagonal["accuracy"],
            "nll": diagonal["nll"],
        },
        "full_degree2_oracle": {"accuracy": oracle["accuracy"], "nll": oracle["nll"]},
    }

    writer_runs: dict[str, list[dict[str, object]]] = {}
    for kind in ("square_writer", "bilinear_writer"):
        runs = []
        scales = (0.1, 0.3, 1.0) if kind == "square_writer" else (1.0, 1.0, 1.0)
        for initialization, scale in enumerate(scales):
            runs.append(
                train_nonlinear_writer(
                    kind,
                    train,
                    validation,
                    test,
                    seed=90_000 + 100 * world + initialization,
                    steps=steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    initialization_scale=scale,
                )
            )
        writer_runs[kind] = runs
        best = min(runs, key=lambda run: float(run["validation_nll"]))
        models[kind] = best

    return {
        "world": world,
        "axis_aligned": axis_aligned,
        "mean_shift": mean_shift,
        "maximum_absolute_linear_mean": float(np.max(np.abs(train_mean))),
        "models": models,
        "writer_initializations": writer_runs,
    }


def confidence_interval(values: list[float]) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half_width = 2.776 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def summarize(rotated_worlds: list[dict[str, object]]) -> dict[str, object]:
    model_names = sorted(rotated_worlds[0]["models"])
    medians = {
        model: {
            "accuracy": float(
                np.median([world["models"][model]["accuracy"] for world in rotated_worlds])
            ),
            "nll": float(
                np.median([world["models"][model]["nll"] for world in rotated_worlds])
            ),
        }
        for model in model_names
    }
    relative_excess_improvements = []
    wins = 0
    strongest_names = []
    for world in rotated_worlds:
        models = world["models"]
        oracle_nll = float(models["full_degree2_oracle"]["nll"])
        strongest_name = min(
            ("square_writer", "bilinear_writer"),
            key=lambda name: float(models[name]["nll"]),
        )
        strongest_names.append(strongest_name)
        strongest_nll = float(models[strongest_name]["nll"])
        candidate_nll = float(models["gfqv_offdiag"]["nll"])
        denominator = max(strongest_nll - oracle_nll, 1e-12)
        relative_excess_improvements.append(
            (strongest_nll - candidate_nll) / denominator
        )
        wins += int(candidate_nll < strongest_nll)
    improvement_mean, improvement_low, improvement_high = confidence_interval(
        relative_excess_improvements
    )
    oracle_valid = min(
        float(world["models"]["full_degree2_oracle"]["accuracy"])
        for world in rotated_worlds
    ) >= 0.82
    no_leakage = max(
        max(
            float(world["models"][baseline]["accuracy"])
            for baseline in ("linear_mean", "post_mean_degree2")
        )
        for world in rotated_worlds
    ) <= 0.52
    gfqv_preferred = (
        oracle_valid
        and no_leakage
        and wins >= 4
        and improvement_mean >= 0.05
        and improvement_low > 0.0
    )
    return {
        "median_rotated_metrics": medians,
        "harness_valid": oracle_valid and no_leakage,
        "checks": {
            "full_covariance_ceiling_at_least_82pct_each_world": oracle_valid,
            "linear_and_post_mean_accuracy_at_most_52pct_each_world": no_leakage,
        },
        "gfqv_vs_strongest_equal_budget": {
            "strongest_writer_by_world": strongest_names,
            "gfqv_win_count": wins,
            "relative_excess_nll_improvement_by_world": relative_excess_improvements,
            "mean": improvement_mean,
            "paired_95pct_ci": [improvement_low, improvement_high],
            "passes_preference_rule": gfqv_preferred,
        },
        "decision": (
            "retain_gfqv_as_preferred_equal_ledger_writer"
            if gfqv_preferred
            else "reject_gfqv_preference_retain_gauge_to_curvature_family"
        ),
    }


def run_gate(
    *,
    worlds: tuple[int, ...] = (0, 1, 2, 3, 4),
    train_samples: int = 2_048,
    validation_samples: int = 512,
    test_samples: int = 4_096,
    steps: int = 800,
    batch_size: int = 256,
    learning_rate: float = 0.03,
    include_controls: bool = True,
) -> dict[str, object]:
    rotated = [
        evaluate_world(
            world,
            axis_aligned=False,
            mean_shift=False,
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        for world in worlds
    ]
    controls: dict[str, object] = {}
    if include_controls:
        controls["axis_aligned"] = evaluate_world(
            100,
            axis_aligned=True,
            mean_shift=False,
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        controls["mean_shift"] = evaluate_world(
            101,
            axis_aligned=False,
            mean_shift=True,
            train_samples=train_samples,
            validation_samples=validation_samples,
            test_samples=test_samples,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    source_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "experiment": "gauge_curvature_covariance_gate",
        "configuration": {
            "worlds": list(worlds),
            "width": 16,
            "independent_draws_per_bag": 8,
            "emitted_tokens_per_bag": 16,
            "covariance_delta": 0.15,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "rotated_worlds": rotated,
        "controls": controls,
        "summary": summarize(rotated),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "gpu_required_now": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    arguments = parser.parse_args()
    if arguments.quick:
        report = run_gate(
            worlds=(0,),
            train_samples=512,
            validation_samples=256,
            test_samples=1_024,
            steps=100,
            include_controls=False,
        )
    else:
        report = run_gate()
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
