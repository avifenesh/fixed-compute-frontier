#!/usr/bin/env python3
"""T19e: collision-free raw lexical key sufficiency oracle."""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import raw_countsketch_payload_t19b as t19b


PREREGISTRATION = ROOT / "results/exact-raw-key-oracle-t19e-preregistration.md"
OUTPUT = ROOT / "results/exact-raw-key-oracle-t19e.json"
DECISION = ROOT / "results/exact-raw-key-oracle-t19e-decision.md"
SCHEMA = "exact-raw-key-oracle-t19e-v1"
L2_COEFFICIENT = 0.01
MAX_ITERATIONS = 200
SHUFFLE_SEED = 19_005
T19B_ACCURACY = 0.6057692307692307
REQUIRED_ACCURACY = 0.75
REQUIRED_GAIN = 0.10


def sha256_file(path: Path) -> str:
    return t19b.sha256_file(path)


@dataclass(frozen=True)
class ExactCorpus:
    feature_to_index: dict[str, int]
    idf: dict[str, float]
    rows: tuple[dict[int, float], ...]

    @property
    def dimensions(self) -> int:
        return len(self.feature_to_index)


@dataclass(frozen=True)
class ReaderData:
    candidate: sparse.csr_matrix
    query_only: sparse.csr_matrix
    targets: np.ndarray
    title_pairs: tuple[tuple[int, int], ...]
    query_rows: tuple[dict[int, float], ...]


def normalize_sparse(values: dict[int, float]) -> dict[int, float]:
    norm = math.sqrt(sum(value * value for value in values.values()))
    if norm == 0.0 or not math.isfinite(norm):
        return {}
    return {index: value / norm for index, value in values.items()}


def build_exact_corpus(documents: list[dict[str, str]]) -> ExactCorpus:
    """Compile collision-free raw keys; this function may see documents only."""
    raw_counts = tuple(t19b.lexical_features(document["text"]) for document in documents)
    frequencies: collections.Counter[str] = collections.Counter()
    for counts in raw_counts:
        frequencies.update(counts.keys())
    features = sorted(frequencies)
    feature_to_index = {feature: index for index, feature in enumerate(features)}
    document_count = len(documents)
    idf = {
        feature: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for feature, frequency in frequencies.items()
    }
    rows: list[dict[int, float]] = []
    for counts in raw_counts:
        values = {
            feature_to_index[feature]: (1.0 + math.log(count)) * idf[feature]
            for feature, count in counts.items()
        }
        normalized = normalize_sparse(values)
        if not normalized:
            raise RuntimeError("raw document produced an empty exact vector")
        rows.append(normalized)
    return ExactCorpus(feature_to_index, idf, tuple(rows))


def encode_query(text: str, corpus: ExactCorpus) -> dict[int, float]:
    counts = t19b.lexical_features(text)
    values = {
        corpus.feature_to_index[feature]: (1.0 + math.log(count)) * corpus.idf[feature]
        for feature, count in counts.items()
        if feature in corpus.feature_to_index
    }
    normalized = normalize_sparse(values)
    if not normalized:
        raise RuntimeError(f"question has no exact raw key: {text!r}")
    return normalized


def sparse_dot(first: dict[int, float], second: dict[int, float]) -> float:
    if len(first) > len(second):
        first, second = second, first
    return sum(value * second.get(index, 0.0) for index, value in first.items())


def symmetric_exact_row(
    query: dict[int, float],
    first: dict[int, float],
    second: dict[int, float],
    dimensions: int,
) -> dict[int, float]:
    row: dict[int, float] = {}
    for index, query_value in query.items():
        first_match = query_value * first.get(index, 0.0)
        second_match = query_value * second.get(index, 0.0)
        minimum = min(first_match, second_match)
        maximum = max(first_match, second_match)
        if minimum:
            row[index] = minimum
        if maximum:
            row[dimensions + index] = maximum
    if len(first) > len(second):
        smaller, larger = second, first
    else:
        smaller, larger = first, second
    for index, first_value in smaller.items():
        product = first_value * larger.get(index, 0.0)
        if product:
            row[2 * dimensions + index] = product
    for index in first.keys() | second.keys():
        difference = abs(first.get(index, 0.0) - second.get(index, 0.0))
        if difference:
            row[3 * dimensions + index] = difference
    first_dot = sparse_dot(query, first)
    second_dot = sparse_dot(query, second)
    scalar_start = 4 * dimensions
    scalars = (
        first_dot,
        second_dot,
        sparse_dot(first, second),
        min(first_dot, second_dot),
        max(first_dot, second_dot),
    )
    for offset, value in enumerate(scalars):
        if value:
            row[scalar_start + offset] = value
    return row


def dictionaries_to_csr(
    rows: list[dict[int, float]], dimensions: int
) -> sparse.csr_matrix:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in rows:
        for index in sorted(row):
            value = row[index]
            if value:
                indices.append(index)
                data.append(value)
        indptr.append(len(data))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(rows), dimensions),
        dtype=np.float64,
    )


def build_reader_data(
    rows: list[dict[str, object]],
    titles: tuple[str, ...],
    corpus: ExactCorpus,
) -> ReaderData:
    candidate_rows: list[dict[int, float]] = []
    query_rows: list[dict[int, float]] = []
    targets: list[float] = []
    pairs: list[tuple[int, int]] = []
    for raw_row in rows:
        question = str(raw_row["question"])
        first_span, second_span, pair = t19b.nonoverlapping_title_pair(question, titles)
        property_text = t19b.remove_spans(question, (first_span, second_span))
        query = encode_query(property_text, corpus)
        candidate_rows.append(
            symmetric_exact_row(
                query,
                corpus.rows[pair[0]],
                corpus.rows[pair[1]],
                corpus.dimensions,
            )
        )
        query_rows.append(query)
        targets.append(float(str(raw_row["answer"]).lower() == "yes"))
        pairs.append(pair)
    return ReaderData(
        candidate=dictionaries_to_csr(candidate_rows, 4 * corpus.dimensions + 5),
        query_only=dictionaries_to_csr(query_rows, corpus.dimensions),
        targets=np.asarray(targets, dtype=np.float64),
        title_pairs=tuple(pairs),
        query_rows=tuple(query_rows),
    )


def build_shuffled_candidate(
    data: ReaderData, corpus: ExactCorpus
) -> tuple[sparse.csr_matrix, tuple[tuple[int, int], ...]]:
    generator = np.random.default_rng(SHUFFLE_SEED)
    flat = np.asarray([value for pair in data.title_pairs for value in pair], dtype=np.int64)
    shuffled = generator.permutation(flat)
    pairs = tuple(
        (int(shuffled[2 * index]), int(shuffled[2 * index + 1]))
        for index in range(len(data.title_pairs))
    )
    rows = [
        symmetric_exact_row(
            query,
            corpus.rows[pair[0]],
            corpus.rows[pair[1]],
            corpus.dimensions,
        )
        for query, pair in zip(data.query_rows, pairs, strict=True)
    ]
    return dictionaries_to_csr(rows, 4 * corpus.dimensions + 5), pairs


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


@dataclass(frozen=True)
class FittedReader:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    optimizer: dict[str, object]


def fit_reader(matrix: sparse.csr_matrix, targets: np.ndarray) -> FittedReader:
    count, dimensions = matrix.shape
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second_moment = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second_moment - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale <= 1e-12] = 1.0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = parameters[:-1]
        bias = parameters[-1]
        scaled_weights = weights / scale
        logits = np.asarray(matrix @ scaled_weights).ravel()
        logits += bias - float(mean @ scaled_weights)
        loss = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
        loss += 0.5 * L2_COEFFICIENT * float(weights @ weights)
        errors = (stable_sigmoid(logits) - targets) / count
        gradient_weights = np.asarray(matrix.T @ errors).ravel()
        gradient_weights -= mean * float(errors.sum())
        gradient_weights /= scale
        gradient_weights += L2_COEFFICIENT * weights
        gradient = np.concatenate((gradient_weights, np.asarray([errors.sum()])))
        return loss, gradient

    started = time.perf_counter()
    result = minimize(
        objective,
        np.zeros(dimensions + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": MAX_ITERATIONS,
            "ftol": 1e-12,
            "gtol": 1e-10,
            "maxls": 20,
        },
    )
    elapsed = time.perf_counter() - started
    if not np.isfinite(result.x).all() or not math.isfinite(float(result.fun)):
        raise RuntimeError("nonfinite exact reader optimization")
    return FittedReader(
        weights=result.x[:-1].copy(),
        bias=float(result.x[-1]),
        mean=mean,
        scale=scale,
        optimizer={
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "terminal_regularized_loss": float(result.fun),
            "seconds": elapsed,
        },
    )


def score_reader(
    reader: FittedReader, matrix: sparse.csr_matrix, targets: np.ndarray
) -> dict[str, object]:
    scaled_weights = reader.weights / reader.scale
    logits = np.asarray(matrix @ scaled_weights).ravel()
    logits += reader.bias - float(reader.mean @ scaled_weights)
    probabilities = stable_sigmoid(logits)
    predictions = probabilities >= 0.5
    labels = targets.astype(bool)
    return {
        "examples": len(targets),
        "accuracy": float(np.mean(predictions == labels)),
        "errors": int(np.sum(predictions != labels)),
        "positive_rate": float(np.mean(targets)),
        "predicted_positive_rate": float(np.mean(predictions)),
        "minimum_logit": float(np.min(logits)),
        "maximum_logit": float(np.max(logits)),
        "finite": bool(np.isfinite(logits).all() and np.isfinite(probabilities).all()),
    }


def verify_inputs() -> dict[str, bool]:
    checks = {
        "candidate_corpus": sha256_file(t19b.t19a.CANDIDATE_CORPUS)
        == t19b.t19a.CANDIDATE_SHA256,
        "train_qa": sha256_file(t19b.TRAIN_QA) == t19b.TRAIN_QA_SHA256,
        "evaluation_qa": sha256_file(t19b.EVALUATION_QA)
        == t19b.t19a.EVALUATOR_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def run() -> dict[str, object]:
    input_checks = verify_inputs()
    documents = t19b.t19a.load_raw_documents(t19b.t19a.CANDIDATE_CORPUS)
    titles = tuple(document["title"] for document in documents)
    corpus = build_exact_corpus(documents)
    train_rows = t19b.load_qa(t19b.TRAIN_QA)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    train = build_reader_data(train_rows, titles, corpus)
    evaluation = build_reader_data(evaluation_rows, titles, corpus)
    shuffled_matrix, shuffled_pairs = build_shuffled_candidate(evaluation, corpus)

    candidate_reader = fit_reader(train.candidate, train.targets)
    query_reader = fit_reader(train.query_only, train.targets)
    candidate = {
        "optimizer": candidate_reader.optimizer,
        "weight_norm": float(np.linalg.norm(candidate_reader.weights)),
        "train": score_reader(candidate_reader, train.candidate, train.targets),
        "evaluation": score_reader(
            candidate_reader, evaluation.candidate, evaluation.targets
        ),
        "shuffled_evaluation": score_reader(
            candidate_reader, shuffled_matrix, evaluation.targets
        ),
    }
    query_only = {
        "optimizer": query_reader.optimizer,
        "weight_norm": float(np.linalg.norm(query_reader.weights)),
        "train": score_reader(query_reader, train.query_only, train.targets),
        "evaluation": score_reader(
            query_reader, evaluation.query_only, evaluation.targets
        ),
    }

    candidate_accuracy = float(candidate["evaluation"]["accuracy"])
    query_accuracy = float(query_only["evaluation"]["accuracy"])
    shuffled_accuracy = float(candidate["shuffled_evaluation"]["accuracy"])
    compiler_fields = set().union(*(document.keys() for document in documents))
    train_classes = set(train.targets.tolist())
    evaluation_classes = set(evaluation.targets.tolist())
    gates = {
        "input_integrity": all(input_checks.values()),
        "raw_only_compiler_fields": compiler_fields
        == {"document_id", "title", "text"},
        "collision_free_coordinates": corpus.dimensions
        == len(corpus.feature_to_index)
        == len(set(corpus.feature_to_index.values())),
        "two_titles_every_question": len(train.title_pairs) == len(train_rows)
        and len(evaluation.title_pairs) == len(evaluation_rows),
        "both_classes": train_classes == {0.0, 1.0}
        and evaluation_classes == {0.0, 1.0},
        "finite_optimization": bool(
            candidate["train"]["finite"]
            and candidate["evaluation"]["finite"]
            and candidate["shuffled_evaluation"]["finite"]
            and query_only["train"]["finite"]
            and query_only["evaluation"]["finite"]
        ),
        "training_accuracy_at_least_95pct": float(
            candidate["train"]["accuracy"]
        )
        >= 0.95,
        "evaluation_accuracy_at_least_75pct": candidate_accuracy
        >= REQUIRED_ACCURACY,
        "gain_over_t19b_at_least_10pp": candidate_accuracy - T19B_ACCURACY
        >= REQUIRED_GAIN,
        "gain_over_query_only_at_least_10pp": candidate_accuracy - query_accuracy
        >= REQUIRED_GAIN,
        "document_shuffle_drop_at_least_10pp": candidate_accuracy
        - shuffled_accuracy
        >= REQUIRED_GAIN,
    }
    return {
        "schema": SCHEMA,
        "input_checks": input_checks,
        "compiler": {
            "document_fields": sorted(compiler_fields),
            "documents": len(documents),
            "exact_dimensions": corpus.dimensions,
            "coordinate_range_is_dense": sorted(corpus.feature_to_index.values())
            == list(range(corpus.dimensions)),
            "function_parameters": list(
                inspect.signature(build_exact_corpus).parameters
            ),
        },
        "reader": {
            "candidate_dimensions": int(train.candidate.shape[1]),
            "query_only_dimensions": int(train.query_only.shape[1]),
            "train_candidate_nonzeros": int(train.candidate.nnz),
            "evaluation_candidate_nonzeros": int(evaluation.candidate.nnz),
            "l2_coefficient": L2_COEFFICIENT,
            "max_iterations": MAX_ITERATIONS,
            "threshold": 0.5,
            "used_qa_fields": ["question", "answer"],
        },
        "candidate": candidate,
        "query_only": query_only,
        "shuffle": {
            "seed": SHUFFLE_SEED,
            "pairs": len(shuffled_pairs),
            "unchanged_pairs": sum(
                observed == shuffled
                for observed, shuffled in zip(
                    evaluation.title_pairs, shuffled_pairs, strict=True
                )
            ),
        },
        "effects": {
            "gain_over_t19b_pp": 100.0
            * (candidate_accuracy - T19B_ACCURACY),
            "gain_over_query_only_pp": 100.0
            * (candidate_accuracy - query_accuracy),
            "document_shuffle_drop_pp": 100.0
            * (candidate_accuracy - shuffled_accuracy),
        },
        "gates": gates,
        "admitted": all(gates.values()),
        "integrity": {
            "candidate_corpus_sha256": sha256_file(t19b.t19a.CANDIDATE_CORPUS),
            "train_qa_sha256": sha256_file(t19b.TRAIN_QA),
            "evaluation_qa_sha256": sha256_file(t19b.EVALUATION_QA),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
        },
        "limits": [
            "This is an uncompressed collision-free sufficiency oracle.",
            "It does not fit the write budget or execute in the served model.",
            "It cannot establish a smarter production model.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run()
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
