#!/usr/bin/env python3
"""T24 decoded-record natural-QA information-sufficiency oracle."""

from __future__ import annotations

import argparse
import collections
import inspect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import exact_raw_key_oracle_t19e as t19e
from experiments import xor_functional_record_t24_stage0 as t24


t19b = t19e.t19b

PREREGISTRATION = ROOT / "results/xor-functional-record-t24-sufficiency-preregistration.md"
PREREGISTRATION_SHA256 = (
    "71e5709e7d8d2da1f1fad7207dbca69e7dde5988b3fffec6390c4c860783b101"
)
T24_RESULT = ROOT / "results/xor-functional-record-t24-stage0.json"
T24_RESULT_SHA256 = (
    "e31fb880785c49c1a4f972cc5d8b1c8f10f8aba5a990c707cf975def35b6c99e"
)
OUTPUT = ROOT / "results/xor-functional-record-t24-sufficiency.json"
SCHEMA = "xor-functional-record-t24-sufficiency-v1"

T19E_ACCURACY = 59 / 104
T19B_ACCURACY = 63 / 104
REQUIRED_ACCURACY = 0.75
REQUIRED_GAIN = 0.10


def sha256_file(path: Path) -> str:
    return t24.sha256_file(path)


def typed_feature_counts(
    collisions: dict[tuple[tuple[int, str], ...], set[tuple[int, str]]],
    documents: int,
) -> tuple[collections.Counter[str], ...]:
    rows = [collections.Counter() for _ in range(documents)]
    for skeleton, values in collisions.items():
        for document, target in values:
            for _, word in skeleton:
                rows[document][f"anchor:{word}"] += 1
            rows[document][f"target:{target}"] += 1
    return tuple(rows)


def build_functional_corpus() -> tuple[t19e.ExactCorpus, dict[str, object]]:
    documents, _, collisions = t24.load_quotient()
    raw_counts = typed_feature_counts(collisions, len(documents))
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
    empty_documents = 0
    for counts in raw_counts:
        values = {
            feature_to_index[feature]: (1.0 + math.log(count)) * idf[feature]
            for feature, count in counts.items()
        }
        normalized = t19e.normalize_sparse(values)
        empty_documents += not bool(normalized)
        rows.append(normalized)
    return (
        t19e.ExactCorpus(feature_to_index, idf, tuple(rows)),
        {
            "documents": len(documents),
            "skeletons": len(collisions),
            "edges": sum(len(values) for values in collisions.values()),
            "features": len(features),
            "anchor_features": sum(feature.startswith("anchor:") for feature in features),
            "target_features": sum(feature.startswith("target:") for feature in features),
            "empty_documents": empty_documents,
            "compiler_fields": sorted(set().union(*(document.keys() for document in documents))),
        },
    )


def encode_query(text: str, corpus: t19e.ExactCorpus) -> dict[int, float]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    counts: collections.Counter[str] = collections.Counter()
    for word in words:
        counts[f"anchor:{word}"] += 1
        counts[f"target:{word}"] += 1
    values = {
        corpus.feature_to_index[feature]: (1.0 + math.log(count)) * corpus.idf[feature]
        for feature, count in counts.items()
        if feature in corpus.feature_to_index
    }
    return t19e.normalize_sparse(values)


def build_reader_data(
    rows: list[dict[str, object]],
    titles: tuple[str, ...],
    corpus: t19e.ExactCorpus,
) -> tuple[t19e.ReaderData, int]:
    candidate_rows: list[dict[int, float]] = []
    query_rows: list[dict[int, float]] = []
    targets: list[float] = []
    pairs: list[tuple[int, int]] = []
    empty_queries = 0
    for raw_row in rows:
        question = str(raw_row["question"])
        first_span, second_span, pair = t19b.nonoverlapping_title_pair(question, titles)
        property_text = t19b.remove_spans(question, (first_span, second_span))
        query = encode_query(property_text, corpus)
        empty_queries += not bool(query)
        candidate_rows.append(
            t19e.symmetric_exact_row(
                query,
                corpus.rows[pair[0]],
                corpus.rows[pair[1]],
                corpus.dimensions,
            )
        )
        query_rows.append(query)
        targets.append(float(str(raw_row["answer"]).lower() == "yes"))
        pairs.append(pair)
    return (
        t19e.ReaderData(
            candidate=t19e.dictionaries_to_csr(
                candidate_rows, 4 * corpus.dimensions + 5
            ),
            query_only=t19e.dictionaries_to_csr(query_rows, corpus.dimensions),
            targets=np.asarray(targets, dtype=np.float64),
            title_pairs=tuple(pairs),
            query_rows=tuple(query_rows),
        ),
        empty_queries,
    )


def verify_inputs() -> dict[str, bool]:
    checks = {
        "preregistration": sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256,
        "t24_stage0_result": sha256_file(T24_RESULT) == T24_RESULT_SHA256,
        "candidate_corpus": sha256_file(t19b.t19a.CANDIDATE_CORPUS)
        == t19b.t19a.CANDIDATE_SHA256,
        "train_qa": sha256_file(t19b.TRAIN_QA) == t19b.TRAIN_QA_SHA256,
        "evaluation_qa": sha256_file(t19b.EVALUATION_QA)
        == t19b.t19a.EVALUATOR_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def run() -> dict[str, object]:
    input_checks = verify_inputs()
    documents = t19b.t19a.load_raw_documents(t19b.t19a.CANDIDATE_CORPUS)
    titles = tuple(document["title"] for document in documents)
    corpus, compiler = build_functional_corpus()
    train_rows = t19b.load_qa(t19b.TRAIN_QA)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    train, train_empty = build_reader_data(train_rows, titles, corpus)
    evaluation, evaluation_empty = build_reader_data(evaluation_rows, titles, corpus)
    shuffled_matrix, shuffled_pairs = t19e.build_shuffled_candidate(evaluation, corpus)

    candidate_reader = t19e.fit_reader(train.candidate, train.targets)
    query_reader = t19e.fit_reader(train.query_only, train.targets)
    candidate = {
        "optimizer": candidate_reader.optimizer,
        "weight_norm": float(np.linalg.norm(candidate_reader.weights)),
        "train": t19e.score_reader(candidate_reader, train.candidate, train.targets),
        "evaluation": t19e.score_reader(
            candidate_reader, evaluation.candidate, evaluation.targets
        ),
        "shuffled_evaluation": t19e.score_reader(
            candidate_reader, shuffled_matrix, evaluation.targets
        ),
    }
    query_only = {
        "optimizer": query_reader.optimizer,
        "weight_norm": float(np.linalg.norm(query_reader.weights)),
        "train": t19e.score_reader(query_reader, train.query_only, train.targets),
        "evaluation": t19e.score_reader(
            query_reader, evaluation.query_only, evaluation.targets
        ),
    }

    accuracy = float(candidate["evaluation"]["accuracy"])
    query_accuracy = float(query_only["evaluation"]["accuracy"])
    shuffled_accuracy = float(candidate["shuffled_evaluation"]["accuracy"])
    gates = {
        "input_integrity": all(input_checks.values()),
        "raw_quotient_exact": compiler["skeletons"] == 3_392
        and compiler["edges"] == 14_146,
        "raw_only_compiler_fields": compiler["compiler_fields"]
        == ["document_id", "text", "title"],
        "two_titles_every_question": len(train.title_pairs) == len(train_rows)
        and len(evaluation.title_pairs) == len(evaluation_rows),
        "both_classes": set(train.targets.tolist()) == {0.0, 1.0}
        and set(evaluation.targets.tolist()) == {0.0, 1.0},
        "finite_optimization": bool(
            candidate["train"]["finite"]
            and candidate["evaluation"]["finite"]
            and candidate["shuffled_evaluation"]["finite"]
            and query_only["train"]["finite"]
            and query_only["evaluation"]["finite"]
        ),
        "training_accuracy_at_least_95pct": float(candidate["train"]["accuracy"])
        >= 0.95,
        "evaluation_accuracy_at_least_75pct": accuracy >= REQUIRED_ACCURACY,
        "gain_over_t19e_at_least_10pp": accuracy - T19E_ACCURACY >= REQUIRED_GAIN,
        "gain_over_t19b_at_least_10pp": accuracy - T19B_ACCURACY >= REQUIRED_GAIN,
        "gain_over_query_only_at_least_10pp": accuracy - query_accuracy
        >= REQUIRED_GAIN,
        "document_shuffle_drop_at_least_10pp": accuracy - shuffled_accuracy
        >= REQUIRED_GAIN,
    }
    admitted = all(gates.values())
    return {
        "schema": SCHEMA,
        "input_checks": input_checks,
        "compiler": compiler,
        "reader": {
            "candidate_dimensions": int(train.candidate.shape[1]),
            "query_only_dimensions": int(train.query_only.shape[1]),
            "train_nonzeros": int(train.candidate.nnz),
            "evaluation_nonzeros": int(evaluation.candidate.nnz),
            "train_empty_queries": train_empty,
            "evaluation_empty_queries": evaluation_empty,
            "l2_coefficient": t19e.L2_COEFFICIENT,
            "max_iterations": t19e.MAX_ITERATIONS,
            "threshold": 0.5,
            "function_parameters": list(inspect.signature(build_functional_corpus).parameters),
        },
        "candidate": candidate,
        "query_only": query_only,
        "shuffle": {
            "seed": t19e.SHUFFLE_SEED,
            "pairs": len(shuffled_pairs),
            "unchanged_pairs": sum(
                observed == shuffled
                for observed, shuffled in zip(
                    evaluation.title_pairs, shuffled_pairs, strict=True
                )
            ),
        },
        "effects": {
            "gain_over_t19e_pp": 100.0 * (accuracy - T19E_ACCURACY),
            "gain_over_t19b_pp": 100.0 * (accuracy - T19B_ACCURACY),
            "gain_over_query_only_pp": 100.0 * (accuracy - query_accuracy),
            "document_shuffle_drop_pp": 100.0 * (accuracy - shuffled_accuracy),
        },
        "gates": gates,
        "admitted": admitted,
        "decision": (
            "admit_one_from_zero_learned_interface_experiment"
            if admitted
            else "close_anchor_window_quotient_for_natural_comparison_qa"
        ),
        "claim_boundary": (
            "Uncompressed decoded-record information oracle only; no serving, "
            "training-efficiency, or smarter-model claim."
        ),
        "provenance": {
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "t24_stage0_result_sha256": sha256_file(T24_RESULT),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
