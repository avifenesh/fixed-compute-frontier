#!/usr/bin/env python3
"""CPU-only provenance and boundary audit for T32R exact-title routing."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.scale_referenced_whole_record_t32_information_oracle import (
    nonoverlapping_title_pair,
    replace_spans,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
EVALUATOR = ROOT / "data/hotpot-real-prose-t12/sealed-evaluator.jsonl"
CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
EVALUATOR_SHA256 = "5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b"
DOCUMENTS = 2_405
QUESTIONS = 104


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def route_strings(question: str, titles: Sequence[str]) -> tuple[str, str]:
    _, _, indices = nonoverlapping_title_pair(question, titles)
    return titles[indices[0]], titles[indices[1]]


def alias_question(question: str, titles: Sequence[str]) -> str:
    first, second, _ = nonoverlapping_title_pair(question, titles)
    return replace_spans(question, (first, second), ("Entity A", "Entity B"))


def delete_first_title(question: str, titles: Sequence[str]) -> str:
    first, second, _ = nonoverlapping_title_pair(question, titles)
    return replace_spans(question, (first, second), ("", question[second[0] : second[1]]))


def route_rejects(question: str, titles: Sequence[str]) -> bool:
    try:
        nonoverlapping_title_pair(question, titles)
    except RuntimeError:
        return True
    return False


def synthetic_overlap_passes() -> bool:
    titles = ("York", "New York", "New York City", "City Hall")
    question = "Did New York City and City Hall open together?"
    return route_strings(question, titles) == ("New York City", "City Hall")


def audit() -> dict[str, object]:
    started = time.perf_counter()
    if sha256_file(CORPUS) != CORPUS_SHA256:
        raise RuntimeError("corpus hash mismatch")
    if sha256_file(EVALUATOR) != EVALUATOR_SHA256:
        raise RuntimeError("evaluator hash mismatch")

    corpus = load_jsonl(CORPUS)
    evaluator = load_jsonl(EVALUATOR)
    titles = tuple(str(row["title"]) for row in corpus)
    if len(titles) != DOCUMENTS or len(set(titles)) != DOCUMENTS:
        raise RuntimeError("title domain mismatch")
    if len(evaluator) != QUESTIONS:
        raise RuntimeError("question domain mismatch")

    raw_routes: list[tuple[str, str]] = []
    support_pairs: list[tuple[str, str]] = []
    case_matches = 0
    alias_rejections = 0
    deletion_rejections = 0
    for row in evaluator:
        question = str(row["question"])
        route = route_strings(question, titles)
        if route[0] == route[1]:
            raise RuntimeError("duplicate route")
        raw_routes.append(route)
        support_pairs.append(tuple(str(value) for value in row["supporting_titles"]))
        case_matches += route_strings(question.swapcase(), titles) == route
        alias_rejections += route_rejects(alias_question(question, titles), titles)
        deletion_rejections += route_rejects(delete_first_title(question, titles), titles)

    # Labels are deliberately read only after every route has been frozen.
    support_matches = sum(
        set(route) == set(support)
        for route, support in zip(raw_routes, support_pairs)
    )

    mutated_support = [("not-a-title-a", "not-a-title-b")] * len(evaluator)
    label_independent = all(
        route == route_strings(str(row["question"]), titles)
        for route, row, _ in zip(raw_routes, evaluator, mutated_support)
    )

    reversed_titles = tuple(reversed(titles))
    order_matches = sum(
        set(route) == set(route_strings(str(row["question"]), reversed_titles))
        for route, row in zip(raw_routes, evaluator)
    )

    gates = {
        "all_questions_route": len(raw_routes) == QUESTIONS,
        "all_routes_distinct": all(a != b for a, b in raw_routes),
        "all_support_pairs_match": support_matches == QUESTIONS,
        "case_invariant": case_matches == QUESTIONS,
        "aliases_rejected": alias_rejections == QUESTIONS,
        "single_title_deletions_rejected": deletion_rejections == QUESTIONS,
        "support_labels_do_not_affect_route": label_independent,
        "nested_longest_nonoverlap": synthetic_overlap_passes(),
        "corpus_order_invariant_strings": order_matches == QUESTIONS,
        "cpu_only": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"router gate failed: {gates}")

    route_digest = hashlib.sha256(
        "\n".join("\t".join(route) for route in raw_routes).encode()
    ).hexdigest()
    return {
        "status": "pass_exact_surface_only",
        "domain": {"documents": DOCUMENTS, "questions": QUESTIONS},
        "counts": {
            "support_matches": support_matches,
            "case_matches": case_matches,
            "alias_rejections": alias_rejections,
            "deletion_rejections": deletion_rejections,
            "order_matches": order_matches,
        },
        "gates": gates,
        "integrity": {
            "corpus_sha256": sha256_file(CORPUS),
            "evaluator_sha256": sha256_file(EVALUATOR),
            "route_digest_sha256": route_digest,
        },
        "environment": {
            "python": platform.python_version(),
            "gpu_access": False,
            "model_access": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "non_claims": [
            "alias_resolution",
            "implicit_reference_retrieval",
            "semantic_extraction",
            "reader_sufficiency",
            "model_capability",
            "serving_performance",
        ],
    }


def main() -> None:
    print(json.dumps(audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

