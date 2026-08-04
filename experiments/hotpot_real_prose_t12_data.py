#!/usr/bin/env python3
"""Prepare a label-blind real-prose T12 corpus and sealed HotpotQA split."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/hotpotqa/validation-00000-of-00001.parquet"
OUTPUT_DIR = ROOT / "data/hotpot-real-prose-t12"
MANIFEST = ROOT / "results/hotpot-real-prose-t12-data-manifest.json"

HF_REPO = "hotpotqa/hotpot_qa"
HF_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HF_FILE = "distractor/validation-00000-of-00001.parquet"
SOURCE_BYTES = 27_452_575
SOURCE_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"

STOPWORDS = frozenset(
    "are is were was both and or do does did have has had the a an of for to "
    "that these those this two their its it be been being whether also all".split()
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(identifier: str, buckets: int = 5) -> int:
    value = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big")
    return value % buckets


def supporting_titles(row: dict[str, object]) -> tuple[str, ...]:
    supporting = row["supporting_facts"]
    assert isinstance(supporting, dict)
    return tuple(sorted(set(supporting["title"])))


def eligible(row: dict[str, object]) -> bool:
    answer = str(row["answer"]).lower()
    question = str(row["question"]).lower()
    titles = supporting_titles(row)
    return (
        row["type"] == "comparison"
        and answer in {"yes", "no"}
        and " both " in question
        and len(titles) == 2
        and all(title.lower() in question for title in titles)
    )


@dataclass(frozen=True)
class Split:
    train: tuple[dict[str, object], ...]
    evaluation: tuple[dict[str, object], ...]
    dropped_title_overlap: tuple[dict[str, object], ...]


def split_rows(rows: Iterable[dict[str, object]]) -> Split:
    selected = sorted((row for row in rows if eligible(row)), key=lambda row: row["id"])
    train = tuple(row for row in selected if stable_bucket(str(row["id"])) >= 2)
    provisional_evaluation = tuple(
        row for row in selected if stable_bucket(str(row["id"])) < 2
    )
    train_titles = {title for row in train for title in supporting_titles(row)}
    evaluation = tuple(
        row
        for row in provisional_evaluation
        if not (set(supporting_titles(row)) & train_titles)
    )
    dropped = tuple(
        row
        for row in provisional_evaluation
        if set(supporting_titles(row)) & train_titles
    )
    return Split(train, evaluation, dropped)


def context_documents(row: dict[str, object]) -> Iterable[tuple[str, str]]:
    context = row["context"]
    assert isinstance(context, dict)
    for title, sentences in zip(
        context["title"], context["sentences"], strict=True
    ):
        yield str(title), "".join(str(sentence) for sentence in sentences).strip()


def candidate_documents(split: Split) -> list[dict[str, str]]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in (*split.train, *split.evaluation):
        for title, text in context_documents(row):
            key = (title, text)
            identifier = hashlib.sha256((title + "\n" + text).encode()).hexdigest()
            unique[key] = {"document_id": identifier, "title": title, "text": text}
    return sorted(unique.values(), key=lambda document: document["document_id"])


def qa_training_rows(split: Split) -> list[dict[str, str]]:
    return [
        {
            "id": str(row["id"]),
            "question": str(row["question"]),
            "answer": str(row["answer"]).lower(),
        }
        for row in split.train
    ]


def evaluator_rows(split: Split) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in split.evaluation:
        supporting = row["supporting_facts"]
        assert isinstance(supporting, dict)
        result.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]).lower(),
                "supporting_titles": supporting_titles(row),
                "supporting_sentence_ids": tuple(int(value) for value in supporting["sent_id"]),
            }
        )
    return result


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def privileged_lexical_probe(split: Split) -> dict[str, object]:
    """Gold-document lexical upper bound; diagnostic only, never a candidate input."""
    all_rows = (*split.train, *split.evaluation)
    document_sets: list[set[str]] = []
    per_row_documents: dict[str, dict[str, str]] = {}
    for row in all_rows:
        documents = dict(context_documents(row))
        per_row_documents[str(row["id"])] = documents
        document_sets.extend(set(word_tokens(text)) for text in documents.values())
    frequencies = collections.Counter(word for words in document_sets for word in words)
    document_count = len(document_sets)

    def score(row: dict[str, object]) -> float:
        question = str(row["question"]).lower()
        for title in sorted(supporting_titles(row), key=len, reverse=True):
            question = question.replace(title.lower(), " ")
        query = [word for word in word_tokens(question) if word not in STOPWORDS]
        denominator = sum(
            math.log((document_count + 1) / (frequencies[word] + 1)) + 1
            for word in query
        ) or 1.0
        documents = per_row_documents[str(row["id"])]
        page_scores = []
        for title in supporting_titles(row):
            words = set(word_tokens(documents[title]))
            page_scores.append(
                sum(
                    math.log((document_count + 1) / (frequencies[word] + 1)) + 1
                    for word in query
                    if word in words
                )
                / denominator
            )
        return min(page_scores)

    train_scores = [
        (score(row), str(row["answer"]).lower() == "yes") for row in split.train
    ]
    candidates = sorted({-1.0, 1.0, *(value for value, _ in train_scores)})
    best: tuple[float, float, bool] | None = None
    for threshold in candidates:
        for greater_equal in (True, False):
            accuracy = sum(
                ((value >= threshold) if greater_equal else (value <= threshold))
                == target
                for value, target in train_scores
            ) / len(train_scores)
            choice = (accuracy, -abs(threshold), greater_equal)
            if best is None or choice > best:
                best = choice
                selected_threshold = threshold
                selected_direction = greater_equal
    assert best is not None

    def evaluate(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
        records = []
        for row in rows:
            value = score(row)
            prediction = (
                value >= selected_threshold
                if selected_direction
                else value <= selected_threshold
            )
            target = str(row["answer"]).lower() == "yes"
            records.append((prediction, target))
        return {
            "examples": len(records),
            "accuracy": sum(prediction == target for prediction, target in records)
            / len(records),
            "positive_rate": sum(target for _, target in records) / len(records),
            "predicted_positive_rate": sum(prediction for prediction, _ in records)
            / len(records),
        }

    return {
        "claim_boundary": (
            "Uses sealed supporting-page identities and fitted train labels; this is "
            "a privileged lexical difficulty probe, not an admissible model."
        ),
        "threshold": selected_threshold,
        "greater_equal": selected_direction,
        "train": evaluate(split.train),
        "evaluation": evaluate(split.evaluation),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )


def run(source: Path, output_dir: Path) -> dict[str, object]:
    if source.stat().st_size != SOURCE_BYTES or sha256_file(source) != SOURCE_SHA256:
        raise RuntimeError("HotpotQA source bytes or checksum mismatch")
    import pyarrow.parquet as pq

    rows = pq.read_table(source).to_pylist()
    split = split_rows(rows)
    documents = candidate_documents(split)
    training = qa_training_rows(split)
    evaluation = evaluator_rows(split)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_raw_prose": output_dir / "candidate-raw-prose.jsonl",
        "router_and_control_train": output_dir / "router-and-control-train.jsonl",
        "sealed_evaluator": output_dir / "sealed-evaluator.jsonl",
    }
    write_jsonl(paths["candidate_raw_prose"], documents)
    write_jsonl(paths["router_and_control_train"], training)
    write_jsonl(paths["sealed_evaluator"], evaluation)

    train_titles = {title for row in split.train for title in supporting_titles(row)}
    evaluation_titles = {
        title for row in split.evaluation for title in supporting_titles(row)
    }
    candidate_fields = set().union(*(document.keys() for document in documents))
    gates = {
        "source_exact": True,
        "candidate_has_no_question_answer_or_support_fields": not (
            candidate_fields & {"question", "answer", "supporting_titles", "sent_id"}
        ),
        "train_evaluation_ids_disjoint": not (
            {row["id"] for row in training} & {row["id"] for row in evaluation}
        ),
        "train_evaluation_support_titles_disjoint": not (
            train_titles & evaluation_titles
        ),
        "evaluation_has_both_classes": {
            row["answer"] for row in evaluation
        }
        == {"yes", "no"},
        "questions_withheld_from_compiler_corpus": all(
            "question" not in document for document in documents
        ),
    }
    return {
        "schema": "hotpot-real-prose-t12-data-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "source": {
            "hub_repo": HF_REPO,
            "hub_revision": HF_REVISION,
            "hub_file": HF_FILE,
            "bytes": SOURCE_BYTES,
            "sha256": SOURCE_SHA256,
            "license": "CC BY-SA 4.0",
        },
        "selection": {
            "definition": (
                "HotpotQA hard comparison, yes/no answer, contains ' both ', "
                "exactly two supporting titles both present verbatim in the question"
            ),
            "eligible": len(split.train)
            + len(split.evaluation)
            + len(split.dropped_title_overlap),
            "router_and_control_train": len(split.train),
            "sealed_evaluation": len(split.evaluation),
            "dropped_for_support_title_overlap": len(split.dropped_title_overlap),
            "train_answer_counts": dict(
                collections.Counter(row["answer"] for row in training)
            ),
            "evaluation_answer_counts": dict(
                collections.Counter(row["answer"] for row in evaluation)
            ),
            "unique_train_support_titles": len(train_titles),
            "unique_evaluation_support_titles": len(evaluation_titles),
        },
        "candidate_corpus": {
            "documents": len(documents),
            "characters": sum(len(document["text"]) for document in documents),
            "fields": sorted(candidate_fields),
        },
        "privileged_lexical_probe": privileged_lexical_probe(split),
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": (
            "This freezes a real-prose, natural-question data boundary. It does not "
            "show autonomous extraction or model capability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    arguments = parser.parse_args()
    result = run(arguments.source, arguments.output_dir)
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
