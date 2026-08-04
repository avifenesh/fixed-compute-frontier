#!/usr/bin/env python3
"""T33 Phase B: sealed structural-coverage evaluator.

This process never constructs or changes a rule.  It opens support metadata
only after a valid Phase-A object exists, maps source sentence identities to
the frozen raw documents, and scores edge coverage.  It never accesses an
answer value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reversible_slot_grammar_plane_t33_stage0 import (
    CORPUS,
    EXPECTED_CORPUS_SHA256,
    PREREGISTRATION,
    canonical_json_bytes,
    sentence_char_spans,
    sha256_file,
)


PHASE_A = ROOT / "results/reversible-slot-grammar-plane-t33-stage0-phase-a.json"
SEALED = ROOT / "data/hotpot-real-prose-t12/sealed-evaluator.jsonl"
SOURCE = ROOT / "data/hotpotqa/validation-00000-of-00001.parquet"
OUTPUT = ROOT / "results/reversible-slot-grammar-plane-t33-stage0-coverage.json"

EXPECTED_SEALED_SHA256 = "5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b"
EXPECTED_SOURCE_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
TITLE_SHUFFLE_SEED = 33_001
RULE_SHUFFLE_SEED = 33_003


@dataclass(frozen=True)
class Edge:
    title: str
    sentence_index: int
    rule_id: int
    raw_start: int
    raw_length: int
    unique: bool


@dataclass(frozen=True)
class SupportItem:
    question_id: str
    title: str
    source_sentence_id: int
    source_char_start: int
    source_char_end: int
    compiler_sentence_indices: tuple[int, ...]


def verify_phase_a_object(payload: dict[str, Any]) -> None:
    if payload.get("status") != "pass" or not payload.get("phase_b_admitted"):
        raise RuntimeError("Phase A did not admit sealed coverage")
    expected = payload.get("object_sha256")
    unhashed = dict(payload)
    unhashed.pop("object_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
    if expected != actual:
        raise RuntimeError("Phase-A object SHA-256 mismatch")
    gates = payload.get("phase_a_gates")
    if not isinstance(gates, dict) or not all(gates.values()):
        raise RuntimeError("Phase-A gate object is not wholly true")


def load_candidate_documents() -> dict[str, dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if set(row) != {"document_id", "title", "text"}:
                raise RuntimeError("candidate corpus field boundary changed")
            title = row["title"]
            if title in documents:
                raise RuntimeError(f"duplicate candidate title: {title}")
            documents[title] = row
    return documents


def load_sealed_identities() -> dict[str, dict[str, Any]]:
    """Parse the sealed rows but copy only identity/support-integrity fields."""
    rows: dict[str, dict[str, Any]] = {}
    with SEALED.open(encoding="utf-8") as handle:
        for line in handle:
            parsed = json.loads(line)
            identifier = str(parsed["id"])
            if identifier in rows:
                raise RuntimeError(f"duplicate sealed ID: {identifier}")
            rows[identifier] = {
                "id": identifier,
                "supporting_titles": tuple(str(value) for value in parsed["supporting_titles"]),
                "supporting_sentence_ids": tuple(
                    int(value) for value in parsed["supporting_sentence_ids"]
                ),
            }
    return rows


def source_sentence_spans(sentences: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    joined = "".join(sentences)
    left_trim = len(joined) - len(joined.lstrip())
    right_limit = len(joined.rstrip())
    text = joined.strip()
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        raw_start, raw_end = cursor, cursor + len(sentence)
        clipped_start = max(raw_start, left_trim)
        clipped_end = min(raw_end, right_limit)
        spans.append(
            (
                max(0, clipped_start - left_trim),
                max(0, clipped_end - left_trim),
            )
        )
        cursor = raw_end
    return text, tuple(spans)


def map_support_items(
    sealed: dict[str, dict[str, Any]],
    candidate_documents: dict[str, dict[str, str]],
) -> tuple[tuple[SupportItem, ...], dict[str, int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(SOURCE, columns=["id", "supporting_facts", "context"])
    source_by_id = {
        str(row["id"]): row for row in table.to_pylist() if str(row["id"]) in sealed
    }
    if set(source_by_id) != set(sealed):
        raise RuntimeError("sealed IDs do not map one-to-one into pinned source")

    output: list[SupportItem] = []
    diagnostics: Counter[str] = Counter()
    for question_id in sorted(sealed):
        sealed_row = sealed[question_id]
        source_row = source_by_id[question_id]
        supporting = source_row["supporting_facts"]
        source_pairs = tuple(
            (str(title), int(sentence_id))
            for title, sentence_id in zip(
                supporting["title"], supporting["sent_id"], strict=True
            )
        )
        unique_pairs = tuple(dict.fromkeys(source_pairs))
        if set(title for title, _ in unique_pairs) != set(sealed_row["supporting_titles"]):
            raise RuntimeError(f"sealed/source support-title mismatch: {question_id}")
        if Counter(sentence_id for _, sentence_id in source_pairs) != Counter(
            sealed_row["supporting_sentence_ids"]
        ):
            raise RuntimeError(f"sealed/source support-ID mismatch: {question_id}")

        context = source_row["context"]
        context_titles = tuple(str(value) for value in context["title"])
        context_sentences = tuple(
            tuple(str(sentence) for sentence in values)
            for values in context["sentences"]
        )
        if len(context_titles) != len(set(context_titles)):
            raise RuntimeError(f"duplicate source context title: {question_id}")
        context_by_title = dict(zip(context_titles, context_sentences, strict=True))
        for title, source_sentence_id in unique_pairs:
            if title not in candidate_documents or title not in context_by_title:
                raise RuntimeError(f"support title missing from exact raw mapping: {title}")
            sentences = context_by_title[title]
            if not 0 <= source_sentence_id < len(sentences):
                raise RuntimeError(f"source support sentence out of range: {question_id}")
            rebuilt_text, source_spans = source_sentence_spans(sentences)
            candidate_text = candidate_documents[title]["text"]
            if rebuilt_text != candidate_text:
                raise RuntimeError(f"candidate/source text mismatch: {title}")
            source_start, source_end = source_spans[source_sentence_id]
            compiler_spans = sentence_char_spans(candidate_text, 0)
            compiler_indices = tuple(
                index
                for index, (start, end) in enumerate(compiler_spans)
                if source_start <= start and end <= source_end
            )
            if not compiler_indices:
                diagnostics["support_items_without_contained_compiler_sentence"] += 1
            output.append(
                SupportItem(
                    question_id=question_id,
                    title=title,
                    source_sentence_id=source_sentence_id,
                    source_char_start=source_start,
                    source_char_end=source_end,
                    compiler_sentence_indices=compiler_indices,
                )
            )
    diagnostics["source_rows_read_without_answer_column"] = len(source_by_id)
    diagnostics["mapped_support_items"] = len(output)
    return tuple(output), dict(diagnostics)


def phase_a_edges(payload: dict[str, Any]) -> tuple[Edge, ...]:
    edges = tuple(
        Edge(
            title=str(row["title"]),
            sentence_index=int(row["sentence_index"]),
            rule_id=int(row["rule_id"]),
            raw_start=int(row["raw_start"]),
            raw_length=int(row["raw_length"]),
            unique=bool(int(row["flags"]) & 1),
        )
        for row in payload["compiler"]["edges"]
    )
    return edges


def recompute_uniqueness(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    values = tuple(edges)
    counts = Counter((edge.title, edge.rule_id) for edge in values)
    return tuple(
        replace(edge, unique=counts[(edge.title, edge.rule_id)] == 1)
        for edge in values
    )


def score_coverage(
    support_items: Sequence[SupportItem], edges: Sequence[Edge]
) -> dict[str, Any]:
    by_title_sentence: dict[tuple[str, int], list[Edge]] = defaultdict(list)
    for edge in edges:
        if edge.unique:
            by_title_sentence[(edge.title, edge.sentence_index)].append(edge)
    item_counts: list[int] = []
    per_question_title: dict[str, dict[str, bool]] = defaultdict(dict)
    middle_lengths: list[int] = []
    for item in support_items:
        usable = [
            edge
            for sentence_index in item.compiler_sentence_indices
            for edge in by_title_sentence.get((item.title, sentence_index), ())
        ]
        item_counts.append(len(usable))
        previous = per_question_title[item.question_id].get(item.title, False)
        per_question_title[item.question_id][item.title] = previous or bool(usable)
        middle_lengths.extend(edge.raw_length for edge in usable)
    question_both = [
        len(title_map) == 2 and all(title_map.values())
        for title_map in per_question_title.values()
    ]
    support_count = len(item_counts)
    question_count = len(question_both)
    sorted_lengths = sorted(middle_lengths)
    return {
        "support_items": support_count,
        "support_items_with_edge": sum(value >= 1 for value in item_counts),
        "support_items_with_exactly_one_edge": sum(value == 1 for value in item_counts),
        "support_edge_coverage": sum(value >= 1 for value in item_counts) / support_count,
        "support_exactly_one_coverage": sum(value == 1 for value in item_counts) / support_count,
        "questions": question_count,
        "questions_with_both_documents": sum(question_both),
        "question_both_document_coverage": sum(question_both) / question_count,
        "usable_middle_lengths": {
            "count": len(sorted_lengths),
            "minimum": min(sorted_lengths) if sorted_lengths else None,
            "median": sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else None,
            "maximum": max(sorted_lengths) if sorted_lengths else None,
        },
    }


def shuffled_title_edges(edges: Sequence[Edge]) -> tuple[Edge, ...]:
    titles = sorted({edge.title for edge in edges})
    shuffled = list(titles)
    random.Random(TITLE_SHUFFLE_SEED).shuffle(shuffled)
    target_for_source = dict(zip(titles, shuffled, strict=True))
    return recompute_uniqueness(
        replace(edge, title=target_for_source[edge.title]) for edge in edges
    )


def shuffled_rule_edges(edges: Sequence[Edge]) -> tuple[Edge, ...]:
    rule_ids = [edge.rule_id for edge in edges]
    random.Random(RULE_SHUFFLE_SEED).shuffle(rule_ids)
    return recompute_uniqueness(
        replace(edge, rule_id=rule_id)
        for edge, rule_id in zip(edges, rule_ids, strict=True)
    )


def run() -> dict[str, Any]:
    started = time.perf_counter()
    input_hashes = {
        "phase_a": sha256_file(PHASE_A),
        "candidate": sha256_file(CORPUS),
        "sealed": sha256_file(SEALED),
        "source": sha256_file(SOURCE),
        "preregistration": sha256_file(PREREGISTRATION),
        "phase_b_source": sha256_file(Path(__file__)),
    }
    if input_hashes["candidate"] != EXPECTED_CORPUS_SHA256:
        raise RuntimeError("candidate hash mismatch")
    if input_hashes["sealed"] != EXPECTED_SEALED_SHA256:
        raise RuntimeError("sealed evaluator hash mismatch")
    if input_hashes["source"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("source parquet hash mismatch")

    phase_a = json.loads(PHASE_A.read_text())
    verify_phase_a_object(phase_a)
    candidates = load_candidate_documents()
    sealed = load_sealed_identities()
    support_items, mapping = map_support_items(sealed, candidates)
    edges = phase_a_edges(phase_a)
    correct = score_coverage(support_items, edges)
    title_shuffle = score_coverage(support_items, shuffled_title_edges(edges))
    rule_shuffle = score_coverage(support_items, shuffled_rule_edges(edges))
    gates = {
        "all_inputs_exact": True,
        "phase_a_valid": True,
        "every_support_item_mapped": mapping["mapped_support_items"] == len(support_items),
        "support_coverage_89_44pct": correct["support_edge_coverage"] >= 0.8944,
        "question_both_80pct": correct["question_both_document_coverage"] >= 0.80,
        "answers_not_loaded_from_source": True,
    }
    payload = {
        "schema": "reversible-slot-grammar-plane-t33-coverage-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "claim_boundary": (
            "Sealed structural coverage only. No answer was accessed and no semantic "
            "sufficiency, query mapping, or model capability is established."
        ),
        "input_hashes": input_hashes,
        "phase_a_object_sha256": phase_a["object_sha256"],
        "mapping": mapping,
        "correct": correct,
        "controls": {
            "title_shuffle_seed_33001": title_shuffle,
            "edge_rule_shuffle_seed_33003": rule_shuffle,
            "note": (
                "A dictionary-consistent global rule-ID permutation is invariant by "
                "construction; the reported rule control shuffles IDs across edges only."
            ),
        },
        "gates": gates,
        "next_stage_admitted": all(gates.values()),
        "wall_seconds": time.perf_counter() - started,
    }
    payload["object_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
