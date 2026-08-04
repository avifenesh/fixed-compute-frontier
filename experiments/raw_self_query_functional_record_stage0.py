#!/usr/bin/env python3
"""Stage 0 for raw-only, query-conditioned functional document records."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
OUTPUT = ROOT / "results/raw-self-query-functional-record-stage0.json"
SCHEMA = "raw-self-query-functional-record-stage0-v1"
WORD_PATTERN = re.compile(r"[a-z0-9]+")
RADIUS = 4
EVALUATION_BUCKETS = 5
EXPECTED_DOCUMENTS = 2_405
CODE_CELLS = 220
BITS_PER_CELL = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def word_tokens(text: str) -> tuple[str, ...]:
    return tuple(WORD_PATTERN.findall(text.lower()))


def frequency_thresholds(document_count: int) -> dict[str, int]:
    if document_count < 2:
        raise ValueError("at least two documents are required")
    return {
        "anchor_min_document_frequency": math.ceil(document_count / 100),
        "anchor_max_document_frequency": document_count // 2,
        "target_min_document_frequency": 1,
        "target_max_document_frequency": math.ceil(document_count / 20),
    }


def document_frequencies(sequences: Iterable[tuple[str, ...]]) -> collections.Counter[str]:
    frequencies: collections.Counter[str] = collections.Counter()
    for sequence in sequences:
        frequencies.update(set(sequence))
    return frequencies


def relation_skeleton(
    sequence: tuple[str, ...],
    target_index: int,
    anchors: frozenset[str],
    radius: int = RADIUS,
) -> tuple[tuple[int, str], ...]:
    """Keep corpus-common relation anchors and delete the target and rare values."""
    start = max(0, target_index - radius)
    stop = min(len(sequence), target_index + radius + 1)
    return tuple(
        (index - target_index, sequence[index])
        for index in range(start, stop)
        if index != target_index and sequence[index] in anchors
    )


@dataclasses.dataclass(frozen=True)
class Probe:
    document_index: int
    document_id: str
    target_index: int
    target: str
    skeleton: tuple[tuple[int, str], ...]
    fallback: bool = False
    split: str = ""

    @property
    def identity(self) -> str:
        return f"{self.document_id}:{self.skeleton}:{self.target}"


def split_document_probes(probes: list[Probe]) -> list[Probe]:
    """Stable 80/20 split, repaired per document without overlapping an occurrence."""
    if len(probes) < 2:
        raise ValueError("each document needs at least two distinct probes")
    ordered = sorted(probes, key=lambda probe: (stable_u64(probe.identity), probe.target_index))
    splits = [
        "evaluation"
        if stable_u64(probe.identity) % EVALUATION_BUCKETS == 0
        else "train"
        for probe in ordered
    ]
    if "evaluation" not in splits:
        splits[0] = "evaluation"
    if "train" not in splits:
        splits[-1] = "train"
    return [dataclasses.replace(probe, split=split) for probe, split in zip(ordered, splits, strict=True)]


def build_probes(
    documents: list[dict[str, str]],
) -> tuple[list[Probe], dict[str, object]]:
    fields = set().union(*(document.keys() for document in documents))
    if fields != {"document_id", "title", "text"}:
        raise ValueError(f"non-raw corpus fields: {sorted(fields)}")
    sequences = [word_tokens(document["text"]) for document in documents]
    frequencies = document_frequencies(sequences)
    thresholds = frequency_thresholds(len(documents))
    anchors = frozenset(
        word
        for word, frequency in frequencies.items()
        if thresholds["anchor_min_document_frequency"]
        <= frequency
        <= thresholds["anchor_max_document_frequency"]
    )
    targets = frozenset(
        word
        for word, frequency in frequencies.items()
        if thresholds["target_min_document_frequency"]
        <= frequency
        <= thresholds["target_max_document_frequency"]
    )

    probes: list[Probe] = []
    fallback_documents = 0
    for document_index, (document, sequence) in enumerate(zip(documents, sequences, strict=True)):
        candidates = [
            Probe(
                document_index=document_index,
                document_id=document["document_id"],
                target_index=index,
                target=word,
                skeleton=relation_skeleton(sequence, index, anchors),
            )
            for index, word in enumerate(sequence)
            if word in targets
        ]
        functional = [probe for probe in candidates if probe.skeleton]
        unique: dict[tuple[tuple[tuple[int, str], ...], str], Probe] = {}
        for probe in functional:
            key = (probe.skeleton, probe.target)
            previous = unique.get(key)
            if previous is None or probe.target_index < previous.target_index:
                unique[key] = probe
        functional = list(unique.values())
        if len(functional) < 2:
            fallback_documents += 1
            # A position-only carrier keeps the record trainable but is excluded
            # from the semantic-collision claim.
            existing = {(probe.skeleton, probe.target) for probe in functional}
            for probe in candidates:
                fallback = dataclasses.replace(
                    probe, skeleton=(), fallback=True
                )
                key = (fallback.skeleton, fallback.target)
                if key not in existing:
                    functional.append(fallback)
                    existing.add(key)
                if len(functional) >= 2:
                    break
        probes.extend(split_document_probes(functional))

    return probes, {
        "thresholds": thresholds,
        "anchor_words": len(anchors),
        "target_words": len(targets),
        "fallback_documents": fallback_documents,
        "raw_fields": sorted(fields),
    }


def conditional_entropy_bits(groups: Iterable[set[tuple[int, str]]]) -> float:
    weighted = 0.0
    count = 0
    for values in groups:
        targets = collections.Counter(target for _, target in values)
        size = len(values)
        entropy = -sum(
            (frequency / size) * math.log2(frequency / size)
            for frequency in targets.values()
        )
        weighted += size * entropy
        count += size
    return weighted / count if count else 0.0


def single_valued_collision_groups(
    probes: Iterable[Probe],
) -> tuple[
    dict[tuple[tuple[int, str], ...], set[tuple[int, str]]],
    int,
    int,
]:
    grouped: dict[
        tuple[tuple[int, str], ...], dict[int, set[str]]
    ] = collections.defaultdict(lambda: collections.defaultdict(set))
    for probe in probes:
        if not probe.fallback and probe.skeleton:
            grouped[probe.skeleton][probe.document_index].add(probe.target)
    ambiguous_document_skeletons = sum(
        len(targets) > 1
        for documents in grouped.values()
        for targets in documents.values()
    )
    single_valued = {
        skeleton: {
            (document, next(iter(targets)))
            for document, targets in documents.items()
            if len(targets) == 1
        }
        for skeleton, documents in grouped.items()
    }
    collisions = {
        skeleton: values
        for skeleton, values in single_valued.items()
        if len({document for document, _ in values}) >= 2
        and len({target for _, target in values}) >= 2
    }
    return collisions, ambiguous_document_skeletons, len(grouped)


def collision_statistics(probes: Iterable[Probe]) -> dict[str, object]:
    collisions, ambiguous_document_skeletons, unique_skeletons = (
        single_valued_collision_groups(probes)
    )
    equality_groups: dict[tuple[tuple[int, str], ...], set[tuple[int, str]]] = {}
    positive_pairs = 0
    negative_pairs = 0
    for skeleton, values in collisions.items():
        by_target: dict[str, set[int]] = collections.defaultdict(set)
        by_document: dict[int, set[str]] = collections.defaultdict(set)
        for document, target in values:
            by_target[target].add(document)
            by_document[document].add(target)
        positives = sum(
            len(documents) * (len(documents) - 1) // 2
            for documents in by_target.values()
        )
        documents = sorted(by_document)
        negatives = sum(
            sum(
                left != right
                for left in by_document[first]
                for right in by_document[second]
            )
            for offset, first in enumerate(documents)
            for second in documents[offset + 1 :]
        )
        if positives and negatives:
            equality_groups[skeleton] = values
            positive_pairs += positives
            negative_pairs += negatives
    collision_values = list(collisions.values())
    return {
        "unique_skeletons": unique_skeletons,
        "ambiguous_document_skeletons_excluded": ambiguous_document_skeletons,
        "collision_groups": len(collisions),
        "collision_probe_pairs": sum(len(values) for values in collision_values),
        "collision_documents": len(
            {document for values in collision_values for document, _ in values}
        ),
        "conditional_target_entropy_bits": conditional_entropy_bits(collision_values),
        "equality_groups": len(equality_groups),
        "equality_documents": len(
            {document for values in equality_groups.values() for document, _ in values}
        ),
        "positive_cross_document_pairs": positive_pairs,
        "negative_cross_document_pairs": negative_pairs,
    }


def run(corpus: Path) -> dict[str, object]:
    documents = [json.loads(line) for line in corpus.read_text().splitlines()]
    probes, vocabulary = build_probes(documents)
    splits = collections.Counter(probe.split for probe in probes)
    split_documents = {
        split: len({probe.document_index for probe in probes if probe.split == split})
        for split in ("train", "evaluation")
    }
    all_collisions = collision_statistics(probes)
    train_collisions = collision_statistics(
        probe for probe in probes if probe.split == "train"
    )
    evaluation_collisions = collision_statistics(
        probe for probe in probes if probe.split == "evaluation"
    )
    gates = {
        "expected_raw_documents_and_fields": len(documents) == EXPECTED_DOCUMENTS
        and vocabulary["raw_fields"] == ["document_id", "text", "title"],
        "every_document_has_disjoint_train_and_evaluation_reads": split_documents
        == {"train": len(documents), "evaluation": len(documents)}
        and len({probe.identity for probe in probes}) == len(probes),
        "collision_queries_cover_at_least_85pct_documents": all_collisions[
            "collision_documents"
        ]
        >= math.ceil(0.85 * len(documents)),
        "collision_target_entropy_at_least_2_bits": all_collisions[
            "conditional_target_entropy_bits"
        ]
        >= 2.0,
        "train_has_at_least_200_equality_groups_and_500_positive_pairs": train_collisions[
            "equality_groups"
        ]
        >= 200
        and train_collisions["positive_cross_document_pairs"] >= 500,
        "evaluation_retains_at_least_10_equality_groups_and_25_positive_pairs": evaluation_collisions[
            "equality_groups"
        ]
        >= 10
        and evaluation_collisions["positive_cross_document_pairs"] >= 25,
        "logical_record_budget_unchanged": len(documents)
        * CODE_CELLS
        * BITS_PER_CELL
        == 2_116_400,
    }
    return {
        "schema": SCHEMA,
        "status": "stage0_only_not_capability_evidence",
        "corpus": {
            "path": str(corpus.relative_to(ROOT)),
            "sha256": sha256_file(corpus),
            "documents": len(documents),
            "fields": vocabulary["raw_fields"],
        },
        "probe_rule": {
            **vocabulary,
            "radius_words_each_side": RADIUS,
            "evaluation_hash_buckets": EVALUATION_BUCKETS,
            "target_is_removed": True,
            "rare_non_target_words_are_masked": True,
            "uses_parser_teacher_qa_or_support_fields": False,
        },
        "probes": {
            "total": len(probes),
            "split_counts": dict(sorted(splits.items())),
            "split_documents": split_documents,
            "fallback_probes": sum(probe.fallback for probe in probes),
        },
        "collisions": {
            "all": all_collisions,
            "train": train_collisions,
            "evaluation": evaluation_collisions,
        },
        "digital_record": {
            "documents": len(documents),
            "cells_per_document": CODE_CELLS,
            "bits_per_cell": BITS_PER_CELL,
            "logical_bits": len(documents) * CODE_CELLS * BITS_PER_CELL,
        },
        "gates": gates,
        "stage0_pass": all(gates.values()),
        "decision": (
            "admit_preregistration_of_a_raw-only_unary-read_and_balanced-pair-equality_GPU_gate"
            if all(gates.values())
            else "stop_before_GPU"
        ),
        "limits": [
            "Stage 0 proves only that the raw corpus supplies ambiguous query families and self-supervised equality pairs.",
            "It does not show that a quantized record learns the reads, transfers to natural QA, or exports into ordinary weights.",
            "The reused Hotpot development split cannot support a final capability claim; an admitted mechanism needs untouched holdout.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(arguments.corpus)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
