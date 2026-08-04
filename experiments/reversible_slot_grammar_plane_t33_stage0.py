#!/usr/bin/env python3
"""T33 Phase A: raw-only reversible one-hole grammar compiler.

The reusable core has no tokenizer, model, Torch, or GPU dependency.  The
natural-corpus entry point imports only a pinned, cached fast tokenizer and
uses it as a coordinate system.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import struct
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results/reversible-slot-grammar-plane-t33-stage0-preregistration.md"
CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
OUTPUT = ROOT / "results/reversible-slot-grammar-plane-t33-stage0-phase-a.json"
IMAGE_OUTPUT = ROOT / "results/reversible-slot-grammar-plane-t33-stage0-phase-a.bin"

EXPECTED_CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TOKENIZER_ARTIFACTS = (
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

SELF_TOKEN = 65_535
TOKEN_SLOTS = 128
MAX_ANCHOR_TOKENS = 127
MAX_MIDDLE_TOKENS = 127
MAX_RULES = 4_096
MAX_OCCURRENCES = 16_383
BITS_PER_TOKEN = 16
RULE_METADATA_BITS = 60
REFERENCE_BITS = 19
STATE_BYTE_CAP = 974_848
ALIGNMENT = 64
IMAGE_MAGIC = b"T33RSGP\0"

Proposal = tuple[tuple[int, ...], tuple[int, ...]]
SentenceKey = tuple[str, int]


@dataclass(frozen=True)
class NormalizedSentence:
    document_index: int
    document_id: str
    title: str
    sentence_index: int
    tokens: tuple[int, ...]
    raw_boundaries: tuple[int, ...]
    raw_record_tokens: tuple[int, ...]
    serialized_char_start: int = 0
    serialized_char_end: int = 0

    def __post_init__(self) -> None:
        if len(self.raw_boundaries) != len(self.tokens) + 1:
            raise ValueError("normalized boundary count mismatch")
        if any(left >= right for left, right in zip(self.raw_boundaries, self.raw_boundaries[1:])):
            raise ValueError("raw boundaries must increase strictly")
        if any(not 0 <= token <= 65_535 for token in self.tokens):
            raise ValueError("normalized token outside uint16 domain")

    @property
    def key(self) -> SentenceKey:
        return self.document_id, self.sentence_index


@dataclass(frozen=True)
class Occurrence:
    sentence_ordinal: int
    middle_start: int
    middle_end: int


@dataclass(frozen=True)
class RetainedRule:
    rule_id: int
    prefix: tuple[int, ...]
    suffix: tuple[int, ...]
    gain_bits: int
    occurrences: tuple[Occurrence, ...]
    crc32: int


@dataclass(frozen=True)
class MaterializedEdge:
    document_index: int
    document_id: str
    title: str
    sentence_index: int
    rule_id: int
    raw_start: int
    raw_length: int
    flags: int
    normalized_middle: tuple[int, ...]
    raw_middle: tuple[int, ...]


@dataclass(frozen=True)
class Compilation:
    rules: tuple[RetainedRule, ...]
    edges: tuple[MaterializedEdge, ...]
    proposals: tuple[Proposal, ...]
    eligible: tuple[Proposal, ...]
    exact_repeat_groups: tuple[tuple[SentenceKey, ...], ...]
    metrics: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def proposal_bytes(proposal: Proposal) -> bytes:
    prefix, suffix = proposal
    return canonical_json_bytes([list(prefix), list(suffix)])


def proposal_sort_key(proposal: Proposal) -> bytes:
    return proposal_bytes(proposal)


def anti_unify(left: Sequence[int], right: Sequence[int]) -> Proposal | None:
    """Return the frozen maximum-total-boundary one-hole proposal."""
    x, y = tuple(left), tuple(right)
    if x == y or min(len(x), len(y)) < 3:
        return None
    common_prefix = 0
    while (
        common_prefix < min(len(x), len(y))
        and x[common_prefix] == y[common_prefix]
    ):
        common_prefix += 1
    common_suffix = 0
    while (
        common_suffix < min(len(x), len(y))
        and x[-1 - common_suffix] == y[-1 - common_suffix]
    ):
        common_suffix += 1
    if common_prefix == 0 or common_suffix == 0:
        return None
    candidates: list[tuple[int, int, bytes, Proposal]] = []
    for prefix_length in range(1, min(common_prefix, MAX_ANCHOR_TOKENS) + 1):
        maximum_suffix = min(
            common_suffix,
            MAX_ANCHOR_TOKENS,
            len(x) - prefix_length - 1,
            len(y) - prefix_length - 1,
        )
        minimum_suffix = max(
            1,
            len(x) - prefix_length - MAX_MIDDLE_TOKENS,
            len(y) - prefix_length - MAX_MIDDLE_TOKENS,
        )
        if maximum_suffix < minimum_suffix:
            continue
        suffix_length = maximum_suffix
        proposal = x[:prefix_length], x[len(x) - suffix_length :]
        candidates.append(
            (
                prefix_length + suffix_length,
                prefix_length,
                proposal_bytes(proposal),
                proposal,
            )
        )
    if not candidates:
        return None
    best_total = max(item[0] for item in candidates)
    best_prefix = max(item[1] for item in candidates if item[0] == best_total)
    return min(
        (
            item
            for item in candidates
            if item[0] == best_total and item[1] == best_prefix
        ),
        key=lambda item: item[2],
    )[3]


def brute_force_anti_unify(left: Sequence[int], right: Sequence[int]) -> Proposal | None:
    """Independent exhaustive reference for the direct anti-unifier."""
    x, y = tuple(left), tuple(right)
    if x == y:
        return None
    choices: list[tuple[int, int, bytes, Proposal]] = []
    for prefix_length in range(1, min(len(x), len(y), MAX_ANCHOR_TOKENS) + 1):
        if x[:prefix_length] != y[:prefix_length]:
            continue
        for suffix_length in range(1, min(len(x), len(y), MAX_ANCHOR_TOKENS) + 1):
            x_middle = len(x) - prefix_length - suffix_length
            y_middle = len(y) - prefix_length - suffix_length
            if not (
                1 <= x_middle <= MAX_MIDDLE_TOKENS
                and 1 <= y_middle <= MAX_MIDDLE_TOKENS
            ):
                continue
            if x[-suffix_length:] != y[-suffix_length:]:
                continue
            proposal = x[:prefix_length], x[-suffix_length:]
            choices.append(
                (
                    prefix_length + suffix_length,
                    prefix_length,
                    proposal_bytes(proposal),
                    proposal,
                )
            )
    if not choices:
        return None
    best_total = max(item[0] for item in choices)
    best_prefix = max(item[1] for item in choices if item[0] == best_total)
    finalists = [
        item
        for item in choices
        if item[0] == best_total and item[1] == best_prefix
    ]
    return min(finalists, key=lambda item: item[2])[3]


def match(proposal: Proposal, tokens: Sequence[int]) -> tuple[int, int] | None:
    prefix, suffix = proposal
    sequence = tuple(tokens)
    middle_start = len(prefix)
    middle_end = len(sequence) - len(suffix)
    if (
        sequence[:middle_start] == prefix
        and sequence[middle_end:] == suffix
        and 1 <= middle_end - middle_start <= MAX_MIDDLE_TOKENS
    ):
        return middle_start, middle_end
    return None


def code_gain_bits(proposal: Proposal, occurrence_count: int) -> int:
    anchor_length = len(proposal[0]) + len(proposal[1])
    return (
        BITS_PER_TOKEN * (occurrence_count - 1) * anchor_length
        - RULE_METADATA_BITS
        - occurrence_count * REFERENCE_BITS
    )


def enumerate_proposals(
    sentences: Sequence[NormalizedSentence],
) -> tuple[set[Proposal], tuple[tuple[SentenceKey, ...], ...], int]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    repeats: dict[tuple[int, ...], set[SentenceKey]] = defaultdict(set)
    for ordinal, sentence in enumerate(sentences):
        if sentence.tokens:
            buckets[(sentence.tokens[0], sentence.tokens[-1])].append(ordinal)
            repeats[sentence.tokens].add(sentence.key)

    proposals: set[Proposal] = set()
    comparisons = 0
    for ordinals in buckets.values():
        for left_ordinal, right_ordinal in itertools.combinations(ordinals, 2):
            left, right = sentences[left_ordinal], sentences[right_ordinal]
            if left.document_id == right.document_id:
                continue
            comparisons += 1
            proposal = anti_unify(left.tokens, right.tokens)
            if proposal is not None:
                proposals.add(proposal)

    exact_groups = tuple(
        sorted(
            (
                tuple(sorted(keys))
                for keys in repeats.values()
                if len({document_id for document_id, _ in keys}) >= 2
            ),
            key=lambda group: canonical_json_bytes(group),
        )
    )
    return proposals, exact_groups, comparisons


def indexed_occurrences(
    sentences: Sequence[NormalizedSentence], proposals: Iterable[Proposal]
) -> dict[Proposal, tuple[Occurrence, ...]]:
    proposal_set = set(proposals)
    result: dict[Proposal, list[Occurrence]] = defaultdict(list)
    for ordinal, sentence in enumerate(sentences):
        sequence = sentence.tokens
        length = len(sequence)
        for prefix_length in range(1, min(MAX_ANCHOR_TOKENS, length - 2) + 1):
            maximum_suffix = min(MAX_ANCHOR_TOKENS, length - prefix_length - 1)
            minimum_suffix = max(1, length - prefix_length - MAX_MIDDLE_TOKENS)
            for suffix_length in range(minimum_suffix, maximum_suffix + 1):
                proposal = (
                    sequence[:prefix_length],
                    sequence[length - suffix_length :],
                )
                if proposal in proposal_set:
                    result[proposal].append(
                        Occurrence(ordinal, prefix_length, length - suffix_length)
                    )
    return {
        proposal: tuple(sorted(occurrences, key=lambda value: value.sentence_ordinal))
        for proposal, occurrences in result.items()
    }


def dense_occurrences(
    sentences: Sequence[NormalizedSentence], proposals: Iterable[Proposal]
) -> dict[Proposal, tuple[Occurrence, ...]]:
    result: dict[Proposal, tuple[Occurrence, ...]] = {}
    for proposal in proposals:
        values = []
        for ordinal, sentence in enumerate(sentences):
            span = match(proposal, sentence.tokens)
            if span is not None:
                values.append(Occurrence(ordinal, *span))
        if values:
            result[proposal] = tuple(values)
    return result


def rule_crc32(proposal: Proposal) -> int:
    flat = proposal[0] + proposal[1]
    payload = struct.pack(f"<{len(flat)}H", *flat) if flat else b""
    return zlib.crc32(payload) & 0xFFFF_FFFF


def select_rules(
    sentences: Sequence[NormalizedSentence],
    proposals: Iterable[Proposal],
    occurrences: dict[Proposal, tuple[Occurrence, ...]],
) -> tuple[tuple[RetainedRule, ...], tuple[Proposal, ...], dict[str, int]]:
    eligible: list[Proposal] = []
    for proposal in proposals:
        values = occurrences.get(proposal, ())
        if len(values) > MAX_OCCURRENCES:
            raise RuntimeError("proposal occurrence count exceeds 14-bit format")
        distinct_documents = {
            sentences[value.sentence_ordinal].document_id for value in values
        }
        if len(values) >= 2 and len(distinct_documents) >= 2 and code_gain_bits(proposal, len(values)) > 0:
            eligible.append(proposal)

    touching: dict[int, set[Proposal]] = defaultdict(set)
    for proposal in eligible:
        for occurrence in occurrences[proposal]:
            touching[occurrence.sentence_ordinal].add(proposal)
    conflicted_sentences = {
        ordinal for ordinal, values in touching.items() if len(values) > 1
    }
    conflicted_proposals = {
        proposal
        for ordinal in conflicted_sentences
        for proposal in touching[ordinal]
    }
    retained_proposals = sorted(
        (proposal for proposal in eligible if proposal not in conflicted_proposals),
        key=proposal_sort_key,
    )
    if len(retained_proposals) > MAX_RULES:
        raise RuntimeError("retained rule count exceeds 12-bit format")
    rules = tuple(
        RetainedRule(
            rule_id=rule_id,
            prefix=proposal[0],
            suffix=proposal[1],
            gain_bits=code_gain_bits(proposal, len(occurrences[proposal])),
            occurrences=occurrences[proposal],
            crc32=rule_crc32(proposal),
        )
        for rule_id, proposal in enumerate(retained_proposals)
    )
    return rules, tuple(sorted(eligible, key=proposal_sort_key)), {
        "conflicted_sentences": len(conflicted_sentences),
        "conflicted_proposals": len(conflicted_proposals),
    }


def materialize_edges(
    sentences: Sequence[NormalizedSentence], rules: Sequence[RetainedRule]
) -> tuple[tuple[MaterializedEdge, ...], dict[str, int]]:
    provisional: list[tuple[RetainedRule, Occurrence, NormalizedSentence, int, int]] = []
    outside = 0
    witness_failures = 0
    for rule in rules:
        proposal = rule.prefix, rule.suffix
        for occurrence in rule.occurrences:
            sentence = sentences[occurrence.sentence_ordinal]
            raw_start = sentence.raw_boundaries[occurrence.middle_start]
            raw_end = sentence.raw_boundaries[occurrence.middle_end]
            normalized_exact = match(proposal, sentence.tokens) == (
                occurrence.middle_start,
                occurrence.middle_end,
            )
            raw_exact = (
                0 <= raw_start < raw_end <= len(sentence.raw_record_tokens)
                and len(sentence.raw_record_tokens[raw_start:raw_end])
                == raw_end - raw_start
            )
            if not normalized_exact or not raw_exact:
                witness_failures += 1
                continue
            if raw_end > TOKEN_SLOTS:
                outside += 1
                continue
            provisional.append((rule, occurrence, sentence, raw_start, raw_end))

    key_counts = Counter(
        (sentence.document_id, rule.rule_id)
        for rule, _, sentence, _, _ in provisional
    )
    edges: list[MaterializedEdge] = []
    for rule, occurrence, sentence, raw_start, raw_end in provisional:
        unique = key_counts[(sentence.document_id, rule.rule_id)] == 1
        flags = (1 if unique else 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        edges.append(
            MaterializedEdge(
                document_index=sentence.document_index,
                document_id=sentence.document_id,
                title=sentence.title,
                sentence_index=sentence.sentence_index,
                rule_id=rule.rule_id,
                raw_start=raw_start,
                raw_length=raw_end - raw_start,
                flags=flags,
                normalized_middle=sentence.tokens[
                    occurrence.middle_start : occurrence.middle_end
                ],
                raw_middle=sentence.raw_record_tokens[raw_start:raw_end],
            )
        )
    edges.sort(
        key=lambda edge: (
            edge.document_index,
            edge.rule_id,
            edge.raw_start,
            edge.raw_length,
            edge.sentence_index,
        )
    )
    return tuple(edges), {
        "pointers_outside_record": outside,
        "witness_failures": witness_failures,
        "duplicate_document_rule_keys": sum(count > 1 for count in key_counts.values()),
    }


def compile_sentences(sentences: Sequence[NormalizedSentence]) -> Compilation:
    ordered = tuple(
        sorted(
            sentences,
            key=lambda sentence: (
                sentence.document_id,
                sentence.sentence_index,
                sentence.tokens,
            ),
        )
    )
    proposals, repeats, comparisons = enumerate_proposals(ordered)
    indexed = indexed_occurrences(ordered, proposals)
    rules, eligible, conflict_metrics = select_rules(ordered, proposals, indexed)
    edges, edge_metrics = materialize_edges(ordered, rules)
    documents_with_edges = Counter(edge.document_id for edge in edges)
    return Compilation(
        rules=rules,
        edges=edges,
        proposals=tuple(sorted(proposals, key=proposal_sort_key)),
        eligible=eligible,
        exact_repeat_groups=repeats,
        metrics={
            "sentences": len(ordered),
            "pair_comparisons": comparisons,
            "proposals": len(proposals),
            "eligible_proposals": len(eligible),
            "retained_rules": len(rules),
            "retained_occurrences": sum(len(rule.occurrences) for rule in rules),
            "materialized_edges": len(edges),
            "exact_repeat_groups": len(repeats),
            "documents_with_edges": len(documents_with_edges),
            "documents_above_24_edges": sum(value > 24 for value in documents_with_edges.values()),
            **conflict_metrics,
            **edge_metrics,
        },
    )


def sentence_char_spans(text: str, serialized_offset: int) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(text):
        if character in ".?!":
            if index + 1 > start:
                spans.append((serialized_offset + start, serialized_offset + index + 1))
            start = index + 1
    if start < len(text):
        spans.append((serialized_offset + start, serialized_offset + len(text)))
    return tuple(span for span in spans if span[1] > span[0])


def nonoverlapping_char_occurrences(
    haystack: str, needle: str, start: int, end: int
) -> tuple[tuple[int, int], ...]:
    if not needle:
        return ()
    output: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        found = haystack.find(needle, cursor, end)
        if found < 0:
            break
        output.append((found, found + len(needle)))
        cursor = found + len(needle)
    return tuple(output)


def observe_encoded_document(
    *,
    document_index: int,
    document_id: str,
    title: str,
    text: str,
    input_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
) -> tuple[tuple[NormalizedSentence, ...], dict[str, int]]:
    serialized = title + "\n" + text
    ids = tuple(int(value) for value in input_ids)
    token_offsets = tuple((int(value[0]), int(value[1])) for value in offsets)
    if len(ids) != len(token_offsets):
        raise RuntimeError("token/offset length mismatch")
    if any(not 0 <= value < 49_152 for value in ids):
        raise RuntimeError("tokenizer ID outside frozen vocabulary")

    diagnostics = {
        "raw_sentences": 0,
        "eligible_sentences": 0,
        "sentence_boundary_rejections": 0,
        "title_occurrences": 0,
        "title_boundary_rejections": 0,
        "title_replacements": 0,
    }
    sentences: list[NormalizedSentence] = []
    for sentence_index, (char_start, char_end) in enumerate(
        sentence_char_spans(text, len(title) + 1)
    ):
        diagnostics["raw_sentences"] += 1
        intersecting = [
            index
            for index, (start, end) in enumerate(token_offsets)
            if start < char_end and end > char_start and end > start
        ]
        crossing = any(
            not (char_start <= token_offsets[index][0] and token_offsets[index][1] <= char_end)
            for index in intersecting
        )
        if crossing or not intersecting or any(
            right != left + 1 for left, right in zip(intersecting, intersecting[1:])
        ):
            diagnostics["sentence_boundary_rejections"] += 1
            continue

        replacement_by_start: dict[int, int] = {}
        for occurrence_start, occurrence_end in nonoverlapping_char_occurrences(
            serialized, title, char_start, char_end
        ):
            diagnostics["title_occurrences"] += 1
            covered = [
                index
                for index in intersecting
                if token_offsets[index][0] >= occurrence_start
                and token_offsets[index][1] <= occurrence_end
                and token_offsets[index][1] > token_offsets[index][0]
            ]
            exact = (
                bool(covered)
                and token_offsets[covered[0]][0] == occurrence_start
                and token_offsets[covered[-1]][1] == occurrence_end
                and all(
                    right == left + 1 for left, right in zip(covered, covered[1:])
                )
                and not any(
                    token_offsets[index][0] < occurrence_end
                    and token_offsets[index][1] > occurrence_start
                    and index not in covered
                    for index in intersecting
                )
            )
            if not exact:
                diagnostics["title_boundary_rejections"] += 1
                continue
            replacement_by_start[covered[0]] = covered[-1] + 1
            diagnostics["title_replacements"] += 1

        normalized: list[int] = []
        boundaries = [intersecting[0]]
        cursor = intersecting[0]
        stop = intersecting[-1] + 1
        while cursor < stop:
            replacement_end = replacement_by_start.get(cursor)
            if replacement_end is not None:
                normalized.append(SELF_TOKEN)
                cursor = replacement_end
            else:
                normalized.append(ids[cursor])
                cursor += 1
            boundaries.append(cursor)
        if not normalized:
            diagnostics["sentence_boundary_rejections"] += 1
            continue
        sentences.append(
            NormalizedSentence(
                document_index=document_index,
                document_id=document_id,
                title=title,
                sentence_index=sentence_index,
                tokens=tuple(normalized),
                raw_boundaries=tuple(boundaries),
                raw_record_tokens=ids,
                serialized_char_start=char_start,
                serialized_char_end=char_end,
            )
        )
        diagnostics["eligible_sentences"] += 1
    return tuple(sentences), diagnostics


def _synthetic_sentence(document: int, tokens: Sequence[int], sentence_index: int = 0) -> NormalizedSentence:
    values = tuple(tokens)
    return NormalizedSentence(
        document_index=document,
        document_id=f"doc-{document:04d}",
        title=f"Title {document}",
        sentence_index=sentence_index,
        tokens=values,
        raw_boundaries=tuple(range(len(values) + 1)),
        raw_record_tokens=values,
    )


def exhaustive_anti_unifier_gate() -> dict[str, int]:
    sequences = [
        sequence
        for length in range(1, 7)
        for sequence in itertools.product(range(3), repeat=length)
    ]
    checked = 0
    for left_index, left in enumerate(sequences):
        for right in sequences[left_index + 1 :]:
            direct = anti_unify(left, right)
            reference = brute_force_anti_unify(left, right)
            if direct != reference:
                raise AssertionError(
                    f"anti-unifier mismatch: {left}, {right}, {direct}, {reference}"
                )
            checked += 1
    return {"sequences": len(sequences), "unordered_pairs": checked}


def _rule_sentences(prefix: Sequence[int], suffix: Sequence[int], values: Sequence[Sequence[int]], start: int) -> list[NormalizedSentence]:
    return [
        _synthetic_sentence(start + offset, tuple(prefix) + tuple(value) + tuple(suffix))
        for offset, value in enumerate(values)
    ]


def run_microbenchmarks() -> dict[str, Any]:
    anti_gate = exhaustive_anti_unifier_gate()
    prefix_a, suffix_a = (1, 2, 3, 4, 5), (6, 7, 8, 9, 10)
    prefix_b, suffix_b = (11, 12, 13, 14, 15), (16, 17, 18, 19, 20)
    world1 = _rule_sentences(prefix_a, suffix_a, ((21,), (22,), (23,)), 0)
    world1 += _rule_sentences(prefix_b, suffix_b, ((31,), (32,), (33,)), 3)
    compiled1 = compile_sentences(world1)
    if len(compiled1.rules) != 2 or compiled1.metrics["witness_failures"] != 0:
        raise AssertionError("world 1 exact recovery failed")

    singleton = compile_sentences([_synthetic_sentence(0, prefix_a + (99,) + suffix_a)])
    if singleton.rules:
        raise AssertionError("world 2 retained a one-shot rule")

    nondiverse = _rule_sentences(prefix_a, suffix_a, ((99, 21), (99, 22), (99, 23)), 0)
    compiled3 = compile_sentences(nondiverse)
    planted = (tuple(prefix_a), tuple(suffix_a))
    if planted in compiled3.proposals:
        raise AssertionError("world 3 falsely recovered a non-identifiable boundary")

    outer_left, outer_right = (40, 41), (48, 49)
    world4 = _rule_sentences(outer_left + (42, 43), (46, 47) + outer_right, ((60,), (61,), (62,)), 0)
    world4 += _rule_sentences(outer_left + (44, 45), (50, 51) + outer_right, ((70,), (71,), (72,)), 3)
    compiled4 = compile_sentences(world4)
    if compiled4.rules or compiled4.metrics["conflicted_sentences"] != 6:
        raise AssertionError("world 4 did not reject broad/narrow conflicts")

    merged_world = _rule_sentences(prefix_a, suffix_a, ((81,), (82,), (83,), (84,)), 0)
    merged_a = compile_sentences(merged_world)
    merged_b = compile_sentences(list(reversed(merged_world)))
    if [(r.prefix, r.suffix, len(r.occurrences)) for r in merged_a.rules] != [
        (r.prefix, r.suffix, len(r.occurrences)) for r in merged_b.rules
    ]:
        raise AssertionError("world 5 observable merge changed with latent labeling/order")

    sentence6 = _synthetic_sentence(0, (1, 2, 3, 4, 5, 6, 7, 8, 9))
    companions = (
        _synthetic_sentence(1, (1, 2, 30, 4, 5, 6, 7, 8, 9)),
        _synthetic_sentence(2, (1, 2, 31, 4, 5, 6, 7, 8, 9)),
        _synthetic_sentence(3, (1, 2, 3, 4, 5, 60, 7, 8, 9)),
        _synthetic_sentence(4, (1, 2, 3, 4, 5, 61, 7, 8, 9)),
    )
    compiled6 = compile_sentences((sentence6, *companions))
    if compiled6.metrics["conflicted_sentences"] == 0:
        raise AssertionError("world 6 failed to expose a two-hole conflict")

    repeated = tuple(_synthetic_sentence(index, (1, 2, 3, 4)) for index in range(3))
    compiled7 = compile_sentences(repeated)
    if compiled7.rules or len(compiled7.exact_repeat_groups) != 1:
        raise AssertionError("world 7 exact-repeat control failed")

    serialized = "Å Å\nÅ Å is here!"
    ids = (100, 101, 102, 103, 104, 105, 106)
    offsets = ((0, 1), (2, 3), (3, 4), (4, 5), (6, 7), (8, 10), (10, 11))
    observed, observer_metrics = observe_encoded_document(
        document_index=0,
        document_id="unicode",
        title="Å Å",
        text="Å Å is here!",
        input_ids=ids,
        offsets=offsets,
    )
    if len(observed) != 1 or observed[0].tokens[0] != SELF_TOKEN:
        raise AssertionError("world 8 title/Unicode observer failed")
    crossing, crossing_metrics = observe_encoded_document(
        document_index=1,
        document_id="crossing",
        title="T",
        text="A! B",
        input_ids=(1, 2, 3),
        offsets=((0, 1), (2, 4), (3, 6)),
    )
    if crossing or crossing_metrics["sentence_boundary_rejections"] == 0:
        raise AssertionError("world 8 crossing-boundary rejection failed")

    if compile_sentences(world1).rules != compile_sentences(tuple(reversed(world1))).rules:
        raise AssertionError("world 9 document order invariance failed")
    proposals1, _, _ = enumerate_proposals(tuple(world1))
    indexed = indexed_occurrences(tuple(world1), proposals1)
    dense = dense_occurrences(tuple(world1), reversed(sorted(proposals1, key=proposal_sort_key)))
    if indexed != dense:
        raise AssertionError("world 9 independent occurrence matchers disagree")

    image, layout = build_candidate_image(
        documents=tuple((sentence.document_id, sentence.raw_record_tokens) for sentence in world1),
        rules=compiled1.rules,
        edges=compiled1.edges,
    )
    verify_candidate_image(image, layout, compiled1.rules, compiled1.edges)
    first_edge = compiled1.edges[0]
    corrupt_witness = replace(
        first_edge,
        raw_middle=(first_edge.raw_middle[0] ^ 1,) + first_edge.raw_middle[1:],
    )
    try:
        verify_candidate_image(
            image,
            layout,
            compiled1.rules,
            (corrupt_witness, *compiled1.edges[1:]),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("world 10 corrupt raw witness was not detected")
    corrupt = bytearray(image)
    corrupt[layout["anchor_offset"]] ^= 1
    corrupt[32:64] = b"\0" * 32
    corrupt[32:64] = hashlib.sha256(corrupt).digest()
    try:
        verify_candidate_image(bytes(corrupt), layout, compiled1.rules, compiled1.edges)
    except RuntimeError:
        pass
    else:
        raise AssertionError("world 10 corruption was not detected")

    length127 = (1,) + tuple(range(2, 128))
    length128 = length127 + (128,)
    if len(length127) != 127 or len(length128) != 128:
        raise AssertionError("world 8 record-boundary fixtures malformed")
    long_proposal = ((1,), (9,))
    long_sentence = _synthetic_sentence(20, (1,) + (2,) * 128 + (9,))
    if indexed_occurrences((long_sentence,), (long_proposal,)) or dense_occurrences(
        (long_sentence,), (long_proposal,)
    ):
        raise AssertionError("world 8 admitted a 128-token hole")

    return {
        "status": "pass",
        "anti_unifier": anti_gate,
        "worlds": 10,
        "world1_rules": len(compiled1.rules),
        "world4_conflicted_sentences": compiled4.metrics["conflicted_sentences"],
        "world5_latent_partition": "non-identifiable",
        "world8_observer": observer_metrics,
        "world10_image_bytes": len(image),
    }


def align64(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def pack_edge(edge: MaterializedEdge) -> int:
    if not (
        0 <= edge.rule_id < 4_096
        and 0 <= edge.raw_start < 128
        and 1 <= edge.raw_length <= 128
        and edge.raw_start + edge.raw_length <= 128
        and 0 <= edge.flags < 64
    ):
        raise RuntimeError(f"edge outside packed domain: {edge}")
    return (
        edge.rule_id
        | (edge.raw_start << 12)
        | ((edge.raw_length - 1) << 19)
        | (edge.flags << 26)
    )


def build_candidate_image(
    *,
    documents: Sequence[tuple[str, Sequence[int]]],
    rules: Sequence[RetainedRule],
    edges: Sequence[MaterializedEdge],
) -> tuple[bytes, dict[str, int]]:
    if len(documents) > 65_535 or len(rules) > MAX_RULES:
        raise RuntimeError("candidate image count outside header domain")
    anchors = tuple(token for rule in rules for token in (rule.prefix + rule.suffix))
    sections: dict[str, int] = {"header_offset": 0}
    cursor = 64
    for name, size in (
        ("raw_token_offset", len(documents) * TOKEN_SLOTS * 2),
        ("raw_length_offset", len(documents) * 2),
        ("rule_offset", len(rules) * 16),
        ("anchor_offset", len(anchors) * 2),
        ("document_offset", len(documents) * 8),
        ("edge_offset", len(edges) * 4),
    ):
        cursor = align64(cursor)
        sections[name] = cursor
        cursor += size
    total = align64(cursor)
    image = bytearray(total)

    for document_index, (_, tokens) in enumerate(documents):
        values = tuple(int(value) for value in tokens[:TOKEN_SLOTS])
        if any(not 0 <= value < 65_536 for value in values):
            raise RuntimeError("raw token outside uint16")
        padded = values + (0,) * (TOKEN_SLOTS - len(values))
        struct.pack_into(
            f"<{TOKEN_SLOTS}H",
            image,
            sections["raw_token_offset"] + document_index * TOKEN_SLOTS * 2,
            *padded,
        )
        struct.pack_into(
            "<H",
            image,
            sections["raw_length_offset"] + document_index * 2,
            len(values),
        )

    anchor_cursor = 0
    for rule_index, rule in enumerate(rules):
        if not (
            0 < len(rule.prefix) <= MAX_ANCHOR_TOKENS
            and 0 < len(rule.suffix) <= MAX_ANCHOR_TOKENS
            and len(rule.occurrences) <= MAX_OCCURRENCES
        ):
            raise RuntimeError("rule outside frozen metadata format")
        struct.pack_into(
            "<IBBHII",
            image,
            sections["rule_offset"] + rule_index * 16,
            anchor_cursor,
            len(rule.prefix),
            len(rule.suffix),
            len(rule.occurrences),
            rule.crc32,
            0,
        )
        anchor_cursor += len(rule.prefix) + len(rule.suffix)
    if anchors:
        struct.pack_into(
            f"<{len(anchors)}H", image, sections["anchor_offset"], *anchors
        )

    by_document: dict[int, list[MaterializedEdge]] = defaultdict(list)
    for edge in edges:
        by_document[edge.document_index].append(edge)
    ordered_edges: list[MaterializedEdge] = []
    for document_index in range(len(documents)):
        values = sorted(
            by_document.get(document_index, ()),
            key=lambda edge: (edge.rule_id, edge.raw_start, edge.raw_length),
        )
        if len(values) > 65_535:
            raise RuntimeError("document edge count outside uint16")
        struct.pack_into(
            "<IHH",
            image,
            sections["document_offset"] + document_index * 8,
            len(ordered_edges),
            len(values),
            0,
        )
        ordered_edges.extend(values)
    if len(ordered_edges) != len(edges):
        raise RuntimeError("edge references unknown document index")
    for edge_index, edge in enumerate(ordered_edges):
        struct.pack_into(
            "<I",
            image,
            sections["edge_offset"] + edge_index * 4,
            pack_edge(edge),
        )

    struct.pack_into(
        "<8sHHHHII8s32s",
        image,
        0,
        IMAGE_MAGIC,
        1,
        0,
        len(documents),
        len(rules),
        len(edges),
        len(anchors),
        b"\0" * 8,
        b"\0" * 32,
    )
    integrity = hashlib.sha256(image).digest()
    image[32:64] = integrity
    sections["total_bytes"] = total
    sections["integrity_offset"] = 32
    return bytes(image), sections


def verify_candidate_image(
    image: bytes,
    layout: dict[str, int],
    rules: Sequence[RetainedRule],
    edges: Sequence[MaterializedEdge],
) -> None:
    if len(image) != layout["total_bytes"]:
        raise RuntimeError("candidate image length mismatch")
    mutable = bytearray(image)
    stored_integrity = bytes(mutable[32:64])
    mutable[32:64] = b"\0" * 32
    if hashlib.sha256(mutable).digest() != stored_integrity:
        raise RuntimeError("candidate image SHA-256 mismatch")
    header = struct.unpack_from("<8sHHHHII8s32s", image, 0)
    if header[0] != IMAGE_MAGIC or header[1] != 1 or header[2] != 0 or header[7] != b"\0" * 8:
        raise RuntimeError("candidate image header mismatch")
    anchor_cursor = 0
    for rule_index, expected in enumerate(rules):
        record = struct.unpack_from("<IBBHII", image, layout["rule_offset"] + 16 * rule_index)
        offset, prefix_length, suffix_length, occurrence_count, checksum, reserved = record
        if (
            offset != anchor_cursor
            or prefix_length != len(expected.prefix)
            or suffix_length != len(expected.suffix)
            or occurrence_count != len(expected.occurrences)
            or checksum != expected.crc32
            or reserved != 0
        ):
            raise RuntimeError("candidate rule record mismatch")
        width = prefix_length + suffix_length
        decoded = struct.unpack_from(
            f"<{width}H", image, layout["anchor_offset"] + 2 * offset
        )
        proposal = expected.prefix, expected.suffix
        if decoded != expected.prefix + expected.suffix or rule_crc32(proposal) != checksum:
            raise RuntimeError("candidate rule checksum mismatch")
        anchor_cursor += width
    by_document: dict[int, list[MaterializedEdge]] = defaultdict(list)
    for edge in edges:
        by_document[edge.document_index].append(edge)
    edge_ordinal = 0
    for document_index in range(header[3]):
        edge_offset, edge_count, document_flags = struct.unpack_from(
            "<IHH", image, layout["document_offset"] + document_index * 8
        )
        if edge_offset != edge_ordinal or document_flags != 0:
            raise RuntimeError("candidate document-directory mismatch")
        expected_edges = sorted(
            by_document.get(document_index, ()),
            key=lambda edge: (edge.rule_id, edge.raw_start, edge.raw_length),
        )
        if edge_count != len(expected_edges):
            raise RuntimeError("candidate document edge count mismatch")
        raw_length = struct.unpack_from(
            "<H", image, layout["raw_length_offset"] + document_index * 2
        )[0]
        for expected in expected_edges:
            packed = struct.unpack_from(
                "<I", image, layout["edge_offset"] + edge_ordinal * 4
            )[0]
            if packed != pack_edge(expected):
                raise RuntimeError("candidate packed-edge mismatch")
            if expected.raw_start + expected.raw_length > raw_length:
                raise RuntimeError("candidate edge exceeds stored raw length")
            pointed = struct.unpack_from(
                f"<{expected.raw_length}H",
                image,
                layout["raw_token_offset"]
                + (document_index * TOKEN_SLOTS + expected.raw_start) * 2,
            )
            if pointed != expected.raw_middle:
                raise RuntimeError("candidate raw witness mismatch")
            edge_ordinal += 1
    if edge_ordinal != len(edges):
        raise RuntimeError("candidate edge count mismatch")


def resolve_tokenizer_artifacts() -> dict[str, dict[str, Any]]:
    from huggingface_hub import try_to_load_from_cache

    artifacts: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_ARTIFACTS:
        cached = try_to_load_from_cache(TOKENIZER, name, revision=TOKENIZER_REVISION)
        if not isinstance(cached, str):
            raise RuntimeError(f"tokenizer artifact not cached: {name}")
        path = Path(cached).absolute()
        if not path.is_file() or TOKENIZER_REVISION not in path.parts:
            raise RuntimeError(f"tokenizer artifact provenance mismatch: {path}")
        artifacts[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return artifacts


def load_natural_observations() -> tuple[
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[NormalizedSentence, ...],
    dict[str, Any],
]:
    if sha256_file(CORPUS) != EXPECTED_CORPUS_SHA256:
        raise RuntimeError("raw corpus checksum mismatch")
    artifacts = resolve_tokenizer_artifacts()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        revision=TOKENIZER_REVISION,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast or int(tokenizer.vocab_size) != 49_152:
        raise RuntimeError("pinned tokenizer shape mismatch")

    documents: list[tuple[str, tuple[int, ...]]] = []
    sentences: list[NormalizedSentence] = []
    totals: Counter[str] = Counter()
    seen_document_ids: set[str] = set()
    seen_titles: set[str] = set()
    with CORPUS.open(encoding="utf-8") as handle:
        for document_index, line in enumerate(handle):
            row = json.loads(line)
            if set(row) != {"document_id", "title", "text"}:
                raise RuntimeError(f"forbidden or missing corpus field at row {document_index}")
            document_id, title, text = row["document_id"], row["title"], row["text"]
            if not all(isinstance(value, str) for value in (document_id, title, text)):
                raise RuntimeError(f"non-string corpus field at row {document_index}")
            if document_id in seen_document_ids or title in seen_titles:
                raise RuntimeError("candidate document IDs and titles must be unique")
            seen_document_ids.add(document_id)
            seen_titles.add(title)
            serialized = title + "\n" + text
            encoded = tokenizer(
                serialized,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = tuple(int(value) for value in encoded["input_ids"])
            observed, diagnostics = observe_encoded_document(
                document_index=document_index,
                document_id=document_id,
                title=title,
                text=text,
                input_ids=ids,
                offsets=encoded["offset_mapping"],
            )
            documents.append((document_id, ids))
            sentences.extend(observed)
            totals.update(diagnostics)
            totals["full_bpe_tokens"] += len(ids)
            totals["stored_bpe_tokens"] += min(len(ids), TOKEN_SLOTS)
    return tuple(documents), tuple(sentences), {
        "documents": len(documents),
        **dict(totals),
        "tokenizer": {
            "name": TOKENIZER,
            "revision": TOKENIZER_REVISION,
            "artifacts": artifacts,
        },
    }


def compilation_json(compilation: Compilation) -> dict[str, Any]:
    return {
        "rules": [
            {
                "rule_id": rule.rule_id,
                "prefix": list(rule.prefix),
                "suffix": list(rule.suffix),
                "gain_bits": rule.gain_bits,
                "occurrence_count": len(rule.occurrences),
                "crc32": rule.crc32,
            }
            for rule in compilation.rules
        ],
        "edges": [
            {
                **asdict(edge),
                "normalized_middle": list(edge.normalized_middle),
                "raw_middle": list(edge.raw_middle),
            }
            for edge in compilation.edges
        ],
        "metrics": compilation.metrics,
        "exact_repeat_groups": [list(group) for group in compilation.exact_repeat_groups],
    }


def run_natural() -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    microbench = run_microbenchmarks()
    documents, sentences, observer = load_natural_observations()
    compilation = compile_sentences(sentences)
    image, layout = build_candidate_image(
        documents=documents,
        rules=compilation.rules,
        edges=compilation.edges,
    )
    verify_candidate_image(image, layout, compilation.rules, compilation.edges)
    IMAGE_OUTPUT.write_bytes(image)
    documents_with_edges = Counter(edge.document_id for edge in compilation.edges)
    edge_bounded_fraction = (
        sum(value <= 24 for value in documents_with_edges.values())
        / len(documents_with_edges)
        if documents_with_edges
        else 0.0
    )
    phase_a_gates = {
        "microbench": microbench["status"] == "pass",
        "retained_rule_domain": 1 <= len(compilation.rules) <= MAX_RULES,
        "zero_witness_failures": compilation.metrics["witness_failures"] == 0,
        "edge_bound_95pct": edge_bounded_fraction >= 0.95,
        "state_byte_cap": len(image) <= STATE_BYTE_CAP,
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
    }
    payload = {
        "schema": "reversible-slot-grammar-plane-t33-phase-a-v1",
        "status": "pass" if all(phase_a_gates.values()) else "fail",
        "claim_boundary": (
            "Raw-only structural census. It does not establish semantic purity, "
            "answer sufficiency, natural query mapping, model quality, or novelty."
        ),
        "inputs": {
            "corpus": {"path": str(CORPUS), "sha256": sha256_file(CORPUS)},
            "preregistration": {
                "path": str(PREREGISTRATION),
                "sha256": sha256_file(PREREGISTRATION),
            },
            "phase_a_source_sha256": sha256_file(Path(__file__)),
        },
        "observer": observer,
        "microbench": microbench,
        "compiler": compilation_json(compilation),
        "image": {
            "path": str(IMAGE_OUTPUT),
            "bytes": len(image),
            "sha256": sha256_file(IMAGE_OUTPUT),
            "layout": layout,
            "edge_bounded_document_fraction": edge_bounded_fraction,
            "mapper_reader_reserve_bytes": 1_105_920 - len(image),
        },
        "phase_a_gates": phase_a_gates,
        "phase_b_admitted": all(phase_a_gates.values()),
        "runtime": {
            "cpu_seconds": time.process_time() - started_cpu,
            "wall_seconds": time.perf_counter() - started_wall,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    payload["object_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbench-only", action="store_true")
    parser.add_argument("--run-natural", action="store_true")
    arguments = parser.parse_args()
    if arguments.run_natural:
        result = run_natural()
    else:
        result = run_microbenchmarks()
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
