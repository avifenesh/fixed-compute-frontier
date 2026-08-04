#!/usr/bin/env python3
"""Privileged T33 postmortem: distinguish selector failure from algebra failure."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import reversible_slot_grammar_plane_t33_coverage as coverage
from experiments import reversible_slot_grammar_plane_t33_stage0 as stage0


OUTPUT = ROOT / "results/reversible-slot-grammar-plane-t33-postmortem.json"


def make_rules(
    proposals: Iterable[stage0.Proposal],
    occurrences: dict[stage0.Proposal, tuple[stage0.Occurrence, ...]],
) -> tuple[stage0.RetainedRule, ...]:
    ordered = sorted(set(proposals), key=stage0.proposal_sort_key)
    if len(ordered) > stage0.MAX_RULES:
        raise RuntimeError("diagnostic proposal set exceeds packed rule domain")
    return tuple(
        stage0.RetainedRule(
            rule_id=rule_id,
            prefix=proposal[0],
            suffix=proposal[1],
            gain_bits=stage0.code_gain_bits(proposal, len(occurrences[proposal])),
            occurrences=occurrences[proposal],
            crc32=stage0.rule_crc32(proposal),
        )
        for rule_id, proposal in enumerate(ordered)
    )


def evaluate_variant(
    name: str,
    proposals: Iterable[stage0.Proposal],
    occurrences: dict[stage0.Proposal, tuple[stage0.Occurrence, ...]],
    sentences: tuple[stage0.NormalizedSentence, ...],
    documents: tuple[tuple[str, tuple[int, ...]], ...],
    support_items: tuple[coverage.SupportItem, ...],
) -> dict[str, Any]:
    rules = make_rules(proposals, occurrences)
    edges, edge_metrics = stage0.materialize_edges(sentences, rules)
    image, layout = stage0.build_candidate_image(
        documents=documents, rules=rules, edges=edges
    )
    stage0.verify_candidate_image(image, layout, rules, edges)
    scored = coverage.score_coverage(
        support_items,
        tuple(
            coverage.Edge(
                title=edge.title,
                sentence_index=edge.sentence_index,
                rule_id=edge.rule_id,
                raw_start=edge.raw_start,
                raw_length=edge.raw_length,
                unique=bool(edge.flags & 1),
            )
            for edge in edges
        ),
    )
    per_document = Counter(edge.document_id for edge in edges)
    bounded = (
        sum(count <= 24 for count in per_document.values()) / len(per_document)
        if per_document
        else 0.0
    )
    return {
        "name": name,
        "rules": len(rules),
        "logical_occurrences": sum(len(rule.occurrences) for rule in rules),
        "materialized_edges": len(edges),
        "documents_with_edges": len(per_document),
        "documents_at_most_24_edge_fraction": bounded,
        "image_bytes": len(image),
        "state_byte_cap_pass": len(image) <= stage0.STATE_BYTE_CAP,
        "edge_bound_pass": bounded >= 0.95,
        "edge_metrics": edge_metrics,
        "coverage": scored,
        "layout": layout,
    }


def direct_sentence_index_control(
    sentences: tuple[stage0.NormalizedSentence, ...],
    documents: tuple[tuple[str, tuple[int, ...]], ...],
    support_items: tuple[coverage.SupportItem, ...],
) -> dict[str, Any]:
    served = tuple(
        sentence
        for sentence in sentences
        if sentence.raw_boundaries[-1] <= stage0.TOKEN_SLOTS
    )
    control_edges = tuple(
        coverage.Edge(
            title=sentence.title,
            sentence_index=sentence.sentence_index,
            rule_id=0,
            raw_start=sentence.raw_boundaries[0],
            raw_length=sentence.raw_boundaries[-1] - sentence.raw_boundaries[0],
            unique=True,
        )
        for sentence in served
    )
    per_document = Counter(sentence.document_id for sentence in served)
    cursor = 64
    layout: dict[str, int] = {}
    for name, size in (
        ("raw_token_offset", len(documents) * stage0.TOKEN_SLOTS * 2),
        ("raw_length_offset", len(documents) * 2),
        ("document_offset", len(documents) * 8),
        ("sentence_span_offset", len(served) * 4),
    ):
        cursor = stage0.align64(cursor)
        layout[name] = cursor
        cursor += size
    total = stage0.align64(cursor)
    bounded = (
        sum(count <= 24 for count in per_document.values()) / len(per_document)
        if per_document
        else 0.0
    )
    return {
        "name": "direct_sentence_index",
        "served_sentence_spans": len(served),
        "documents_with_spans": len(per_document),
        "documents_at_most_24_span_fraction": bounded,
        "image_bytes": total,
        "state_byte_cap_pass": total <= stage0.STATE_BYTE_CAP,
        "coverage": coverage.score_coverage(support_items, control_edges),
        "layout": {**layout, "total_bytes": total},
        "claim_boundary": (
            "Stores one raw start/length per compiler sentence and no grammar rule. "
            "This is the mandatory simpler structural control, not a model result."
        ),
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    phase_a = json.loads(coverage.PHASE_A.read_text())
    coverage.verify_phase_a_object(phase_a)
    documents, raw_sentences, observer = stage0.load_natural_observations()
    sentences = tuple(
        sorted(
            raw_sentences,
            key=lambda sentence: (
                sentence.document_id,
                sentence.sentence_index,
                sentence.tokens,
            ),
        )
    )
    proposals, _, comparisons = stage0.enumerate_proposals(sentences)
    occurrences = stage0.indexed_occurrences(sentences, proposals)

    positive = tuple(
        proposal
        for proposal in proposals
        if len(occurrences.get(proposal, ())) >= 2
        and len(
            {
                sentences[value.sentence_ordinal].document_id
                for value in occurrences[proposal]
            }
        )
        >= 2
        and stage0.code_gain_bits(proposal, len(occurrences[proposal])) > 0
    )
    recurring = tuple(
        proposal
        for proposal in proposals
        if len(occurrences.get(proposal, ())) >= 2
        and len(
            {
                sentences[value.sentence_ordinal].document_id
                for value in occurrences[proposal]
            }
        )
        >= 2
    )

    candidates = coverage.load_candidate_documents()
    sealed = coverage.load_sealed_identities()
    support_items, mapping = coverage.map_support_items(sealed, candidates)
    variants = {
        "all_positive_ambiguous": evaluate_variant(
            "all_positive_ambiguous",
            positive,
            occurrences,
            sentences,
            documents,
            support_items,
        ),
        "all_recurring_ignore_mdl": evaluate_variant(
            "all_recurring_ignore_mdl",
            recurring,
            occurrences,
            sentences,
            documents,
            support_items,
        ),
        "direct_sentence_index": direct_sentence_index_control(
            sentences, documents, support_items
        ),
    }
    payload = {
        "schema": "reversible-slot-grammar-plane-t33-privileged-postmortem-v1",
        "status": "diagnostic-only",
        "claim_boundary": (
            "Support locations were already unsealed. These ceilings may diagnose the "
            "closed lane but may not select, tune, or promote a T33 variant."
        ),
        "purpose": (
            "If ambiguity rejection alone caused the miss, all positive rules should "
            "recover structural coverage within the same byte/edge envelope. If every "
            "recurring proposal still misses or exceeds the envelope, the observable "
            "one-hole algebra is insufficient."
        ),
        "inputs": {
            "phase_a_object_sha256": phase_a["object_sha256"],
            "coverage_object_sha256": json.loads(coverage.OUTPUT.read_text())[
                "object_sha256"
            ],
            "source_sha256": stage0.sha256_file(Path(__file__)),
        },
        "observer_documents": observer["documents"],
        "sentences": len(sentences),
        "pair_comparisons": comparisons,
        "raw_proposals": len(proposals),
        "positive_proposals": len(positive),
        "recurring_proposals": len(recurring),
        "mapping": mapping,
        "variants": variants,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    payload["object_sha256"] = hashlib.sha256(
        stage0.canonical_json_bytes(payload)
    ).hexdigest()
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
