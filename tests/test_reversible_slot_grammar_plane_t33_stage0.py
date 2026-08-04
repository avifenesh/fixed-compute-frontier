from dataclasses import replace

import pytest

from experiments.reversible_slot_grammar_plane_t33_stage0 import (
    SELF_TOKEN,
    NormalizedSentence,
    anti_unify,
    brute_force_anti_unify,
    build_candidate_image,
    code_gain_bits,
    compile_sentences,
    dense_occurrences,
    indexed_occurrences,
    observe_encoded_document,
    run_microbenchmarks,
    verify_candidate_image,
)


def sentence(document: int, tokens: tuple[int, ...]) -> NormalizedSentence:
    return NormalizedSentence(
        document_index=document,
        document_id=f"doc-{document}",
        title=f"Title {document}",
        sentence_index=0,
        tokens=tokens,
        raw_boundaries=tuple(range(len(tokens) + 1)),
        raw_record_tokens=tokens,
    )


def test_joint_boundary_maximization_handles_nested_repetition():
    left = (0, 0, 0)
    right = (0, 0, 0, 0)
    assert anti_unify(left, right) == ((0,), (0,))
    assert anti_unify(left, right) == brute_force_anti_unify(left, right)


def test_code_gain_is_exactly_the_frozen_two_part_formula():
    proposal = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    assert code_gain_bits(proposal, 3) == 16 * 2 * 10 - 60 - 3 * 19


def test_singleton_cannot_create_a_rule():
    compiled = compile_sentences((sentence(0, (1, 2, 3, 4, 5)),))
    assert compiled.rules == ()


def test_eligible_overlapping_proposals_are_all_rejected():
    rows = (
        sentence(0, (1, 2, 3, 4, 5, 6, 7, 8, 9)),
        sentence(1, (1, 2, 30, 4, 5, 6, 7, 8, 9)),
        sentence(2, (1, 2, 31, 4, 5, 6, 7, 8, 9)),
        sentence(3, (1, 2, 3, 4, 5, 60, 7, 8, 9)),
        sentence(4, (1, 2, 3, 4, 5, 61, 7, 8, 9)),
    )
    compiled = compile_sentences(rows)
    assert compiled.metrics["conflicted_sentences"] > 0
    assert all(
        len(
            {
                rule.rule_id
                for rule in compiled.rules
                if any(
                    rows[occurrence.sentence_ordinal].document_id == row.document_id
                    for occurrence in rule.occurrences
                )
            }
        )
        <= 1
        for row in rows
    )


def test_exact_title_normalization_retains_raw_boundary_map():
    observed, diagnostics = observe_encoded_document(
        document_index=0,
        document_id="unicode",
        title="Å Å",
        text="Å Å is here!",
        input_ids=(100, 101, 102, 103, 104, 105, 106),
        offsets=((0, 1), (2, 3), (3, 4), (4, 5), (6, 7), (8, 10), (10, 11)),
    )
    assert diagnostics["title_replacements"] == 1
    assert observed[0].tokens[0] == SELF_TOKEN
    assert observed[0].raw_boundaries[:2] == (3, 5)


def test_cross_sentence_token_is_rejected():
    observed, diagnostics = observe_encoded_document(
        document_index=0,
        document_id="crossing",
        title="T",
        text="A! B",
        input_ids=(1, 2, 3),
        offsets=((0, 1), (2, 4), (3, 6)),
    )
    assert observed == ()
    assert diagnostics["sentence_boundary_rejections"] == 2


def test_indexed_and_dense_matchers_reject_a_128_token_hole():
    row = sentence(0, (1,) + (2,) * 128 + (9,))
    proposal = ((1,), (9,))
    assert indexed_occurrences((row,), (proposal,)) == {}
    assert dense_occurrences((row,), (proposal,)) == {}


def test_packed_image_detects_corrupt_raw_witness():
    prefix, suffix = (1, 2, 3, 4, 5), (6, 7, 8, 9, 10)
    rows = tuple(
        sentence(index, prefix + (20 + index,) + suffix) for index in range(3)
    )
    compiled = compile_sentences(rows)
    image, layout = build_candidate_image(
        documents=tuple((row.document_id, row.raw_record_tokens) for row in rows),
        rules=compiled.rules,
        edges=compiled.edges,
    )
    verify_candidate_image(image, layout, compiled.rules, compiled.edges)
    edge = compiled.edges[0]
    corrupt = replace(edge, raw_middle=(edge.raw_middle[0] ^ 1,))
    with pytest.raises(RuntimeError, match="raw witness"):
        verify_candidate_image(
            image, layout, compiled.rules, (corrupt, *compiled.edges[1:])
        )


def test_all_frozen_synthetic_worlds_pass():
    result = run_microbenchmarks()
    assert result["status"] == "pass"
    assert result["anti_unifier"]["unordered_pairs"] == 595_686
    assert result["world5_latent_partition"] == "non-identifiable"
