from experiments.reversible_slot_grammar_plane_t33_coverage import (
    Edge,
    SupportItem,
    score_coverage,
    source_sentence_spans,
)


def test_source_sentence_spans_account_for_outer_strip_only():
    text, spans = source_sentence_spans(("  Alpha. ", "Beta.  "))
    assert text == "Alpha. Beta."
    assert text[slice(*spans[0])] == "Alpha. "
    assert text[slice(*spans[1])] == "Beta."


def test_coverage_requires_unique_edge_in_contained_compiler_sentence():
    items = (
        SupportItem("q1", "Alpha", 0, 0, 10, (0,)),
        SupportItem("q1", "Beta", 0, 0, 10, (1,)),
    )
    edges = (
        Edge("Alpha", 0, 1, 3, 2, True),
        Edge("Beta", 1, 2, 5, 1, True),
    )
    result = score_coverage(items, edges)
    assert result["support_edge_coverage"] == 1.0
    assert result["question_both_document_coverage"] == 1.0


def test_duplicate_or_wrong_sentence_edge_is_not_usable():
    items = (SupportItem("q1", "Alpha", 0, 0, 10, (0,)),)
    edges = (
        Edge("Alpha", 0, 1, 3, 2, False),
        Edge("Alpha", 1, 2, 5, 1, True),
    )
    result = score_coverage(items, edges)
    assert result["support_edge_coverage"] == 0.0
