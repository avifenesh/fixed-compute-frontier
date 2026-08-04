from collections import Counter

from experiments.scale_referenced_whole_record_t32_information_oracle import (
    DOCUMENT_COUNT,
    LOCAL_WINDOW,
    SHUFFLE_OFFSET,
    alias_question,
    build_user_message,
    nonoverlapping_title_pair,
    paired_changes,
    replace_title,
    select_local_window,
    selector_question,
    shuffled_title_map,
)


def test_longest_nonoverlapping_title_route_preserves_question_order():
    titles = ("Alpha", "Alpha Beta", "Gamma", "Unrelated")
    question = "Were Gamma and Alpha Beta both writers?"
    first, second, indices = nonoverlapping_title_pair(question, titles)
    assert question[first[0] : first[1]] == "Gamma"
    assert question[second[0] : second[1]] == "Alpha Beta"
    assert indices == (2, 1)


def test_alias_and_selector_question_use_same_exact_spans():
    question = "Were Gamma and Alpha Beta both writers?"
    spans = ((5, 10), (15, 25))
    assert alias_question(question, spans) == "Were Entity A and Entity B both writers?"
    assert selector_question(question, spans) == "Were  and  both writers?"


def test_replace_title_is_case_insensitive():
    assert replace_title("ALPHA was called Alpha.", "Alpha", "Entity A") == (
        "Entity A was called Entity A."
    )


def test_fixed_shuffle_is_bijective_derangement():
    titles = tuple(f"Title {index:04d}" for index in range(DOCUMENT_COUNT))
    mapping = shuffled_title_map(titles)
    assert SHUFFLE_OFFSET == 997
    assert set(mapping) == set(titles)
    assert set(mapping.values()) == set(titles)
    assert all(source != target for source, target in mapping.items())


def test_local_selector_uses_rarest_matching_token_and_earliest_tie():
    record = tuple(range(10))
    query = (6, 9)
    frequencies = Counter({6: 100, 9: 2})
    selection = select_local_window(record, query, frequencies)
    assert LOCAL_WINDOW == 48
    assert selection.start == 0
    assert selection.ids == record
    assert selection.matched_ids == (6, 9)

    longer = tuple(range(70))
    selection = select_local_window(longer, (60,), Counter({60: 1}))
    assert selection.start == 13
    assert selection.ids[0] == 13
    assert selection.ids[-1] == 60


def test_local_selector_filters_high_document_frequency_and_falls_back_to_start():
    record = tuple(range(70))
    selection = select_local_window(
        record,
        (60,),
        Counter({60: int(0.25 * DOCUMENT_COUNT) + 1}),
    )
    assert selection.start == 0
    assert selection.score == 0.0
    assert selection.matched_ids == ()


def test_paired_changes_counts_direction():
    ids = ("a", "b", "c")
    labels = {"a": "yes", "b": "yes", "c": "no"}
    predictions = {
        ("a", "candidate"): "yes",
        ("a", "control"): "no",
        ("b", "candidate"): "no",
        ("b", "control"): "yes",
        ("c", "candidate"): "no",
        ("c", "control"): "no",
    }
    assert paired_changes(ids, labels, predictions, "candidate", "control") == {
        "gained": 1,
        "lost": 1,
        "unchanged": 1,
    }


def test_user_message_shape_is_frozen():
    assert build_user_message("Q?", "A evidence", "B evidence") == (
        "Question: Q?\n\n"
        "Evidence for Entity A:\nA evidence\n\n"
        "Evidence for Entity B:\nB evidence\n\n"
        "Answer (yes or no):"
    )
