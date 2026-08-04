from experiments.packed_token_evidence_plane_t27_prefix_oracle import (
    SHUFFLE_OFFSET,
    alias_question,
    build_user_message,
    nonoverlapping_title_pair,
    replace_title,
    shuffled_title_map,
)


def test_longest_nonoverlapping_title_route_preserves_question_order():
    titles = ("Alpha", "Alpha Beta", "Gamma", "Unrelated")
    question = "Were Gamma and Alpha Beta both writers?"
    first, second, indices = nonoverlapping_title_pair(question, titles)
    assert question[first[0] : first[1]] == "Gamma"
    assert question[second[0] : second[1]] == "Alpha Beta"
    assert indices == (2, 1)


def test_alias_question_replaces_exact_spans():
    question = "Were Gamma and Alpha Beta both writers?"
    aliased = alias_question(question, ((5, 10), (15, 25)))
    assert aliased == "Were Entity A and Entity B both writers?"


def test_replace_title_is_case_insensitive():
    assert replace_title("ALPHA was also called Alpha.", "Alpha", "Entity A") == (
        "Entity A was also called Entity A."
    )


def test_fixed_shuffle_is_derangement_and_bijection():
    titles = tuple(f"Title {index:04d}" for index in range(2_405))
    mapping = shuffled_title_map(titles)
    assert SHUFFLE_OFFSET == 997
    assert set(mapping) == set(titles)
    assert set(mapping.values()) == set(titles)
    assert all(source != target for source, target in mapping.items())


def test_user_message_is_frozen_shape():
    message = build_user_message(
        "Are Entity A and Entity B writers?",
        "Entity A was an author.",
        "Entity B was a novelist.",
    )
    assert message == (
        "Question: Are Entity A and Entity B writers?\n\n"
        "Evidence for Entity A:\nEntity A was an author.\n\n"
        "Evidence for Entity B:\nEntity B was a novelist.\n\n"
        "Answer (yes or no):"
    )
