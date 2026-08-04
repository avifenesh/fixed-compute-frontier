import numpy as np
import torch

from experiments.title_triggered_affine_prefix_t29_stage0 import (
    A_LEVELS,
    B_LEVELS,
    TOKEN_CODEBOOK,
    apply_pair,
    compose,
    fold_balanced,
    fold_black_box,
    fold_left,
    nearest_level,
    sequential_state,
)


def test_compose_matches_two_sequential_steps():
    first = TOKEN_CODEBOOK[0]
    second = TOKEN_CODEBOOK[2]
    combined = compose(second, first)
    for state in (-2.0, 0.0, 2.0):
        assert apply_pair(combined, state) == apply_pair(second, apply_pair(first, state))


def test_empty_sequence_is_identity():
    assert fold_left(()) == (1.0, 0.0)
    assert fold_balanced(()) == (1.0, 0.0)
    assert fold_black_box(()) == (1.0, 0.0)


def test_fold_forms_match_known_sequence():
    sequence = TOKEN_CODEBOOK
    left = fold_left(sequence)
    assert np.allclose(left, fold_balanced(sequence), atol=1e-12, rtol=0.0)
    assert np.allclose(left, fold_black_box(sequence), atol=1e-12, rtol=0.0)
    for state in (-2.0, -1.0, 0.0, 1.0, 2.0):
        assert abs(sequential_state(sequence, state) - apply_pair(left, state)) <= 1e-12


def test_title_substitution_matches_virtual_insertion():
    prefix = TOKEN_CODEBOOK[:2]
    title = TOKEN_CODEBOOK[2]
    document = TOKEN_CODEBOOK[1:]
    suffix = TOKEN_CODEBOOK[:1]
    compiled_title = compose(fold_left(document), title)
    for state in (-2.0, 0.0, 2.0):
        expanded = sequential_state(prefix + (title,) + document + suffix, state)
        compiled = sequential_state(prefix + (compiled_title,) + suffix, state)
        assert abs(expanded - compiled) <= 1e-12


def test_repeated_title_substitutions_match_two_virtual_insertions():
    prefix = TOKEN_CODEBOOK[:1]
    title = TOKEN_CODEBOOK[2]
    first_document = TOKEN_CODEBOOK[1:3]
    middle = TOKEN_CODEBOOK[3:]
    second_document = TOKEN_CODEBOOK[:2]
    suffix = TOKEN_CODEBOOK[2:3]
    expanded = (
        prefix
        + (title,)
        + first_document
        + middle
        + (title,)
        + second_document
        + suffix
    )
    compiled = (
        prefix
        + (compose(fold_left(first_document), title),)
        + middle
        + (compose(fold_left(second_document), title),)
        + suffix
    )
    for state in (-2.0, 0.0, 2.0):
        assert abs(sequential_state(expanded, state) - sequential_state(compiled, state)) <= 1e-12


def test_reversing_transition_order_changes_the_operator():
    first = TOKEN_CODEBOOK[0]
    second = TOKEN_CODEBOOK[1]
    forward = compose(second, first)
    reversed_order = compose(first, second)
    assert forward != reversed_order
    assert any(
        apply_pair(forward, state) != apply_pair(reversed_order, state)
        for state in (-2.0, 0.0, 2.0)
    )


def test_codec_contains_both_affine_identity_coordinates():
    assert 1.0 in A_LEVELS
    assert 0.0 in B_LEVELS
    assert len(A_LEVELS) == 16
    assert len(B_LEVELS) == 16


def test_codec_minimum_and_maximum_levels_round_trip_through_bf16():
    endpoints = (A_LEVELS[0], A_LEVELS[-1], B_LEVELS[0], B_LEVELS[-1])
    decoded = torch.tensor(endpoints, dtype=torch.float32).to(torch.bfloat16).to(torch.float64)
    assert tuple(decoded.tolist()) == endpoints
    for levels in (A_LEVELS, B_LEVELS):
        for expected_index in (0, len(levels) - 1):
            index, value, clipped = nearest_level(levels[expected_index], levels)
            assert index == expected_index
            assert value == levels[expected_index]
            assert not clipped


def test_nearest_level_tie_uses_lower_index():
    index, value, clipped = nearest_level(0.03125, A_LEVELS)
    assert index == 0
    assert value == 0.0
    assert not clipped


def test_dense_transition_order_is_not_elementwise_commutative():
    first = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    second = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    assert not np.array_equal(second @ first, first @ second)
