import numpy as np
import pytest
import torch

from experiments.dihedral_monomial_prefix_t30_stage0 import (
    A_LEVELS,
    B_LEVELS,
    IDENTITY_GROUP,
    STATES,
    TOKENS,
    WIDTH,
    apply_operator,
    compose,
    compose_group,
    decode_group,
    encode_group,
    fold_left,
    group_from_nibbles,
    group_nibbles,
    identity_operator,
    inverse_group,
    lazy_sequence,
    permute,
    sequential_state,
)


def test_group_identity_inverse_and_code_roundtrip():
    for code in range(2 * WIDTH):
        group = decode_group(code)
        assert encode_group(group) == code
        assert group_from_nibbles(*group_nibbles(group)) == group
        inverse = inverse_group(group)
        assert compose_group(group, inverse) == IDENTITY_GROUP
        assert compose_group(inverse, group) == IDENTITY_GROUP


def test_invalid_group_codes_are_rejected():
    for code in (-1, 2 * WIDTH, 255):
        with pytest.raises(ValueError):
            decode_group(code)
    with pytest.raises(ValueError):
        group_from_nibbles(16, 0)


def test_rotation_and_reflection_do_not_commute():
    rho = (1, 1)
    tau = (-1, 0)
    assert compose_group(tau, compose_group(rho, tau)) == inverse_group(rho)
    assert compose_group(tau, rho) != compose_group(rho, tau)
    assert not np.array_equal(
        permute(compose_group(tau, rho), STATES[-1]),
        permute(compose_group(rho, tau), STATES[-1]),
    )


def test_empty_document_is_identity():
    identity = fold_left(())
    expected = identity_operator()
    assert identity[0] == expected[0]
    assert np.array_equal(identity[1], expected[1])
    assert np.array_equal(identity[2], expected[2])


def test_reversed_operator_order_changes_result():
    first = TOKENS[1]
    second = TOKENS[2]
    forward = compose(second, first)
    reverse = compose(first, second)
    assert forward[0] != reverse[0]
    assert any(
        not np.array_equal(apply_operator(forward, state), apply_operator(reverse, state))
        for state in STATES
    )


def test_repeated_titles_match_expanded_execution():
    title = TOKENS[2]
    first_document = TOKENS[:2]
    second_document = TOKENS[2:]
    expanded = (title,) + first_document + (title,) + second_document
    compiled = (
        compose(fold_left(first_document), title),
        compose(fold_left(second_document), title),
    )
    for state in STATES:
        assert np.allclose(
            sequential_state(expanded, state),
            sequential_state(compiled, state),
            atol=1e-12,
            rtol=0.0,
        )


def test_codec_endpoints_survive_bf16():
    endpoints = (A_LEVELS[0], A_LEVELS[-1], B_LEVELS[0], B_LEVELS[-1])
    decoded = torch.tensor(endpoints, dtype=torch.float32).to(torch.bfloat16).to(torch.float64)
    assert tuple(decoded.tolist()) == endpoints


def test_lazy_and_materialized_execution_match():
    sequence = (TOKENS[1], TOKENS[2], TOKENS[3], TOKENS[0])
    expected_group = fold_left(sequence)[0]
    for state in STATES:
        frame, _, logical = lazy_sequence(sequence, state)
        assert frame == expected_group
        assert np.allclose(
            logical,
            sequential_state(sequence, state),
            atol=1e-12,
            rtol=0.0,
        )
