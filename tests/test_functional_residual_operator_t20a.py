from __future__ import annotations

import inspect

import torch

from experiments import functional_residual_operator_t20a as t20a


def test_transition_batch_is_causal_and_masked() -> None:
    class Tokenizer:
        eos_token_id = 9

        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [int(value) for value in text.split()]

    inputs, targets, mask = t20a.transition_batch(["1 2 3", "4"], Tokenizer())
    assert inputs.shape == targets.shape == mask.shape == (2, t20a.CONTEXT)
    assert inputs[0, :3].tolist() == [1, 2, 3]
    assert targets[0, :3].tolist() == [2, 3, 9]
    assert mask[0, :3].tolist() == [True, True, True]
    assert int(mask[1].sum()) == 1


def test_random_bases_are_orthonormal_and_deterministic() -> None:
    first = t20a.random_basis(32, 7, 123)
    second = t20a.random_basis(32, 7, 123)
    assert torch.equal(first, second)
    assert t20a.orthonormal_error(first) < 1e-5


def test_projection_has_frozen_shape_and_unit_norm() -> None:
    generator = torch.Generator().manual_seed(7)
    operators = torch.randn(3, 32, 32, generator=generator)
    operators = torch.nn.functional.normalize(operators.flatten(1), dim=1).view(
        3, 32, 32
    )
    left = t20a.random_basis(32, t20a.LEFT_RANK, 11)
    right = t20a.random_basis(32, t20a.RIGHT_RANK, 13)
    projected = t20a.project_operators(operators, left, right)
    assert projected.shape == (3, t20a.PAYLOAD_DIMS)
    assert torch.allclose(projected.norm(dim=1), torch.ones(3), atol=1e-6)


def test_exact_physical_ledger() -> None:
    writes = (
        4_150 * t20a.t19b.t19a.CODE_DIMS
        + 2_405
        * (t20a.t19b.t19a.CODE_DIMS + 1 + 1 + t20a.PAYLOAD_DIMS)
    )
    assert t20a.PAYLOAD_DIMS == 220
    assert writes == t20a.t19b.EXPECTED_WRITES == 743_670
    assert writes <= t20a.t19b.WRITE_BUDGET


def test_compiler_signature_and_frozen_hashes() -> None:
    assert tuple(inspect.signature(t20a.build_raw_operator_plane).parameters) == (
        "model",
        "documents",
        "tokenizer",
        "device",
    )
    assert t20a.sha256_file(t20a.CHECKPOINT) == t20a.CHECKPOINT_SHA256
    assert (
        t20a.sha256_file(t20a.t19b.t19a.CANDIDATE_CORPUS)
        == t20a.t19b.t19a.CANDIDATE_SHA256
    )
    assert (
        t20a.sha256_file(t20a.t19b.EVALUATION_QA)
        == t20a.t19b.t19a.EVALUATOR_SHA256
    )
