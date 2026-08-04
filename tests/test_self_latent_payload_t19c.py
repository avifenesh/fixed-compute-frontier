from __future__ import annotations

import inspect

import torch

from experiments import self_latent_payload_t19c as t19c


def test_padded_tokens_preserves_lengths_and_context() -> None:
    class Tokenizer:
        eos_token_id = 7

        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return list(range(1, len(text) + 1))

    rows, lengths = t19c.padded_tokens(["abc", "x"], Tokenizer())
    assert rows.shape == (2, t19c.CONTEXT)
    assert lengths.tolist() == [3, 1]
    assert rows[0, :3].tolist() == [1, 2, 3]
    assert rows[1, 1:].eq(7).all()


def test_writer_signature_excludes_qa() -> None:
    assert tuple(inspect.signature(t19c.encode_document_payloads).parameters) == (
        "model",
        "documents",
        "tokenizer",
        "device",
    )


def test_checkpoint_and_frozen_hashes() -> None:
    assert t19c.sha256_file(t19c.CHECKPOINT) == t19c.CHECKPOINT_SHA256
    assert (
        t19c.sha256_file(t19c.t19b.t19a.CANDIDATE_CORPUS)
        == t19c.t19b.t19a.CANDIDATE_SHA256
    )
    assert t19c.sha256_file(t19c.t19b.TRAIN_QA) == t19c.t19b.TRAIN_QA_SHA256


def test_physical_ledger_is_reused_exactly() -> None:
    assert t19c.t19b.EXPECTED_WRITES == 743_670
    assert t19c.t19b.EXPECTED_WRITES <= t19c.t19b.WRITE_BUDGET


def test_reader_payload_dimensions_match() -> None:
    query = torch.randn(t19c.PAYLOAD_DIMS)
    first = torch.randn(t19c.PAYLOAD_DIMS)
    second = torch.randn(t19c.PAYLOAD_DIMS)
    values = t19c.t19b.symmetric_reader_features(query, first, second)
    assert values.shape == (4 * t19c.PAYLOAD_DIMS + 5,)
