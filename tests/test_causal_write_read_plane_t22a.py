from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from experiments import causal_write_read_plane_t22a as t22a


def test_json_native_converts_scalars_and_rejects_tensor_payloads() -> None:
    converted = t22a.json_native(
        {"tensor": torch.tensor(True), "numpy": np.int64(3)}
    )
    assert json.loads(json.dumps(converted)) == {"tensor": True, "numpy": 3}
    try:
        t22a.json_native({"bad": torch.zeros(2)})
    except TypeError as error:
        assert "$.bad" in str(error)
    else:
        raise AssertionError("non-scalar tensor was silently serialized")


class TinyTokenizer:
    eos_token_id = 99

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        values = {
            "Alpha\n": [1, 2],
            "Beta\n": [6],
            "first body": [3, 4, 5],
            "second body": [7, 8, 9, 10],
        }
        return values[text]


def tiny_corpus() -> t22a.RawCorpus:
    return t22a.build_raw_corpus(
        [
            {
                "document_id": "a",
                "title": "Alpha",
                "text": "first body",
            },
            {
                "document_id": "b",
                "title": "Beta",
                "text": "second body",
            },
        ],
        TinyTokenizer(),
    )


def test_raw_corpus_separates_title_body_and_causal_targets() -> None:
    corpus = tiny_corpus()
    assert corpus.writer_inputs.shape == (2, t22a.CONTEXT)
    assert corpus.writer_lengths.tolist() == [6, 6]
    assert corpus.title_ends.tolist() == [1, 0]
    assert corpus.body_positions[0].tolist() == [2, 3, 4]
    assert corpus.body_positions[1].tolist() == [1, 2, 3, 4]
    assert corpus.causal_inputs[0, :5].tolist() == [1, 2, 3, 4, 5]
    assert corpus.causal_targets[0, :5].tolist() == [2, 3, 4, 5, 99]
    assert int(corpus.causal_mask[0].sum()) == 5


def test_probe_hides_target_and_appends_answer_carrier() -> None:
    corpus = tiny_corpus()
    special = t22a.SpecialTokens(90, 91, 92, 3)
    indices = torch.tensor([0, 1])
    probes = t22a.build_probes(corpus, indices, 123, special, 4)
    assert probes.inputs.shape == (8, t22a.CONTEXT)
    assert probes.document_indices.tolist() == [0] * 4 + [1] * 4
    for row in range(len(probes.inputs)):
        position = int(probes.target_positions[row])
        document = int(probes.document_indices[row])
        assert int(probes.inputs[row, position]) == special.target_mask
        assert int(probes.targets[row]) == int(corpus.writer_inputs[document, position])
        assert (
            int(probes.inputs[row, int(probes.lengths[row]) - 1])
            == special.answer_carrier
        )


def test_fixed_random_codes_are_deterministic_q16_and_bf16_stable() -> None:
    first = t22a.fixed_random_code("document-a")
    second = t22a.fixed_random_code("document-a")
    other = t22a.fixed_random_code("document-b")
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.shape == (t22a.CODE_DIMS,)
    indices = t22a.decoded_level_indices(first)
    assert set(indices.tolist()).issubset(set(range(16)))
    assert torch.equal(
        indices,
        t22a.decoded_level_indices(first.to(torch.bfloat16).float()),
    )


def test_writer_uses_only_frozen_contextual_slice() -> None:
    corpus = tiny_corpus()

    class Writer(nn.Module):
        def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
            batch, length = tokens.shape
            hidden = torch.zeros(batch, length, t22a.core.HIDDEN)
            values = torch.linspace(-2.0, 2.0, t22a.CODE_DIMS)
            hidden[:, :, t22a.CODE_START : t22a.CODE_START + t22a.CODE_DIMS] = values
            return hidden

    codes = t22a.writer_codes(Writer(), corpus, torch.tensor([0, 1]), torch.device("cpu"))
    expected, _ = t22a.quantize_16_ste(torch.linspace(-2.0, 2.0, t22a.CODE_DIMS))
    assert torch.equal(codes[0], expected)
    assert torch.equal(codes[1], expected)


def test_writer_split_and_frozen_compute_ledger() -> None:
    corpus = tiny_corpus()
    training, heldout = t22a.split_document_indices(corpus, frozenset({"Beta"}))
    assert training.tolist() == [0]
    assert heldout.tolist() == [1]
    candidate = (
        t22a.READ_STEPS
        * (t22a.BATCH + t22a.BATCH * t22a.PROBES_PER_DOCUMENT)
        * t22a.CONTEXT
    )
    dense = t22a.DENSE_STEPS * t22a.BATCH * t22a.CONTEXT
    assert candidate == dense == 6_144_000
    assert 2_405 * t22a.CODE_DIMS == 529_100
    assert t22a.t19b.EXPECTED_WRITES == 743_670


def test_frozen_input_and_preregistration_hashes() -> None:
    assert t22a.verify_inputs() == {
        "candidate_corpus": True,
        "train_qa": True,
        "evaluation_qa": True,
        "natural_train": True,
        "natural_validation": True,
        "preregistration_exists": True,
        "parent_preregistration": True,
        "v2_preregistration": True,
        "root_preregistration": True,
    }
    assert (
        t22a.sha256_file(t22a.PREREGISTRATION)
        == "f3dd0a9634efd1aca50c6c267c9980f8a14487737e8b0e4a5b20e1df3a7ea3e9"
    )
