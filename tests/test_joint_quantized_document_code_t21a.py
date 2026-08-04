from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from experiments import joint_quantized_document_code_t21a as t21a


def test_json_native_converts_scalar_tensors_and_numpy_values() -> None:
    value = {
        "tensor": torch.tensor(True),
        "numpy": np.float64(1.25),
        "nested": (torch.tensor(3),),
    }
    converted = t21a.json_native(value)
    assert converted == {"tensor": True, "numpy": 1.25, "nested": [3]}
    assert json.loads(json.dumps(converted)) == converted


def test_json_native_rejects_accidental_tensor_payload() -> None:
    try:
        t21a.json_native({"bad": torch.zeros(2)})
    except TypeError as error:
        assert "$.bad" in str(error)
    else:
        raise AssertionError("non-scalar tensor was silently serialized")


def test_quantizer_uses_16_levels_and_bf16_preserves_indices() -> None:
    values = torch.linspace(-4.0, 4.0, 10_000).view(100, 100)
    codes, indices = t21a.quantize_16_ste(values)
    assert set(indices.flatten().tolist()) == set(range(16))
    assert torch.equal(indices, t21a.decoded_level_indices(codes))
    assert torch.equal(
        indices,
        t21a.decoded_level_indices(codes.to(torch.bfloat16).float()),
    )


def test_record_injection_changes_only_selected_positions_and_coordinates() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = nn.Embedding(8, t21a.core.HIDDEN)
            nn.init.zeros_(self.token.weight)
            self.blocks = nn.ModuleList()
            self.final_norm = nn.Identity()

    model = Model()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    records = torch.stack(
        (torch.ones(2, t21a.CODE_DIMS), 2.0 * torch.ones(2, t21a.CODE_DIMS)),
        dim=1,
    )
    positions = torch.tensor([[0, 2], [1, 2]])
    hidden = t21a.hidden_with_records(model, tokens, records, positions)
    assert torch.all(hidden[0, 0, t21a.CODE_START : t21a.CODE_START + t21a.CODE_DIMS] == 1)
    assert torch.all(hidden[0, 2, t21a.CODE_START : t21a.CODE_START + t21a.CODE_DIMS] == 2)
    assert torch.all(hidden[1, 1, t21a.CODE_START : t21a.CODE_START + t21a.CODE_DIMS] == 1)
    assert torch.all(hidden[1, 2, t21a.CODE_START : t21a.CODE_START + t21a.CODE_DIMS] == 2)
    assert torch.count_nonzero(hidden[..., : t21a.CODE_START]) == 0


def test_document_corpus_masks_padding() -> None:
    class Tokenizer:
        eos_token_id = 9

        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [1, 2, 3]

    documents = [{"document_id": "x", "title": "X", "text": "text"}]
    corpus = t21a.build_document_corpus(documents, Tokenizer())
    assert corpus.inputs.shape == corpus.targets.shape == corpus.mask.shape == (
        1,
        t21a.CONTEXT,
    )
    assert corpus.inputs[0, :3].tolist() == [1, 2, 3]
    assert corpus.targets[0, :3].tolist() == [2, 3, 9]
    assert int(corpus.mask.sum()) == 3


def test_frozen_pretraining_schedule_counts() -> None:
    one_x = sum(
        t21a.arm_is_document_step("joint_code_1x", step)
        for step in range(1, t21a.PRETRAIN_STEPS_1X + 1)
    )
    two_x = sum(
        t21a.arm_is_document_step("dense_2x_doc", step)
        for step in range(1, t21a.PRETRAIN_STEPS_2X + 1)
    )
    assert one_x == 1_000
    assert two_x == 2_000
    assert t21a.PRETRAIN_STEPS_1X - one_x == 19_000
    assert t21a.PRETRAIN_STEPS_2X - two_x == 19_000


def test_physical_ledger_and_frozen_hashes() -> None:
    writes = (
        4_150 * t21a.t19a.CODE_DIMS
        + 2_405 * (t21a.t19a.CODE_DIMS + 1 + 1 + t21a.CODE_DIMS)
    )
    assert writes == t21a.t19b.EXPECTED_WRITES == 743_670
    assert writes <= t21a.t19b.WRITE_BUDGET
    assert t21a.sha256_file(t21a.t19a.CANDIDATE_CORPUS) == t21a.t19a.CANDIDATE_SHA256
    assert t21a.sha256_file(t21a.t19b.TRAIN_QA) == t21a.t19b.TRAIN_QA_SHA256
    assert (
        t21a.sha256_file(t21a.t19b.EVALUATION_QA)
        == t21a.t19a.EVALUATOR_SHA256
    )
