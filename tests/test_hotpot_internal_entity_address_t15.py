from __future__ import annotations

import inspect

import torch

from experiments import hotpot_internal_entity_address_t15 as experiment


def test_t15_has_no_training_or_answer_surface() -> None:
    source = inspect.getsource(experiment.run)
    assert "optimizer" not in source
    assert ".backward(" not in source
    probe_source = inspect.getsource(experiment.surface_probe)
    assert '["answer"]' not in probe_source


def test_title_span_uses_final_overlapping_token() -> None:
    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text: str, **_kwargs: object) -> dict[str, object]:
            assert text == "Are Alpha Beta and Gamma both old?"
            return {
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "offset_mapping": [
                    (0, 3), (4, 9), (10, 14), (15, 18),
                    (19, 24), (25, 29), (30, 33), (33, 34),
                ],
            }

    row = {
        "question": "Are Alpha Beta and Gamma both old?",
        "supporting_facts": {"title": ["Alpha Beta", "Gamma"], "sent_id": [0, 0]},
    }
    identifiers, indices = experiment.question_tokens_and_title_indices(Tokenizer(), row)
    assert identifiers == [1, 2, 3, 4, 5, 6, 7, 8]
    assert indices == {"Alpha Beta": 2, "Gamma": 4}


def test_frozen_t14_anchor() -> None:
    assert experiment.T14_RESULT_SHA256 == (
        "4e80b58fb9d2d6372b8093792ba57c133eb6ea27b78ec01b63330325cd019839"
    )


def test_canonical_interface_rows_are_normalized() -> None:
    device = torch.device("cpu")
    model = experiment.t10.core.build_model(811, device).eval()

    class Tokenizer:
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [11, 12] if "One" in text else [21, 22, 23]

    keys = experiment.canonical_keys(model, Tokenizer(), ["One", "Two"], device)
    assert torch.allclose(keys.norm(dim=1), torch.ones(2), atol=1e-6)
