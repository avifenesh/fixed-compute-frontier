from __future__ import annotations

import inspect

import torch
from torch.nn import functional as F

from experiments import hotpot_folded_entity_address_t16 as experiment


def test_compiler_training_cannot_accept_questions_or_labels() -> None:
    assert tuple(inspect.signature(experiment.train_projection).parameters) == (
        "canonical",
        "train_views",
        "device",
    )
    source = inspect.getsource(experiment.train_projection)
    assert "question" not in source
    assert "answer" not in source


def test_frozen_projection_ledger() -> None:
    assert experiment.PROJECTION_DIMS == 64
    assert experiment.PREFIX_TOKENS == 32
    assert experiment.TRAIN_VIEWS == 4
    assert experiment.HELD_VIEWS == 1
    assert experiment.PROJECTION_SEED == 9_109
    assert experiment.PROJECTION_STEPS == 800
    assert experiment.PROJECTION_BATCH == 256
    assert experiment.PROJECTION_LR == 0.003
    assert experiment.TEMPERATURE == 0.05


def test_folded_key_algebra_is_exact() -> None:
    device = torch.device("cpu")
    projection = experiment.initialize_projection(device)
    generator = torch.Generator().manual_seed(9_113)
    canonical = torch.randn(7, experiment.t10.core.small.HIDDEN, generator=generator)
    query = torch.randn(experiment.t10.core.small.HIDDEN, generator=generator)
    addresses = F.normalize(projection(canonical), dim=1)
    keys = experiment.folded_keys(projection, canonical)
    served = query @ keys.T
    explicit = projection(query) @ addresses.T
    assert torch.allclose(served, explicit, atol=1e-6, rtol=1e-5)


def test_raw_prefix_views_are_deterministic_and_bounded() -> None:
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            base = sum(text.encode()) % 97 + 3
            return [base + index for index in range(max(len(text.split()), 1))]

    documents = [
        {"title": f"D{index}", "text": "one two three four five six seven"}
        for index in range(9)
    ]
    first = experiment.raw_prefix_views(Tokenizer(), documents, ["Alpha", "Beta"])
    second = experiment.raw_prefix_views(Tokenizer(), documents, ["Alpha", "Beta"])
    assert first == second
    canonical, views = first
    assert len(canonical) == 2
    assert all(len(title_views) == 5 for title_views in views)
    assert all(
        0 < len(sequence) <= experiment.t13.CHUNK_TOKENS
        for title_views in views
        for sequence in title_views
    )


def test_t15_integrity_anchor() -> None:
    assert experiment.T15_RESULT_SHA256 == (
        "b88c42915fd4b7e594a3fb337d3153c90db25be33fb3fce5a5a538c2044c36a0"
    )
