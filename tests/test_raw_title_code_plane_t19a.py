from __future__ import annotations

import inspect
from pathlib import Path

import torch

from experiments import raw_title_code_plane_t19a as t19a


def test_token_code_is_deterministic_bipolar() -> None:
    first = t19a.deterministic_token_code(17)
    second = t19a.deterministic_token_code(17)
    other = t19a.deterministic_token_code(18)
    assert torch.equal(first, second)
    assert set(first.tolist()) == {-1.0, 1.0}
    assert not torch.equal(first, other)


def test_raw_documents_have_only_allowed_fields() -> None:
    documents = t19a.load_raw_documents(t19a.CANDIDATE_CORPUS)
    assert len(documents) == 2_405
    assert all(set(document) == {"document_id", "title", "text"} for document in documents)
    assert len({document["title"] for document in documents}) == 2_405


def test_compiler_signature_excludes_questions_and_answers() -> None:
    assert tuple(inspect.signature(t19a.compile_token_codes).parameters) == (
        "model",
        "title_data",
    )


def test_frozen_files_match_preregistration() -> None:
    assert t19a.sha256_file(t19a.CANDIDATE_CORPUS) == t19a.CANDIDATE_SHA256
    assert t19a.sha256_file(t19a.DEVELOPMENT_EVALUATOR) == t19a.EVALUATOR_SHA256
    assert t19a.sha256_file(t19a.t10.TRAIN_FILE) == t19a.NATURAL_TRAIN_SHA256
    assert t19a.sha256_file(t19a.t10.VALIDATION_FILE) == t19a.NATURAL_VALIDATION_SHA256
    assert Path(t19a.PREREGISTRATION).exists()


def test_schedule_is_positive_and_decays() -> None:
    assert t19a.schedule_fraction(1, t19a.STEPS_1X) > 0.0
    assert t19a.schedule_fraction(t19a.WARMUP_STEPS, t19a.STEPS_1X) == 1.0
    assert t19a.schedule_fraction(t19a.STEPS_1X, t19a.STEPS_1X) == 0.1
