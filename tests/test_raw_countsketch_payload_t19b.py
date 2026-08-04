from __future__ import annotations

import inspect

from experiments import raw_countsketch_payload_t19b as t19b


def test_lexical_features_include_required_families() -> None:
    features = t19b.lexical_features("Born in 1962. Alpha beta")
    assert features["u:born"] == 1
    assert features["b:alpha\x1fbeta"] == 1
    assert features["y:1962"] == 1
    assert features["d:1960"] == 1


def test_hash_is_deterministic_and_bounded() -> None:
    first = t19b.feature_bucket_and_sign("u:composer")
    second = t19b.feature_bucket_and_sign("u:composer")
    assert first == second
    assert 0 <= first[0] < t19b.PAYLOAD_DIMS
    assert first[1] in {-1.0, 1.0}


def test_exact_write_ledger() -> None:
    writes = (
        4_150 * t19b.t19a.CODE_DIMS
        + 2_405 * (t19b.t19a.CODE_DIMS + 1 + 1 + t19b.PAYLOAD_DIMS)
    )
    assert writes == t19b.EXPECTED_WRITES == 743_670
    assert writes <= t19b.WRITE_BUDGET


def test_compiler_accepts_only_documents() -> None:
    assert tuple(inspect.signature(t19b.build_sketch_corpus).parameters) == (
        "documents",
    )


def test_frozen_hashes() -> None:
    assert t19b.sha256_file(t19b.t19a.CANDIDATE_CORPUS) == t19b.t19a.CANDIDATE_SHA256
    assert t19b.sha256_file(t19b.TRAIN_QA) == t19b.TRAIN_QA_SHA256
    assert t19b.sha256_file(t19b.EVALUATION_QA) == t19b.t19a.EVALUATOR_SHA256
