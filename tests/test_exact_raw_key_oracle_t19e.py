from __future__ import annotations

import inspect

import numpy as np

from experiments import exact_raw_key_oracle_t19e as t19e


def test_exact_compiler_has_one_coordinate_per_raw_key() -> None:
    documents = [
        {"document_id": "a", "title": "A", "text": "Alpha beta 1962"},
        {"document_id": "b", "title": "B", "text": "Beta gamma 1971"},
    ]
    corpus = t19e.build_exact_corpus(documents)
    expected = set()
    for document in documents:
        expected.update(t19e.t19b.lexical_features(document["text"]))
    assert set(corpus.feature_to_index) == expected
    assert len(set(corpus.feature_to_index.values())) == len(expected)
    assert sorted(corpus.feature_to_index.values()) == list(range(len(expected)))


def test_symmetric_coordinate_blocks_and_ordered_scalar_contract() -> None:
    query = {0: 0.8, 1: 0.6}
    first = {0: 1.0}
    second = {1: 1.0}
    forward = t19e.symmetric_exact_row(query, first, second, 2)
    reverse = t19e.symmetric_exact_row(query, second, first, 2)
    assert {key: value for key, value in forward.items() if key < 8} == {
        key: value for key, value in reverse.items() if key < 8
    }
    assert forward[8] == reverse[9]
    assert forward[9] == reverse[8]
    assert forward.get(10, 0.0) == reverse.get(10, 0.0)
    assert forward.get(11, 0.0) == reverse.get(11, 0.0)
    assert forward.get(12, 0.0) == reverse.get(12, 0.0)


def test_sparse_standardized_reader_fits_simple_signal() -> None:
    rows = [
        {0: 1.0},
        {0: 0.8},
        {1: 1.0},
        {1: 0.8},
    ]
    matrix = t19e.dictionaries_to_csr(rows, 2)
    targets = np.asarray([1.0, 1.0, 0.0, 0.0])
    reader = t19e.fit_reader(matrix, targets)
    score = t19e.score_reader(reader, matrix, targets)
    assert score["accuracy"] == 1.0
    assert score["finite"]


def test_compiler_signature_excludes_questions_and_labels() -> None:
    assert tuple(inspect.signature(t19e.build_exact_corpus).parameters) == (
        "documents",
    )


def test_frozen_hashes() -> None:
    assert (
        t19e.sha256_file(t19e.t19b.t19a.CANDIDATE_CORPUS)
        == t19e.t19b.t19a.CANDIDATE_SHA256
    )
    assert t19e.sha256_file(t19e.t19b.TRAIN_QA) == t19e.t19b.TRAIN_QA_SHA256
    assert (
        t19e.sha256_file(t19e.t19b.EVALUATION_QA)
        == t19e.t19b.t19a.EVALUATOR_SHA256
    )
