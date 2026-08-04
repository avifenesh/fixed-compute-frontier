from __future__ import annotations

import pytest

from experiments import consensus_permutation_raw_prose as experiment


def test_swiglu_two_channel_product_is_exact_on_bipolar_bits() -> None:
    assert experiment.swiglu_product_error() < 1e-12


def test_compiler_recovers_small_raw_prose_world_without_latent_input() -> None:
    corpus = experiment.make_corpus(
        seed=91,
        entities=32,
        relations=4,
        values=8,
        views=2,
        paraphrases=2,
        repeats=31,
        corruption_rate=0.05,
    )
    compiled = experiment.compile_corpus(corpus.sentences)
    metrics = experiment.evaluate_recovery(corpus, compiled)
    assert metrics["direct_fact_accuracy"] == 1.0
    assert metrics["entity_alias_accuracy"] == 1.0
    assert metrics["relation_paraphrase_accuracy"] == 1.0
    assert metrics["compiler_derived_equality_accuracy"] == 1.0


def test_compiler_rejects_nonidentifiable_duplicate_signatures() -> None:
    corpus = experiment.make_corpus(
        seed=92,
        entities=24,
        relations=4,
        values=8,
        views=2,
        paraphrases=2,
        repeats=15,
        corruption_rate=0.0,
        duplicate_entity_signature=True,
    )
    with pytest.raises(experiment.NonIdentifiable, match="duplicate complete"):
        experiment.compile_corpus(corpus.sentences)
