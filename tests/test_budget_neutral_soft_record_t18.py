from __future__ import annotations

import inspect

import pytest
import torch

from experiments import budget_neutral_soft_record_t18 as experiment


def test_exact_matrix_budget_identity() -> None:
    assert experiment.BASELINE_REPLACED_SCALARS == 2_359_296
    assert experiment.CANDIDATE_REPLACED_SCALARS == 2_359_296
    assert experiment.BASELINE_REPLACED_SCALARS == experiment.CANDIDATE_REPLACED_SCALARS
    ledger = experiment.matrix_ledger()
    assert ledger["baseline_two_swiglu_parameters"] == ledger["candidate_total_parameters"]
    assert ledger["baseline_two_swiglu_macs_per_token"] == ledger["candidate_total_macs_per_token"]


def test_full_model_parameter_count_is_exact() -> None:
    device = torch.device("cpu")
    baseline = experiment.t10.core.build_model(17, device)
    candidate = experiment.build_candidate(17, device)
    assert experiment.parameter_count(baseline) == experiment.parameter_count(candidate)
    assert experiment.parameter_count(candidate) == 36_577_152


def test_raw_compiler_signature_and_extra_field_rejection() -> None:
    assert tuple(inspect.signature(experiment.compile_raw_records).parameters) == (
        "model",
        "tokenizer",
        "documents",
    )
    model = experiment.build_candidate(19, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="document_id/title/text"):
        experiment.compile_raw_records(
            model,
            object(),
            [{"document_id": "x", "title": "T", "text": "P", "answer": "x"}],
        )


def test_record_forward_and_gradients_match_dense_reference() -> None:
    result = experiment.bounded_reference(torch.device("cpu"))
    assert max(result["maximum_errors"].values()) <= 2e-6
    assert result["null_output_max_abs"] < 1e-5


def test_candidate_shape_matches_baseline() -> None:
    device = torch.device("cpu")
    candidate = experiment.build_candidate(23, device).eval()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        logits = candidate(tokens)
    assert logits.shape == (1, 4, experiment.t10.core.small.VOCAB)


def test_activation_ledger_prices_softmax_materialization() -> None:
    ledger = experiment.activation_ledger(2, 128)
    assert ledger["candidate_materialized_score_probability_scalars"] > 0
    assert ledger["candidate_softmax_exponentials_per_token"] == 2_406
    assert ledger["candidate_materialized_score_probability_bytes"] > ledger[
        "baseline_peak_gate_up_product_bytes"
    ]
