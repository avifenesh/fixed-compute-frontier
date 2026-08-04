from __future__ import annotations

import inspect

import torch

from experiments import hotpot_predictive_metric_t14 as experiment


def test_t14_has_no_training_surface() -> None:
    source = inspect.getsource(experiment.run)
    assert "optimizer" not in source
    assert ".backward(" not in source


def test_output_metric_equals_full_logit_dot_product() -> None:
    model = experiment.t10.core.build_model(711, torch.device("cpu")).eval()
    generator = torch.Generator().manual_seed(713)
    q = torch.randn(experiment.t10.core.small.HIDDEN, generator=generator)
    d = torch.randn(experiment.t10.core.small.HIDDEN, generator=generator)
    metric = experiment.output_metric(model)
    algebraic = q @ metric @ d
    q_logits = q @ model.token.weight.T
    d_logits = d @ model.token.weight.T
    explicit = (q_logits @ d_logits) / experiment.t10.core.small.VOCAB
    assert torch.allclose(algebraic, explicit, atol=1e-6, rtol=1e-5)


def test_right_padding_cannot_change_real_final_states() -> None:
    device = torch.device("cpu")
    model = experiment.t10.core.build_model(719, device).eval()
    sequences = [[41, 42, 43], [51, 52, 53, 54, 55]]
    batched = experiment.encode_final_sequences(model, sequences, 2, device, 2)
    separate = [
        experiment.encode_final_sequences(model, [sequence], 2, device, 1)[0]
        for sequence in sequences
    ]
    for left, right in zip(batched, separate, strict=True):
        # Different CPU GEMM batch shapes can change the last few FP32 bits.
        # The measured worst case for this contract is 2.27e-6; keep a narrow
        # bound that still rejects any causal dependence on appended padding.
        assert torch.allclose(left, right, atol=5e-6, rtol=1e-5)


def test_frozen_references() -> None:
    assert experiment.T13_RESULT_SHA256 == (
        "a09fc367ad52bfe0b357690e9b0aed6e252c5c774b9efaeae8b8f8c906347329"
    )
    assert experiment.STATIC_REFERENCE_ACCURACY == 0.6057692307692307
    assert experiment.CONTEXTUAL_REFERENCE_ACCURACY == 0.5384615384615384
