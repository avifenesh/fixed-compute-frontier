from __future__ import annotations

import torch

from experiments import budget_neutral_soft_record_t18_runtime as experiment


def test_block_pair_parameter_counts_are_exact() -> None:
    baseline = experiment.BaselineFFNPair()
    candidate = experiment.CandidateRecordPair()
    assert experiment.module_parameters(baseline) == experiment.module_parameters(candidate)
    assert experiment.module_parameters(baseline) == (
        experiment.t18.BASELINE_REPLACED_SCALARS + 2 * experiment.t18.HIDDEN
    )


def test_block_pair_shapes_match() -> None:
    hidden = torch.randn(2, 7, experiment.t18.HIDDEN)
    baseline = experiment.BaselineFFNPair().eval()
    candidate = experiment.CandidateRecordPair().eval()
    with torch.inference_mode():
        assert baseline(hidden).shape == candidate(hidden).shape == hidden.shape


def test_runtime_protocol_is_frozen() -> None:
    assert experiment.TRIALS == 5
    assert experiment.WARMUPS == 20
    assert experiment.BLOCK_SURFACES == ((1, 1), (1, 128), (8, 128), (32, 128))
    assert experiment.MODEL_SURFACES == ((1, 1), (1, 128), (8, 128))
    assert experiment.STAGE0_SHA256 == (
        "ba2ac04c370a4c000b513438ac30ce5b5fccd0d6856ba98861198ea83db47126"
    )
