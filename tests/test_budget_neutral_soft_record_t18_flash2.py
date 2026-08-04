from __future__ import annotations

import torch

from experiments import budget_neutral_soft_record_t18 as t18
from experiments import budget_neutral_soft_record_t18_flash2 as experiment
from experiments import budget_neutral_soft_record_t18_runtime as runtime


def test_two_head_matrix_ledger_is_exact() -> None:
    record_macs = (
        experiment.HEADS
        * 2
        * t18.RECORD_SLOTS
        * experiment.HEAD_DIM
    )
    assert record_macs == 1_847_808
    assert record_macs + 3 * t18.HIDDEN * t18.SMALL_FFN == 2_359_296
    assert t18.BASELINE_REPLACED_SCALARS == 2_359_296


def test_cpu_forward_and_gradients_match_explicit_two_head_reference() -> None:
    result = experiment.bounded_reference(torch.device("cpu"))
    assert max(result["maximum_errors"].values()) <= 2e-6
    assert result["null_output_max_abs"] < 1e-5


def test_candidate_parameter_counts_remain_exact() -> None:
    baseline_pair = runtime.BaselineFFNPair()
    candidate_pair = runtime.CandidateRecordPair()
    candidate_pair.memory = experiment.Flash2RecordMemory()
    assert runtime.module_parameters(baseline_pair) == runtime.module_parameters(
        candidate_pair
    )
    model = experiment.build_candidate(29, torch.device("cpu"))
    assert t18.parameter_count(model) == 36_577_152
