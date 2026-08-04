import torch

from experiments.phase_elastic_v3_50m_phase_faithful import (
    CACHED_BATCHES,
    CACHED_BATCH_SIZE,
    CACHED_BOUNDARIES,
    SEED,
    STEPS,
    paired_t_interval,
)


def test_long_horizon_uses_fresh_seed_and_50m_tokens():
    assert SEED == 9901
    assert STEPS == 1525
    assert STEPS * 2 * 32 * 512 == 49_971_200


def test_cached_first_logit_alignment_is_zero_based_position_p():
    first_full = 480
    inputs = torch.arange(512)
    targets = torch.arange(1, 513)
    assert inputs[first_full].item() == 480
    assert targets[first_full].item() == 481


def test_cached_evaluation_has_two_boundaries_and_128_sequence_pairs():
    assert CACHED_BOUNDARIES == (256, 480)
    assert CACHED_BATCHES == 16
    assert CACHED_BATCH_SIZE == 8
    assert CACHED_BATCHES * CACHED_BATCH_SIZE == 128


def test_cached_interval_uses_student_t_127():
    reference = {"per_batch_loss": [5.0] * 128}
    candidate = {"per_batch_loss": [4.9 + index / 10000 for index in range(128)]}
    interval = paired_t_interval(candidate, reference)
    assert interval["degrees_of_freedom"] == 127
    assert interval["critical_95"] == 1.978819534
