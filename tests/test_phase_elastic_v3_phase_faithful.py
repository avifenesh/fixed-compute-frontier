import torch

from experiments.phase_elastic_v3_phase_faithful import (
    EVAL_FIRST_FULL,
    TRAIN_FIRST_FULL,
    PhaseFaithfulV2MLP,
    aggregate_bucket,
    bucket_slices,
    paired_noninferior,
    phase_mask,
    set_phase_mask,
)
from experiments.phase_elastic_residual_precision_v2_lm_screen import serving_ledger


def test_phase_mask_uses_one_shared_boundary_per_sequence():
    first_full = torch.tensor([1, 3])
    mask = phase_mask(first_full, sequence_length=5, device=torch.device("cpu"))
    assert mask.shape == (2, 5, 1)
    assert mask[:, :, 0].tolist() == [
        [False, True, True, True, True],
        [False, False, False, True, True],
    ]


def test_phase_boundaries_are_frozen_and_cover_short_decode():
    assert TRAIN_FIRST_FULL == (64, 128, 256, 384, 448, 480, 496)
    assert EVAL_FIRST_FULL == (256, 384, 448, 480, 496)
    assert serving_ledger()["fits_bf16_bytes"]


def test_phase_faithful_module_rejects_missing_or_misaligned_mask():
    module = object.__new__(PhaseFaithfulV2MLP)
    torch.nn.Module.__init__(module)
    module.mode = "split"
    module.phase_mask = None
    hidden = torch.zeros(2, 5, 3)
    try:
        module._mask_for(hidden)
    except RuntimeError as error:
        assert "phase mask" in str(error)
    else:
        raise AssertionError("missing mask was accepted")

    module.phase_mask = torch.zeros(2, 4, 1, dtype=torch.bool)
    try:
        module._mask_for(hidden)
    except RuntimeError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("misaligned mask was accepted")


def test_bucket_alignment_starts_first_generation_loss_at_boundary_logit():
    buckets = bucket_slices(first_full=480, sequence_length=512)
    assert buckets["first_token"] == slice(480, 481)
    assert buckets["offsets_2_8"] == slice(481, 488)
    assert buckets["offsets_9_32"] == slice(488, 512)
    assert buckets["offsets_33_plus"] == slice(512, 512)
    assert buckets["all_suffix"] == slice(480, 512)


def test_exact_same_mask_object_is_installed_in_every_layer():
    class Stub:
        def set_phase_mask(self, mask):
            self.phase_mask = mask

    modules = [Stub(), Stub(), Stub()]
    mask = torch.ones(2, 5, 1, dtype=torch.bool)
    set_phase_mask(modules, mask)
    assert all(module.phase_mask is mask for module in modules)


def test_aggregate_bucket_weights_boundaries_equally_then_preserves_pairs():
    grid = {}
    for index, boundary in enumerate(EVAL_FIRST_FULL):
        grid[str(boundary)] = {
            "all_suffix": {
                "loss": float(index),
                "per_batch_loss": [float(index), float(index + 10)],
                "prediction_tokens": index + 1,
            }
        }
    aggregate = aggregate_bucket(grid, "all_suffix")
    assert aggregate["per_batch_loss"] == [2.0, 12.0]
    assert aggregate["loss"] == 7.0
    assert aggregate["prediction_tokens"] == 15


def test_first_token_noninferiority_requires_confidence_bound_too():
    assert paired_noninferior(
        candidate_loss=5.001,
        reference_loss=5.0,
        interval={"upper_95": 0.002},
        relative_margin=0.0005,
    )
    assert not paired_noninferior(
        candidate_loss=5.001,
        reference_loss=5.0,
        interval={"upper_95": 0.003},
        relative_margin=0.0005,
    )
