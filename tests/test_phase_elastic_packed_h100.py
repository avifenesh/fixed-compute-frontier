import torch

from experiments.phase_elastic_packed_h100 import (
    BASE_WIDTH,
    PACKED_COMPONENT_KEYS,
    BRANCH_WIDTH,
    D,
    DECODE_ROWS,
    LAYERS,
    PREFILL_ROWS,
    TOTAL_WIDTH,
    bootstrap_ratio,
    bootstrap_service_ratio,
    layer_buffers,
    packed_layout_audit,
)
from experiments.phase_elastic_residual_precision_v2_lm_screen import serving_ledger


def test_physical_shapes_and_mac_ratio_match_quality_candidate():
    assert (D, BASE_WIDTH, BRANCH_WIDTH, TOTAL_WIDTH) == (384, 1024, 1472, 2496)
    assert TOTAL_WIDTH / BASE_WIDTH == 2.4375
    assert DECODE_ROWS == (1, 8, 32)
    assert PREFILL_ROWS == (128, 512, 2048)


def test_model_slack_funds_combined_decode_workspace_through_row_32():
    ledger = serving_ledger()
    model_slack = LAYERS * ledger["storage_slack_bytes_per_layer"]
    extra_row_bytes = 2 * (TOTAL_WIDTH - BASE_WIDTH)
    assert model_slack >= 32 * extra_row_bytes
    assert model_slack < 128 * extra_row_bytes


def test_every_packed_component_is_naturally_128_byte_aligned():
    component_bytes = (
        393_216, 98_304, 2_048, 2_048,
        393_216, 98_304, 2_048, 2_048,
        393_216, 98_304, 768, 768,
        282_624, 2_944, 282_624, 2_944, 282_624, 768,
    )
    assert sum(component_bytes) == serving_ledger()["candidate_ffn_bytes_per_layer"]
    assert all(byte_count % 128 == 0 for byte_count in component_bytes)


def test_monolithic_layout_has_exact_offsets_and_shared_storage():
    specifications = {
        "bg_q8": ((BASE_WIDTH * D,), torch.int8),
        "bg_t": ((BASE_WIDTH * D // 4,), torch.uint8),
        "bg_s8": ((BASE_WIDTH,), torch.bfloat16),
        "bg_st": ((BASE_WIDTH,), torch.bfloat16),
        "bu_q8": ((BASE_WIDTH * D,), torch.int8),
        "bu_t": ((BASE_WIDTH * D // 4,), torch.uint8),
        "bu_s8": ((BASE_WIDTH,), torch.bfloat16),
        "bu_st": ((BASE_WIDTH,), torch.bfloat16),
        "bd_q8": ((D * BASE_WIDTH,), torch.int8),
        "bd_t": ((D * BASE_WIDTH // 4,), torch.uint8),
        "bd_s8": ((D,), torch.bfloat16),
        "bd_st": ((D,), torch.bfloat16),
        "rg_q4": ((BRANCH_WIDTH * D // 2,), torch.uint8),
        "rg_s": ((BRANCH_WIDTH,), torch.bfloat16),
        "ru_q4": ((BRANCH_WIDTH * D // 2,), torch.uint8),
        "ru_s": ((BRANCH_WIDTH,), torch.bfloat16),
        "rd_q4": ((D * BRANCH_WIDTH // 2,), torch.uint8),
        "rd_s": ((D,), torch.bfloat16),
    }
    prefix = "model.layers.0.mlp."
    state = {
        prefix + PACKED_COMPONENT_KEYS[name]: torch.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in specifications.items()
    }
    audit = packed_layout_audit(layer_buffers(state))
    assert audit["logical_bytes"] == serving_ledger()["candidate_ffn_bytes_per_layer"]
    assert audit["storage_bytes"] == audit["logical_bytes"]
    assert all(segment["shared_blob_storage"] for segment in audit["segments"])
    assert all(segment["expected_offset"] for segment in audit["segments"])


def test_paired_bootstraps_are_exact_for_identical_samples():
    samples = [0.010 + index * 1e-6 for index in range(100)]
    ratio = bootstrap_ratio(samples, samples, seed=1)
    service = bootstrap_service_ratio(
        samples, samples, samples, samples, decode_steps=128, seed=2
    )
    assert ratio == {"median_ratio": 1.0, "lower_95": 1.0, "upper_95": 1.0}
    assert service == {"median_ratio": 1.0, "lower_95": 1.0, "upper_95": 1.0}
