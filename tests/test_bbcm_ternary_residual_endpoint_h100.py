import torch

from experiments.bbcm_ternary_residual_endpoint_h100 import (
    quantize_int8_plus_ternary_residual,
)


def test_ternary_residual_strictly_reduces_error_on_random_weight():
    torch.manual_seed(11)
    weight = torch.randn(23, 41)
    decoded, statistics = quantize_int8_plus_ternary_residual(weight)
    assert decoded.shape == weight.shape
    assert statistics["combined_relative_l2_error"] < statistics["base_relative_l2_error"]
    assert 0.0 < statistics["zero_code_fraction"] < 1.0
    assert statistics["positive_code_fraction"] > 0.0
    assert statistics["negative_code_fraction"] > 0.0
    assert statistics["base_scales_exactly_bfloat16_representable"]
    assert statistics["delta_scales_exactly_bfloat16_representable"]


def test_zero_weight_remains_exact_and_finite():
    weight = torch.zeros(3, 7)
    decoded, statistics = quantize_int8_plus_ternary_residual(weight)
    torch.testing.assert_close(decoded, weight, rtol=0.0, atol=0.0)
    assert torch.isfinite(decoded).all()
    assert statistics["combined_relative_l2_error"] == 0.0
    assert statistics["base_scales_exactly_bfloat16_representable"]
    assert statistics["delta_scales_exactly_bfloat16_representable"]
