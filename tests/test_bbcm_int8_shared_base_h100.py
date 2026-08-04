import torch

from experiments.bbcm_int8_shared_base_h100 import quantize_int8_per_output


def test_per_output_int8_quantization_is_finite_and_bounded():
    torch.manual_seed(7)
    weight = torch.randn(17, 31)
    decoded, stats = quantize_int8_per_output(weight)
    assert decoded.shape == weight.shape
    assert torch.isfinite(decoded).all()
    assert stats["weights"] == weight.numel()
    assert stats["output_rows"] == weight.shape[0]
    raw_scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
    scale = raw_scale.to(torch.bfloat16).float()
    expected = torch.round(weight / scale).clamp(-127, 127) * scale
    torch.testing.assert_close(decoded, expected, rtol=0.0, atol=0.0)
    assert stats["scale_storage_dtype"] == "bfloat16"
    assert stats["scales_exactly_bfloat16_representable"]


def test_zero_rows_remain_zero():
    weight = torch.zeros(3, 5)
    decoded, stats = quantize_int8_per_output(weight)
    torch.testing.assert_close(decoded, weight, rtol=0.0, atol=0.0)
    assert stats["mean_squared_error"] == 0.0
    assert stats["scales_exactly_bfloat16_representable"]
