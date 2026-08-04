import pytest

torch = pytest.importorskip("torch")

from experiments.self_product_ffn_scale_fused_h100 import (
    build_fused,
    exhaustive_bf16_activation_equivalence,
    self_product_inplace,
    silu_inplace,
    swiglu_inplace,
)
from experiments.self_product_ffn_scale_h100 import build_folded as build_eager, service_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_inplace_kernels_match_bf16_references():
    gate = torch.randn(7, 513, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    expected = torch.nn.functional.silu(gate.float()) * up.float()
    actual = swiglu_inplace(gate.clone(), up)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
    values = torch.randn(7, 769, device="cuda", dtype=torch.bfloat16)
    expected_self = torch.nn.functional.silu(values.float()) * values.float()
    actual_self = self_product_inplace(values.clone())
    torch.testing.assert_close(actual_self.float(), expected_self, atol=0.03, rtol=0.02)
    expected_silu = torch.nn.functional.silu(values.float())
    actual_silu = silu_inplace(values.clone())
    torch.testing.assert_close(actual_silu.float(), expected_silu, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exhaustive_finite_bf16_activation_equivalence():
    result = exhaustive_bf16_activation_equivalence(torch.device("cuda"))
    assert result["finite_bf16_values"] == 65280
    assert result["pass"]
    assert all(result["checks"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_models_have_equal_static_counts_and_live_cached_decode():
    counts = []
    cases = (
        ("parallel_swiglu", False),
        ("parallel_swiglu", True),
        ("parallel_wide_silu", False),
        ("parallel_self_product", False),
    )
    for arm, packed_swiglu in cases:
        model = build_fused(torch.device("cuda"), arm, packed_swiglu=packed_swiglu)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
        prompt = torch.randint(0, 1000, (1, 16), device="cuda")
        logits, cache = service_forward(model, prompt, use_cache=True)
        assert logits.shape == (1, 1, 49152)
        logits, cache = service_forward(model, prompt[:, :1], cache, use_cache=True)
        assert cache.get_seq_length() == 17
        del logits, cache, prompt, model; torch.cuda.empty_cache()
    assert len(set(counts)) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_model_logits_and_cached_decode_match_corresponding_eager_folded_model():
    prompt = torch.randint(0, 1000, (2, 16), device="cuda")
    next_token = torch.randint(0, 1000, (2, 1), device="cuda")
    cases = (
        ("parallel_swiglu", False),
        ("parallel_swiglu", True),
        ("parallel_wide_silu", False),
        ("parallel_self_product", False),
    )
    for arm, packed_swiglu in cases:
        eager, modules = build_eager(torch.device("cuda"), arm)
        fused = build_fused(torch.device("cuda"), arm, packed_swiglu=packed_swiglu)
        expected, eager_cache = service_forward(eager, prompt, use_cache=True)
        actual, fused_cache = service_forward(fused, prompt, use_cache=True)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.03, rtol=0.02)
        eager_legacy = eager_cache.to_legacy_cache()
        fused_legacy = fused_cache.to_legacy_cache()
        assert len(fused_legacy) == len(eager_legacy)
        for (eager_key, eager_value), (fused_key, fused_value) in zip(eager_legacy, fused_legacy):
            torch.testing.assert_close(fused_key.float(), eager_key.float(), atol=0.03, rtol=0.02)
            torch.testing.assert_close(fused_value.float(), eager_value.float(), atol=0.03, rtol=0.02)
        expected_next, eager_cache = service_forward(eager, next_token, eager_cache, use_cache=True)
        actual_next, fused_cache = service_forward(fused, next_token, fused_cache, use_cache=True)
        torch.testing.assert_close(actual_next.float(), expected_next.float(), atol=0.03, rtol=0.02)
        assert eager_cache.get_seq_length() == fused_cache.get_seq_length() == 17
        eager_updated_legacy = eager_cache.to_legacy_cache()
        fused_updated_legacy = fused_cache.to_legacy_cache()
        assert len(fused_updated_legacy) == len(eager_updated_legacy)
        for (eager_key, eager_value), (fused_key, fused_value) in zip(
            eager_updated_legacy, fused_updated_legacy
        ):
            torch.testing.assert_close(fused_key.float(), eager_key.float(), atol=0.03, rtol=0.02)
            torch.testing.assert_close(fused_value.float(), eager_value.float(), atol=0.03, rtol=0.02)
        del expected, actual, expected_next, actual_next, eager_cache, fused_cache
        del eager_legacy, fused_legacy, eager_updated_legacy, fused_updated_legacy
        del eager, modules, fused
        torch.cuda.empty_cache()
    del prompt, next_token
