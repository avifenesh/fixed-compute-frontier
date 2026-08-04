import pytest

torch = pytest.importorskip("torch")

from experiments.self_product_ffn_scale_h100 import build_folded, service_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_h100_build_is_folded_and_equal_count():
    device = torch.device("cuda")
    counts = []
    for arm in ("parallel_swiglu", "parallel_wide_silu", "parallel_self_product"):
        model, modules = build_folded(device, arm)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        assert all(module.deployment_folded for module in modules)
        assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
        prompt = torch.randint(0, 1000, (1, 16), device=device)
        logits, cache = service_forward(model, prompt, use_cache=True)
        assert logits.shape == (1, 1, 49152)
        logits, cache = service_forward(model, prompt[:, :1], cache, use_cache=True)
        assert logits.shape == (1, 1, 49152)
        assert cache.get_seq_length() == 17
        cache.crop(16)
        assert cache.get_seq_length() == 16
        del logits, cache, prompt, model, modules; torch.cuda.empty_cache()
    assert len(set(counts)) == 1
