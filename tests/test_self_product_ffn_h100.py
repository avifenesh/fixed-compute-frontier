import pytest


torch = pytest.importorskip("torch")

from experiments.self_product_ffn_h100 import build_folded  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_folded_models_have_equal_counts_and_live_outputs():
    device = torch.device("cuda")
    counts = {}
    for arm in ("parallel_swiglu", "parallel_wide_silu", "parallel_self_product"):
        model, modules = build_folded(device, arm)
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        if modules:
            assert all(module.deployment_folded for module in modules)
        inputs = torch.randint(0, 1000, (2, 16), device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits
        assert torch.isfinite(logits).all()
        del model, modules; torch.cuda.empty_cache()
    assert len(set(counts.values())) == 1
