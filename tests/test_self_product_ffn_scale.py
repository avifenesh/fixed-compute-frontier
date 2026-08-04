import pytest

torch = pytest.importorskip("torch")

from experiments.self_product_ffn_scale import (
    BASELINE_INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
    LAYERS,
    VALID_ARMS,
    WIDE,
    build_scale_model,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scale_models_have_exact_equal_counts_and_live_gradients():
    device = torch.device("cuda")
    counts = {}
    for arm in VALID_ARMS:
        model, modules = build_scale_model(device, arm)
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        inputs = torch.randint(0, 1000, (2, 16), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=inputs, use_cache=False).logits.float().square().mean()
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        if arm != "parallel_swiglu": assert len(modules) == LAYERS
        del model, modules; torch.cuda.empty_cache()
    assert len(set(counts.values())) == 1
    assert 3 * HIDDEN_SIZE * BASELINE_INTERMEDIATE_SIZE == 2 * HIDDEN_SIZE * WIDE == 3_440_640
