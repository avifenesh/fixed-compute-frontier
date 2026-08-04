import pytest


torch = pytest.importorskip("torch")

from experiments.generator_edge_ffn_h100 import (  # noqa: E402
    ARMS,
    GeneratorEdgeMLP,
    SelfProductMLP,
    build_arm,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_model_parameter_counts_and_live_gradients():
    device = torch.device("cuda")
    counts = {}
    for arm in ARMS:
        model = build_arm(device, arm)
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        inputs = torch.randint(0, 1000, (2, 16), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = logits.float().square().mean()
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        del model
        torch.cuda.empty_cache()
    assert len(set(counts.values())) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_candidate_and_controls_have_frozen_shapes():
    hidden = torch.randn(2, 7, 384, device="cuda")
    self_product = SelfProductMLP(1 / 384**0.5).cuda()
    candidate = GeneratorEdgeMLP(1 / 384**0.5).cuda()
    duplicate = GeneratorEdgeMLP(1 / 384**0.5, duplicate=True).cuda()
    assert self_product(hidden).shape == hidden.shape
    assert candidate(hidden).shape == hidden.shape
    assert duplicate(hidden).shape == hidden.shape
    assert candidate.features(hidden).shape[-1] == 2048
    duplicated = duplicate.features(hidden)
    torch.testing.assert_close(duplicated[..., :1024], duplicated[..., 1024:], atol=0, rtol=0)
