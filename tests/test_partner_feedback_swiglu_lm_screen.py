import pytest


torch = pytest.importorskip("torch")

from experiments.partner_feedback_swiglu_lm_screen import (  # noqa: E402
    CHART,
    VALID_ARMS,
    build_model,
)


def build_seeded(arm, device):
    torch.manual_seed(809)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(809)
    return build_model(device, arm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_endpoint_and_parameter_counts():
    device = torch.device("cuda")
    input_ids = torch.randint(0, 1000, (2, 16), device=device)
    outputs = {}
    counts = {}
    for arm in VALID_ARMS:
        model, modules = build_seeded(arm, device)
        counts[arm] = (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs[arm] = model(input_ids=input_ids, use_cache=False).logits.float()
        if arm == "raw_baseline":
            for layer in model.model.layers:
                assert float(layer.mlp.up_proj.weight[0, 0]) == pytest.approx(CHART)
        else:
            for layer in model.model.layers:
                assert float(layer.mlp.canonical_up_weight()[0, 0]) == pytest.approx(CHART)
                assert float(layer.mlp.carrier_beta) == 0.0
        del model, modules
    assert len(set(counts.values())) == 1
    for arm in VALID_ARMS[1:]:
        torch.testing.assert_close(outputs[arm], outputs["raw_baseline"], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_carrier_gradient_and_ablation_modes_are_live():
    device = torch.device("cuda")
    model, modules = build_seeded("partner_feedback", device)
    with torch.no_grad():
        for module in modules:
            module.carrier_beta.fill_(0.05)
    hidden = torch.randn(3, 7, 384, device=device, requires_grad=True)
    full = modules[0](hidden)
    modules[0].ablation_mode = "zero"
    zero = modules[0](hidden)
    modules[0].ablation_mode = "self"
    self_value = modules[0](hidden)
    modules[0].ablation_mode = "shifted"
    shifted = modules[0](hidden)
    assert not torch.equal(full, zero)
    assert not torch.equal(full, self_value)
    assert not torch.equal(full, shifted)
    modules[0].ablation_mode = "full"
    full.square().mean().backward()
    gradient = modules[0].carrier_beta.grad
    assert torch.isfinite(gradient) and float(gradient.abs()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_training_forward_matches_folded_deployment_weights():
    device = torch.device("cuda")
    _, modules = build_seeded("partner_feedback", device)
    module = modules[0]
    with torch.no_grad():
        module.carrier_beta.fill_(0.05)
    hidden = torch.randn(2, 11, 384, device=device)
    actual = module(hidden)
    coefficient = module.lambda_value(hidden.dtype)
    scale = 1.0 + 4.0 * coefficient
    deployed_up = module.canonical_up_weight().clone()
    deployed_up[0] *= scale
    deployed_down = module.down_proj.weight.clone()
    deployed_down[:, 0] /= scale
    gate = module.gate_proj(hidden)
    up = torch.nn.functional.linear(hidden, deployed_up)
    z0 = torch.nn.functional.silu(gate) * up
    partner = module.partner_values(z0, "partner")
    activation = torch.nn.functional.silu(
        gate + coefficient * partner.clamp(-1, 1)
    ) * up
    expected = torch.nn.functional.linear(activation, deployed_down)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
