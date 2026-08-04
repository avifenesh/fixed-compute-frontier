import pytest


torch = pytest.importorskip("torch")

from experiments.self_product_ffn_lm_screen import (  # noqa: E402
    PLAIN_SCALE,
    SELF_PRODUCT_SCALE,
    VALID_ARMS,
    WIDE,
    activation_diagnostics,
    build_model,
    ffn_dense_parameters,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_exact_counts_forward_and_gradients():
    device = torch.device("cuda")
    counts = {}
    for arm in VALID_ARMS:
        torch.manual_seed(1907); torch.cuda.manual_seed_all(1907)
        model, modules = build_model(device, arm)
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        inputs = torch.randint(0, 1000, (2, 16), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = logits.float().square().mean()
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        if arm != "parallel_swiglu":
            assert len(modules) == 12
        del model, modules; torch.cuda.empty_cache()
    assert len(set(counts.values())) == 1
    assert len({ffn_dense_parameters(arm) for arm in VALID_ARMS}) == 1
    assert WIDE == 1536
    assert PLAIN_SCALE == pytest.approx((2 / 3) ** 0.5)
    assert SELF_PRODUCT_SCALE == pytest.approx(0.44642046792894413)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_folded_deployment_and_plain_ablation_are_exact_and_live():
    device = torch.device("cuda")
    _, modules = build_model(device, "parallel_self_product")
    module = modules[0]
    hidden = torch.randn(2, 11, 384, device=device)
    generated = module.generator_proj(hidden)
    actual = module(hidden)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(generated) * generated,
        module.folded_down_weight(),
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    module.ablation_plain = True
    ablated_unfolded = module(hidden)
    module.ablation_plain = False
    module.fold_for_deployment()
    folded = module(hidden)
    torch.testing.assert_close(actual, folded, atol=2e-5, rtol=2e-5)
    assert module.deployment_folded
    module.ablation_plain = True
    ablated_folded = module(hidden)
    torch.testing.assert_close(ablated_unfolded, ablated_folded, atol=2e-5, rtol=2e-5)
    assert not torch.equal(actual, ablated_folded)

    _, wide_modules = build_model(device, "parallel_wide_silu")
    wide_module = wide_modules[0]
    wide_actual = wide_module(hidden)
    wide_module.fold_for_deployment()
    wide_folded = wide_module(hidden)
    torch.testing.assert_close(wide_actual, wide_folded, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_envelope_diagnostics_cover_all_layers():
    device = torch.device("cuda")

    class Validation:
        def batch(self, _index, batch_size, requested_device):
            values = torch.randint(0, 1000, (batch_size, 16), device=requested_device)
            return values, values

    baseline, _ = build_model(device, "parallel_swiglu")
    baseline_record = activation_diagnostics(baseline, [], Validation(), device)
    assert baseline_record["activation_nonfinite_fraction_layer_max"] == 0.0
    reference = baseline_record["activation_rms_layer_median"]
    del baseline; torch.cuda.empty_cache()
    candidate, modules = build_model(device, "parallel_self_product")
    record = activation_diagnostics(candidate, modules, Validation(), device, reference)
    assert record["activation_nonfinite_fraction_layer_max"] == 0.0
    assert 0 <= record["activation_outside_4x_reference_rms_fraction_layer_max"] <= 1
