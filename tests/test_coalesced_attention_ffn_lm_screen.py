import pytest


torch = pytest.importorskip("torch")

from experiments.coalesced_attention_ffn_lm_screen import (  # noqa: E402
    BASELINE_INTERMEDIATE_SIZE,
    CANDIDATE_WIDTH,
    CoalescedDecoderLayer,
    HIDDEN_SIZE,
    VALID_ARMS,
    activation_diagnostics,
    build_model,
    dense_projection_parameters,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_arms_forward_backward_and_counts():
    device = torch.device("cuda")
    counts = {}
    for arm in VALID_ARMS:
        torch.manual_seed(1601)
        torch.cuda.manual_seed_all(1601)
        model, modules = build_model(device, arm)
        inputs = torch.randint(0, 1000, (2, 16), device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = logits.float().square().mean()
        loss.backward()
        assert logits.shape[:2] == inputs.shape
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        if arm.startswith("coalesced"):
            assert len(modules) == 12
            assert all(isinstance(module, CoalescedDecoderLayer) for module in modules)
        del model, modules
        torch.cuda.empty_cache()
    assert dense_projection_parameters("sequential_baseline") == 1_572_864
    assert dense_projection_parameters("parallel_baseline") == 1_572_864
    assert dense_projection_parameters("coalesced_1024") == 3 * HIDDEN_SIZE * BASELINE_INTERMEDIATE_SIZE
    assert dense_projection_parameters("coalesced_1344") == 1_548_288
    assert CANDIDATE_WIDTH == 1344
    assert counts["coalesced_1344"] < counts["sequential_baseline"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coalesced_model_is_causal_and_ablation_live():
    device = torch.device("cuda")
    torch.manual_seed(1601)
    model, modules = build_model(device, "coalesced_1344")
    model.eval()
    inputs = torch.randint(0, 1000, (2, 12), device=device)
    changed = inputs.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 1000
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        original = model(input_ids=inputs, use_cache=False).logits.float()
        perturbed = model(input_ids=changed, use_cache=False).logits.float()
        for module in modules:
            module.zero_attention_local_products = True
        ablated = model(input_ids=inputs, use_cache=False).logits.float()
    torch.testing.assert_close(original[:, :-1], perturbed[:, :-1], atol=0, rtol=0)
    assert not torch.equal(original, ablated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_layer_activation_envelope_diagnostics():
    device = torch.device("cuda")

    class Validation:
        def batch(self, _index, batch_size, requested_device):
            inputs = torch.randint(0, 1000, (batch_size, 16), device=requested_device)
            return inputs, inputs

    torch.manual_seed(1601)
    baseline, _ = build_model(device, "sequential_baseline")
    baseline_diagnostics = activation_diagnostics(
        baseline, [], Validation(), device
    )
    assert baseline_diagnostics["joint_nonfinite_fraction_layer_max"] == 0.0
    assert baseline_diagnostics["joint_abs_max_layer_max"] >= baseline_diagnostics["joint_abs_p99_layer_max"]
    reference_rms = baseline_diagnostics["joint_rms_layer_median"]
    del baseline
    torch.cuda.empty_cache()

    torch.manual_seed(1601)
    candidate, modules = build_model(device, "coalesced_1344")
    candidate_diagnostics = activation_diagnostics(
        candidate, modules, Validation(), device, reference_rms
    )
    assert candidate_diagnostics["joint_nonfinite_fraction_layer_max"] == 0.0
    assert 0.0 <= candidate_diagnostics[
        "joint_outside_4x_reference_rms_fraction_layer_max"
    ] <= 1.0
