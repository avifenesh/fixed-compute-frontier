import pytest
import torch

from experiments.gauge_zero_g1_lm_pilot import (
    ARMS,
    build_model,
    coefficient_serving_diagnostics,
    evaluate_disable_shear,
    optimizer_for,
    optimizer_state_bytes,
)


def test_all_lm_arms_match_parameter_value_and_tensor_counts():
    values = []
    tensors = []
    for arm in ARMS:
        model, modules = build_model(torch.device("cpu"), arm)
        values.append(sum(parameter.numel() for parameter in model.parameters()))
        tensors.append(len(list(model.parameters())))
        for module in modules:
            assert module.dense_serialized_values().numel() == sum(
                parameter.numel() for parameter in module.parameters()
            )
    assert len(set(values)) == 1
    assert len(set(tensors)) == 1


def test_optimizer_uses_one_identical_weight_decay_group():
    model, _ = build_model(torch.device("cpu"), "gauge_zero_g1")
    optimizer = optimizer_for(model, 3e-4, 0.1)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in model.parameters()
    }


def test_bf16_serving_coefficient_reads_the_same_dense_weight():
    model, modules = build_model(torch.device("cpu"), "gauge_zero_g1")
    with torch.no_grad():
        for layer, module in enumerate(modules):
            module.coefficient_physical_weights().fill_((layer + 1) * 1e-4)
    diagnostics = coefficient_serving_diagnostics(modules)
    assert diagnostics["bf16_export_coefficient_bit_exact"]
    assert diagnostics["coefficient_bf16_nonzero_fraction"] == 1.0


def test_disable_shear_ablation_preserves_weights_and_restores_candidate(monkeypatch):
    model, modules = build_model(torch.device("cpu"), "gauge_zero_g1")
    before = [module.key_weight.detach().clone() for module in modules]

    def fake_evaluate(model, validation_file, eval_batches, eval_batch_size, device):
        assert all(module.arm == "canonical_bilinear_control" for module in modules)
        assert all(
            torch.equal(module.key_weight, expected)
            for module, expected in zip(modules, before)
        )
        return {"loss": 1.0, "batch_losses": [1.0]}

    monkeypatch.setattr(
        "experiments.gauge_zero_g1_lm_pilot.evaluate", fake_evaluate
    )
    args = type("Args", (), {"eval_batches": 1, "eval_batch_size": 1})()
    result = evaluate_disable_shear(
        model, modules, None, args, torch.device("cpu")
    )

    assert result["loss"] == 1.0
    assert all(module.arm == "gauge_zero_g1" for module in modules)
    assert all(
        torch.equal(module.key_weight, expected)
        for module, expected in zip(modules, before)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA optimizer-state test")
def test_optimizer_state_bytes_match_after_one_real_step():
    records = []
    inputs = torch.randint(0, 32000, (1, 8), device="cuda")
    for arm in ARMS:
        torch.manual_seed(313)
        torch.cuda.manual_seed_all(313)
        model, _ = build_model(torch.device("cuda"), arm)
        optimizer = optimizer_for(model, 3e-4, 0.1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = logits.float().square().mean()
        loss.backward()
        optimizer.step()
        records.append(optimizer_state_bytes(optimizer))
        del optimizer, model
        torch.cuda.empty_cache()
    assert len(set(records)) == 1
