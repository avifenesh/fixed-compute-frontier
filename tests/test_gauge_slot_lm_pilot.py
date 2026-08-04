import pytest
import torch
import torch.nn.functional as F

from experiments.gauge_slot_lm_pilot import ARMS, build_model, optimizer_for


def test_all_lm_arms_match_parameter_values_and_tensor_counts():
    value_counts = []
    tensor_counts = []
    for arm in ARMS:
        model, modules = build_model(torch.device("cpu"), arm)
        value_counts.append(sum(parameter.numel() for parameter in model.parameters()))
        tensor_counts.append(len(list(model.parameters())))
        for module in modules:
            assert module.packed_serialized_values().numel() == (
                module.chart_parameter_count()
            )
    assert len(set(value_counts)) == 1
    assert len(set(tensor_counts)) == 1


def test_optimizer_has_one_matched_group_with_every_parameter():
    model, _ = build_model(torch.device("cpu"), "gauge_slot_g2")
    optimizer = optimizer_for(model, 3e-4, 0.1)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in model.parameters()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 parity test")
def test_full_model_candidate_and_control_are_bf16_identical_at_initialization():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(1709)
    inputs = torch.randint(0, 32000, (2, 32), generator=generator).to(device)

    def result_for(arm: str) -> tuple[torch.Tensor, float]:
        torch.manual_seed(991)
        torch.cuda.manual_seed_all(991)
        model, _ = build_model(device, arm)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                inputs[:, 1:].reshape(-1),
            )
        sample = logits[:, :, :128].float().cpu()
        value = float(loss)
        del model, logits, loss
        torch.cuda.empty_cache()
        return sample, value

    control_logits, control_nll = result_for("canonical_bilinear_control")
    candidate_logits, candidate_nll = result_for("gauge_slot_g2")
    torch.testing.assert_close(candidate_logits, control_logits, atol=0.0, rtol=0.0)
    assert candidate_nll == control_nll
