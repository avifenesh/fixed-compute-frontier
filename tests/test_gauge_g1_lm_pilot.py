import pytest
import torch
import torch.nn.functional as F

from experiments.gauge_g1_lm_pilot import ARMS, build_model, optimizer_for


def test_all_lm_arms_have_exact_same_parameter_count():
    counts = []
    for arm in ARMS:
        model, modules = build_model(torch.device("cpu"), arm)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        if modules:
            for module in modules:
                assert module.physical_serialized_values().numel() == sum(
                    parameter.numel()
                    for parameter in (
                        module.query_weight,
                        module.key_nonpivot,
                        module.polar,
                        module.v_proj.weight,
                        module.o_proj.weight,
                    )
                )
    assert len(set(counts)) == 1


def test_polar_optimizer_partition_is_exact_and_exempts_only_polar_from_decay():
    model, modules = build_model(torch.device("cpu"), "physical_scale_g1")
    optimizer = optimizer_for(model, modules, 3e-4, 0.1)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    grouped = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert grouped == {id(parameter) for parameter in model.parameters()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 parity test")
def test_full_model_bf16_initial_nll_parity_on_actual_tokens():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(1709)
    inputs = torch.randint(0, 32000, (2, 32), generator=generator).to(device)

    def logits_for(arm: str) -> torch.Tensor:
        torch.manual_seed(991)
        torch.cuda.manual_seed_all(991)
        model, _ = build_model(device, arm)
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs, use_cache=False).logits.float().cpu()
        del model
        torch.cuda.empty_cache()
        return logits

    raw = logits_for("raw_baseline")
    control = logits_for("polar_bilinear_control")
    candidate = logits_for("physical_scale_g1")

    def nll(logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            inputs[:, 1:].cpu().reshape(-1),
        )

    # A tiny sample is a numerical smoke test, not the aggregate decision gate.
    # The formal pilot separately requires <=1e-4 mean held-out NLL drift.
    assert abs(float(nll(control) - nll(raw))) <= 1e-3
    torch.testing.assert_close(candidate, control, atol=0.0, rtol=0.0)
