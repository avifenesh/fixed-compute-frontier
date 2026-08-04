from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bbcm_matched_learning_screen import (
    BBCMWeightBank,
    IndependentW4WeightBank,
    RoutedQuantizedMLP,
    fake_quant_symmetric,
    fake_quant_ternary,
    initial_symmetric_scale,
    initial_ternary_scale,
    serving_ledger,
)


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)
        self.act_fn = F.silu


def test_fake_quant_forward_is_exact_and_backward_exists():
    torch.manual_seed(3)
    shadow = torch.randn(5, 7, requires_grad=True)
    scale = initial_symmetric_scale(shadow, 7)
    decoded = fake_quant_symmetric(shadow, scale.log(), 7)
    exact = torch.round(shadow.detach() / scale).clamp(-7, 7) * scale
    torch.testing.assert_close(decoded.detach(), exact, rtol=0.0, atol=0.0)
    decoded.sum().backward()
    assert shadow.grad is not None
    assert torch.isfinite(shadow.grad).all()


def test_ternary_forward_uses_only_three_codes_and_backward_exists():
    torch.manual_seed(5)
    shadow = torch.randn(4, 11, requires_grad=True)
    scale = initial_ternary_scale(shadow)
    decoded = fake_quant_ternary(shadow, scale.log())
    codes = decoded.detach() / scale
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    decoded.square().mean().backward()
    assert shadow.grad is not None


def test_weight_banks_have_identical_routes_at_initialization():
    torch.manual_seed(7)
    target = torch.randn(13, 17)
    for bank in (BBCMWeightBank(target, 4), IndependentW4WeightBank(target, 4)):
        reference = bank.exact_weight(0)
        for route in range(1, 4):
            torch.testing.assert_close(bank.exact_weight(route), reference, rtol=0.0, atol=0.0)


def test_every_quantized_bank_master_and_scale_receives_finite_nonzero_gradient():
    torch.manual_seed(9)
    target = torch.randn(13, 17)
    bbcm = BBCMWeightBank(target, 4)
    bbcm_loss = sum(bbcm.weight(route).square().mean() for route in range(4))
    bbcm_loss.backward()
    for parameter in (
        bbcm.base_shadow,
        bbcm.base_log_scale,
        bbcm.delta_shadow,
        bbcm.delta_log_scale,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert bool((parameter.grad != 0).any())
    independent = IndependentW4WeightBank(target, 4)
    independent_loss = sum(independent.weight(route).square().mean() for route in range(4))
    independent_loss.backward()
    for parameter in (independent.shadow, independent.log_scale):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert bool((parameter.grad != 0).any())


def test_small_model_serving_ledgers_are_exactly_funded():
    bbcm = serving_ledger("bbcm", 576, 1536, 1520, 4)
    independent = serving_ledger("independent_k4_w4", 576, 1536, 1520, 4)
    assert bbcm["resident_ffn_bytes_per_layer"] == 5293896
    assert independent["resident_ffn_bytes_per_layer"] == 5286664
    assert bbcm["fits_bytes"] and bbcm["fits_macs_including_router"]
    assert independent["fits_bytes"] and independent["fits_macs_including_router"]


def test_sparse_route_grouping_restores_original_token_order():
    torch.manual_seed(13)
    module = RoutedQuantizedMLP(
        TinyMLP(),
        arm="bbcm",
        routes=4,
        target_hidden=6,
        layer_index=0,
        warmup_steps=0,
        keep=torch.arange(6),
        routing_seed=211,
    )
    module.eval()
    module.router_weight.data.copy_(torch.eye(4))
    module.router_bias.data.zero_()
    inputs = torch.tensor(
        [[[3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 6.0]]]
    )
    actual = module(inputs)
    base_gate = module.gate_bank.base_weight()
    base_up = module.up_bank.base_weight()
    base_down = module.down_bank.base_weight()
    expected_rows = []
    for route, row in enumerate(inputs.reshape(-1, 4)):
        gate = F.linear(row, module.gate_bank.weight(route, base_gate))
        up = F.linear(row, module.up_bank.weight(route, base_up))
        expected_rows.append(F.linear(F.silu(gate) * up, module.down_bank.weight(route, base_down)))
    expected = torch.stack(expected_rows).reshape_as(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.square().mean().backward()
    assert module.router_weight.grad is not None
    assert torch.isfinite(module.router_weight.grad).all()
    assert bool((module.router_weight.grad != 0).any())
