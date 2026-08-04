from __future__ import annotations

import torch

from experiments.interference_expert_lm_pilot import (
    SharedScaleInterferenceWeight,
    fake_quant_activation,
    serving_ledger,
)


def test_fake_quant_activation_is_bounded_and_ste_live() -> None:
    value = torch.linspace(-2.0, 2.0, 33, requires_grad=True).reshape(3, 11)
    quantized = fake_quant_activation(value, 7)
    assert torch.isfinite(quantized).all()
    quantized.sum().backward()
    assert value.grad is not None or value._base.grad is not None


def test_interference_bank_starts_with_zero_secondary_codes() -> None:
    generator = torch.Generator().manual_seed(2833)
    target = torch.randn(17, 13, generator=generator)
    bank = SharedScaleInterferenceWeight(target)
    primary, secondary = bank.weights()
    assert primary.shape == target.shape
    torch.testing.assert_close(secondary, torch.zeros_like(secondary), rtol=0, atol=0)
    assert bank.code_diagnostics()["secondary_code_nonzero_fraction"] == 0.0


def test_serving_ledger_matches_bytes_and_ideal_instruction_units() -> None:
    ledger = serving_ledger()
    baseline = ledger["w8_baseline"]
    candidate = ledger["interference"]
    assert candidate["resident_code_bytes_per_layer"] == baseline["resident_code_bytes_per_layer"]
    assert candidate["bf16_weight_scales_per_layer"] == baseline["bf16_weight_scales_per_layer"]
    assert candidate["ideal_gate_up_mma_k32_units"] == baseline["ideal_gate_up_mma_k32_units"]
    assert candidate["mathematical_products_per_layer"] > baseline["mathematical_products_per_layer"]
