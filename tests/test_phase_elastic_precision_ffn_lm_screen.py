import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.phase_elastic_precision_ffn_lm_screen import (
    BRANCH_WIDTH, D, M, PhaseElasticMLP, QuantizedWeight, serving_ledger,
)


class TinyOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D, M, bias=False)
        self.up_proj = nn.Linear(D, M, bias=False)
        self.down_proj = nn.Linear(M, D, bias=False)
        self.act_fn = F.silu


def test_exact_serving_ledger_fits_bf16_payload():
    ledger = serving_ledger()
    assert BRANCH_WIDTH == 1984
    assert ledger["baseline_bf16_ffn_bytes_per_layer"] == 2_359_296
    assert ledger["base_w8_code_bytes_per_layer"] == 1_179_648
    assert ledger["base_bf16_scale_bytes_per_layer"] == 4_864
    assert ledger["residual_w4_code_bytes_per_layer"] == 1_142_784
    assert ledger["residual_bf16_scale_bytes_per_layer"] == 8_704
    assert ledger["candidate_ffn_bytes_per_layer"] == 2_336_000
    assert ledger["storage_slack_bytes_per_layer"] == 23_296
    assert ledger["decode_to_baseline_mac_ratio"] == 2.9375
    assert ledger["fits_bf16_bytes"]


def test_fake_quant_codes_are_bounded_and_gradients_exist():
    torch.manual_seed(3)
    bank = QuantizedWeight(torch.randn(9, 13), 7)
    decoded = bank()
    audit = bank.audit()
    assert audit["code_min"] >= -7 and audit["code_max"] <= 7
    decoded.square().mean().backward()
    assert bank.shadow.grad is not None and torch.isfinite(bank.shadow.grad).all()
    assert bank.log_scale.grad is not None and torch.isfinite(bank.log_scale.grad).all()


def test_base_and_full_modes_are_distinct_and_base_skips_branch_gradients():
    torch.manual_seed(5)
    module = PhaseElasticMLP(TinyOriginal(), layer_index=0, std=0.02)
    inputs = torch.randn(2, 3, D)
    module.eval()
    module.set_mode("base")
    base = module(inputs)
    module.set_mode("full")
    full = module(inputs)
    assert base.shape == full.shape == inputs.shape
    assert not torch.equal(base, full)

    module.zero_grad(set_to_none=True)
    module.set_mode("base")
    module(inputs).square().mean().backward()
    assert module.base_gate.shadow.grad is not None
    assert module.branch_gate.shadow.grad is None

