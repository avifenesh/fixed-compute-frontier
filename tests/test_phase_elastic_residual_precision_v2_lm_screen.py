import torch

from experiments.phase_elastic_residual_precision_v2_lm_screen import (
    BRANCH_WIDTH, ResidualQuantizedWeight, serving_ledger,
)


def test_v2_ledger_reallocates_bits_without_exceeding_bf16():
    ledger=serving_ledger()
    assert BRANCH_WIDTH==1472
    assert ledger["base_int8_ternary_code_bytes_per_layer"]==1_474_560
    assert ledger["base_bf16_scale_bytes_per_layer"]==9_728
    assert ledger["residual_w4_code_bytes_per_layer"]==847_872
    assert ledger["residual_bf16_scale_bytes_per_layer"]==6_656
    assert ledger["candidate_ffn_bytes_per_layer"]==2_338_816
    assert ledger["storage_slack_bytes_per_layer"]==20_480
    assert ledger["decode_to_baseline_mac_ratio"]==2.4375
    assert ledger["fits_bf16_bytes"]


def test_residual_quantized_weight_has_exact_codebooks_and_gradients():
    torch.manual_seed(11)
    bank=ResidualQuantizedWeight(torch.randn(13,17))
    audit=bank.audit()
    assert audit["base_code_min"]>=-127 and audit["base_code_max"]<=127
    assert set(audit["delta_code_values"])<= {-1,0,1}
    decoded=bank();decoded.square().mean().backward()
    for parameter in bank.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

