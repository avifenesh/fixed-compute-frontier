import torch

from experiments.coalesced_attention_ffn_gate import (
    causal_gqa,
    coalesced_forward,
    dedicated_endpoint,
    resource_ledger,
    run_stage0,
)


def test_frozen_ledger():
    ledger = resource_ledger(384, 1024, 6, 2, 64)
    assert ledger.kv_dim == 128
    assert ledger.candidate_width == 1344
    assert ledger.baseline_input_rows == ledger.candidate_input_rows == 2688
    assert ledger.baseline_output_columns == 1408
    assert ledger.candidate_output_columns == 1344
    assert ledger.baseline_dense_parameters == 1_572_864
    assert ledger.candidate_dense_parameters == 1_548_288
    assert ledger.candidate_to_baseline == 0.984375
    assert ledger.independent_swiglu_floor == 704


def test_dedicated_containment_endpoint_and_fused_down():
    result = run_stage0()
    assert result["stage0_pass"]
    assert result["checks"]["dedicated_endpoint_max_abs_error"] < 1e-10
    assert result["checks"]["fused_down_max_abs_error"] < 1e-10


def test_causality_and_gradients():
    torch.manual_seed(4)
    batch, tokens, hidden = 2, 6, 24
    query_heads, kv_heads, head_dim = 3, 1, 8
    kv_dim = kv_heads * head_dim
    width = 48
    x = torch.randn(batch, tokens, hidden, dtype=torch.float64, requires_grad=True)
    gate = torch.randn(width, hidden, dtype=torch.float64, requires_grad=True)
    up = torch.randn(width, hidden, dtype=torch.float64, requires_grad=True)
    down = torch.randn(hidden, width, dtype=torch.float64, requires_grad=True)
    output, _ = coalesced_forward(x, gate, up, down, query_heads, kv_heads, head_dim)
    perturbed = x.detach().clone()
    perturbed[:, -1] += torch.randn_like(perturbed[:, -1])
    changed, _ = coalesced_forward(perturbed, gate, up, down, query_heads, kv_heads, head_dim)
    torch.testing.assert_close(output[:, :-1], changed[:, :-1], atol=0, rtol=0)
    output.square().mean().backward()
    for parameter in (x, gate, up, down):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0


def test_gqa_reference_shapes():
    query = torch.randn(2, 5, 24)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)
    output = causal_gqa(query, key, value, 3, 1, 8)
    assert output.shape == query.shape


def test_dual_use_ablation_is_live():
    ledger = resource_ledger(24, 32, 3, 1, 8)
    torch.manual_seed(8)
    x = torch.randn(2, 5, 24)
    gate = torch.randn(ledger.candidate_width, 24)
    up = torch.randn_like(gate)
    down = torch.randn(24, ledger.candidate_width)
    full, _ = coalesced_forward(x, gate, up, down, 3, 1, 8)
    ablated, _ = coalesced_forward(
        x, gate, up, down, 3, 1, 8, zero_attention_local_products=True
    )
    assert not torch.equal(full, ablated)
