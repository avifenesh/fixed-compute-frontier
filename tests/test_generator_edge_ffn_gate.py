import torch

from experiments.generator_edge_ffn_gate import (
    edge_indices,
    generator_edge_features,
    generator_edge_forward,
    graph_checks,
    parameter_ledger,
    run_stage0,
)


def test_exact_ledger_and_graph():
    ledger = parameter_ledger(384, 1024)
    assert ledger["swiglu_parameters"] == 1_179_648
    assert len({value for key, value in ledger.items() if key.endswith("parameters")}) == 1
    graph = graph_checks(1024)
    assert graph["edges"] == graph["unique_edges"] == 2048
    assert graph["out_degree_min"] == graph["out_degree_max"] == 2
    assert graph["in_degree_min"] == graph["in_degree_max"] == 2
    assert graph["first_route_is_permutation"]
    assert graph["second_route_is_permutation"]


def test_reference_forward_and_gradients():
    torch.manual_seed(5)
    hidden = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
    bank = torch.randn(16, 8, dtype=torch.float64, requires_grad=True)
    down = torch.randn(8, 32, dtype=torch.float64, requires_grad=True)
    actual = generator_edge_forward(hidden, bank, down)
    generated = hidden @ bank.T
    first, second = edge_indices(16)
    expected_features = torch.cat(
        (torch.nn.functional.silu(generated) * generated[..., first],
         torch.nn.functional.silu(generated) * generated[..., second]),
        dim=-1,
    )
    expected = expected_features @ down.T
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    for tensor in (hidden, bank, down):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0


def test_duplicate_control_is_distinct_but_shape_matched():
    generated = torch.randn(3, 16)
    full = generator_edge_features(generated)
    duplicate = generator_edge_features(generated, duplicate=True)
    assert full.shape == duplicate.shape == (3, 32)
    assert not torch.equal(full, duplicate)
    torch.testing.assert_close(duplicate[..., :16], duplicate[..., 16:], atol=0, rtol=0)


def test_functional_rank_gate():
    result = run_stage0()
    assert result["stage0_cpu_pass"]
    ranks = result["functional_ranks"]
    assert ranks["generator_edge"]["rank"] > ranks["swiglu"]["rank"]
