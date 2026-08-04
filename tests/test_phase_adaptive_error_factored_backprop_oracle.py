import torch

from experiments.phase_adaptive_error_factored_backprop_oracle import (
    aggregate,
    combined_direction_metrics,
    error_factored_backward_oracle,
    frontier_projection,
)


def test_combined_direction_metric_handles_joint_linear_gradients() -> None:
    exact_input = torch.tensor([[1.0, 0.0]])
    exact_weight = torch.tensor([[0.0, 2.0]])
    same = combined_direction_metrics(
        exact_input,
        exact_weight,
        3.0 * exact_input,
        3.0 * exact_weight,
    )
    assert abs(same["cosine_squared"] - 1.0) < 1e-7


def test_rank_one_error_reproduces_exact_backward_cheaply() -> None:
    generator = torch.Generator().manual_seed(3)
    left = torch.randn(64, 1, generator=generator)
    right = torch.randn(32, 1, generator=generator)
    error = left @ right.T
    inputs = torch.randn(64, 32, generator=generator)
    weight = torch.randn(32, 32, generator=generator)
    result = error_factored_backward_oracle(inputs, error, weight)
    selected = result["selected"]
    assert selected["rank"] == 1
    assert not selected["dense_fallback"]
    assert selected["cosine_squared"] > 0.999999
    assert selected["charged_backward_cost_ratio"] < 0.08


def test_isotropic_error_uses_dense_fallback_at_small_shape() -> None:
    generator = torch.Generator().manual_seed(5)
    error = torch.randn(64, 32, generator=generator)
    inputs = torch.randn(64, 32, generator=generator)
    weight = torch.randn(32, 32, generator=generator)
    result = error_factored_backward_oracle(inputs, error, weight)
    assert result["selected"]["cosine_squared"] >= 0.99
    assert result["selected"]["charged_backward_cost_ratio"] <= 1.0


def test_aggregate_and_frontier_projection_are_consistent() -> None:
    generator = torch.Generator().manual_seed(9)
    left = torch.randn(64, 1, generator=generator)
    right = torch.randn(32, 1, generator=generator)
    row = error_factored_backward_oracle(
        torch.randn(64, 32, generator=generator),
        left @ right.T,
        torch.randn(32, 32, generator=generator),
    )
    summary = aggregate({"one": row})
    projected = frontier_projection(summary)
    assert summary["charged_backward_cost_ratio"] < 0.08
    assert summary["energy_weighted_cosine_squared"] >= 0.99
    assert projected["optimistic_descent_per_compute"] > 1.0
