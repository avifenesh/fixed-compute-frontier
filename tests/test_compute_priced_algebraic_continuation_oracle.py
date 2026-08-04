import torch

from experiments.compute_priced_algebraic_continuation_oracle import (
    BLOCK,
    aggregate_matrix_results,
    analyze_matrix,
    block_sparse_direction,
    direction_metrics,
    factor_pair,
    frontier_projection,
    kronecker_approximation,
)


def test_factor_pair_is_balanced_and_exact() -> None:
    assert factor_pair(48) == (6, 8)
    assert factor_pair(64) == (8, 8)
    assert factor_pair(17) == (1, 17)


def test_direction_metric_is_squared_cosine() -> None:
    gradient = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    same = direction_metrics(gradient, 3.0 * gradient)
    orthogonal = direction_metrics(
        gradient, torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    )
    assert abs(same["cosine_squared"] - 1.0) < 1e-12
    assert orthogonal["cosine_squared"] == 0.0


def test_block_sparse_direction_selects_highest_energy_block() -> None:
    matrix = torch.zeros(2 * BLOCK, 2 * BLOCK)
    matrix[:BLOCK, BLOCK:] = 2.0
    matrix[BLOCK:, :BLOCK] = 1.0
    sparse, nonzero = block_sparse_direction(matrix, 0.25)
    assert nonzero == BLOCK * BLOCK
    torch.testing.assert_close(sparse[:BLOCK, BLOCK:], matrix[:BLOCK, BLOCK:])
    assert int(torch.count_nonzero(sparse[BLOCK:, :BLOCK])) == 0


def test_rank_one_kronecker_matrix_is_recovered() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(4, 4, generator=generator)
    right = torch.randn(4, 4, generator=generator)
    matrix = torch.kron(left, right)
    approximation, cost, metadata = kronecker_approximation(matrix, 1)
    torch.testing.assert_close(approximation, matrix, rtol=2e-5, atol=2e-5)
    assert cost == 0.5
    assert metadata["rank"] == 1


def test_oracle_falls_back_to_dense_when_structures_are_not_cheap_enough() -> None:
    generator = torch.Generator().manual_seed(11)
    gradient = torch.randn(32, 32, generator=generator)
    result = analyze_matrix(gradient, include_candidates=False)
    assert result["best_at_quality"]["cosine_squared"] >= 0.90
    assert result["best_at_quality"]["cost_ratio"] <= 1.0
    assert result["best_low_rank_at_quality"]["cost_ratio"] <= 1.0


def test_aggregate_and_frontier_ledger_are_bounded() -> None:
    matrix = torch.eye(32)
    row = analyze_matrix(matrix, include_candidates=False)
    aggregate = aggregate_matrix_results({"one": row})
    projected = frontier_projection(aggregate)
    assert 0.0 < aggregate["mixed_cost_ratio"] <= 1.0
    assert 0.0 < projected["whole_step_cost_ratio_with_refresh"] <= 1.0
    assert projected["optimistic_descent_per_compute"] > 0.0
