from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "sparse_cayley_program.py"
SPEC = importlib.util.spec_from_file_location("sparse_cayley_program", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def dense_generator(bank, expert: int) -> torch.Tensor:
    matrix = torch.zeros(bank.width, bank.width)
    permutations = bank.permutations[expert]
    weights = bank.bounded_edge_weights()[expert].detach()
    for matching in range(bank.degree):
        for pair in range(bank.width // 2):
            first = int(permutations[matching, 2 * pair])
            second = int(permutations[matching, 2 * pair + 1])
            weight = weights[matching, pair]
            matrix[first, second] += weight
            matrix[second, first] -= weight
    return matrix


def test_sparse_application_matches_dense_generators() -> None:
    torch.manual_seed(3)
    bank = MODULE.SparseSkewExpertBank(16, 3, 3, 0.25, 7)
    values = torch.randn(11, 16)
    actual = bank.apply_all(values)
    for expert in range(3):
        expected = values @ dense_generator(bank, expert).T
        torch.testing.assert_close(actual[:, expert], expected, rtol=1e-6, atol=1e-6)


def test_truncated_cayley_is_close_to_exact_and_has_gradients() -> None:
    torch.manual_seed(5)
    projection = MODULE.CayleyProgramProjection(
        16, experts=2, degree=3, route_length=1, neumann_order=4, seed=11
    )
    projection.force_route = 0
    values = torch.randn(13, 16, requires_grad=True)
    actual = projection(values)
    generator = dense_generator(projection.bank, 0)
    identity = torch.eye(16)
    exact_q = (identity - generator) @ torch.linalg.inv(identity + generator)
    expected = values.detach() @ exact_q.T
    relative_error = torch.linalg.norm(actual.detach() - expected) / torch.linalg.norm(expected)
    assert float(relative_error) <= 0.003
    actual.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert projection.bank.raw_edge_weights.grad is not None


def test_dynamic_program_has_router_gradients_and_changes_with_forced_route() -> None:
    torch.manual_seed(13)
    projection = MODULE.CayleyProgramProjection(16, experts=3, seed=17)
    values = torch.randn(7, 16)
    dynamic = projection(values)
    dynamic.square().mean().backward()
    assert projection.router.grad is not None
    assert float(projection.router.grad.abs().sum()) > 0.0
    projection.force_route = 0
    first = projection(values).detach()
    projection.force_route = 1
    second = projection(values).detach()
    assert float(torch.linalg.norm(first - second)) > 1e-4


def test_frozen_ledger_matches_equal_dense_width() -> None:
    ledger = MODULE.logical_ledger()
    assert ledger["candidate_ffn_parameters"] == 36_864
    assert ledger["equal_dense_swiglu_width"] == 32
    assert ledger["equal_dense_swiglu_parameters"] == 36_864
    assert np.isclose(ledger["candidate_to_baseline_ideal_ratio"], 0.06640625)
