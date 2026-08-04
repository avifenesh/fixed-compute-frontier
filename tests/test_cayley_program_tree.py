from __future__ import annotations

from collections import deque

import torch

from experiments.cayley_program_tree import (
    CayleyProgramTreeMLP,
    ImplicitSparseSkewBasis,
    exact_ledger,
)


def _topology_diameter(width: int, topology_offset: int) -> int:
    basis = ImplicitSparseSkewBasis(
        width,
        degree=3,
        alpha=0.25,
        topology_offset=topology_offset,
    )
    positions = torch.arange(width)
    neighbors = [set() for _ in range(width)]
    edge_sets = []
    for matching in range(3):
        permutation = basis._permutation(positions, matching).tolist()
        assert len(set(permutation)) == width
        edges = {
            tuple(sorted((permutation[index], permutation[index + 1])))
            for index in range(0, width, 2)
        }
        assert len(edges) == width // 2
        edge_sets.append(edges)
        for left, right in edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
    assert all(not (left & right) for index, left in enumerate(edge_sets) for right in edge_sets[index + 1 :])
    assert {len(row) for row in neighbors} == {3}
    diameter = 0
    for source in range(width):
        distances = [-1] * width
        distances[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        assert min(distances) >= 0
        diameter = max(diameter, max(distances))
    return diameter


def test_exact_frozen_parameter_ledger() -> None:
    module = CayleyProgramTreeMLP()
    ledger = exact_ledger()
    assert module.parameter_count() == ledger["candidate_parameters"] == 1_179_583
    assert ledger["parameter_difference"] == -65
    assert ledger["ideal_active_ratio"] == 0.08203125


def test_tree_forward_backward_is_finite() -> None:
    torch.manual_seed(3)
    module = CayleyProgramTreeMLP(width=16, depth=4, seed=11)
    values = torch.randn(9, 16, requires_grad=True)
    output = module(values)
    assert output.shape == values.shape
    assert torch.isfinite(output).all()
    (output.square().mean() + 0.01 * module.last_router_auxiliary).backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert module.payload.grad is not None
    assert module.threshold.grad is not None
    assert all(bank.raw_edge_weights.grad is not None for bank in module.bases.banks)


def test_checkpoint_contains_no_index_tables() -> None:
    module = CayleyProgramTreeMLP()
    state = module.state_dict()
    assert all(tensor.is_floating_point() for tensor in state.values())
    assert sum(tensor.numel() for tensor in state.values()) == module.parameter_count()
    assert not any("permutation" in key or "route_coordinates" in key for key in state)


def test_route_formula_has_no_node_collisions_at_frozen_width() -> None:
    module = CayleyProgramTreeMLP()
    for depth in range(module.depth):
        first_node = 2**depth - 1
        nodes = torch.arange(first_node, first_node + 2**depth)
        coordinates = module._route_coordinates(nodes, depth)
        assert torch.unique(coordinates).numel() == nodes.numel()


def test_formula_topologies_are_connected_at_pilot_and_production_widths() -> None:
    assert [_topology_diameter(384, offset) for offset in (0, 3, 6)] == [
        10,
        10,
        10,
    ]
    assert [_topology_diameter(4096, offset) for offset in (0, 3, 6)] == [
        16,
        15,
        15,
    ]


def test_frozen_width_initialization_stays_bounded() -> None:
    torch.manual_seed(815)
    module = CayleyProgramTreeMLP(seed=815)
    values = torch.randn(64, 384)
    output = module(values)
    output_rms = float(output.detach().square().mean().sqrt())
    assert 0.05 <= output_rms <= 0.50


def test_forced_paths_change_the_function() -> None:
    torch.manual_seed(5)
    module = CayleyProgramTreeMLP(width=16, depth=4, seed=13).eval()
    values = torch.randn(7, 16)
    module.force_bits = (0, 0, 0, 0)
    first = module(values)
    module.force_bits = (1, 1, 1, 1)
    second = module(values)
    assert float(torch.linalg.norm(first - second).detach()) > 1e-3


def test_transpose_basis_approximately_inverts_forward_basis() -> None:
    torch.manual_seed(7)
    module = CayleyProgramTreeMLP(width=16, depth=3, seed=17)
    values = torch.randn(11, 16)
    transformed = module.bases.apply(values, 0)
    recovered = module.bases.apply(transformed, 0, transpose=True)
    relative_error = torch.linalg.norm(recovered - values) / torch.linalg.norm(values)
    assert float(relative_error.detach()) <= 0.01


def test_runtime_transpose_is_the_exact_matrix_transpose() -> None:
    torch.manual_seed(19)
    module = CayleyProgramTreeMLP(width=16, depth=3, seed=23)
    identity = torch.eye(16)
    forward_rows = module.bases.apply(identity, 0)
    transpose_rows = module.bases.apply(identity, 0, transpose=True)
    assert torch.allclose(transpose_rows, forward_rows.T, atol=1e-6, rtol=1e-6)
