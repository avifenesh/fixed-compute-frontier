from __future__ import annotations

import torch

from experiments import heterogeneous_algebraic_compilation_t1 as base
from experiments import heterogeneous_algebraic_compilation_t1b as extension


def test_mixing_is_dense_invertible_and_not_a_permutation() -> None:
    world = extension.make_mixed_world(731)
    matrix = extension.mixing_tensor(world)
    solution, rank, _ = base.solve_gf2(
        matrix, torch.arange(base.INPUT_DIM).remainder(2).to(torch.uint8)
    )
    assert solution is not None
    assert rank == base.INPUT_DIM
    assert float(matrix.float().mean()) > 0.2
    assert not bool((matrix.sum(0) == 1).all() and (matrix.sum(1) == 1).all())


def test_mixed_observed_mask_reproduces_latent_target() -> None:
    device = torch.device("cpu")
    world = extension.make_mixed_world(947)
    hidden, target, _ = extension.make_mixed_batch(
        world, 4096, 1234, device, route_one_only=True
    )
    observed_bits = (hidden[:, : base.INPUT_DIM] < 0).to(torch.uint8)
    mask = torch.zeros(base.INPUT_DIM, dtype=torch.uint8)
    mask[list(world.observed_support)] = 1
    predicted = (observed_bits.to(torch.int16) @ mask.to(torch.int16)).remainder(2)
    torch.testing.assert_close(predicted, (target < 0).to(torch.int16))


def test_solver_and_compiler_pass_dense_mixing() -> None:
    device = torch.device("cpu")
    world = extension.make_mixed_world(1213)
    result = extension.candidate(
        world, "mixed", extension.make_mixed_batch, 8213, device
    )
    assert result["support_exact"]
    assert result["evaluation"]["accuracy"] == 1.0
    assert result["evaluation"]["protected_accuracy"] == 1.0


def test_consecutive_recurrence_prefix_is_full_rank_and_exact() -> None:
    world = extension.make_mixed_world(731)
    windows, targets, _ = extension.full_rank_recurrence_prefix(world)
    solution, rank, _ = base.solve_gf2(windows, targets)
    assert solution is not None
    assert rank == base.INPUT_DIM
    recovered = tuple(torch.nonzero(solution, as_tuple=False).flatten().tolist())
    assert recovered == world.observed_support


def test_solver_and_compiler_pass_raw_recurrence() -> None:
    device = torch.device("cpu")
    world = extension.make_mixed_world(947)
    result = extension.candidate(
        world, "recurrence", extension.make_recurrence_batch, 7947, device
    )
    assert result["support_exact"]
    assert result["evaluation"]["accuracy"] == 1.0
    assert result["evaluation"]["parity_accuracy"] == 1.0


def test_rademacher_rebirth_is_data_independent_and_changes_weights() -> None:
    device = torch.device("cpu")
    first = base.build_model(7731, device)
    second = base.build_model(7731, device)
    initial = base.state_sha256(first)
    extension.rademacher_rebirth(first, 123, 0.5)
    extension.rademacher_rebirth(second, 123, 0.5)
    assert base.state_sha256(first) == base.state_sha256(second)
    assert base.state_sha256(first) != initial

