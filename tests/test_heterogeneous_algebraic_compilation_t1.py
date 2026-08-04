from __future__ import annotations

import torch
from torch.nn import functional as F

from experiments.heterogeneous_algebraic_compilation_t1 import (
    INPUT_DIM,
    PREFIX_EXAMPLES,
    ResidualSwiGLUParity,
    build_model,
    compile_support,
    evaluate,
    make_batch,
    make_world,
    signs_to_gf2,
    solve_gf2,
    state_sha256,
)


def test_silu_pair_is_exact_multiplication_identity() -> None:
    values = torch.linspace(-20, 20, 10_001, dtype=torch.float64)
    torch.testing.assert_close(F.silu(values) - F.silu(-values), values)


def test_gf2_solver_recovers_hidden_support() -> None:
    device = torch.device("cpu")
    world = make_world(731)
    hidden, target, _ = make_batch(
        world, PREFIX_EXAMPLES, 40_731, device, route_one_only=True
    )
    matrix, rhs = signs_to_gf2(hidden, target)
    solution, rank, _ = solve_gf2(matrix, rhs)
    assert rank == INPUT_DIM
    assert solution is not None
    recovered = tuple(torch.nonzero(solution, as_tuple=False).flatten().tolist())
    assert recovered == world.support


def test_gf2_solver_abstains_on_inconsistent_random_labels() -> None:
    device = torch.device("cpu")
    world = make_world(947)
    hidden, target, _ = make_batch(
        world,
        PREFIX_EXAMPLES,
        50_947,
        device,
        route_one_only=True,
        random_labels=True,
    )
    matrix, rhs = signs_to_gf2(hidden, target)
    solution, rank, _ = solve_gf2(matrix, rhs)
    assert rank == INPUT_DIM
    assert solution is None


def test_compiled_swiglu_is_exact_on_parity_and_protected_routes() -> None:
    device = torch.device("cpu")
    world = make_world(1213)
    model = build_model(8213, device)
    compile_support(model, world.support)
    metrics = evaluate(model, world, device, seed=91_213, count=8_192)
    assert metrics["accuracy"] == 1.0
    assert metrics["parity_accuracy"] == 1.0
    assert metrics["protected_accuracy"] == 1.0


def test_candidate_and_control_start_byte_identical() -> None:
    device = torch.device("cpu")
    first = build_model(7731, device)
    second = build_model(7731, device)
    assert state_sha256(first) == state_sha256(second)
    compile_support(first, make_world(731).support)
    assert state_sha256(first) != state_sha256(second)


def test_architecture_has_only_dense_bias_free_linear_parameters() -> None:
    model = ResidualSwiGLUParity()
    assert all(parameter.ndim == 2 for parameter in model.parameters())
    assert not any("bias" in name for name, _ in model.named_parameters())

