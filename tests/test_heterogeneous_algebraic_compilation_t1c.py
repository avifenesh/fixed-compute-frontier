from __future__ import annotations

import torch

from experiments import heterogeneous_algebraic_compilation_t1 as t1
from experiments import heterogeneous_algebraic_compilation_t1b as t1b
from experiments import heterogeneous_algebraic_compilation_t1c as causal


def test_interpolator_matches_every_hamming_count() -> None:
    for support_size in (13, 17, 18, 24):
        _, _, fitted = causal.interpolation_coefficients(support_size)
        counts = torch.arange(support_size, -1, -1)
        targets = torch.where(counts.remainder(2) == 0, 1.0, -1.0).double()
        torch.testing.assert_close(fitted, targets, atol=1e-8, rtol=1e-8)


def test_causal_prefix_recovers_mixed_support() -> None:
    world = t1b.make_mixed_world(731)
    support, rank, _ = causal.solve_prefix(world, torch.device("cpu"))
    assert rank == t1.INPUT_DIM
    assert support == world.observed_support


def test_compiled_causal_decoder_solves_both_queries() -> None:
    device = torch.device("cpu")
    for seed in t1.WORLD_SEEDS:
        world = t1b.make_mixed_world(seed)
        model = causal.build_model(seed + 9_000, device)
        causal.compile_causal(model, world.observed_support)
        metrics = causal.evaluate(model, world, device, seed + 77_000, count=4_096)
        assert metrics["accuracy"] == 1.0
        assert metrics["parity_accuracy"] == 1.0
        assert metrics["protected_accuracy"] == 1.0


def test_attention_is_causally_masked() -> None:
    source = causal.CausalAttention.forward.__code__.co_names
    assert "triu" in source
    assert "masked_fill" in source


def test_candidate_and_controls_start_identically() -> None:
    device = torch.device("cpu")
    first = causal.build_model(9731, device)
    second = causal.build_model(9731, device)
    assert causal.state_sha256(first) == causal.state_sha256(second)
    causal.rademacher_rebirth(first, 123, 1.0)
    assert causal.state_sha256(first) != causal.state_sha256(second)


def test_all_parameters_are_dense_two_dimensional_matrices() -> None:
    model = causal.CausalSwiGLUDecoder()
    assert all(parameter.ndim == 2 for parameter in model.parameters())
    assert causal.parameter_count() == sum(parameter.numel() for parameter in model.parameters())

