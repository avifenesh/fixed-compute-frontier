from __future__ import annotations

import torch

from experiments import heterogeneous_algebraic_compilation_t1b as t1b
from experiments import heterogeneous_algebraic_compilation_t2_lm as lm


def test_parameter_count_and_tied_output() -> None:
    model = lm.CoexistenceLM()
    assert lm.parameter_count() == 36_619_776
    tokens = torch.zeros(2, 4, dtype=torch.long)
    logits = model(tokens)
    assert logits.shape == (2, 4, lm.VOCAB)


def test_compiled_partition_solves_algebra_in_full_lm() -> None:
    device = torch.device("cpu")
    world = t1b.make_mixed_world(731)
    model = lm.build_model(20_731, device)
    support, rank, _ = lm.solve_prefix(world, device)
    frozen, compiler = lm.compile_partition(model, support)
    assert rank == 32
    assert support == world.observed_support
    assert compiler["reserved_hidden_coordinates"] == 9 + len(world.observed_support)
    assert compiler["bf16_stable_product_tree"]
    metrics = lm.evaluate_algebra(model, world, device, 12345, count=512)
    assert metrics["parity_accuracy"] == 1.0
    assert metrics["protected_accuracy"] == 1.0
    frozen.enforce(model)


def test_frozen_entries_restore_after_arbitrary_mutation() -> None:
    device = torch.device("cpu")
    world = t1b.make_mixed_world(947)
    model = lm.build_model(20_947, device)
    frozen, _ = lm.compile_partition(model, world.observed_support)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))
    frozen.enforce(model)
    for name, parameter in model.named_parameters():
        mask = frozen.masks[name]
        torch.testing.assert_close(parameter[mask], before[name][mask])


def test_frozen_gradients_are_zeroed_before_clipping() -> None:
    model = lm.CoexistenceLM()
    world = t1b.make_mixed_world(731)
    frozen, _ = lm.compile_partition(model, world.observed_support)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    frozen.mask_gradients(model)
    for name, parameter in model.named_parameters():
        assert torch.count_nonzero(parameter.grad[frozen.masks[name]]) == 0


def test_natural_token_rows_ignore_compiler_coordinates() -> None:
    model = lm.CoexistenceLM()
    world = t1b.make_mixed_world(731)
    _, compiler = lm.compile_partition(model, world.observed_support)
    compiler_dims = compiler["reserved_hidden_coordinates"]
    assert torch.count_nonzero(model.token.weight[: lm.BASE_VOCAB, :compiler_dims]) == 0
    assert torch.count_nonzero(model.token.weight[lm.BASE_VOCAB :, compiler_dims:]) == 0


def test_candidate_and_controls_start_identically() -> None:
    first = lm.build_model(20_731, torch.device("cpu"))
    second = lm.build_model(20_731, torch.device("cpu"))
    assert lm.state_sha256(first) == lm.state_sha256(second)
    lm.compile_partition(first, t1b.make_mixed_world(731).observed_support)
    assert lm.state_sha256(first) != lm.state_sha256(second)


def test_common_schedule_warms_up_and_decays() -> None:
    assert lm.scheduled_learning_rate(1) == lm.PEAK_LEARNING_RATE / lm.WARMUP_STEPS
    assert lm.scheduled_learning_rate(lm.WARMUP_STEPS) == lm.PEAK_LEARNING_RATE
    assert lm.scheduled_learning_rate(1000) < lm.PEAK_LEARNING_RATE
    assert abs(lm.scheduled_learning_rate(2000) - 0.1 * lm.PEAK_LEARNING_RATE) < 1e-12
