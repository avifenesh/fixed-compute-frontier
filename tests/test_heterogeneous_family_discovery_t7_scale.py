from __future__ import annotations

import torch

from experiments import heterogeneous_family_discovery_t7_scale as experiment


def fake_layout() -> experiment.TokenLayout:
    count = experiment.t6.TASKS + 2 * experiment.t6.INPUT_DIM + 4
    selected = tuple(range(count))
    cursor = 0
    tasks = selected[cursor : cursor + experiment.t6.TASKS]
    cursor += experiment.t6.TASKS
    zero = selected[cursor : cursor + experiment.t6.INPUT_DIM]
    cursor += experiment.t6.INPUT_DIM
    one = selected[cursor : cursor + experiment.t6.INPUT_DIM]
    cursor += experiment.t6.INPUT_DIM
    return experiment.TokenLayout(tasks, zero, one, *selected[cursor : cursor + 4])


def test_adjacent_scale_shape_and_parameter_count() -> None:
    assert experiment.HIDDEN == 640
    assert experiment.LAYERS == 16
    assert experiment.HEADS == 10
    assert experiment.HEAD_DIM == 64
    assert experiment.parameter_count() == 110_776_960


def test_world_is_untouched_and_exactly_discoverable() -> None:
    assert experiment.WORLD_SEED != 131_071
    world = experiment.t6.make_world(experiment.WORLD_SEED)
    supports, families, accepted = experiment.t6.discover(world)
    assert accepted[: experiment.t6.STRUCTURED_TASKS].all()
    assert not accepted[experiment.t6.STRUCTURED_TASKS :].any()
    assert torch.equal(
        supports[: experiment.t6.STRUCTURED_TASKS],
        world.supports[: experiment.t6.STRUCTURED_TASKS],
    )
    assert torch.equal(
        families[: experiment.t6.STRUCTURED_TASKS],
        world.families[: experiment.t6.STRUCTURED_TASKS],
    )


def test_muon_flop_ledger_is_distinct_from_model_training() -> None:
    model = experiment.build_model(11, torch.device("cpu"))
    counted = sum(parameter.numel() for parameter in model.parameters())
    assert counted == experiment.parameter_count()
    assert experiment.muon_step_flops(model) > 0
    assert experiment.training_flops(16, 128, binary_head=False) > 0


def test_compiler_uses_same_model_shape_at_adjacent_scale() -> None:
    world = experiment.t6.make_world(experiment.WORLD_SEED)
    supports, families, accepted = experiment.t6.discover(world)
    model = experiment.build_model(13, torch.device("cpu"))
    before = sum(parameter.numel() for parameter in model.parameters())
    frozen, metadata = experiment.compile_interpreter(
        model,
        supports,
        families,
        accepted,
        fake_layout(),
        world,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == before
    assert metadata["accepted_tasks"] == experiment.t6.STRUCTURED_TASKS
    assert metadata["accepted_random_tasks"] == 0
    assert metadata["exact_supports"] == experiment.t6.STRUCTURED_TASKS
    assert metadata["exact_families"] == experiment.t6.STRUCTURED_TASKS
    assert metadata["used_decoder_channels"] <= experiment.PROGRAM_CHANNELS
    assert frozen.count > metadata["fixed_nonzero_entries"]
