from __future__ import annotations

import torch

from experiments import heterogeneous_family_discovery_t6 as experiment


def fake_layout() -> experiment.TokenLayout:
    count = experiment.TASKS + 2 * experiment.INPUT_DIM + 4
    selected = tuple(range(count))
    cursor = 0
    tasks = selected[cursor : cursor + experiment.TASKS]
    cursor += experiment.TASKS
    zero = selected[cursor : cursor + experiment.INPUT_DIM]
    cursor += experiment.INPUT_DIM
    one = selected[cursor : cursor + experiment.INPUT_DIM]
    cursor += experiment.INPUT_DIM
    return experiment.TokenLayout(tasks, zero, one, *selected[cursor : cursor + 4])


def test_candidate_support_space_is_complete() -> None:
    candidates = experiment.candidate_supports()
    assert candidates.shape == (12_870, experiment.INPUT_DIM)
    assert torch.all(candidates.sum(1) == experiment.SUPPORT_SIZE)
    assert len({tuple(row.tolist()) for row in candidates}) == len(candidates)


def test_four_family_truth_tables_are_distinct() -> None:
    counts = torch.arange(experiment.SUPPORT_SIZE + 1)
    tables = [tuple(experiment.family_labels(counts, family).tolist()) for family in range(4)]
    assert len(set(tables)) == 4
    assert tables[0] == (False, True, False, True, False, True, False, True, False)


def test_world_is_uniquely_identifiable_and_random_tasks_abstain() -> None:
    world = experiment.make_world()
    supports, families, accepted = experiment.discover(world)
    assert accepted[: experiment.STRUCTURED_TASKS].all()
    assert not accepted[experiment.STRUCTURED_TASKS :].any()
    assert torch.equal(
        supports[: experiment.STRUCTURED_TASKS],
        world.supports[: experiment.STRUCTURED_TASKS],
    )
    assert torch.equal(
        families[: experiment.STRUCTURED_TASKS],
        world.families[: experiment.STRUCTURED_TASKS],
    )


def test_task_families_are_balanced() -> None:
    world = experiment.make_world()
    for family in range(4):
        assert int((world.families == family).sum()) == experiment.TASKS_PER_FAMILY
    assert int((world.families == -1).sum()) == experiment.RANDOM_TASKS


def test_compiler_records_exact_discovery_and_fixed_width() -> None:
    world = experiment.make_world()
    supports, families, accepted = experiment.discover(world)
    model = experiment.t3.build_model(31, torch.device("cpu"))
    frozen, metadata = experiment.compile_interpreter(
        model, supports, families, accepted, fake_layout()
    )
    assert metadata["accepted_tasks"] == experiment.STRUCTURED_TASKS
    assert metadata["accepted_random_tasks"] == 0
    assert metadata["exact_supports"] == experiment.STRUCTURED_TASKS
    assert metadata["exact_families"] == experiment.STRUCTURED_TASKS
    assert metadata["shared_width_independent_of_tasks"] is True
    assert metadata["used_decoder_channels"] <= experiment.PROGRAM_CHANNELS
    assert frozen.count > metadata["fixed_nonzero_entries"]


def test_model_shape_is_arm_independent() -> None:
    model = experiment.t3.build_model(37, torch.device("cpu"))
    before = experiment.t3.parameter_count()
    world = experiment.make_world()
    supports, families, accepted = experiment.discover(world)
    experiment.compile_interpreter(model, supports, families, accepted, fake_layout())
    assert sum(parameter.numel() for parameter in model.parameters()) == before
