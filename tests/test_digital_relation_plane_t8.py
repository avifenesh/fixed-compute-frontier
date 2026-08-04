from __future__ import annotations

import torch

from experiments import digital_relation_plane_t8 as experiment


def fake_layout() -> experiment.RelationLayout:
    count = experiment.ENTITIES + experiment.RELATIONS + experiment.VALUES
    selected = tuple(range(count))
    cursor = 0
    entities = selected[cursor : cursor + experiment.ENTITIES]
    cursor += experiment.ENTITIES
    queries = selected[cursor : cursor + experiment.RELATIONS]
    cursor += experiment.RELATIONS
    results = selected[cursor : cursor + experiment.VALUES]
    return experiment.RelationLayout(entities, queries, results)


def test_relation_codes_are_unique_and_constant_norm() -> None:
    codes = [tuple(experiment.relation_code(index).tolist()) for index in range(64)]
    assert len(set(codes)) == 64
    assert all(sum(value * value for value in code) == 6 for code in codes)


def test_value_levels_and_codes_are_distinct() -> None:
    levels = experiment.value_level(torch.arange(experiment.VALUES))
    assert levels.tolist() == list(range(-15, 16, 2))
    codes = [tuple(experiment.value_code(index).tolist()) for index in range(16)]
    assert len(set(codes)) == 16


def test_every_compiled_entity_has_identical_program_norm() -> None:
    world = experiment.make_world()
    levels = experiment.value_level(world.values)
    for row in levels:
        used = float(row.pow(2).sum().item()) + 1.0
        compensation = max(experiment.ENTITY_NORM_SQUARED - used, 0.0) ** 0.5
        assert abs(used + compensation**2 - experiment.ENTITY_NORM_SQUARED) < 1e-5


def test_compilation_keeps_deployed_parameter_count() -> None:
    model = experiment.t7.build_model(17, torch.device("cpu"))
    before = sum(parameter.numel() for parameter in model.parameters())
    frozen, metadata = experiment.compile_relation_plane(
        model,
        experiment.make_world(),
        fake_layout(),
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == before
    assert metadata["compiled_facts"] == 32_768
    assert metadata["payload_scalars"] == 32_768
    assert metadata["bits_per_payload_scalar"] == 4.0
    assert metadata["decoder_channels"] == 48
    assert frozen.count > metadata["fixed_nonzero_entries"]
