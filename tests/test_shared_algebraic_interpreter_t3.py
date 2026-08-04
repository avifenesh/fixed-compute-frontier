from __future__ import annotations

import torch

from experiments import shared_algebraic_interpreter_t3 as experiment


def fake_layout() -> experiment.TokenLayout:
    selected = tuple(range(experiment.RULES + 2 * experiment.INPUT_DIM + 4))
    cursor = 0
    rules = selected[cursor : cursor + experiment.RULES]
    cursor += experiment.RULES
    zero = selected[cursor : cursor + experiment.INPUT_DIM]
    cursor += experiment.INPUT_DIM
    one = selected[cursor : cursor + experiment.INPUT_DIM]
    cursor += experiment.INPUT_DIM
    return experiment.TokenLayout(rules, zero, one, *selected[cursor : cursor + 4])


def test_supports_are_unique_fixed_weight_rules() -> None:
    supports = experiment.make_supports()
    assert supports.shape == (experiment.RULES, experiment.INPUT_DIM)
    assert torch.all(supports.sum(1) == experiment.SUPPORT_SIZE)
    assert len({tuple(row.tolist()) for row in supports}) == experiment.RULES


def test_prefix_recovers_every_rule_exactly() -> None:
    supports = experiment.make_supports()
    bits, targets, rules = experiment.make_prefix(supports)
    recovered, operations = experiment.solve_prefix(bits, targets, rules)
    assert torch.equal(recovered, supports)
    assert operations > 0


def test_layout_reuses_fixed_vocabulary_rows() -> None:
    layout = fake_layout()
    assert len(layout.rule) == experiment.RULES
    assert len(layout.zero) == experiment.INPUT_DIM
    assert len(layout.one) == experiment.INPUT_DIM
    assert len(layout.all_special) == experiment.RULES + 2 * experiment.INPUT_DIM + 4
    assert len(set(layout.all_special)) == len(layout.all_special)
    assert max(layout.all_special) < experiment.VOCAB


def test_algorithm_encoding_is_position_specific() -> None:
    layout = fake_layout()
    bits = torch.tensor([[0, 1] + [0] * (experiment.INPUT_DIM - 2)], dtype=torch.uint8)
    rules = torch.tensor([7])
    route = torch.tensor([True])
    tokens = experiment.encode_algorithm(bits, rules, route, layout, torch.device("cpu"))
    assert tokens.shape == (1, experiment.INPUT_DIM + 2)
    assert tokens[0, 0].item() == layout.rule[7]
    assert tokens[0, 1].item() == layout.zero[0]
    assert tokens[0, 2].item() == layout.one[1]
    assert tokens[0, -1].item() == layout.parity_query


def test_silu_pair_is_exact_multiplication_identity() -> None:
    values = torch.linspace(-20, 20, 1001)
    recovered = torch.nn.functional.silu(values) - torch.nn.functional.silu(-values)
    assert torch.allclose(recovered, values, atol=2e-6, rtol=2e-6)


def test_compiler_width_and_description_do_not_scale_as_per_rule_circuits() -> None:
    supports = experiment.make_supports()
    model = experiment.build_model(5, torch.device("cpu"))
    frozen, metadata = experiment.compile_interpreter(model, supports, fake_layout())
    assert metadata["description_entries"] == experiment.RULES * experiment.INPUT_DIM
    assert metadata["program_hidden_coordinates"] == experiment.PROGRAM_DIMS
    assert metadata["shared_width_independent_of_rules"] is True
    assert metadata["used_decoder_channels"] <= experiment.PROGRAM_CHANNELS
    assert frozen.count > metadata["description_entries"]


def test_model_shape_is_arm_independent() -> None:
    first = experiment.build_model(11, torch.device("cpu"))
    second = experiment.build_model(11, torch.device("cpu"))
    initial_hash = experiment.state_sha256(first)
    experiment.compile_interpreter(first, experiment.make_supports(), fake_layout())
    assert experiment.parameter_count() == sum(parameter.numel() for parameter in first.parameters())
    assert initial_hash == experiment.state_sha256(second)
