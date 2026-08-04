from __future__ import annotations

import torch

from experiments import packed_rule_memory_t4 as experiment


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


def test_supports_are_unique_and_fixed_weight() -> None:
    supports = experiment.make_supports()
    assert supports.shape == (experiment.RULES, experiment.INPUT_DIM)
    assert torch.all(supports.sum(1) == experiment.SUPPORT_SIZE)
    assert len({tuple(row.tolist()) for row in supports}) == experiment.RULES


def test_nibble_pack_round_trip_and_density() -> None:
    supports = experiment.make_supports()
    packed = experiment.pack_supports(supports)
    assert packed.shape == (experiment.RULES, experiment.NIBBLES)
    assert int(packed.max()) <= 15
    assert torch.equal(experiment.unpack_supports(packed), supports)
    assert packed.numel() == 6144
    assert supports.numel() == 24576


def test_prefix_recovers_all_768_rules() -> None:
    supports = experiment.make_supports()
    bits, targets, rules = experiment.make_prefix(supports)
    recovered, operations = experiment.solve_prefix(bits, targets, rules)
    assert torch.equal(recovered, supports)
    assert operations > 0


def test_piecewise_bit_functions_match_all_nibbles() -> None:
    for bit in range(4):
        values = [experiment.bit_sign(value, bit) for value in range(16)]
        assert set(values) == {-1.0, 1.0}
        for value, sign in enumerate(values):
            assert sign == (1.0 if ((value >> bit) & 1) else -1.0)


def test_layout_fits_the_sealed_vocabulary() -> None:
    layout = fake_layout()
    assert len(layout.all_special) == experiment.RULES + 2 * experiment.INPUT_DIM + 4
    assert len(set(layout.all_special)) == len(layout.all_special)
    assert max(layout.all_special) < experiment.VOCAB


def test_compiler_uses_fewer_description_scalars_than_t3() -> None:
    supports = experiment.make_supports()
    model = experiment.t3.build_model(17, torch.device("cpu"))
    frozen, metadata = experiment.compile_packed_interpreter(model, supports, fake_layout())
    assert metadata["description_entries"] == 6144
    assert metadata["description_entries"] < 8192
    assert metadata["decoded_support_bits"] == 24576
    assert metadata["decoder_channels"] == 130
    assert metadata["decoder_channels"] <= experiment.DECODE_CHANNELS
    assert metadata["execute_decoder_channels"] <= experiment.EXECUTE_CHANNELS
    assert metadata["shared_width_independent_of_rules"] is True
    assert frozen.count > metadata["fixed_nonzero_entries"]


def test_model_shape_is_unchanged_from_t3() -> None:
    model = experiment.t3.build_model(19, torch.device("cpu"))
    before = experiment.t3.parameter_count()
    experiment.compile_packed_interpreter(model, experiment.make_supports(), fake_layout())
    assert sum(parameter.numel() for parameter in model.parameters()) == before
