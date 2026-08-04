from __future__ import annotations

import torch

from experiments import regenerated_packed_memory_t5 as experiment


def fake_layout() -> experiment.t4.TokenLayout:
    count = experiment.RULES + 2 * experiment.INPUT_DIM + 4
    selected = tuple(range(count))
    cursor = 0
    rules = selected[cursor : cursor + experiment.RULES]
    cursor += experiment.RULES
    zero = selected[cursor : cursor + experiment.INPUT_DIM]
    cursor += experiment.INPUT_DIM
    one = selected[cursor : cursor + experiment.INPUT_DIM]
    cursor += experiment.INPUT_DIM
    return experiment.t4.TokenLayout(rules, zero, one, *selected[cursor : cursor + 4])


def test_clip_level_is_inside_the_sealed_t4_margin() -> None:
    assert experiment.CLIP_LEVEL == 1 / 16
    assert 2 * experiment.CLIP_LEVEL < 0.1416015625


def test_ideal_regeneration_is_constant_amplitude() -> None:
    values = torch.tensor([-1.6016, -0.1417, 0.1417, 1.6016])
    a = experiment.CLIP_LEVEL
    regenerated = -a + torch.relu(values + a) - torch.relu(values - a)
    assert torch.allclose(regenerated, torch.tensor([-a, -a, a, a]), atol=1e-6)


def test_packed_table_still_has_12x_t3_rule_density() -> None:
    supports = experiment.t4.make_supports()
    packed = experiment.t4.pack_supports(supports)
    assert packed.numel() == 6144
    assert supports.numel() == 24576
    assert 24576 / 8192 == 3
    assert (24576 / 6144) * (experiment.RULES / 256) == 12


def test_compiler_regenerates_every_sign_on_cpu() -> None:
    supports = experiment.t4.make_supports()
    model = experiment.t4.t3.build_model(23, torch.device("cpu"))
    frozen, metadata = experiment.compile_regenerated_interpreter(
        model, supports, fake_layout()
    )
    assert metadata["raw_sign_accuracy"] == 1.0
    assert metadata["canonical_sign_accuracy"] == 1.0
    assert metadata["maximum_canonical_magnitude_ratio"] <= 1.05
    assert metadata["decoder_channels"] == 130
    assert metadata["regeneration_channels"] == 66
    assert metadata["execute_decoder_channels"] <= experiment.EXECUTE_CHANNELS
    assert frozen.count > metadata["fixed_nonzero_entries"]


def test_model_shape_is_identical_to_t3_and_t4() -> None:
    model = experiment.t4.t3.build_model(29, torch.device("cpu"))
    before = experiment.t4.t3.parameter_count()
    experiment.compile_regenerated_interpreter(model, experiment.t4.make_supports(), fake_layout())
    assert sum(parameter.numel() for parameter in model.parameters()) == before


def test_program_fits_existing_hidden_and_ffn_widths() -> None:
    assert experiment.PROGRAM_DIMS < experiment.HIDDEN
    assert experiment.DECODE_CHANNELS <= experiment.t4.FFN_WIDTH
    assert experiment.REGENERATE_CHANNELS <= experiment.t4.FFN_WIDTH
    assert experiment.EXECUTE_CHANNELS <= experiment.t4.FFN_WIDTH
