import math

import torch

from experiments.cycle_factor_ffn_lm_screen import (
    M, NARROW_WIDTHS, SELF_SCALE, narrow_scale, route_values,
)


def test_exact_aggregate_2dm_control():
    assert len(NARROW_WIDTHS) == 12
    assert 3 * sum(NARROW_WIDTHS) == 12 * 2 * M


def test_frozen_cycles_fit_width():
    assert M % 4 == 0
    assert NARROW_WIDTHS == (683, 683, 682) * 4


def test_route_orientation_and_ablation():
    values = torch.arange(M, dtype=torch.float32).reshape(1, 1, M)
    assert route_values(values)[0, 0, :4].tolist() == [1.0, 2.0, 3.0, 0.0]
    assert route_values(values, "second_neighbor")[0, 0, :4].tolist() == [2.0, 3.0, 0.0, 1.0]
    assert torch.equal(route_values(values, "self"), values)


def test_variance_scales_are_frozen():
    assert math.isclose(narrow_scale(683), math.sqrt(1024 / 683))
    assert math.isclose(narrow_scale(682), math.sqrt(1024 / 682))
    assert math.isclose(SELF_SCALE, 0.44642046792894413 / math.sqrt(2 / 3))
