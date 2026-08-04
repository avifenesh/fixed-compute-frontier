from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "experiments" / "feedback_swiglu_h100.py"
SPEC = importlib.util.spec_from_file_location("feedback_swiglu_h100", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_zero_feedback_exactly_contains_baseline() -> None:
    torch.manual_seed(3)
    x = torch.randn(5, 7)
    gate = torch.randn(11, 7)
    up = torch.randn(11, 7)
    down = torch.randn(7, 11)
    compress = torch.randn(1, 11, 3)
    expand = torch.zeros(1, 11, 3)
    baseline = MODULE.baseline_swiglu(x, gate, up, down)
    feedback = MODULE.feedback_swiglu(x, gate, up, down, compress, expand)
    torch.testing.assert_close(feedback, baseline, rtol=0.0, atol=0.0)


def test_equal_parameter_shape_is_under_budget() -> None:
    dimension, hidden, state = 4096, 14336, 8
    exact = (3 * dimension * hidden) // (3 * dimension + 2 * state)
    groups = 8
    aligned = (exact // (8 * groups)) * (8 * groups)
    assert exact == 14317
    assert aligned == 14272
    assert aligned * (3 * dimension + 2 * state) <= 3 * dimension * hidden
