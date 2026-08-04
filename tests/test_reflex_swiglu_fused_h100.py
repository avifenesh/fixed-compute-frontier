from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "experiments" / "reflex_swiglu_fused_h100.py"
SPEC = importlib.util.spec_from_file_location("reflex_swiglu_fused_h100", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_alpha_contains_fused_baseline() -> None:
    torch.manual_seed(11)
    gate = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    alpha = torch.zeros(64, device="cuda", dtype=torch.bfloat16)
    baseline = MODULE.baseline_activation(gate, up)
    reflex = MODULE.reflex_activation(gate, up, alpha)
    torch.testing.assert_close(reflex, baseline, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nonzero_alpha_matches_reference_across_clip_regimes() -> None:
    result = MODULE.nonzero_alpha_reference_check(torch.device("cuda"))
    assert result["max_row_relative_error"] <= 0.01
    assert result["negative_clipped_coordinates"] > 0
    assert result["unclipped_coordinates"] > 0
    assert result["positive_clipped_coordinates"] > 0


def test_equal_parameter_width() -> None:
    dimension, hidden = 4096, 14336
    exact = (3 * dimension * hidden) // (3 * dimension + 1)
    assert exact == 14334
    assert (exact // 8) * 8 == 14328
