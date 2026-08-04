from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


SCRIPT = Path(__file__).parents[1] / "experiments" / "feedback_swiglu_fused_h100.py"
SPEC = importlib.util.spec_from_file_location("feedback_swiglu_fused_h100", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_baseline_matches_torch() -> None:
    torch.manual_seed(5)
    gate = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    actual = MODULE.baseline_activation(gate, up)
    expected = F.silu(gate.float()) * up.float()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_zero_feedback_contains_fused_baseline() -> None:
    torch.manual_seed(7)
    gate = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    compress = torch.randn(4, 16, 3, device="cuda", dtype=torch.bfloat16)
    expand = torch.zeros_like(compress)
    baseline = MODULE.baseline_activation(gate, up)
    feedback = MODULE.feedback_activation(gate, up, compress, expand)
    torch.testing.assert_close(feedback, baseline, rtol=0.0, atol=0.0)

