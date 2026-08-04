from __future__ import annotations

import torch

from experiments.balanced_radix_projection_h100 import benchmark_cell


def test_packed_h100_kernel_matches_fused_dual_at_real_k() -> None:
    assert torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)
    result = benchmark_cell(rows=16, columns=64, reduction=4096, seed=2207)
    assert result["max_abs_error_low"] == 0
    assert result["max_abs_error_high"] == 0
