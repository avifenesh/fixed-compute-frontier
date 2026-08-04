import math
from pathlib import Path


def test_frozen_gate_is_symmetric_and_cost_is_lower():
    slope = 0.14709223807891283
    q = lambda z: min(1.0, max(0.0, 0.5 + slope * z))
    for z in (-10.0, -3.4, -1.0, 0.0, 1.0, 3.4, 10.0):
        assert math.isclose(q(-z), 1.0 - q(z), abs_tol=1e-15)
    assert 2 * 640 * 2560 < 3 * 640 * 1792


def test_protocol_has_no_width_or_slope_sweep():
    source = Path("experiments/hard_gated_self_product_h100.py").read_text()
    assert "HARD_SLOPE = 0.14709223807891283" in source
    assert "LATENCY_MATCHED_WIDE" in source
    assert "for slope" not in source
    assert "for width" not in source
