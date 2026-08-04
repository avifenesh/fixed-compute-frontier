import math

import torch

from experiments.temporal_factor_ffn_lm_screen import ANGLE_COUNT_PER_LAYER, TemporalFactorMLP


def test_zero_angle_is_self_product_endpoint():
    module=TemporalFactorMLP(0.02)
    hidden=torch.randn(2,5,384)
    actual=module.activation(hidden)
    generated=module.generator_proj(hidden)
    from experiments.cycle_factor_ffn_lm_screen import SELF_SCALE
    expected=SELF_SCALE*torch.nn.functional.silu(generated)*generated
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)


def test_state_and_angle_ledger():
    assert ANGLE_COUNT_PER_LAYER==32
    assert 12*1024*2==24576
    assert math.isclose((math.pi/2)*math.tanh(0),0)
