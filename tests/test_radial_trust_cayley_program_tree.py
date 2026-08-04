import math

import torch

from experiments.cayley_program_tree import CayleyProgramTreeMLP
from experiments.radial_trust_cayley_program_tree import (
    RadialTrustCayleyProgramTreeMLP,
)


def rms(values: torch.Tensor) -> torch.Tensor:
    return values.double().square().mean(dim=-1).sqrt()


def test_radial_trust_region_is_strictly_bounded_and_full_rank() -> None:
    module = RadialTrustCayleyProgramTreeMLP(width=8, depth=3).double()
    values = torch.randn(5, 8, dtype=torch.float64) * 100.0
    assert bool((rms(module.radial_trust_region(values)) < 1.0).all())
    point = torch.randn(8, dtype=torch.float64) * 5.0
    jacobian = torch.autograd.functional.jacobian(
        lambda value: module.radial_trust_region(value[None])[0], point
    )
    assert int(torch.linalg.matrix_rank(jacobian)) == 8


def test_radial_trust_tree_has_the_same_checkpoint_budget() -> None:
    base = CayleyProgramTreeMLP()
    candidate = RadialTrustCayleyProgramTreeMLP()
    assert candidate.parameter_count() == base.parameter_count()
    assert set(candidate.state_dict()) == set(base.state_dict())


def test_adversarial_rms_one_input_obeys_depth_bound() -> None:
    width = 384
    depth = 9
    module = RadialTrustCayleyProgramTreeMLP(width=width, depth=depth)
    module.eval()
    module.record_diagnostics = True
    values = torch.zeros(4, width)
    values[:, 0] = math.sqrt(width)
    output = module(values)
    assert bool(torch.isfinite(output).all())
    assert max(row["trusted_delta_rms"] for row in module.last_trust_diagnostics) < 1.0
    assert max(row["hidden_rms"] for row in module.last_trust_diagnostics) <= 4.00001
