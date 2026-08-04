import math

import torch

from experiments.radial_trust_cayley_conditioning_b1 import (
    R_AMP,
    R_GRAD,
    all_forced_bits,
    radial_fp64,
    resource_ceiling_ok,
    trace_tree,
    wilson_interval,
)
from experiments.radial_trust_cayley_program_tree import (
    RadialTrustCayleyProgramTreeMLP,
)


def test_precision_surfaces_match_definitions() -> None:
    trusted_radius = lambda r: r / math.sqrt(1.0 + r * r)
    relative_doubling = trusted_radius(2 * R_AMP) / trusted_radius(R_AMP) - 1.0
    assert math.isclose(relative_doubling, 2.0**-7, rel_tol=1e-13)
    assert math.isclose(1.0 / (1.0 + R_GRAD**2), 2.0**-7, rel_tol=1e-13)


def test_trace_matches_candidate_at_unit_multiplier() -> None:
    torch.manual_seed(17)
    module = RadialTrustCayleyProgramTreeMLP(
        width=8, depth=3, trust_rms=1.0, seed=17
    )
    module.train()
    inputs = torch.randn(5, 8)
    traced, trace, nodes_ok = trace_tree(module, inputs, 1.0)
    with torch.no_grad():
        expected = module(inputs)
    assert torch.equal(traced, expected)
    assert trace["r"].shape == (5, 3)
    assert nodes_ok


def test_vectorized_forced_routes_match_module() -> None:
    torch.manual_seed(19)
    module = RadialTrustCayleyProgramTreeMLP(
        width=8, depth=3, trust_rms=1.0, seed=19
    )
    module.train()
    source = torch.randn(2, 8)
    bits = all_forced_bits(3)
    repeated = source.repeat_interleave(8, dim=0)
    forced = bits.repeat(2, 1)
    traced, _, nodes_ok = trace_tree(module, repeated, 1.0, forced_bits=forced)
    assert nodes_ok
    for code in range(8):
        module.force_bits = tuple(int(value) for value in bits[code])
        with torch.no_grad():
            expected = module(source)
        selected = traced[torch.arange(2) * 8 + code]
        assert torch.equal(selected, expected)
    module.force_bits = None


def test_radial_fp64_has_expected_eigenvalues() -> None:
    point = torch.tensor([math.sqrt(2.0), 0.0], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: radial_fp64(value[None])[0], point
    )
    singular = torch.linalg.svdvals(jacobian)
    expected = torch.tensor(
        [1.0 / math.sqrt(2.0), 1.0 / (2.0 * math.sqrt(2.0))],
        dtype=torch.float64,
    )
    torch.testing.assert_close(singular, expected, rtol=1e-12, atol=1e-12)


def test_wilson_interval_contains_observed_fraction() -> None:
    low, high = wilson_interval(128, 256)
    assert low < 0.5 < high


def test_final_resource_measurement_can_overrule_prewrite_pass() -> None:
    assert resource_ceiling_ok(1794.9, 7.9, 1024, 1024)
    assert not resource_ceiling_ok(1800.1, 7.9, 1024, 1024)
