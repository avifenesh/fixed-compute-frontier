import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "triangular_microdepth_gate.py"
)
SPEC = importlib.util.spec_from_file_location("triangular_microdepth_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

compiled_group_program = MODULE.compiled_group_program
count_linear_segments_on_unit_interval = MODULE.count_linear_segments_on_unit_interval
direct_group_program = MODULE.direct_group_program
matched_width = MODULE.matched_width
packed_edges = MODULE.packed_edges
partial_bias_indices = MODULE.partial_bias_indices
resource_ledger = MODULE.resource_ledger
run_gate = MODULE.run_gate
silu = MODULE.silu
triangular_coefficients = MODULE.triangular_coefficients


def test_compiled_program_matches_direct() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 13))
    u = rng.normal(size=(2, 8, 13))
    v = rng.normal(size=(2, 8, 13))
    np.testing.assert_allclose(
        compiled_group_program(x, u, v),
        direct_group_program(x, u, v),
        atol=1e-11,
        rtol=1e-11,
    )


def test_zero_coupling_is_plain_silu() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(size=(4, 7, 8))
    coupling = np.zeros((7, packed_edges(8)))
    np.testing.assert_array_equal(
        triangular_coefficients(base, coupling), silu(base)
    )


def test_tent_witness_beats_shallow_width_eight_region_limit() -> None:
    assert count_linear_segments_on_unit_interval() == 16
    assert count_linear_segments_on_unit_interval() > 9


def test_exact_small_and_target_budgets() -> None:
    assert matched_width(384, 1024, 8) == (1528, 796)
    assert resource_ledger()["parameter_difference"] == 0
    assert resource_ledger(4096, 11008, 8)["parameter_difference"] == 0


def test_bias_mask_puts_four_thresholds_in_every_group() -> None:
    indices = partial_bias_indices(1528, 8, 796)
    assert len(indices) == len(np.unique(indices)) == 796
    present = set(indices.tolist())
    for group in range(191):
        assert {group * 8 + offset for offset in (1, 3, 5, 7)} <= present


def test_gate_passes() -> None:
    assert run_gate(trials=4)["all_gates_pass"]
