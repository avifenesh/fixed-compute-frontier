from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "cayley_program_tree_stage0.py"
)
SPEC = importlib.util.spec_from_file_location("cayley_program_tree_stage0", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_edge_jacobian_matches_finite_difference() -> None:
    rng = np.random.default_rng(7)
    basis = MODULE.make_basis(8, 3, 0.25, rng)
    hidden = rng.normal(size=8)
    gate = 1.0 + 0.2 * rng.normal(size=8)
    up = 1.0 + 0.2 * rng.normal(size=8)
    down = 0.1 * rng.normal(size=8)
    _, analytic = MODULE.apply_edge(hidden, basis, gate, up, down, 6)
    finite = MODULE.finite_difference_jacobian(
        lambda values: MODULE.apply_edge(values, basis, gate, up, down, 6)[0],
        hidden,
    )
    relative = np.linalg.norm(analytic - finite) / np.linalg.norm(analytic)
    assert relative <= 1e-6


def test_exact_width_384_ledger() -> None:
    ledger = MODULE.width_384_ledger()
    assert ledger["candidate_ffn_parameters"] == 1_179_583
    assert ledger["baseline_swiglu_parameters_and_macs"] == 1_179_648
    assert ledger["candidate_minus_baseline_parameters"] == -65
    assert np.isclose(ledger["candidate_to_dense_active_ratio"], 0.08203125)


def test_frozen_stage0_passes() -> None:
    report = MODULE.build_report()
    assert report["all_gates_pass"]
    assert report["paths"]["jacobian_span_dimension"] == 64
