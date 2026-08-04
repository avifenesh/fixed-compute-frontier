from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "cayley_conditional_expert_rank_gate.py"
)
SPEC = importlib.util.spec_from_file_location("cayley_rank_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_additive_rank_bound() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=12)
    for active in (1, 3, 7):
        keys = rng.normal(size=(active, 12))
        outputs = rng.normal(size=(active, 12))
        _, jacobian = MODULE.additive_memory(values, keys, outputs)
        assert np.linalg.matrix_rank(jacobian) <= active


def test_cayley_edge_analytic_jacobian() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=12)
    basis = MODULE.make_implicit_basis(12, 3, 0.25, 0, rng)
    gate = 1.0 + 0.2 * rng.normal(size=12)
    up = 1.0 + 0.2 * rng.normal(size=12)
    down = rng.normal(size=12)
    _, analytic = MODULE.cayley_edge(values, basis, gate, up, down)
    finite = MODULE.finite_difference_jacobian(
        lambda probe: MODULE.cayley_edge(probe, basis, gate, up, down)[0], values
    )
    relative = np.linalg.norm(analytic - finite) / np.linalg.norm(analytic)
    assert relative <= 1e-6
    assert np.linalg.matrix_rank(analytic) == 12


def test_frozen_report_passes() -> None:
    report = MODULE.build_report()
    assert report["all_gates_pass"]
    assert report["cayley_expert"]["update_jacobian_rank"] == MODULE.WIDTH
