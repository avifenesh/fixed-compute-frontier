from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "sparse_cayley_global_program_stage0.py"
)
SPEC = importlib.util.spec_from_file_location("sparse_cayley_stage0", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cayley_identity_and_truncation_bound() -> None:
    generators = MODULE.make_sparse_skew_generators(16, 3, 3, 0.25, 19)
    rng = np.random.default_rng(23)
    values = rng.normal(size=(16, 32))
    bound = 2.0 * 0.25**5 / 0.75
    for generator in generators:
        exact = MODULE.cayley(generator)
        assert np.linalg.norm(exact.T @ exact - np.eye(16)) <= 1e-12
        approximate = MODULE.neumann_cayley_apply(generator, values, 4)
        relative_error = np.linalg.norm(approximate - exact @ values) / np.linalg.norm(values)
        assert relative_error <= bound


def test_order_matters_but_addition_only_sees_multisets() -> None:
    generators = MODULE.make_sparse_skew_generators(16, 2, 3, 0.25, 29)
    experts = [MODULE.cayley(generator) for generator in generators]
    forward = MODULE.compose((0, 1), experts)
    reverse = MODULE.compose((1, 0), experts)
    assert np.linalg.norm(forward - reverse) >= 1e-3
    assert np.array_equal(
        MODULE.additive((0, 1), experts), MODULE.additive((1, 0), experts)
    )


def test_frozen_report_passes_every_gate() -> None:
    report = MODULE.build_report()
    assert report["all_gates_pass"]
    assert report["programs"]["sequential_span_dimension"] == 256
    assert report["programs"]["additive_span_dimension"] <= 6
