from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments" / "feedback_swiglu_stage0.py"
SPEC = importlib.util.spec_from_file_location("feedback_swiglu_stage0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stage0_frozen_gate_passes() -> None:
    payload = MODULE.run(seed=37)
    assert payload["all_gates_pass"]


def test_equal_parameter_ledger_does_not_exceed_baseline() -> None:
    ledger = MODULE.resource_ledger(4096, 14336, 8)
    assert ledger["equal_parameter_candidate_parameters"] <= ledger["baseline_parameters"]
    assert ledger["equal_parameter_hidden"] == 14317


def test_feedback_zero_expansion_contains_swiglu() -> None:
    payload = MODULE.run(seed=37)
    assert payload["inclusion"]["max_absolute_error"] <= 1e-12

