from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments" / "reflex_swiglu_stage0.py"
SPEC = importlib.util.spec_from_file_location("reflex_swiglu_stage0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stage0_gate_passes() -> None:
    assert MODULE.run(seed=47)["all_gates_pass"]


def test_reference_ledger() -> None:
    result = MODULE.ledger(4096, 14336)
    assert result["exact_equal_parameter_hidden"] == 14334
    assert result["equal_parameter_candidate_parameters"] <= result["baseline_parameters"]

