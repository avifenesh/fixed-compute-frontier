import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments" / "dynamic_binding_swiglu.py"
SPEC = importlib.util.spec_from_file_location("dynamic_binding_swiglu", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage0_algebra_passes() -> None:
    result = MODULE.run()
    assert result["pass"], result


def test_target_ledger_matches_dense_projections() -> None:
    ledger = MODULE.resource_ledger(MODULE.Shape(4096, 14336, 8))
    assert ledger["baseline_learned_scalars"] == ledger["candidate_learned_scalars"]
    assert ledger["baseline_dense_macs_per_token"] == ledger["candidate_dense_macs_per_token"]
    assert ledger["extra_model_selector_bits"] == 0


def test_nonanalytic_witness_is_not_degenerate() -> None:
    witness = MODULE.nonanalytic_witness()
    assert witness["left_quartic_correction"] != witness["right_quartic_correction"]
    assert witness["fourth_derivative_jump"] != 0.0
