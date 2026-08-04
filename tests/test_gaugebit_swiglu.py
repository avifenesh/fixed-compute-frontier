import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments" / "gaugebit_swiglu.py"
SPEC = importlib.util.spec_from_file_location("gaugebit_swiglu", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stage0_algebra_and_ledger_pass() -> None:
    result = MODULE.run()
    assert result["pass"]
    assert all(result["gates"].values())


def test_target_shape_has_exact_learned_and_dense_ledgers() -> None:
    ledger = MODULE.resource_ledger(MODULE.Shape(4096, 14336, 32))
    assert ledger["baseline_learned_scalars"] == ledger["candidate_learned_scalars"]
    assert ledger["baseline_dense_macs_per_token"] == ledger["candidate_dense_macs_per_token"]
    assert ledger["extra_serialized_selector_bits_if_signed_scales_exist"] == 0


def test_decode_rejects_mixed_signs_inside_serving_group() -> None:
    scales = np.ones(64)
    scales[7] = -1.0
    with pytest.raises(ValueError, match="mixed selector signs"):
        MODULE.decode_group_bits(scales, 32)
