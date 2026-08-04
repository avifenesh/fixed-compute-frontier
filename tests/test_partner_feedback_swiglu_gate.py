import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments" / "partner_feedback_swiglu_gate.py"
SPEC = importlib.util.spec_from_file_location("partner_feedback_swiglu_gate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_stage0_passes():
    payload = MODULE.run()
    assert payload["all_gates_pass"], payload["gates"]


def test_target_ledger():
    ledger = MODULE.ledger(4096, 14336)
    assert ledger["candidate_serialized_words"] == 176_160_768
    assert ledger["candidate_trainable_parameters"] == 176_160_768
    assert ledger["candidate_dense_macs_per_token"] == 176_160_768
    assert ledger["per_feature_coefficient_words"] == 0
    assert ledger["serial_feature_dependencies"] == 0


def test_partner_is_fixed_point_free():
    indices = np.arange(1024)
    partner = indices ^ 1
    assert np.all(partner != indices)
    np.testing.assert_array_equal(partner[partner], indices)


def test_square_witness():
    assert MODULE.square_witness(0.25) == {
        "degree_3": 1.0,
        "degree_5": 0.5,
        "degree_7": 0.0625,
    }

