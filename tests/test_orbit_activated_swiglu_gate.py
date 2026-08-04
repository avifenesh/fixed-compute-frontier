import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "orbit_activated_swiglu_gate.py"
SPEC = importlib.util.spec_from_file_location("orbit_activated_swiglu_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_stage0_passes() -> None:
    result = MODULE.run()
    assert result["all_gates_pass"], result["gates"]


def test_ledger_is_exact_at_target_shape() -> None:
    ledger = MODULE.resource_ledger(MODULE.Shape(4096, 14336))
    assert ledger["baseline_trainable_parameters"] == 176_160_768
    assert ledger["candidate_trainable_parameters"] == 176_160_768
    assert ledger["candidate_serialized_words"] == 176_160_768
    assert ledger["candidate_dense_macs_per_token"] == 176_160_768
    assert ledger["maximum_serial_predecessor_steps"] == 7
    assert np.isclose(ledger["candidate_pivot_read_fraction_of_weight_words"], 1 / (3 * 4096))


def test_fold_and_decode_round_trip() -> None:
    rng = np.random.default_rng(7)
    d, m = 5, 10
    pivots, chart = MODULE.pivots_and_chart(d, m, 0.25)
    up = rng.normal(size=(m, d))
    up[np.arange(m), pivots] = chart
    down = rng.normal(size=(d, m))
    coefficients = rng.uniform(-0.04, 0.04, size=m)
    folded_up, folded_down, scales = MODULE.fold_carrier(up, down, coefficients, 8.0)
    decoded = MODULE.decode_carrier(folded_up, pivots, chart, 8.0)
    np.testing.assert_allclose(decoded, coefficients, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(folded_up, up * scales[:, None])
    np.testing.assert_allclose(folded_down, down / scales[None, :])


def test_square_witness_has_strictly_higher_degree() -> None:
    witness = MODULE.square_activation_witness(0.25)
    assert witness == {"degree_3": 1.0, "degree_5": 0.5, "degree_7": 0.0625}


def test_bf16_round_is_idempotent() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=1000).astype(np.float32)
    once = MODULE.bf16_round(values)
    twice = MODULE.bf16_round(once)
    np.testing.assert_array_equal(once, twice)
