import importlib.util
from pathlib import Path
import sys
import ast
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments" / "heterogeneous_swiglu_long_horizon.py"
SPEC = importlib.util.spec_from_file_location("heterogeneous_swiglu_long_horizon", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE


def test_student_t_interval_uses_five_seeds() -> None:
    # Execute only the pure function without importing CUDA/Transformers.
    tree = ast.parse(PATH.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "student_t_interval"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np, "math": math, "list": list, "float": float, "dict": dict, "ValueError": ValueError}
    exec(compile(module, str(PATH), "exec"), namespace)
    interval = namespace["student_t_interval"]([0.1, 0.2, 0.3, 0.4, 0.5])
    assert interval["mean"] == 0.3
    assert interval["lower_95"] < interval["mean"] < interval["upper_95"]


def test_even_gate_is_rms_matched_under_frozen_reference() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=2_000_000)
    silu = values / (1.0 + np.exp(-values))
    even = 0.6854475404927584 * values * np.tanh(values)
    ratio = np.sqrt(np.mean(even**2) / np.mean(silu**2))
    assert abs(ratio - 1.0) < 0.003


def test_frozen_seed_and_arm_contract_is_present() -> None:
    source = PATH.read_text()
    assert "SEEDS = (2741, 2753, 2767, 2789, 2801)" in source
    assert 'ARMS = ("raw_baseline", "canonical_null", "mixed50_rms")' in source
