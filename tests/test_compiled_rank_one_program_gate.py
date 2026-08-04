import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "compiled_rank_one_program_gate.py"
)
SPEC = importlib.util.spec_from_file_location("compiled_rank_one_program_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compiled_evaluator_matches_direct_for_all_activations():
    result = MODULE.run_gate(seed=11, trials=12, dimension=32, length=7)
    assert result["algebra"]["max_relative_error"] < 1e-12


def test_order_separates_program_from_additive_mixture():
    result = MODULE.run_gate(seed=3, trials=2, dimension=16, length=4)
    assert result["algebra"]["additive_order_distance"] < 1e-12
    assert result["algebra"]["program_order_distance"] > 1e-3


def test_compiler_uses_only_strictly_prior_coefficients():
    rng = np.random.default_rng(19)
    x = rng.normal(size=10)
    u = rng.normal(size=(5, 10))
    v = rng.normal(size=(5, 10))
    bias = rng.normal(size=5)
    scale = rng.uniform(0.1, 0.9, size=5)
    gram = MODULE.compile_program(u, v)
    corrupted = gram.copy()
    corrupted[np.triu_indices(5)] += 10_000.0
    clean, _ = MODULE.compiled_program(x, u, v, bias, scale, np.tanh, gram)
    dirty, _ = MODULE.compiled_program(x, u, v, bias, scale, np.tanh, corrupted)
    np.testing.assert_allclose(clean, dirty, rtol=0.0, atol=0.0)


def test_frozen_resource_ledger_is_honest_about_added_state_and_work():
    ledger = MODULE.resource_ledger(
        dimension=4096,
        bank_size=16384,
        program_count=1_000_000,
        program_length=8,
    )
    assert ledger["physical_program_record_bytes"] == 72
    assert ledger["compiled_program_table_bytes"] == 72_000_000
    assert ledger["active_expert_bytes_per_token"] == 131_104
    assert ledger["scalar_recurrence_macs_per_token"] == 28
    assert ledger["scalar_recurrence_overhead_fraction"] < 0.001
    assert ledger["compression_vs_singleton_peer"] > 40.0


def test_all_frozen_cpu_gates_pass():
    result = MODULE.run_gate(seed=7, trials=8, dimension=24, length=6)
    assert result["all_gates_pass"]
