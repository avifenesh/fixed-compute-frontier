import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "triangular_microprogram_gate.py"
)
SPEC = importlib.util.spec_from_file_location("triangular_microprogram_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_additive_peer_is_exact_endpoint():
    result = MODULE.run_gate(seed=1, trials=8, dimension=16, length=5)
    assert result["metrics"]["max_zero_coupling_absolute_error"] < 1e-12


def test_sequential_rank_one_program_is_exact_endpoint():
    result = MODULE.run_gate(seed=2, trials=8, dimension=16, length=5)
    assert result["metrics"]["max_compiled_sequential_relative_error"] < 1e-12


def test_square_activation_degree_doubles_each_step():
    for length in range(1, 9):
        additive, triangular = MODULE.effective_polynomial_degree(length)
        assert additive == 2
        assert triangular == 2**length


def test_equal_byte_control_is_larger_additive_bank():
    ledger = MODULE.resource_ledger()
    assert ledger["program_record_bytes"] == 72
    assert ledger["equal_byte_additive_peer_bank_size"] > ledger["bank_size"]
    assert ledger["equal_byte_additive_peer_bank_size"] == 20777


def test_all_frozen_gates_pass():
    assert MODULE.run_gate(seed=31, trials=12, dimension=24, length=8)[
        "all_gates_pass"
    ]
