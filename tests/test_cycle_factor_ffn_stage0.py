from experiments.cycle_factor_ffn_stage0 import cycle_indices, parameter_ledger


def test_cycle_is_fixed_point_free_one_regular():
    route = cycle_indices(1024)
    assert route.unique().numel() == 1024
    assert (route != route.new_tensor(range(1024))).all()


def test_exact_cost_control():
    ledger = parameter_ledger()
    assert ledger["cycle"] == ledger["self_product"] == ledger["narrow_swiglu"]
    assert 3 * ledger["cycle"] == 2 * ledger["full_swiglu"]
