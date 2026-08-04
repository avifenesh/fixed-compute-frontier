from __future__ import annotations

from experiments.full_rank_conditional_tree_scaling import build_payload, ledger


def test_frozen_width_384_ledger_matches_tree_implementation() -> None:
    row = ledger(384, 1024)
    assert row["internal_nodes"] == 511
    assert row["maximum_balanced_depth"] == 9
    assert row["candidate_parameters"] == 1_179_583
    assert row["active_multiply_like_upper_bound"] == 96_768
    assert row["active_ratio"] == 0.08203125


def test_production_width_ledger_is_large_separation() -> None:
    row = ledger(4096, 14336)
    assert row["internal_nodes"] == 7166
    assert row["maximum_balanced_depth"] == 13
    assert row["active_ratio"] < 0.0085
    assert row["ideal_reduction"] > 118
    assert row["tree_local_update_rank"] == 4096
    assert row["optimistic_equal_active_moe_width_and_rank_upper_bound"] == 121


def test_all_deterministic_gates_pass() -> None:
    payload = build_payload()
    assert payload["all_gates_pass"]
