from argparse import Namespace

import pytest

from experiments.bit_budgeted_conditional_matrix_stage0 import (
    candidate_storage_bytes,
    piecewise_witness,
    run,
)


def frozen_args():
    return Namespace(
        dimension=4096,
        hidden=14336,
        experts=4,
        baseline_bits=16,
        base_bits=8,
        delta_bits=2,
        scale_bits=16,
        router_bits=16,
        alignment=16,
        output="unused.json",
    )


def test_exact_frozen_ledger_passes():
    payload = run(frozen_args())
    assert payload["all_gates_pass"]
    assert payload["ledger"]["equal_storage_aligned_hidden"] == 14320
    assert payload["ledger"]["unused_bytes"] == 33080


def test_candidate_storage_includes_scales_and_router():
    args = frozen_args()
    payload_bytes = 3 * args.dimension * 14320 * 2
    assert candidate_storage_bytes(
        args.dimension,
        14320,
        args.experts,
        args.base_bits,
        args.delta_bits,
        args.scale_bits,
        args.router_bits,
    ) > payload_bytes


def test_piecewise_map_uses_deployed_ternary_alphabet_and_separates_routes():
    witness = piecewise_witness()
    assert witness["actual_affine_router_routes"] == witness["expected_routes"]
    assert witness["delta_two_bit_alphabet"] == [-1, 0, 1]
    assert all(code in {-1, 0, 1} for route in witness["delta_codes"] for code in route)
    assert witness["routed"]["mse"] == 0.0
    assert witness["best_single_bias_free_linear"]["mse"] == 4.0


def test_witness_rejects_other_bit_allocations():
    args = frozen_args()
    args.base_bits = 12
    args.delta_bits = 1
    with pytest.raises(ValueError, match="frozen specifically"):
        run(args)
