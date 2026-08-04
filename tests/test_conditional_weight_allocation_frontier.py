from argparse import Namespace

from experiments.conditional_weight_allocation_frontier import (
    allocation_neutral_witness,
    high_rate_crossover,
    run,
)


def frozen_args():
    return Namespace(
        dimension=4096,
        hidden=14336,
        baseline_bits=16,
        scale_bits=16,
        router_bits=16,
        alignment=16,
        output="unused.json",
    )


def test_equal_budget_layouts_and_control():
    payload = run(frozen_args())
    assert payload["all_gates_pass"]
    layouts = {record["name"]: record for record in payload["layouts"]}
    assert layouts["shared_base_8_plus_4x2_ternary"]["aligned_hidden"] == 14320
    assert layouts["independent_k4_w4"]["resident_bytes"] == 352222984
    assert layouts["independent_k4_w4"]["active_payload_bits_per_coordinate"] == 4
    assert layouts["shared_base_8_plus_4x2_ternary"]["active_payload_bits_per_coordinate"] == 10


def test_high_rate_crossover_reproduces_closed_form_threshold():
    crossover = high_rate_crossover(8, 2)
    assert abs(crossover["maximum_delta_to_base_variance_ratio_for_structured_win"] - 255 / 3840) < 1e-15
    assert crossover["minimum_pairwise_route_correlation_for_structured_win"] > 0.93


def test_corner_witness_is_neutral_across_frozen_codebooks():
    witness = allocation_neutral_witness()
    assert witness["actual_routes"] == [0, 1, 2, 3]
    assert witness["routed_mse"] == 0
    assert witness["best_single_bias_free_linear"]["mse"] == 4
    assert all(code in {-1, 1} for delta in witness["route_deltas"] for code in delta)
