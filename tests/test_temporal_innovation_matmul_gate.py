from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments.temporal_innovation_matmul_gate import (
    BITS,
    LANES,
    exact_algebra_check,
    gate_decision,
    projection_records,
    quantize_per_coordinate,
    resource_ledger,
    selected_layer_indices,
    summarize_records,
)


class TemporalInnovationMatmulGateTests(unittest.TestCase):
    def test_exact_dense_and_rmsnorm_updates(self) -> None:
        result = exact_algebra_check()
        self.assertTrue(result["passes"])
        self.assertLess(result["dense_relative_error"], 1e-6)
        self.assertLess(result["rmsnorm_relative_error"], 1e-6)
        self.assertEqual(result["fanout_nonzero_fraction"], 1.0)

    def test_layer_selection_is_frozen_by_relative_depth(self) -> None:
        self.assertEqual(selected_layer_indices(range(32)), (0, 8, 16, 23, 31))
        self.assertEqual(selected_layer_indices(range(4)), (0, 1, 2, 3))

    def test_per_coordinate_quantizer_is_deterministic(self) -> None:
        inputs = torch.tensor(
            [
                [[0.0, -1.0], [0.5, 1.0]],
                [[1.0, 0.0], [-1.0, 0.5]],
            ]
        )
        first = quantize_per_coordinate(inputs, 4)
        second = quantize_per_coordinate(inputs, 4)
        for left, right in zip(first, second, strict=True):
            torch.testing.assert_close(left, right)
        self.assertEqual(first[0].dtype, torch.int16)

    def test_projection_records_cover_all_lanes_and_metrics(self) -> None:
        torch.manual_seed(4)
        inputs = torch.randn(len(LANES), 20, 12)
        module = nn.Linear(12, 9, bias=False)
        records = projection_records(
            inputs=inputs,
            module=module,
            layer=3,
            projection="q_proj",
        )
        self.assertEqual(len(records), len(LANES) * 16)
        self.assertEqual({record["lane"] for record in records}, set(LANES))
        self.assertTrue(all(record["group"] == "attention" for record in records))
        self.assertEqual(set(records[0]["quantized"]), {str(bits) for bits in BITS})

    def test_gate_rejects_dense_temporal_changes(self) -> None:
        torch.manual_seed(5)
        module = nn.Linear(10, 8, bias=False)
        records = []
        for projection in ("q_proj", "down_proj"):
            records.extend(
                projection_records(
                    inputs=torch.randn(len(LANES), 20, 10),
                    module=module,
                    layer=0,
                    projection=projection,
                )
            )
        decision = gate_decision(summarize_records(records))
        self.assertEqual(decision["status"], "reject_exact_and_pretrained_slack")
        self.assertFalse(decision["free_exact_reuse"])
        self.assertFalse(decision["gpu_rental_justified"])

    def test_resource_ledger_counts_shared_inputs_and_existing_kv(self) -> None:
        modules = {
            "q_proj": nn.Linear(8, 8, bias=False),
            "k_proj": nn.Linear(8, 4, bias=False),
            "v_proj": nn.Linear(8, 4, bias=False),
            "o_proj": nn.Linear(8, 8, bias=False),
            "gate_proj": nn.Linear(8, 16, bias=False),
            "up_proj": nn.Linear(8, 16, bias=False),
            "down_proj": nn.Linear(16, 8, bias=False),
        }
        ledger = resource_ledger(modules, decoder_layers=3)
        self.assertEqual(ledger["per_layer"]["unique_previous_input_elements"], 40)
        self.assertEqual(ledger["per_layer"]["previous_output_elements"], 64)
        self.assertEqual(ledger["per_layer"]["naive_incremental_state_elements"], 104)
        self.assertEqual(
            ledger["per_layer"]["lower_bound_elements_excluding_existing_kv"], 96
        )
        self.assertEqual(ledger["all_layers"]["naive_bf16_bytes"], 624)


if __name__ == "__main__":
    unittest.main()
