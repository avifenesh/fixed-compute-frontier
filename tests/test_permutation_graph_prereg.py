from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads(
    (ROOT / "manifests/permutation-graph-refinement.pre004.json").read_text()
)


class PermutationGraphPreregistrationTests(unittest.TestCase):
    def test_remains_pre_candidate_gpu_free_and_training_locked(self) -> None:
        self.assertEqual(PLAN["status"], "pre_candidate_falsification")
        self.assertIsNone(PLAN["candidate_number"])
        self.assertIsNone(PLAN["a_e_claim"])
        self.assertFalse(PLAN["gpu_authorized"])
        self.assertFalse(PLAN["training_authorized"])
        self.assertFalse(PLAN["freeze"]["stage_1_authorized"])
        self.assertFalse(PLAN["promotion"]["may_assign_candidate_004"])

    def test_streaming_contract_closes_hidden_history_loophole(self) -> None:
        stream = PLAN["streaming_contract"]
        self.assertTrue(stream["commands_visible_one_at_a_time"])
        self.assertEqual(stream["controller_calls_per_command"], 1)
        self.assertFalse(stream["past_command_reread"])
        self.assertFalse(stream["persistent_encoder_activations"])
        self.assertTrue(stream["all_cross_step_bits_must_be_in_state_arena"])

    def test_primary_machine_and_arena_are_frozen(self) -> None:
        mechanism = PLAN["mechanism"]
        arena = PLAN["primary_state_arena"]
        self.assertEqual(mechanism["node_capacity"], 64)
        self.assertEqual(mechanism["primary_degree"], 2)
        self.assertEqual(mechanism["refinement_sweeps"], 8)
        self.assertEqual(mechanism["relation_schedule"], [0, 1] * 4)
        self.assertEqual(mechanism["full_model_calls_per_rewrite"], 0)
        self.assertFalse(mechanism["exact_attention_fallback"])
        self.assertFalse(mechanism["physical_in_place_claim_before_stage_4"])
        self.assertEqual(arena["allocated_bytes_per_episode"], 4096)
        self.assertEqual(sum(view["bytes"] for view in arena["views"]), 4096)
        self.assertFalse(arena["side_episode_state_allowed"])

    def test_resource_lanes_do_not_use_dummy_cross_dtype_matching(self) -> None:
        budgets = PLAN["budgets"]
        lanes = PLAN["comparison_lanes"]
        self.assertEqual(budgets["primary_persistent_state_allocated_bytes_cap"], 4096)
        self.assertEqual(budgets["learned_parameters_cap"], 262144)
        self.assertFalse(budgets["dummy_work_for_matching"])
        self.assertFalse(lanes["single_cross_dtype_operation_equivalence"])
        self.assertIn("hash", budgets["typed_operation_columns"])
        self.assertIn("gather", budgets["typed_operation_columns"])

    def test_length_and_effective_hop_extrapolation_are_locked(self) -> None:
        workload = PLAN["workload"]
        self.assertEqual(workload["train_max_mutations"], 32)
        self.assertEqual(workload["evaluation_mutations"], [64, 256, 1024])
        self.assertGreaterEqual(workload["minimum_relational_operand_fraction"], 0.5)
        self.assertEqual(workload["minimum_distinct_nodes_for_eight_hop"], 9)
        self.assertEqual(workload["minimum_distinct_nodes_for_thirty_two_hop"], 33)
        self.assertTrue(workload["query_pointer_reads_counted"])

    def test_ancestor_axis_is_separate_and_includes_strong_exact_controls(self) -> None:
        axis = PLAN["ancestor_axis"]
        self.assertFalse(axis["inherits_primary_4096_byte_cap"])
        modes = {mode["name"]: mode for mode in axis["modes"]}
        self.assertEqual(
            set(modes),
            {
                "parent_recompute",
                "exact_interval_rebuild",
                "materialized_closure_bitset",
                "ancestor_bloom64_rebuild",
                "continuous_ancestor_bf16x16",
            },
        )
        self.assertEqual(modes["exact_interval_rebuild"]["batch_one_allocated_bytes"], "6*N")
        self.assertFalse(modes["ancestor_bloom64_rebuild"]["counters"])
        self.assertFalse(modes["ancestor_bloom64_rebuild"]["hidden_child_index"])
        self.assertFalse(axis["path_aggregate_claim_allowed_for_bloom_or_continuous_modes"])
        self.assertTrue(axis["closure_compression_is_not_memory_superiority"])
        self.assertEqual(
            axis["expected_batch_one_allocated_bytes_by_n"]["1024"],
            {
                "parent_recompute": 2048,
                "exact_interval_rebuild": 6144,
                "materialized_closure_bitset": 133120,
                "ancestor_bloom64_rebuild": 10272,
                "continuous_ancestor_bf16x16": 34816,
            },
        )

    def test_controls_receive_same_exact_primitives(self) -> None:
        variants = PLAN["variants"]
        self.assertEqual(
            set(variants["envelope_controls"]),
            {
                "flat_exact_registers",
                "continuous_recurrent_graph",
                "one_pass_discrete_graph",
                "tiny_transformer_or_recurrent_scratchpad",
            },
        )
        self.assertTrue(variants["all_receive_same_exact_global_primitive_bank"])
        self.assertTrue(variants["same_training_examples_optimizer_updates_and_tuning_trials"])

    def test_stage_order_and_causal_gates_are_locked(self) -> None:
        self.assertEqual(
            PLAN["stage_order"],
            [
                "oracle_executor_generator_and_accounting",
                "oracle_bound_frozen_correction",
                "learned_canonical_command_routing",
                "heldout_synthetic_language_routing",
                "hardware_feasibility",
            ],
        )
        gates = PLAN["gates"]
        self.assertEqual(gates["stage_0_oracle_bitwise_accuracy"], 1.0)
        self.assertGreaterEqual(gates["learned_trajectory_accuracy_at_1024"], 0.99)
        self.assertGreaterEqual(gates["stage_1_minimum_complete_trajectory_advantage_points"], 5.0)
        self.assertFalse(gates["stage_2_superiority_required"])
        self.assertEqual(gates["invariant_violations_allowed"], 0)
        self.assertTrue(gates["hardware_stage_requires_all_prior_stages"])

    def test_stage_zero_gold_interpreter_is_independent(self) -> None:
        gold = PLAN["stage_0_gold_validation"]
        requirements = PLAN["stage_0_child_manifest_requirements"]
        self.assertTrue(gold["reference_interpreter_must_not_share_transition_code"])
        self.assertEqual(gold["exhaustive_small_n_max"], 6)
        self.assertIn("node_renaming_equivariance", gold["metamorphic_invariants"])
        self.assertIn("independent_reference_interpreter_sha256", requirements)

    def test_corruption_and_statistics_cannot_be_selected_post_hoc(self) -> None:
        corruption = PLAN["corruption_grid"]
        statistics = PLAN["statistics"]
        self.assertFalse(corruption["post_result_cell_selection"])
        self.assertFalse(corruption["corrupts_hidden_task_state"])
        self.assertTrue(corruption["identical_evidence_for_every_variant"])
        self.assertEqual(statistics["bootstrap_resamples"], 10000)
        self.assertEqual(statistics["bootstrap_rng_seed"], 20260725)
        self.assertEqual(len(set(statistics["training_seeds"])), 3)
        self.assertEqual(statistics["simultaneous_method"], "bootstrap_max_statistic")


if __name__ == "__main__":
    unittest.main()
