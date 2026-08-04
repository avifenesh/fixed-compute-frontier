from __future__ import annotations

import unittest

import numpy as np

from experiments.gauge_curvature_learned_routing import (
    CONFIG,
    VARIANTS,
    binary_nll,
    constrained_marginal_logits,
    data_invariants,
    generate_scenes,
    materialize_inputs,
    screen_decision,
    variant_ledger,
)


class GaugeCurvatureLearnedRoutingTests(unittest.TestCase):
    def test_scene_closes_global_label_and_address_shortcuts(self) -> None:
        data = generate_scenes(
            scenes=64, world_seed=17011, split_seed=1701103
        )
        invariants = data_invariants(data)
        self.assertEqual(invariants["balanced_labels_max_abs_error"], 0.0)
        self.assertLess(invariants["paired_content_mean_max_abs"], 1e-6)
        self.assertLess(invariants["address_orthogonality_max_abs_error"], 1e-5)
        self.assertGreater(invariants["oracle_accuracy"], 0.85)
        self.assertGreater(invariants["oracle_accuracy"], 0.91)
        self.assertLess(invariants["oracle_nll"], 0.21)
        self.assertEqual(invariants["uniform_all_bags_accuracy"], 0.5)

    def test_materialized_inputs_keep_streams_disjoint(self) -> None:
        data = generate_scenes(
            scenes=8, world_seed=17027, split_seed=1702701
        )
        inputs = materialize_inputs(data, (0, 3, 7))
        self.assertEqual(inputs.shape, (3, CONFIG.sequence_length, CONFIG.model_width))
        self.assertTrue(
            np.all(inputs[:, CONFIG.source_tokens :, : CONFIG.content_width] == 0.0)
        )
        self.assertTrue(np.all(inputs[:, :, 2 * CONFIG.content_width + 2 :] == 0.0))
        self.assertTrue(
            np.all(inputs[:, : CONFIG.source_tokens, 2 * CONFIG.content_width] == 1.0)
        )
        self.assertTrue(
            np.all(inputs[:, CONFIG.source_tokens :, 2 * CONFIG.content_width + 1] == 1.0)
        )

    def test_resource_ledgers_match_frozen_budget(self) -> None:
        dense = variant_ledger("dense")
        for variant in (
            "cyclic",
            "source_mlp",
            "g2_matched",
            "g2_full",
            "glu",
            "g1",
        ):
            current = variant_ledger(variant)
            self.assertEqual(
                current["attention_plus_ffn_parameters_two_layers"],
                dense["attention_plus_ffn_parameters_two_layers"],
                variant,
            )
        self.assertEqual(
            variant_ledger("cyclic")[
                "attention_matrix_multiplications_per_token_layer"
            ]
            + variant_ledger("cyclic")[
                "attention_explicit_products_per_token_layer"
            ],
            dense["attention_matrix_multiplications_per_token_layer"],
        )
        self.assertEqual(variant_ledger("glu")["kv_cache_scalars_per_token_layer"], 84)
        self.assertEqual(dense["kv_cache_scalars_per_token_layer"], 96)
        self.assertLess(
            variant_ledger("square")["actual_trainable_parameters_expected"],
            dense["actual_trainable_parameters_expected"],
        )
        self.assertEqual(set(VARIANTS), {
            "dense", "bda", "cyclic", "square", "source_mlp",
            "g2_matched", "g2_full", "glu", "g1"
        })

    def test_screen_decision_is_incomplete_without_results(self) -> None:
        decision = screen_decision({"runs": []})
        self.assertEqual(decision["status"], "incomplete")
        self.assertEqual(len(decision["missing"]), 5 * len(VARIANTS))

    def test_generation_is_deterministic_and_splits_change(self) -> None:
        first = generate_scenes(
            scenes=16, world_seed=17041, split_seed=1704101
        )
        repeated = generate_scenes(
            scenes=16, world_seed=17041, split_seed=1704101
        )
        changed = generate_scenes(
            scenes=16, world_seed=17041, split_seed=1704102
        )
        for key in first:
            np.testing.assert_array_equal(first[key], repeated[key])
        self.assertFalse(np.array_equal(first["source_content"], changed["source_content"]))

    def test_constrained_oracle_uses_other_bags(self) -> None:
        individual = np.array(((2.0, 1.0, -0.5, -1.0),), dtype=np.float64)
        constrained = constrained_marginal_logits(individual)
        self.assertEqual(constrained.shape, individual.shape)
        marginal = 1.0 / (1.0 + np.exp(-constrained))
        self.assertAlmostEqual(float(marginal.sum()), 2.0, places=12)
        labels = np.array(((1.0, 1.0, 0.0, 0.0),), dtype=np.float64)
        self.assertLess(binary_nll(constrained, labels), binary_nll(individual, labels))


if __name__ == "__main__":
    unittest.main()
