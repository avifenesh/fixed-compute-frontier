from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.feature_dag_matmul_gate import (
    CONFIG,
    TARGETS,
    VARIANTS,
    WORLD_SEEDS,
    build_model,
    generate_target_data,
    prefix_algebra_check,
    screen_decision,
    train_once,
    variant_ledger,
)


class FeatureDAGMatmulGateTests(unittest.TestCase):
    def test_prefix_circuit_is_exact_and_full_rank(self) -> None:
        result = prefix_algebra_check()
        self.assertEqual(result["matrix_rank"], CONFIG.width)
        self.assertEqual(result["serial_prefix_additions"], CONFIG.width - 1)
        self.assertEqual(result["dense_nonzero_terms"], 528)
        self.assertLess(result["max_abs_error"], 1e-12)

    def test_all_variants_have_the_frozen_parameter_count(self) -> None:
        for variant in VARIANTS:
            model = build_model(variant, seed=17)
            actual = sum(parameter.numel() for parameter in model.parameters())
            self.assertEqual(actual, 1024, variant)
            self.assertEqual(variant_ledger(variant)["trainable_scalars"], actual)
        self.assertEqual(variant_ledger("lookup")["parameter_scalar_reads"], 128)
        self.assertEqual(variant_ledger("lookup")["additions"], 128)
        self.assertEqual(variant_ledger("lookup")["runtime_fixed_buffer_bytes"], 120)
        self.assertEqual(
            variant_ledger("butterfly")["runtime_fixed_buffer_bytes"], 4096
        )
        self.assertEqual(variant_ledger("feature_dag")["multiplications"], 832)

    def test_data_is_deterministic_split_independent_and_train_normalized(self) -> None:
        first = generate_target_data(WORLD_SEEDS[0], "mixed")
        repeated = generate_target_data(WORLD_SEEDS[0], "mixed")
        changed = generate_target_data(WORLD_SEEDS[1], "mixed")
        for key in first:
            np.testing.assert_array_equal(first[key], repeated[key])
        self.assertFalse(np.array_equal(first["train_x"], first["validation_x"][:4096]))
        self.assertFalse(np.array_equal(first["train_y"], changed["train_y"]))
        np.testing.assert_allclose(first["train_y"].mean(axis=0), 0.0, atol=2e-6)
        np.testing.assert_allclose(first["train_y"].std(axis=0), 1.0, atol=2e-6)

    def test_every_variant_has_the_expected_shape_and_gradient(self) -> None:
        inputs = torch.randn(9, CONFIG.width)
        for variant in VARIANTS:
            model = build_model(variant, seed=23)
            output = model(inputs)
            self.assertEqual(tuple(output.shape), (9, CONFIG.width), variant)
            output.square().mean().backward()
            gradients = [parameter.grad for parameter in model.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients), variant)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients), variant
            )

    def test_decision_requires_every_frozen_run(self) -> None:
        result = screen_decision({"runs": []})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(
            len(result["missing"]), len(WORLD_SEEDS) * len(TARGETS) * len(VARIANTS)
        )

    def test_decision_labels_mechanism_only_without_haar_noninferiority(self) -> None:
        runs = []
        for world in WORLD_SEEDS:
            for target in TARGETS:
                for variant in VARIANTS:
                    value = 0.30
                    if target == "prefix" and variant == "feature_dag":
                        value = 0.01
                    elif target == "hierarchical" and variant == "feature_dag":
                        value = 0.20
                    elif target == "mixed":
                        value = 0.20 if variant != "feature_dag" else 0.205
                    elif target == "haar":
                        value = 0.10 if variant == "dense_linear" else 0.40
                    runs.append(
                        {
                            "world_seed": world,
                            "target": target,
                            "variant": variant,
                            "test_nrmse": value,
                        }
                    )
        decision = screen_decision({"runs": runs})
        self.assertEqual(decision["status"], "mechanism_only")
        self.assertEqual(decision["hierarchy_world_wins"], 5)
        self.assertEqual(decision["mixed_noninferior_worlds"], 5)
        self.assertFalse(decision["gpu_rental_justified"])

    def test_two_step_training_smoke_stays_on_cpu(self) -> None:
        run = train_once(
            WORLD_SEEDS[0],
            "hierarchical",
            "feature_dag",
            steps=2,
            train_examples=64,
        )
        self.assertEqual(run["steps"], 2)
        self.assertTrue(np.isfinite(run["test_nrmse"]))


if __name__ == "__main__":
    unittest.main()
