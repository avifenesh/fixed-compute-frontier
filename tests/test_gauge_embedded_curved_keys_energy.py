import math

import numpy as np
import torch
import torch.nn.functional as F

from experiments.gauge_embedded_curved_keys_energy import (
    ARMS,
    HEAD_DIMENSION,
    PAIRS,
    QUERY_TYPES,
    TAU,
    attention_logits,
    balanced_energy_split,
    canonicalize,
    independent_sign_matrix,
    make_initial_parameters,
    physical_serialized_parameters,
    paired_bootstrap_primary_advantage,
    quadratic_derangements,
    recursive_numbers_finite,
    rope_frequencies,
    split_balance_checks,
)


def physical_bilinear_logits(query, key, query_types, records):
    query_one_hot = F.one_hot(query_types, QUERY_TYPES).to(records.dtype)
    query_vectors = torch.einsum("rpot,bt->rbpo", query, query_one_hot)
    raw_keys = torch.einsum("rpoc,bic->rbipo", key, records)
    even = raw_keys[..., 0]
    odd = raw_keys[..., 1]
    items = records.shape[1]
    frequencies = rope_frequencies(records.device, records.dtype)
    relative_positions = torch.arange(items, dtype=records.dtype) - items
    angles = relative_positions[:, None] * frequencies[None]
    cosine = torch.cos(angles)[None, None]
    sine = torch.sin(angles)[None, None]
    rotated_even = cosine * even - sine * odd
    rotated_odd = sine * even + cosine * odd
    return torch.sum(
        query_vectors[..., 0][:, :, None] * rotated_even
        + query_vectors[..., 1][:, :, None] * rotated_odd,
        dim=-1,
    )


def test_balanced_split_has_exact_joint_balance_and_energy_labels():
    split = balanced_energy_split(7, 64, 4)
    assert all(split_balance_checks(split, 4).values())


def test_physical_parameter_serialization_is_exactly_40_fp32_values():
    query, key, polar = make_initial_parameters(11, 3, torch.device("cpu"))
    serialized = physical_serialized_parameters(query, key, polar)
    assert serialized.shape == (3, 40)
    assert serialized.element_size() * serialized.shape[-1] == 160
    physical_key = serialized[:, 16:].reshape(3, HEAD_DIMENSION, 3)
    np.testing.assert_allclose(
        physical_key[:, 0::2, 2].numpy(), 0.0, atol=0.0
    )
    np.testing.assert_allclose(
        physical_key[:, 1::2, 2].numpy(), TAU, atol=0.0
    )


def test_canonicalization_preserves_every_ordinary_rope_logit():
    query, key, polar = make_initial_parameters(13, 2, torch.device("cpu"))
    polar = polar + torch.tensor([0.31, -0.27])[None, None]
    records = torch.randn(17, 4, 2, generator=torch.Generator().manual_seed(17))
    query_types = torch.arange(17) % 2
    compiled = attention_logits(
        query, key, polar, query_types, records, "bilinear"
    )
    physical = physical_bilinear_logits(query, key, query_types, records)
    torch.testing.assert_close(compiled, physical, atol=2e-6, rtol=2e-6)


def test_all_arms_start_on_exact_same_nonlinear_off_slice():
    query, key, polar = make_initial_parameters(19, 2, torch.device("cpu"))
    records = torch.randn(23, 4, 2, generator=torch.Generator().manual_seed(23))
    query_types = torch.arange(23) % 2
    logits = [
        attention_logits(query, key, polar, query_types, records, arm)
        for arm in ARMS
    ]
    for candidate in logits[1:]:
        torch.testing.assert_close(candidate, logits[0], atol=0.0, rtol=0.0)


def test_centered_and_uncentered_scale_funded_g1_contain_bilinear_when_log_scale_is_zero():
    query, key, polar = make_initial_parameters(29, 2, torch.device("cpu"))
    polar[..., 0] = torch.tensor([[0.2, -0.3, 0.4, -0.5]])
    polar[..., 1] = 0.0
    records = torch.randn(31, 4, 2, generator=torch.Generator().manual_seed(31))
    query_types = torch.arange(31) % 2
    bilinear = attention_logits(
        query, key, polar, query_types, records, "bilinear"
    )
    for arm in ("centered_scale_only_geck", "full_scale_only_geck"):
        candidate = attention_logits(
            query, key, polar, query_types, records, arm
        )
        torch.testing.assert_close(candidate, bilinear, atol=0.0, rtol=0.0)


def test_every_arm_has_finite_gradients_away_from_off_slice():
    query, key, polar = make_initial_parameters(37, 1, torch.device("cpu"))
    records = torch.randn(32, 4, 2, generator=torch.Generator().manual_seed(41))
    query_types = torch.arange(32) % 2
    targets = torch.arange(32) % 4
    for arm in ARMS:
        arm_query = query.clone().requires_grad_(True)
        arm_key = key.clone().requires_grad_(True)
        arm_polar = (polar + 0.1).clone().requires_grad_(True)
        logits = attention_logits(
            arm_query, arm_key, arm_polar, query_types, records, arm
        )
        loss = F.cross_entropy(logits[0], targets)
        loss.backward()
        assert math.isfinite(float(loss))
        assert torch.isfinite(arm_query.grad).all()
        assert torch.isfinite(arm_key.grad).all()
        assert torch.isfinite(arm_polar.grad).all()


def test_counterfactual_generators_preserve_labels_and_move_every_square_block():
    split = balanced_energy_split(43, 64, 4)
    signs = independent_sign_matrix(47, split["keys"].shape)
    flipped = split["keys"] * signs
    selected = flipped[
        np.arange(64)[:, None],
        np.arange(4)[None, :],
        split["query_types"][:, None],
    ]
    assert np.array_equal(
        np.argmax(np.square(selected), axis=-1), split["targets"]
    )
    derangements = quadratic_derangements(53, 64, 4)
    assert derangements.shape == (64, 4)
    assert np.all(derangements != np.arange(4)[None])
    for permutation in derangements:
        assert np.array_equal(np.sort(permutation), np.arange(4))


def test_paired_bootstrap_recomputes_best_nonquadratic_competitor():
    samples = 128
    targets = np.arange(samples) % 4

    def logits_with_correct(correct_mask):
        logits = torch.zeros(1, samples, 4)
        predictions = np.where(correct_mask, targets, (targets + 1) % 4)
        logits[0, np.arange(samples), predictions] = 6.0
        return {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=-1),
            "nll": torch.zeros(1),
            "accuracy": torch.zeros(1),
            "per_type_accuracy": torch.zeros(1, 2),
        }

    evaluations = {
        "full_scale_only_geck": logits_with_correct(np.ones(samples, dtype=bool)),
        "bilinear": logits_with_correct(np.arange(samples) % 4 == 0),
        "gauge_linear_control": logits_with_correct(np.arange(samples) % 4 == 1),
        "position_only_mean_square_control": logits_with_correct(np.arange(samples) % 4 == 2),
    }
    result = paired_bootstrap_primary_advantage(evaluations, targets, 59)
    assert result["replicates"] == 2_000
    assert result["accuracy_advantage_lower_95_percent"] > 0.60
    assert result["nll_advantage_lower_95_percent"] > 3.0


def test_recursive_finiteness_rejects_hidden_counterfactual_nan():
    assert recursive_numbers_finite({
        "arms": [{"nll": 0.1, "parameters": [1.0, 2.0]}],
        "counterfactuals": {"accuracy": 0.9},
    })
    assert not recursive_numbers_finite({
        "arms": [{"nll": 0.1}],
        "counterfactuals": {"nested": [0.9, float("nan")]},
    })
