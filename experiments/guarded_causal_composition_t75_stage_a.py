#!/usr/bin/env python3
"""Model-free T75 Stage A validator. No training and no GPU."""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.stats import beta


M = 33
EPSILON = 0.1
WORLDS = [(t, s) for t in range(1, M) for s in (0, 1)]
ORIENTATION = np.array(
    [[s ^ int(u >= t) for u in range(M)] for t, s in WORLDS], dtype=np.int8
)
P_Y1 = np.where(ORIENTATION == 0, EPSILON, 0.5)


def binomial_classification_errors(repetitions: int) -> tuple[float, float]:
    """Errors for equal-prior LRT; an exact likelihood tie predicts class 0."""
    error_0 = 0.0
    error_1 = 0.0
    for successes in range(repetitions + 1):
        coefficient = math.comb(repetitions, successes)
        likelihood_0 = coefficient * 0.1**successes * 0.9 ** (repetitions - successes)
        likelihood_1 = coefficient * 0.5**repetitions
        if likelihood_1 > likelihood_0:
            error_0 += likelihood_0
        else:
            error_1 += likelihood_1
    return error_0, error_1


def repetitions_for(decisions: int) -> tuple[int, tuple[float, float]]:
    for repetitions in range(1, 10_000):
        errors = binomial_classification_errors(repetitions)
        if decisions * max(errors) <= 0.05:
            return repetitions, errors
    raise AssertionError("repetition search did not terminate")


def information_floor() -> dict[str, float]:
    with localcontext() as context:
        context.prec = 80
        one = Decimal(1)
        two = Decimal(2)
        epsilon = Decimal(1) / Decimal(10)
        gap = Decimal(4) / Decimal(10)
        ln_two = two.ln()

        def entropy(probability: Decimal) -> Decimal:
            if probability in (0, 1):
                return Decimal(0)
            return -(
                probability * probability.ln()
                + (one - probability) * (one - probability).ln()
            ) / ln_two

        h_epsilon = entropy(epsilon)
        exponent = (one - h_epsilon) / gap
        p_star = one / (one + (exponent * ln_two).exp())
        q_star = (p_star - epsilon) / gap
        capacity = entropy(p_star) - (one - q_star) * h_epsilon - q_star
        error = Decimal(5) / Decimal(100)
        fano_numerator = (
            Decimal(6) - entropy(error) - error * (Decimal(63).ln() / ln_two)
        )
        fano_budget = fano_numerator / capacity

    q_value = float(q_star)
    capacity_value = float(capacity)
    budget_value = float(fano_budget)
    assert 0.462312 < q_value < 0.462314
    assert 0.1475894 < capacity_value < 0.1475895
    assert 36.68 < budget_value < 36.70
    return {
        "q_star": q_value,
        "capacity_bits_per_probe": capacity_value,
        "fano_fixed_real_budget_floor": budget_value,
        "fano_fixed_integer_budget_floor": math.ceil(budget_value),
        "fano_variable_expected_budget_floor": budget_value,
    }


def posterior_float(history: list[tuple[int, int]]) -> np.ndarray:
    posterior = np.full(len(WORLDS), 1.0 / len(WORLDS))
    for selector, outcome in history:
        likelihood = P_Y1[:, selector] if outcome else 1.0 - P_Y1[:, selector]
        posterior *= likelihood
        posterior /= posterior.sum()
    return posterior


def posterior_fraction(history: list[tuple[int, int]]) -> tuple[Fraction, ...]:
    weights = [Fraction(1, len(WORLDS)) for _ in WORLDS]
    for selector, outcome in history:
        for index in range(len(WORLDS)):
            p_one = Fraction(1, 10) if ORIENTATION[index, selector] == 0 else Fraction(1, 2)
            weights[index] *= p_one if outcome else 1 - p_one
        normalizer = sum(weights)
        weights = [weight / normalizer for weight in weights]
    return tuple(weights)


def invariant_checks() -> dict[str, bool]:
    strings = {tuple(row.tolist()) for row in ORIENTATION}
    assert len(WORLDS) == 64 and len(strings) == 64
    assert np.all(P_Y1 > 0) and np.all(P_Y1 < 1)
    assert np.array_equal(P_Y1 + (1.0 - P_Y1), np.ones_like(P_Y1))

    prior = np.full(64, 1.0 / 64)
    passive = prior * np.ones(64)
    passive /= passive.sum()
    assert np.array_equal(passive, prior)

    history = [(0, 1), (16, 0), (7, 1), (16, 1), (31, 0)]
    float_posterior = posterior_float(history)
    rational_posterior = np.array([float(value) for value in posterior_fraction(history)])
    assert np.allclose(float_posterior, rational_posterior, atol=1e-15, rtol=1e-14)
    assert np.allclose(float_posterior, posterior_float(list(reversed(history))), atol=1e-15)

    for world_index, (threshold, low_orientation) in enumerate(WORLDS):
        reflected_world = (M - threshold, 1 - low_orientation)
        reflected_index = WORLDS.index(reflected_world)
        assert np.array_equal(ORIENTATION[world_index, ::-1], ORIENTATION[reflected_index])

    fixed_strings = {tuple(row[: M - 1].tolist()) for row in ORIENTATION}
    assert len(fixed_strings) == 64
    for omitted in range(1, M - 1):
        left = WORLDS.index((omitted, 0))
        right = WORLDS.index((omitted + 1, 0))
        kept = [u for u in range(M) if u != omitted]
        assert np.array_equal(ORIENTATION[left, kept], ORIENTATION[right, kept])
    interior = list(range(1, M - 1))
    edge_a = WORLDS.index((1, 0))
    edge_b = WORLDS.index((M - 1, 1))
    assert np.array_equal(ORIENTATION[edge_a, interior], ORIENTATION[edge_b, interior])

    simulator_table = np.array(
        [
            [0.1 if (s ^ int(u >= t)) == 0 else 0.5 for u in range(M)]
            for t, s in WORLDS
        ]
    )
    assert np.array_equal(simulator_table, P_Y1)
    return {
        "world_injectivity": True,
        "likelihood_normalization": True,
        "passive_zero_information": True,
        "rational_float_filter_agreement": True,
        "evidence_order_invariance": True,
        "selector_reflection_equivariance": True,
        "nonadaptive_location_witnesses": True,
        "simulator_filter_identity": True,
    }


def constructive_path(threshold: int, low_orientation: int) -> list[int]:
    selectors = [0]
    low = 1
    high = M - 1
    while low < high:
        midpoint = (low + high) // 2
        selectors.append(midpoint)
        local_orientation = low_orientation ^ int(midpoint >= threshold)
        if local_orientation != low_orientation:
            high = midpoint
        else:
            low = midpoint + 1
    assert low == threshold and len(selectors) == 6
    return selectors


def analytic_references() -> dict[str, object]:
    adaptive_r, adaptive_errors = repetitions_for(6)
    nonadaptive_r, nonadaptive_errors = repetitions_for(32)
    assert adaptive_r == 28
    assert nonadaptive_r == 40

    adaptive_accuracy = []
    nonadaptive_accuracy = []
    for threshold, low_orientation in WORLDS:
        path = constructive_path(threshold, low_orientation)
        path_orientations = [low_orientation ^ int(u >= threshold) for u in path]
        adaptive_accuracy.append(
            math.prod(1.0 - adaptive_errors[orientation] for orientation in path_orientations)
        )

        if threshold < M - 1:
            probability = (1.0 - nonadaptive_errors[low_orientation]) ** threshold
            probability *= 1.0 - nonadaptive_errors[1 - low_orientation]
        else:
            probability = (1.0 - nonadaptive_errors[low_orientation]) ** (M - 1)
        nonadaptive_accuracy.append(probability)

    adaptive_average = float(np.mean(adaptive_accuracy))
    adaptive_worst = min(adaptive_accuracy)
    nonadaptive_average = float(np.mean(nonadaptive_accuracy))
    nonadaptive_worst = min(nonadaptive_accuracy)
    assert adaptive_average >= 0.95 and adaptive_worst >= 0.95
    assert nonadaptive_average >= 0.95 and nonadaptive_worst >= 0.95
    return {
        "constructive_adaptive": {
            "repetitions_per_decision": adaptive_r,
            "class_conditional_errors": list(adaptive_errors),
            "probes": 6 * adaptive_r,
            "average_accuracy_exact": adaptive_average,
            "worst_world_accuracy_exact": adaptive_worst,
        },
        "fixed_nonadaptive": {
            "repetitions_per_selector": nonadaptive_r,
            "class_conditional_errors": list(nonadaptive_errors),
            "probes": 32 * nonadaptive_r,
            "average_accuracy_exact": nonadaptive_average,
            "worst_world_accuracy_exact": nonadaptive_worst,
        },
    }


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-300, 1.0 - 1e-15)
    return -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))


def run_entropy_greedy(true_worlds: np.ndarray, outcome_seed: int) -> dict[str, object]:
    episode_count = len(true_worlds)
    posterior = np.full((episode_count, len(WORLDS)), 1.0 / len(WORLDS))
    actions = np.zeros(episode_count, dtype=np.int32)
    stopped = np.zeros(episode_count, dtype=bool)
    rng = np.random.default_rng(outcome_seed)
    orientation_float = ORIENTATION.astype(np.float64)
    h_epsilon = float(binary_entropy(np.array([EPSILON]))[0])

    for _ in range(168):
        active = np.flatnonzero(~stopped)
        if len(active) == 0:
            break
        active_posterior = posterior[active]
        q_orientation_1 = active_posterior @ orientation_float
        p_y1 = EPSILON + (0.5 - EPSILON) * q_orientation_1
        information_gain = (
            binary_entropy(p_y1)
            - (1.0 - q_orientation_1) * h_epsilon
            - q_orientation_1
        )
        selectors = np.argmax(information_gain, axis=1)
        true_orientation = ORIENTATION[true_worlds[active], selectors]
        true_p_y1 = np.where(true_orientation == 0, EPSILON, 0.5)
        outcomes = rng.random(len(active)) < true_p_y1

        selected_p_y1 = P_Y1[:, selectors].T
        likelihood = np.where(outcomes[:, None], selected_p_y1, 1.0 - selected_p_y1)
        active_posterior *= likelihood
        active_posterior /= active_posterior.sum(axis=1, keepdims=True)
        posterior[active] = active_posterior
        actions[active] += 1
        stopped[active] = np.max(active_posterior, axis=1) >= 0.95

    predictions = np.argmax(posterior, axis=1)
    correct = predictions == true_worlds
    censored = np.max(posterior, axis=1) < 0.95
    return {
        "correct": correct,
        "actions": actions,
        "censored": censored,
        "max_posterior": np.max(posterior, axis=1),
    }


def clopper_pearson(successes: int, total: int, alpha: float) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def summarize_greedy() -> dict[str, object]:
    world_rng = np.random.default_rng(75001)
    average_worlds = world_rng.integers(0, len(WORLDS), size=8192)
    average = run_entropy_greedy(average_worlds, 75002)
    average_successes = int(np.sum(average["correct"]))

    stratified_worlds = np.repeat(np.arange(len(WORLDS)), 128)
    stratified = run_entropy_greedy(stratified_worlds, 75003)
    per_world_accuracy = []
    per_world_intervals = []
    for world_index in range(len(WORLDS)):
        mask = stratified_worlds == world_index
        successes = int(np.sum(stratified["correct"][mask]))
        per_world_accuracy.append(successes / 128)
        per_world_intervals.append(clopper_pearson(successes, 128, 0.05 / len(WORLDS)))

    action_counts = average["actions"]
    return {
        "policy": "exact_posterior_one_step_entropy_greedy_not_globally_optimal",
        "horizon": 168,
        "posterior_stop": 0.95,
        "average_episodes": 8192,
        "average_accuracy": average_successes / 8192,
        "average_accuracy_cp95": list(clopper_pearson(average_successes, 8192, 0.05)),
        "average_censored": int(np.sum(average["censored"])),
        "actions": {
            "mean": float(np.mean(action_counts)),
            "median": float(np.median(action_counts)),
            "p95": float(np.quantile(action_counts, 0.95, method="higher")),
            "max": int(np.max(action_counts)),
        },
        "stratified_episodes_per_world": 128,
        "worst_world_accuracy": min(per_world_accuracy),
        "worst_bonferroni_cp95_lower": min(interval[0] for interval in per_world_intervals),
        "stratified_censored": int(np.sum(stratified["censored"])),
    }


def small_exact_dp() -> dict[str, object]:
    small_m = 3
    small_worlds = [(t, s) for t in range(1, small_m) for s in (0, 1)]
    action_cost = Fraction(1, 100)

    def p_one(world_index: int, selector: int) -> Fraction:
        threshold, low_orientation = small_worlds[world_index]
        orientation = low_orientation ^ int(selector >= threshold)
        return Fraction(1, 10) if orientation == 0 else Fraction(1, 2)

    def transition(
        belief: tuple[Fraction, ...], selector: int, outcome: int
    ) -> tuple[Fraction, tuple[Fraction, ...]]:
        weights = []
        for world_index, prior_weight in enumerate(belief):
            likelihood_one = p_one(world_index, selector)
            likelihood = likelihood_one if outcome else 1 - likelihood_one
            weights.append(prior_weight * likelihood)
        probability = sum(weights)
        return probability, tuple(weight / probability for weight in weights)

    @lru_cache(maxsize=None)
    def value(belief: tuple[Fraction, ...], horizon: int) -> Fraction:
        stop = 1 - max(belief)
        if horizon == 0:
            return stop
        action_values = []
        for selector in range(small_m):
            expected = action_cost
            for outcome in (0, 1):
                probability, next_belief = transition(belief, selector, outcome)
                expected += probability * value(next_belief, horizon - 1)
            action_values.append(expected)
        return min([stop, *action_values])

    prior = tuple(Fraction(1, len(small_worlds)) for _ in small_worlds)
    values = [value(prior, horizon) for horizon in range(7)]
    assert all(values[index + 1] <= values[index] for index in range(len(values) - 1))
    return {
        "m": small_m,
        "horizons": list(range(7)),
        "lambda": float(action_cost),
        "values": [float(item) for item in values],
        "values_exact": [str(item) for item in values],
        "cached_states": value.cache_info().currsize,
        "monotone_nonincreasing": True,
    }


def main() -> None:
    report = {
        "experiment": "t75-stage-a-model-free-v0",
        "scope": "mechanics_only_no_neural_claim",
        "world": {"m": M, "epsilon": EPSILON, "worlds": len(WORLDS)},
        "invariants": invariant_checks(),
        "information_floor": information_floor(),
        "analytic_references": analytic_references(),
        "entropy_greedy_reference": summarize_greedy(),
        "small_exact_dp": small_exact_dp(),
        "status": "pass",
        "admission": "no_neural_no_gpu",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
