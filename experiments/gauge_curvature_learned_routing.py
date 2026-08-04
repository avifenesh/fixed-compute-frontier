"""Frozen learned-Q/K gate for gauge-funded nonlinear value writers.

The data and resource accounting in this module intentionally do not import
PyTorch, so the algebra and leakage checks run on a CPU-only development
machine.  ``gauge_curvature_learned_routing_torch`` owns the GPU model and
training loop and is imported only by the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = (
    ROOT_DIR / "results" / "gauge-curvature-learned-routing-preregistration.md"
)
DEFAULT_SCREEN_PATH = (
    ROOT_DIR / "results" / "gauge-curvature-learned-routing-screen.json"
)

WORLD_SEEDS = (17011, 17027, 17041, 17053, 17077)
SCREEN_SEEDS = (314159,)
CONFIRMATION_SEEDS = (271828, 161803)
VARIANTS = (
    "dense",
    "bda",
    "cyclic",
    "square",
    "source_mlp",
    "g2_matched",
    "g2_full",
    "glu",
    "g1",
)

Array = np.ndarray


@dataclass(frozen=True)
class ExperimentConfig:
    model_width: int = 96
    content_width: int = 16
    address_width: int = 16
    query_heads: int = 6
    kv_heads: int = 3
    head_width: int = 16
    layers: int = 2
    bags: int = 4
    independent_draws: int = 8
    source_tokens_per_bag: int = 16
    query_tokens: int = 4
    address_scale: float = 4.0
    covariance_delta: float = 0.15
    base_ffn_width: int = 256
    train_scenes: int = 8_192
    validation_scenes: int = 1_024
    test_scenes: int = 2_048
    batch_size: int = 128
    steps: int = 1_500
    validation_interval: int = 150
    learning_rate: float = 2e-3
    minimum_learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    route_loss_weight: float = 0.05
    gradient_clip: float = 1.0

    @property
    def source_tokens(self) -> int:
        return self.bags * self.source_tokens_per_bag

    @property
    def sequence_length(self) -> int:
        return self.source_tokens + self.query_tokens


CONFIG = ExperimentConfig()


def orthogonal(rng: np.random.Generator, width: int) -> Array:
    matrix, upper = np.linalg.qr(rng.normal(size=(width, width)))
    signs = np.sign(np.diag(upper))
    signs[signs == 0.0] = 1.0
    return (matrix * signs).astype(np.float32)


def world_rotation(world_seed: int, width: int = CONFIG.content_width) -> Array:
    return orthogonal(np.random.default_rng(world_seed), width)


def _balanced_scene_labels(rng: np.random.Generator, scenes: int) -> Array:
    labels = np.empty((scenes, CONFIG.bags), dtype=np.float32)
    base = np.array((0.0, 0.0, 1.0, 1.0), dtype=np.float32)
    for scene in range(scenes):
        labels[scene] = base[rng.permutation(CONFIG.bags)]
    return labels


def _fresh_addresses(rng: np.random.Generator, scenes: int) -> Array:
    raw = rng.normal(size=(scenes, CONFIG.address_width, CONFIG.bags))
    basis, upper = np.linalg.qr(raw)
    diagonal = np.diagonal(upper, axis1=-2, axis2=-1)
    signs = np.sign(diagonal)
    signs[signs == 0.0] = 1.0
    basis *= signs[:, None, :]
    return np.transpose(basis, (0, 2, 1)).astype(np.float32)


def _shuffle_sources(
    rng: np.random.Generator, content: Array
) -> tuple[Array, Array]:
    scenes = content.shape[0]
    flat_content = content.reshape(
        scenes, CONFIG.source_tokens, CONFIG.content_width
    )
    flat_bags = np.broadcast_to(
        np.arange(CONFIG.bags, dtype=np.int64)[None, :, None],
        (scenes, CONFIG.bags, CONFIG.source_tokens_per_bag),
    ).reshape(scenes, CONFIG.source_tokens)
    order = np.argsort(rng.random((scenes, CONFIG.source_tokens)), axis=1)
    shuffled_content = np.take_along_axis(flat_content, order[:, :, None], axis=1)
    shuffled_bags = np.take_along_axis(flat_bags, order, axis=1)
    return shuffled_content.astype(np.float32), shuffled_bags


def constrained_marginal_logits(individual_logits: Array) -> Array:
    """Condition four independent bag LLRs on exactly two positive labels."""

    if individual_logits.ndim != 2 or individual_logits.shape[1] != CONFIG.bags:
        raise ValueError("individual logits must have shape [scenes, four bags]")
    assignments = np.asarray(
        [
            [1.0 if bag in positive else 0.0 for bag in range(CONFIG.bags)]
            for positive in (
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            )
        ],
        dtype=np.float64,
    )
    assignment_scores = individual_logits.astype(np.float64) @ assignments.T
    assignment_scores -= assignment_scores.max(axis=1, keepdims=True)
    assignment_probability = np.exp(assignment_scores)
    assignment_probability /= assignment_probability.sum(axis=1, keepdims=True)
    marginal = assignment_probability @ assignments
    marginal = np.clip(marginal, 1e-12, 1.0 - 1e-12)
    return np.log(marginal) - np.log1p(-marginal)


def generate_scenes(
    *,
    scenes: int,
    world_seed: int,
    split_seed: int,
    condition: str = "paired_covariance",
) -> dict[str, Array]:
    """Generate one deterministic split without materializing 96-wide inputs."""

    if scenes <= 0:
        raise ValueError("scenes must be positive")
    if condition not in {"paired_covariance", "unpaired_covariance", "mean"}:
        raise ValueError(f"unknown condition: {condition}")

    rng = np.random.default_rng(split_seed)
    rotation = world_rotation(world_seed)
    labels_by_bag = _balanced_scene_labels(rng, scenes)
    signs = 2.0 * labels_by_bag - 1.0
    addresses = _fresh_addresses(rng, scenes)
    spectrum_sign = np.concatenate(
        (
            np.ones(CONFIG.content_width // 2),
            -np.ones(CONFIG.content_width // 2),
        )
    ).astype(np.float64)

    if condition in {"paired_covariance", "unpaired_covariance"}:
        draws_per_bag = (
            CONFIG.independent_draws
            if condition == "paired_covariance"
            else CONFIG.source_tokens_per_bag
        )
        eigenvalues = (
            1.0
            + CONFIG.covariance_delta
            * signs[:, :, None]
            * spectrum_sign[None, None, :]
        )
        standard = rng.normal(
            size=(scenes, CONFIG.bags, draws_per_bag, CONFIG.content_width)
        )
        eigenbasis_draws = standard * np.sqrt(eigenvalues)[:, :, None, :]
        draws = np.einsum(
            "nbki,ji->nbkj", eigenbasis_draws, rotation, optimize=True
        )
        if condition == "paired_covariance":
            content = np.concatenate((draws, -draws), axis=2)
            oracle_basis_draws = eigenbasis_draws
        else:
            content = draws
            oracle_basis_draws = eigenbasis_draws
        inverse_difference = (
            1.0 / (1.0 + CONFIG.covariance_delta * spectrum_sign)
            - 1.0 / (1.0 - CONFIG.covariance_delta * spectrum_sign)
        )
        individual_oracle_by_bag = -0.5 * np.einsum(
            "nbki,i,nbki->nb",
            oracle_basis_draws,
            inverse_difference,
            oracle_basis_draws,
            optimize=True,
        )
    else:
        deviations = rng.normal(
            size=(
                scenes,
                CONFIG.bags,
                CONFIG.independent_draws,
                CONFIG.content_width,
            )
        )
        direction = rotation[:, 0]
        mean = 0.35 * signs[:, :, None, None] * direction[None, None, None, :]
        content = np.concatenate((mean + deviations, mean - deviations), axis=2)
        # Pairing makes the class mean noiseless; finite logits avoid infinities.
        individual_oracle_by_bag = signs * 30.0

    source_content, source_bag = _shuffle_sources(rng, content)
    query_bag = np.argsort(rng.random((scenes, CONFIG.bags)), axis=1).astype(
        np.int64
    )
    labels = np.take_along_axis(labels_by_bag, query_bag, axis=1)
    constrained_oracle_by_bag = constrained_marginal_logits(
        individual_oracle_by_bag
    )
    oracle_logits = np.take_along_axis(
        constrained_oracle_by_bag, query_bag, axis=1
    )
    return {
        "source_content": source_content,
        "source_bag": source_bag,
        "addresses": addresses,
        "query_bag": query_bag,
        "labels": labels.astype(np.float32),
        "oracle_logits": oracle_logits.astype(np.float32),
    }


def materialize_inputs(data: dict[str, Array], indices: Sequence[int]) -> Array:
    chosen = np.asarray(indices, dtype=np.int64)
    content = data["source_content"][chosen]
    source_bag = data["source_bag"][chosen]
    query_bag = data["query_bag"][chosen]
    addresses = data["addresses"][chosen]
    batch = chosen.size
    inputs = np.zeros(
        (batch, CONFIG.sequence_length, CONFIG.model_width), dtype=np.float32
    )
    inputs[:, : CONFIG.source_tokens, : CONFIG.content_width] = content
    scene_index = np.arange(batch)[:, None]
    inputs[
        :, : CONFIG.source_tokens, CONFIG.content_width : 2 * CONFIG.content_width
    ] = CONFIG.address_scale * addresses[scene_index, source_bag]
    inputs[
        :, CONFIG.source_tokens :, CONFIG.content_width : 2 * CONFIG.content_width
    ] = CONFIG.address_scale * addresses[scene_index, query_bag]
    inputs[:, : CONFIG.source_tokens, 2 * CONFIG.content_width] = 1.0
    inputs[:, CONFIG.source_tokens :, 2 * CONFIG.content_width + 1] = 1.0
    return inputs


def binary_nll(logits: Array, labels: Array) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def binary_accuracy(logits: Array, labels: Array) -> float:
    return float(np.mean((logits >= 0.0) == (labels >= 0.5)))


def data_invariants(data: dict[str, Array]) -> dict[str, float]:
    scenes = data["labels"].shape[0]
    source_means = np.zeros(
        (scenes, CONFIG.bags, CONFIG.content_width), dtype=np.float64
    )
    for bag in range(CONFIG.bags):
        mask = data["source_bag"] == bag
        source_means[:, bag] = np.einsum(
            "ns,nsi->ni", mask, data["source_content"], optimize=True
        ) / CONFIG.source_tokens_per_bag
    address_gram = np.einsum(
        "nbi,nci->nbc", data["addresses"], data["addresses"], optimize=True
    )
    identity = np.eye(CONFIG.bags)[None, :, :]
    labels_by_scene = data["labels"].sum(axis=1)
    return {
        "paired_content_mean_max_abs": float(np.max(np.abs(source_means))),
        "address_orthogonality_max_abs_error": float(
            np.max(np.abs(address_gram - identity))
        ),
        "balanced_labels_max_abs_error": float(
            np.max(np.abs(labels_by_scene - 2.0))
        ),
        "oracle_nll": binary_nll(data["oracle_logits"], data["labels"]),
        "oracle_accuracy": binary_accuracy(data["oracle_logits"], data["labels"]),
        "uniform_all_bags_nll": math.log(2.0),
        "uniform_all_bags_accuracy": 0.5,
    }


def variant_ledger(variant: str) -> dict[str, int]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    d = CONFIG.model_width
    r = CONFIG.head_width
    hq = CONFIG.query_heads
    hkv = CONFIG.kv_heads
    q_params = d * hq * r
    k_params = d * hkv * r
    output_width = hq * r
    ffn_width = CONFIG.base_ffn_width
    explicit_products = 0
    activation_elements = 0

    if variant == "dense":
        value_params = d * hkv * r
        output_params = d * output_width
        attention_params = q_params + k_params + value_params + output_params
        attention_matrix_mults = attention_params
        value_cache = hkv * r
    elif variant == "bda":
        value_params = hkv * (d - r) * r
        output_params = d * output_width
        attention_params = q_params + k_params + value_params + output_params
        attention_matrix_mults = attention_params
        value_cache = hkv * r
    elif variant in {"cyclic", "square"}:
        base = hkv * (d - r) * r
        live_gate = hkv * r * (r - 1)
        scale = hkv * r if variant == "cyclic" else 0
        value_params = base + live_gate + scale
        output_params = d * output_width
        attention_params = q_params + k_params + value_params + output_params
        attention_matrix_mults = q_params + k_params + base + live_gate + output_params
        explicit_products = hkv * r
        value_cache = hkv * r
    elif variant in {"source_mlp", "g2_matched"}:
        base = hkv * (d - r) * r
        rank = 5
        low_rank = d * rank + rank * hkv * r
        final_vector = hkv * r
        value_params = base + low_rank + final_vector
        output_params = d * output_width
        attention_params = q_params + k_params + value_params + output_params
        attention_matrix_mults = q_params + k_params + base + low_rank + output_params
        explicit_products = final_vector
        activation_elements = rank if variant == "source_mlp" else hkv * r
        value_cache = hkv * r
    elif variant == "g2_full":
        value_params = d * hkv * r
        gate_params = value_params
        output_params = d * output_width
        attention_params = q_params + k_params + value_params + gate_params + output_params
        attention_matrix_mults = attention_params
        explicit_products = hkv * r
        activation_elements = hkv * r
        value_cache = hkv * r
        ffn_width = 240
    elif variant == "glu":
        value_width = 12
        value_params = 2 * d * hkv * value_width
        output_params = d * hq * value_width
        attention_params = q_params + k_params + value_params + output_params
        attention_matrix_mults = attention_params
        explicit_products = hkv * value_width
        activation_elements = hkv * value_width
        value_cache = hkv * value_width
    else:  # g1
        value_params = d * hkv * r
        output_params = d * output_width
        gate_params = d * output_width
        attention_params = q_params + k_params + value_params + output_params + gate_params
        attention_matrix_mults = attention_params
        explicit_products = output_width
        activation_elements = output_width
        value_cache = hkv * r
        ffn_width = 224

    ffn_matrix_params = 3 * d * ffn_width
    two_layer_matrix_params = CONFIG.layers * (
        attention_params + ffn_matrix_params
    )
    common_nonmatrix_params = CONFIG.layers * 2 * d + d + d
    return {
        "attention_parameters_per_layer": attention_params,
        "ffn_width": ffn_width,
        "ffn_matrix_parameters_per_layer": ffn_matrix_params,
        "attention_plus_ffn_parameters_two_layers": two_layer_matrix_params,
        "actual_trainable_parameters_expected": (
            two_layer_matrix_params + common_nonmatrix_params
        ),
        "attention_matrix_multiplications_per_token_layer": attention_matrix_mults,
        "attention_explicit_products_per_token_layer": explicit_products,
        "attention_activation_elements_per_token_layer": activation_elements,
        "key_cache_scalars_per_token_layer": hkv * r,
        "value_cache_scalars_per_token_layer": value_cache,
        "kv_cache_scalars_per_token_layer": hkv * r + value_cache,
    }


def all_ledgers() -> dict[str, dict[str, int]]:
    return {variant: variant_ledger(variant) for variant in VARIANTS}


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def screen_decision(report: dict[str, Any]) -> dict[str, Any]:
    runs = report.get("runs", [])
    complete = {
        (int(run["world_seed"]), str(run["variant"])): run
        for run in runs
        if int(run["init_seed"]) == SCREEN_SEEDS[0]
    }
    missing = [
        (world, variant)
        for world in WORLD_SEEDS
        for variant in VARIANTS
        if (world, variant) not in complete
    ]
    if missing:
        return {"status": "incomplete", "missing": missing}

    best_accuracy = max(float(run["test_accuracy"]) for run in complete.values())
    oracle_accuracies = [
        float(report["worlds"][str(world)]["test_invariants"]["oracle_accuracy"])
        for world in WORLD_SEEDS
    ]
    leakage = report.get("leakage_controls", {})
    validity_failures: list[str] = []
    if min(oracle_accuracies) < 0.85:
        validity_failures.append("bayes_accuracy_below_85_percent")
    if best_accuracy < 0.75:
        validity_failures.append("best_model_accuracy_below_75_percent")
    for name in ("address_only_accuracy", "uniform_all_bags_accuracy"):
        if float(leakage.get(name, 1.0)) > 0.52:
            validity_failures.append(f"{name}_above_52_percent")
    if float(leakage.get("broken_address_accuracy", 1.0)) > 0.52:
        validity_failures.append("broken_address_accuracy_above_52_percent")
    if max(
        float(
            report["worlds"][str(world)]["test_invariants"][
                "paired_content_mean_max_abs"
            ]
        )
        for world in WORLD_SEEDS
    ) > 1e-6:
        validity_failures.append("paired_content_mean_not_zero")
    if any(not bool(run.get("parameter_ledger_matches", False)) for run in complete.values()):
        validity_failures.append("parameter_ledger_mismatch")
    controls = tuple(variant for variant in VARIANTS if variant != "cyclic")
    control_medians = {
        variant: _median(
            [float(complete[(world, variant)]["test_nll"]) for world in WORLD_SEEDS]
        )
        for variant in controls
    }
    comparator = min(control_medians, key=control_medians.get)
    route_advantages = []
    for world in WORLD_SEEDS:
        for variant in ("cyclic", comparator):
            run = complete[(world, variant)]
            if float(run["test_target_bag_mass"]) < 0.90:
                validity_failures.append(
                    f"{variant}_target_bag_mass_below_90_percent_world_{world}"
                )
            if float(run["test_bag_argmax_accuracy"]) < 0.99:
                validity_failures.append(
                    f"{variant}_bag_argmax_below_99_percent_world_{world}"
                )
        route_advantages.append(
            float(complete[(world, "cyclic")]["test_target_bag_mass"])
            - float(complete[(world, comparator)]["test_target_bag_mass"])
        )
    if validity_failures:
        return {
            "status": "invalid",
            "comparator": comparator,
            "failures": validity_failures,
        }

    wins = 0
    improvements = []
    paired = []
    for world in WORLD_SEEDS:
        cyclic_nll = float(complete[(world, "cyclic")]["test_nll"])
        comparator_nll = float(complete[(world, comparator)]["test_nll"])
        bayes_nll = float(
            report["worlds"][str(world)]["test_invariants"]["oracle_nll"]
        )
        if cyclic_nll < comparator_nll:
            wins += 1
        denominator = max(comparator_nll - bayes_nll, 1e-9)
        relative = (comparator_nll - cyclic_nll) / denominator
        improvements.append(relative)
        paired.append(
            {
                "world_seed": world,
                "cyclic_nll": cyclic_nll,
                "comparator_nll": comparator_nll,
                "bayes_nll": bayes_nll,
                "relative_excess_nll_improvement": relative,
            }
        )
    passed = wins >= 4 and _median(improvements) > 0.0
    frozen_router_required = max(route_advantages) > 0.02
    return {
        "status": (
            "pass_requires_frozen_router"
            if passed and frozen_router_required
            else "pass_expand"
            if passed
            else "reject"
        ),
        "comparator": comparator,
        "control_median_nll": control_medians,
        "cyclic_world_wins": wins,
        "median_relative_excess_nll_improvement": _median(improvements),
        "cyclic_minus_comparator_target_bag_mass_by_world": route_advantages,
        "frozen_router_required": frozen_router_required,
        "physical_cost_claim_permitted": False,
        "physical_cost_audit_required": True,
        "scope": "mechanism_quality_screen_only",
        "paired_worlds": paired,
        "next_action": (
            "run_frozen_router_diagnostic_before_confirmation"
            if passed and frozen_router_required
            else "run_confirmation_seeds_and_controls"
            if passed
            else "destroy_gpu_and_retain_only_the_negative_result"
        ),
    }


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("gauge_curvature_learned_routing_torch.py"),
        PREREGISTRATION_PATH,
    )
    return {
        str(path.relative_to(ROOT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(item for item in value.split(",") if item)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {unknown}")
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("self-test", "smoke", "screen", "confirm"), default="screen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_SCREEN_PATH)
    parser.add_argument("--variants", type=_parse_variants, default=VARIANTS)
    parser.add_argument("--world-seeds", type=_parse_ints, default=WORLD_SEEDS)
    parser.add_argument("--init-seeds", type=_parse_ints, default=SCREEN_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from experiments.gauge_curvature_learned_routing_torch import run_cli
    except ModuleNotFoundError as error:
        if error.name != "experiments":
            raise
        from gauge_curvature_learned_routing_torch import run_cli

    report = run_cli(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
