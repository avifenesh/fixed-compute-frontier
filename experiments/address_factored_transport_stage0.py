#!/usr/bin/env python3
"""Frozen Stage-0 gate for address-factored transport attention (AFTA)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


SEEDS = (11, 23, 37, 53, 71)
NOISES = (0.75, 1.0, 1.5)


@dataclass(frozen=True)
class Config:
    records: int = 128
    relations: int = 4
    group_dim: int = 16
    model_dim: int = 256
    trials: int = 8192
    batch_size: int = 256
    relation_offsets: tuple[int, ...] = (0, 1, 2, 3)

    @property
    def total_dim(self) -> int:
        return self.relations * self.group_dim


def rademacher(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    bits = torch.randint(0, 2, shape, generator=generator, device=device)
    return bits.to(torch.float32) * 2.0 - 1.0


def unit_gaussian(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    value = torch.randn(shape, generator=generator, device=device)
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def resource_ledger(config: Config, queries: int = 97, keys: int = 131) -> dict[str, Any]:
    r = config.relations
    g = config.group_dim
    d = config.total_dim
    m = config.model_dim
    afta_parameter_counts = {"q": m * d, "k": m * d, "v": m * d, "o": d * m}
    split_parameter_counts = {
        "q": r * m * g,
        "k": r * m * g,
        "v": r * m * g,
        "o": r * g * m,
    }
    afta = {
        "q_width": d,
        "k_width": d,
        "v_width": d,
        "qk_macs": queries * keys * d,
        "pv_macs": queries * keys * d,
        "projection_output_width": 3 * d,
        "projection_shapes": {"q": [m, d], "k": [m, d], "v": [m, d], "o": [d, m]},
        "logical_head_shapes": {"q": [[m, d]], "k": [[m, d]], "v": [[m, d]]},
        "projection_parameter_counts": afta_parameter_counts,
        "projection_parameters_total": sum(afta_parameter_counts.values()),
        "k_cache_scalars_per_token": d,
        "v_cache_scalars_per_token": d,
        "softmax_maps": 1,
        "materialized_shifted_v_scalars": 0,
        "offset_metadata_integers": r,
        "transport_bounds_predicates": queries * keys * r,
        "v_scalar_loads": queries * keys * d,
        "v_load_regions_total": r,
        "v_load_regions_per_map": r,
    }
    split = {
        "q_width": r * g,
        "k_width": r * g,
        "v_width": r * g,
        "qk_macs": r * queries * keys * g,
        "pv_macs": r * queries * keys * g,
        "projection_output_width": r * 3 * g,
        "projection_shapes": {"q": [m, r * g], "k": [m, r * g], "v": [m, r * g], "o": [r * g, m]},
        "logical_head_shapes": {
            "q_per_head": [[m, g] for _ in range(r)],
            "k_per_head": [[m, g] for _ in range(r)],
            "v_per_head": [[m, g] for _ in range(r)],
        },
        "projection_parameter_counts": split_parameter_counts,
        "projection_parameters_total": sum(split_parameter_counts.values()),
        "k_cache_scalars_per_token": r * g,
        "v_cache_scalars_per_token": r * g,
        "softmax_maps": r,
        "materialized_shifted_v_scalars": 0,
        "offset_metadata_integers": r,
        "transport_bounds_predicates": queries * keys * r,
        "v_scalar_loads": queries * keys * d,
        "v_load_regions_total": r,
        "v_load_regions_per_map": 1,
    }
    equality_fields = (
        "q_width",
        "k_width",
        "v_width",
        "qk_macs",
        "pv_macs",
        "projection_output_width",
        "projection_shapes",
        "projection_parameter_counts",
        "projection_parameters_total",
        "k_cache_scalars_per_token",
        "v_cache_scalars_per_token",
        "materialized_shifted_v_scalars",
        "offset_metadata_integers",
        "transport_bounds_predicates",
        "v_scalar_loads",
        "v_load_regions_total",
    )
    return {
        "afta": afta,
        "split": split,
        "equality_fields": equality_fields,
        "equal": all(afta[key] == split[key] for key in equality_fields),
        "afta_softmax_reduction_factor": split["softmax_maps"] / afta["softmax_maps"],
    }


def prefix_transport(
    attention: torch.Tensor, values: torch.Tensor, offsets: tuple[int, ...]
) -> torch.Tensor:
    """Reference causal partial shifts; invalid loads are zero, never renormalized."""
    length, relations, group_dim = values.shape
    if attention.shape != (length, length):
        raise ValueError("attention must be square and match values")
    if len(offsets) != relations:
        raise ValueError("one offset is required per value group")
    rows = []
    for t in range(length):
        groups = []
        for r, delta in enumerate(offsets):
            terms = []
            for j in range(t + 1):
                payload = j + delta
                if 0 <= payload <= t:
                    terms.append(attention[t, j] * values[payload, r])
            groups.append(torch.stack(terms).sum(dim=0) if terms else values[0, r] * 0.0)
        rows.append(torch.stack(groups, dim=0))
    return torch.stack(rows, dim=0)


def causal_operator_audits(config: Config, device: torch.device) -> dict[str, float | bool]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260727)
    length = 17
    logits = torch.randn((length, length), generator=generator, device=device)
    causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
    attention = torch.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)
    values = torch.randn(
        (length, config.relations, config.group_dim), generator=generator, device=device
    )
    output = prefix_transport(attention, values, config.relation_offsets)

    future_error = 0.0
    for t in range(length):
        perturbed = values.clone()
        if t + 1 < length:
            perturbation = 100.0 * torch.randn(
                perturbed[t + 1 :].shape, generator=generator, device=device
            )
            perturbed[t + 1 :] += perturbation
        changed = prefix_transport(attention, perturbed, config.relation_offsets)
        future_error = max(future_error, float((changed[t] - output[t]).abs().max().item()))

    zero_offsets = tuple(0 for _ in range(config.relations))
    zero_output = prefix_transport(attention, values, zero_offsets)
    ordinary = torch.einsum("ij,jrg->irg", attention, values)
    zero_offset_error = float((zero_output - ordinary).abs().max().item())

    ones = torch.ones_like(values)
    transported_ones = prefix_transport(attention, ones, config.relation_offsets)[..., 0]
    expected_mass = torch.zeros_like(transported_ones)
    effective_reference_error = 0.0
    for t in range(length):
        for r, delta in enumerate(config.relation_offsets):
            effective = torch.zeros((length,), device=device)
            for j in range(t + 1):
                payload = j + delta
                if 0 <= payload <= t:
                    effective[payload] += attention[t, j]
            expected_mass[t, r] = effective.sum()
            dense = effective @ values[:, r]
            effective_reference_error = max(
                effective_reference_error,
                float((dense - output[t, r]).abs().max().item()),
            )
    no_renormalization_error = float((transported_ones - expected_mass).abs().max().item())

    transport_mask = torch.zeros(
        (length, config.relations, length, length), device=device, dtype=values.dtype
    )
    for t in range(length):
        for r, delta in enumerate(config.relation_offsets):
            for j in range(t + 1):
                payload = j + delta
                if 0 <= payload <= t:
                    transport_mask[t, r, j, payload] = 1.0
    audit_head_dim = 8
    base_q = torch.randn((length, audit_head_dim), generator=generator, device=device)
    base_k = torch.randn((length, audit_head_dim), generator=generator, device=device)
    grad_q = base_q.detach().clone().requires_grad_(True)
    grad_k = base_k.detach().clone().requires_grad_(True)
    grad_values = values.detach().clone().requires_grad_(True)
    grad_logits = grad_q @ grad_k.transpose(0, 1) / math.sqrt(audit_head_dim)
    grad_map = torch.softmax(grad_logits.masked_fill(~causal, float("-inf")), dim=-1)
    functional = prefix_transport(grad_map, grad_values, config.relation_offsets)
    probe = torch.randn(functional.shape, generator=generator, device=device)
    functional_loss = (functional * probe).sum()
    functional_grads = torch.autograd.grad(functional_loss, (grad_q, grad_k, grad_values))
    dense_q = base_q.detach().clone().requires_grad_(True)
    dense_k = base_k.detach().clone().requires_grad_(True)
    dense_values = values.detach().clone().requires_grad_(True)
    dense_logits = dense_q @ dense_k.transpose(0, 1) / math.sqrt(audit_head_dim)
    dense_attention = torch.softmax(
        dense_logits.masked_fill(~causal, float("-inf")), dim=-1
    )
    dense_output = torch.einsum(
        "tj,trjp,prg->trg", dense_attention, transport_mask, dense_values
    )
    dense_loss = (dense_output * probe).sum()
    dense_grads = torch.autograd.grad(dense_loss, (dense_q, dense_k, dense_values))
    gradient_reference_error = max(
        float((functional_grads[0] - dense_grads[0]).abs().max().item()),
        float((functional_grads[1] - dense_grads[1]).abs().max().item()),
        float((functional_grads[2] - dense_grads[2]).abs().max().item()),
    )
    return {
        "future_value_perturbation_max_error": future_error,
        "zero_offset_ordinary_max_error": zero_offset_error,
        "effective_map_reference_max_error": effective_reference_error,
        "no_renormalization_max_error": no_renormalization_error,
        "gradient_reference_max_error": gradient_reference_error,
        "some_boundary_mass_below_one": bool((expected_mass < 1.0 - 1e-6).any().item()),
    }


def decode_payload(output: torch.Tensor, payloads: torch.Tensor) -> torch.Tensor:
    similarity = torch.einsum("brg,bnrg->brn", output, payloads)
    return similarity.argmax(dim=-1)


@torch.inference_mode()
def run_condition(
    *,
    config: Config,
    seed: int,
    noise: float,
    independent_targets: bool,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + (10_000 if independent_targets else 0) + int(noise * 100))
    totals = {
        "afta_fields": 0,
        "split_fields": 0,
        "common_fields": 0,
        "ordinary_fields": 0,
        "afta_records": 0,
        "split_records": 0,
        "common_records": 0,
        "ordinary_records": 0,
    }
    max_algebra_error = 0.0
    processed = 0

    while processed < config.trials:
        batch = min(config.batch_size, config.trials - processed)
        keys = rademacher(
            (batch, config.records, config.relations, config.group_dim), generator, device
        )
        if independent_targets:
            targets = torch.stack(
                [
                    torch.randint(
                        max(0, -delta),
                        min(config.records, config.records - delta),
                        (batch,),
                        generator=generator,
                        device=device,
                    )
                    for delta in config.relation_offsets
                ],
                dim=1,
            )
        else:
            common_target = torch.randint(
                max(0, -min(config.relation_offsets)),
                min(config.records, config.records - max(config.relation_offsets)),
                (batch, 1),
                generator=generator,
                device=device,
            )
            targets = common_target.expand(-1, config.relations)

        batch_indices = torch.arange(batch, device=device)[:, None]
        relation_indices = torch.arange(config.relations, device=device)[None, :]
        query = keys[batch_indices, targets, relation_indices]
        query = query + noise * torch.randn(
            query.shape, generator=generator, device=device, dtype=query.dtype
        )

        partial_raw = torch.einsum("brg,bnrg->brn", query, keys)
        partial_logits = partial_raw / math.sqrt(config.group_dim)
        wide_logits = partial_raw.sum(dim=1) / math.sqrt(config.total_dim)
        afta_attention = torch.softmax(wide_logits, dim=-1)
        split_attention = torch.softmax(partial_logits, dim=-1)

        if processed == 0:
            query64 = query[:4].to(torch.float64).reshape(4, config.total_dim)
            keys64 = keys[:4].to(torch.float64).reshape(4, config.records, config.total_dim)
            wide64 = torch.einsum("bd,bnd->bn", query64, keys64)
            partial64 = torch.einsum(
                "brg,bnrg->brn", query[:4].to(torch.float64), keys[:4].to(torch.float64)
            ).sum(dim=1)
            max_algebra_error = float((wide64 - partial64).abs().max().item())

        payloads = unit_gaussian(
            (batch, config.records, config.relations, config.group_dim), generator, device
        )
        values = unit_gaussian(
            (batch, config.records, config.relations, config.group_dim), generator, device
        )
        for r, delta in enumerate(config.relation_offsets):
            source_start = max(0, -delta)
            source_end = min(config.records, config.records - delta)
            values[:, source_start + delta : source_end + delta, r] = payloads[
                :, source_start:source_end, r
            ]

        afta_groups = []
        split_groups = []
        common_groups = []
        ordinary_delta = 0
        common_delta = config.relation_offsets[1]
        ordinary_groups = []
        for r, delta in enumerate(config.relation_offsets):
            source_start = max(0, -delta)
            source_end = min(config.records, config.records - delta)
            afta_groups.append(
                torch.einsum(
                    "bn,bng->bg",
                    afta_attention[:, source_start:source_end],
                    values[:, source_start + delta : source_end + delta, r],
                )
            )
            split_groups.append(
                torch.einsum(
                    "bn,bng->bg",
                    split_attention[:, r, source_start:source_end],
                    values[:, source_start + delta : source_end + delta, r],
                )
            )
            common_start = max(0, -common_delta)
            common_end = min(config.records, config.records - common_delta)
            common_groups.append(
                torch.einsum(
                    "bn,bng->bg",
                    afta_attention[:, common_start:common_end],
                    values[:, common_start + common_delta : common_end + common_delta, r],
                )
            )
            ordinary_start = max(0, -ordinary_delta)
            ordinary_end = min(config.records, config.records - ordinary_delta)
            ordinary_groups.append(
                torch.einsum(
                    "bn,bng->bg",
                    afta_attention[:, ordinary_start:ordinary_end],
                    values[
                        :,
                        ordinary_start + ordinary_delta : ordinary_end + ordinary_delta,
                        r,
                    ],
                )
            )
        afta_output = torch.stack(afta_groups, dim=1)
        split_output = torch.stack(split_groups, dim=1)
        common_output = torch.stack(common_groups, dim=1)
        ordinary_output = torch.stack(ordinary_groups, dim=1)
        predictions = {
            "afta": decode_payload(afta_output, payloads),
            "split": decode_payload(split_output, payloads),
            "common": decode_payload(common_output, payloads),
            "ordinary": decode_payload(ordinary_output, payloads),
        }
        for name, predicted in predictions.items():
            correct = predicted.eq(targets)
            totals[f"{name}_fields"] += int(correct.sum().item())
            totals[f"{name}_records"] += int(correct.all(dim=1).sum().item())
        processed += batch

    field_denominator = config.trials * config.relations
    return {
        "afta_field_accuracy": totals["afta_fields"] / field_denominator,
        "split_field_accuracy": totals["split_fields"] / field_denominator,
        "common_delay_field_accuracy": totals["common_fields"] / field_denominator,
        "wide_ordinary_field_accuracy": totals["ordinary_fields"] / field_denominator,
        "afta_exact_record_accuracy": totals["afta_records"] / config.trials,
        "split_exact_record_accuracy": totals["split_records"] / config.trials,
        "common_delay_exact_record_accuracy": totals["common_records"] / config.trials,
        "wide_ordinary_exact_record_accuracy": totals["ordinary_records"] / config.trials,
        "max_algebra_error": max_algebra_error,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, bool):
        return True
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def evaluate_gates(
    runs: list[dict[str, Any]], ledger: dict[str, Any], audits: dict[str, Any]
) -> dict[str, bool]:
    shared_10 = [r for r in runs if r["condition"] == "shared" and r["noise"] == 1.0]
    shared_15 = [r for r in runs if r["condition"] == "shared" and r["noise"] == 1.5]
    independent_10 = [
        r for r in runs if r["condition"] == "independent" and r["noise"] == 1.0
    ]
    return {
        "algebra": max(run["max_algebra_error"] for run in runs) <= 2e-10,
        "ledger": bool(ledger["equal"]),
        "shared_reliability": all(r["afta_exact_record_accuracy"] >= 0.90 for r in shared_10),
        "matched_advantage": all(
            r["afta_exact_record_accuracy"] - r["split_exact_record_accuracy"] >= 0.25
            for r in shared_10
        ),
        "noise_robustness": mean([r["afta_exact_record_accuracy"] for r in shared_15])
        - mean([r["split_exact_record_accuracy"] for r in shared_15])
        >= 0.20,
        "predicted_limitation": all(
            r["split_field_accuracy"] - r["afta_field_accuracy"] >= 0.20
            for r in independent_10
        ),
        "causal_operator": (
            audits["future_value_perturbation_max_error"] <= 2e-6
            and audits["zero_offset_ordinary_max_error"] <= 2e-6
            and audits["effective_map_reference_max_error"] <= 2e-6
            and audits["no_renormalization_max_error"] <= 2e-6
            and audits["gradient_reference_max_error"] <= 2e-6
            and audits["some_boundary_mass_below_one"]
        ),
        "common_delay_distinction": all(
            r["afta_exact_record_accuracy"] - r["common_delay_exact_record_accuracy"] >= 0.50
            for r in shared_10
        ),
        "wide_ordinary_distinction": all(
            r["afta_exact_record_accuracy"] - r["wide_ordinary_exact_record_accuracy"] >= 0.50
            for r in shared_10
        ),
        "finite": finite_tree(runs) and finite_tree(ledger) and finite_tree(audits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/address-factored-transport-stage0.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    config = Config()
    if len(config.relation_offsets) != config.relations:
        raise ValueError("relation_offsets must match relations")
    ledger = resource_ledger(config)
    audits = causal_operator_audits(config, device)
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for noise in NOISES:
            for condition, independent in (("shared", False), ("independent", True)):
                metrics = run_condition(
                    config=config,
                    seed=seed,
                    noise=noise,
                    independent_targets=independent,
                    device=device,
                )
                row = {"seed": seed, "noise": noise, "condition": condition, **metrics}
                runs.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    gates = evaluate_gates(runs, ledger, audits)
    result = {
        "schema": "address-factored-transport-stage0-v2",
        "config": asdict(config) | {"total_dim": config.total_dim},
        "seeds": SEEDS,
        "noises": NOISES,
        "device": str(device),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "ledger": ledger,
        "operator_audits": audits,
        "runs": runs,
        "gates": gates,
        "pass": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "pass": result["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
