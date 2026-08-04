#!/usr/bin/env python3
"""Frozen-model upper bound for algebraic replacements of full attention.

The six full-attention layers in Qwen3.5-0.8B are intercepted only during a
teacher-forced continuation.  Exact softmax attention is compared with:

* renormalized top-k attention (tropical/sparse selection);
* top-k exact mass plus a uniform tail value (selection + additive summary);
* zeroth- and first-order recurrent moment reads;
* recent-window, bottom-k, and zero-output sensitivity controls.

Every intervention still materializes the exact score matrix.  This is an
oracle anatomy test, not an efficient implementation or a serving claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_modeling

from qwen35_hybrid_state_anatomy import (
    build_cases,
    build_heldout_cases,
    cache_ledger,
    distribution_metrics,
)


ORIGINAL_EAGER_ATTENTION = qwen_modeling.eager_attention_forward


@dataclass(frozen=True)
class Intervention:
    family: str
    k: int | None = None
    target_layer: int | None = None
    target_kv_group: int | None = None


def parse_intervention(name: str) -> Intervention:
    base, separator, target = name.partition("@")
    target_layer: int | None = None
    target_kv_group: int | None = None
    if separator:
        if not target.startswith("L") or "G" not in target:
            raise ValueError(f"invalid intervention target: {name}")
        layer_text, group_text = target[1:].split("G", 1)
        target_layer = int(layer_text)
        target_kv_group = int(group_text)

    for prefix in ("topk", "tailmean", "recent", "bottom"):
        if base.startswith(prefix):
            return Intervention(
                prefix,
                int(base[len(prefix) :]),
                target_layer,
                target_kv_group,
            )
    if base in {"exact", "uniform", "firstorder", "zero"}:
        return Intervention(base, None, target_layer, target_kv_group)
    raise ValueError(f"unknown intervention: {name}")


def intervention_name(intervention: Intervention) -> str:
    base = (
        intervention.family
        if intervention.k is None
        else f"{intervention.family}{intervention.k}"
    )
    if intervention.target_layer is None:
        return base
    return f"{base}@L{intervention.target_layer}G{intervention.target_kv_group}"


def tensor_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "p90": float(torch.quantile(tensor, 0.9).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


class AttentionController:
    def __init__(self, diagnostic_ks: list[int]) -> None:
        self.intervention = Intervention("exact")
        self.capture = False
        self.diagnostic_ks = diagnostic_ks
        self._stats: dict[int, dict[int, dict[str, list[float]]]] = {}

    def reset_stats(self) -> None:
        self._stats = {}

    def add_stat(self, layer: int, head: int, name: str, value: torch.Tensor) -> None:
        by_head = self._stats.setdefault(layer, {}).setdefault(head, {})
        by_head.setdefault(name, []).extend(
            float(item) for item in value.detach().float().flatten().cpu().tolist()
        )

    def summarized_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for layer, heads in sorted(self._stats.items()):
            layer_result: dict[str, Any] = {}
            for head, metrics in sorted(heads.items()):
                head_result = {name: tensor_summary(values) for name, values in metrics.items()}
                for k in self.diagnostic_ks:
                    mass = metrics[f"top{k}_mass"]
                    top_error = metrics[f"top{k}_relative_output_error"]
                    tail_error = metrics[f"tailmean{k}_relative_output_error"]
                    head_result[f"top{k}_mass"]["fraction_ge_0_99"] = sum(
                        value >= 0.99 for value in mass
                    ) / len(mass)
                    head_result[f"top{k}_relative_output_error"]["fraction_le_0_01"] = sum(
                        value <= 0.01 for value in top_error
                    ) / len(top_error)
                    head_result[f"tailmean{k}_relative_output_error"]["fraction_le_0_01"] = sum(
                        value <= 0.01 for value in tail_error
                    ) / len(tail_error)
                layer_result[str(head)] = head_result
            result[str(layer)] = layer_result
        return result


CONTROLLER = AttentionController([1, 4, 16, 64, 256])


def expand_valid_mask(
    attention_mask: torch.Tensor | None,
    scores: torch.Tensor,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(
            (scores.shape[0], 1, scores.shape[2], scores.shape[3]),
            dtype=torch.bool,
            device=scores.device,
        )
    mask = attention_mask[..., : scores.shape[-1]]
    return mask > -1.0e4


def gather_values(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    query_length = indices.shape[-2]
    expanded_values = values.unsqueeze(2).expand(
        -1, -1, query_length, -1, -1
    )
    return torch.gather(
        expanded_values,
        3,
        indices.unsqueeze(-1).expand(*indices.shape, values.shape[-1]),
    )


def selected_output(
    masked_scores: torch.Tensor,
    values: torch.Tensor,
    k: int,
    largest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    retained = min(k, masked_scores.shape[-1])
    scores, indices = torch.topk(masked_scores, retained, dim=-1, largest=largest)
    selected_values = gather_values(values, indices)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(values.dtype)
    output = (weights.unsqueeze(-1) * selected_values).sum(dim=-2)
    return output, indices, selected_values


def tailmean_output(
    probabilities: torch.Tensor,
    masked_scores: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
    k: int,
    exact_output: torch.Tensor,
) -> torch.Tensor:
    retained = min(k, masked_scores.shape[-1])
    _, indices = torch.topk(masked_scores, retained, dim=-1)
    selected_values = gather_values(values, indices)
    selected_probabilities = torch.gather(probabilities, -1, indices)

    selected_contribution = (
        selected_probabilities.unsqueeze(-1) * selected_values
    ).sum(dim=-2)
    selected_value_sum = selected_values.sum(dim=-2)
    valid_values = torch.matmul(valid.to(values.dtype), values)
    valid_count = valid.sum(dim=-1, keepdim=True).to(values.dtype)
    tail_count = valid_count - retained
    tail_mean = (valid_values - selected_value_sum) / tail_count.clamp_min(1)
    tail_mass = 1.0 - selected_probabilities.sum(dim=-1, keepdim=True)
    approximation = selected_contribution + tail_mass * tail_mean
    return torch.where(tail_count > 0, approximation, exact_output)


def recent_output(
    masked_scores: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
    k: int,
) -> torch.Tensor:
    retained = min(k, masked_scores.shape[-1])
    counts = valid.sum(dim=-1).expand(-1, masked_scores.shape[1], -1)
    offsets = torch.arange(retained, device=masked_scores.device)
    indices = counts.unsqueeze(-1) - retained + offsets
    indices = indices.clamp_min(0)
    scores = torch.gather(masked_scores, -1, indices)
    selected_values = gather_values(values, indices)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(values.dtype)
    return (weights.unsqueeze(-1) * selected_values).sum(dim=-2)


def moment_output(
    raw_scores: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
    order: int,
) -> torch.Tensor:
    valid_float = valid.to(values.dtype)
    count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1)
    mean_value = torch.matmul(valid_float, values) / count
    if order == 0:
        return mean_value
    masked_scores = torch.where(valid, raw_scores, torch.zeros_like(raw_scores))
    mean_score = masked_scores.sum(dim=-1, keepdim=True) / count
    centered_scores = torch.where(
        valid,
        raw_scores - mean_score,
        torch.zeros_like(raw_scores),
    )
    correction = torch.matmul(centered_scores.to(values.dtype), values) / count
    return mean_value + correction


def capture_attention_stats(
    module: Any,
    raw_scores: torch.Tensor,
    masked_scores: torch.Tensor,
    probabilities: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
    exact_output: torch.Tensor,
) -> None:
    probability_float = probabilities.float()
    entropy = -(
        probability_float
        * probability_float.clamp_min(1.0e-30).log()
    ).sum(dim=-1)
    effective_support = entropy.exp()
    top_two = torch.topk(masked_scores, 2, dim=-1).values.float()
    gap = top_two[..., 0] - top_two[..., 1]

    for head in range(probabilities.shape[1]):
        CONTROLLER.add_stat(module.layer_idx, head, "entropy", entropy[:, head])
        CONTROLLER.add_stat(
            module.layer_idx, head, "effective_support", effective_support[:, head]
        )
        CONTROLLER.add_stat(module.layer_idx, head, "top_score_gap", gap[:, head])

    exact_norm = exact_output.float().norm(dim=-1).clamp_min(1.0e-6)
    for k in CONTROLLER.diagnostic_ks:
        retained = min(k, probabilities.shape[-1])
        top_probabilities, _ = torch.topk(probability_float, retained, dim=-1)
        mass = top_probabilities.sum(dim=-1)
        top_output, _, _ = selected_output(masked_scores, values, retained)
        tail_output = tailmean_output(
            probabilities,
            masked_scores,
            values,
            valid,
            retained,
            exact_output,
        )
        top_error = (top_output.float() - exact_output.float()).norm(dim=-1) / exact_norm
        tail_error = (tail_output.float() - exact_output.float()).norm(dim=-1) / exact_norm
        for head in range(probabilities.shape[1]):
            CONTROLLER.add_stat(
                module.layer_idx, head, f"top{k}_mass", mass[:, head]
            )
            CONTROLLER.add_stat(
                module.layer_idx,
                head,
                f"top{k}_relative_output_error",
                top_error[:, head],
            )
            CONTROLLER.add_stat(
                module.layer_idx,
                head,
                f"tailmean{k}_relative_output_error",
                tail_error[:, head],
            )


def algebra_eager_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = qwen_modeling.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen_modeling.repeat_kv(value, module.num_key_value_groups)
    raw_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    masked_scores = raw_scores
    if attention_mask is not None:
        masked_scores = masked_scores + attention_mask[..., : key_states.shape[-2]]
    probabilities = F.softmax(masked_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    exact_output = torch.matmul(probabilities, value_states)
    valid = expand_valid_mask(attention_mask, raw_scores)

    if CONTROLLER.capture:
        capture_attention_stats(
            module,
            raw_scores,
            masked_scores,
            probabilities,
            value_states,
            valid,
            exact_output,
        )

    intervention = CONTROLLER.intervention
    if intervention.family == "exact":
        output = exact_output
    elif intervention.family == "topk":
        output, _, _ = selected_output(
            masked_scores, value_states, intervention.k or 1
        )
    elif intervention.family == "tailmean":
        output = tailmean_output(
            probabilities,
            masked_scores,
            value_states,
            valid,
            intervention.k or 1,
            exact_output,
        )
    elif intervention.family == "recent":
        output = recent_output(
            masked_scores, value_states, valid, intervention.k or 1
        )
    elif intervention.family == "bottom":
        bottom_scores = raw_scores.masked_fill(~valid, torch.inf)
        output, _, _ = selected_output(
            bottom_scores, value_states, intervention.k or 1, largest=False
        )
    elif intervention.family == "uniform":
        output = moment_output(raw_scores, value_states, valid, order=0)
    elif intervention.family == "firstorder":
        output = moment_output(raw_scores, value_states, valid, order=1)
    elif intervention.family == "zero":
        output = torch.zeros_like(exact_output)
    else:
        raise RuntimeError(f"unhandled intervention: {intervention}")

    if intervention.target_layer is not None:
        if module.layer_idx != intervention.target_layer:
            output = exact_output
        else:
            if intervention.target_kv_group is None:
                raise RuntimeError("target layer requires a KV group")
            heads_per_kv_group = query.shape[1] // key.shape[1]
            head_group = (
                torch.arange(query.shape[1], device=query.device)
                // heads_per_kv_group
            )
            target_heads = head_group == intervention.target_kv_group
            if not bool(target_heads.any()):
                raise RuntimeError(
                    f"KV group {intervention.target_kv_group} is out of range"
                )
            output = torch.where(
                target_heads.view(1, -1, 1, 1),
                output,
                exact_output,
            )

    return output.transpose(1, 2).contiguous(), probabilities


def run_continuation(
    model: Any,
    continuation: torch.Tensor,
    cache: Any,
    intervention: Intervention,
    capture: bool = False,
) -> tuple[torch.Tensor, float]:
    CONTROLLER.intervention = intervention
    CONTROLLER.capture = capture
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(
            input_ids=continuation,
            past_key_values=cache,
            use_cache=True,
        ).logits.detach()
    torch.cuda.synchronize(continuation.device)
    elapsed = time.perf_counter() - started
    CONTROLLER.capture = False
    return logits, elapsed


def run_case(
    model: Any,
    ids: torch.Tensor,
    prompt_length: int,
    continuation_length: int,
    interventions: list[Intervention],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prompt = ids[:prompt_length].unsqueeze(0).to(device)
    continuation = ids[
        prompt_length : prompt_length + continuation_length
    ].unsqueeze(0).to(device)
    targets = ids[
        prompt_length + 1 : prompt_length + continuation_length + 1
    ].unsqueeze(0).to(device)

    CONTROLLER.intervention = Intervention("exact")
    CONTROLLER.capture = False
    qwen_modeling.eager_attention_forward = ORIGINAL_EAGER_ATTENTION
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prefill = model(input_ids=prompt, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - started
    pristine_cache = prefill.past_key_values
    ledger = cache_ledger(pristine_cache)

    # The baseline uses the library implementation; every intervention uses the
    # local reimplementation.  The exact intervention is therefore a numerical
    # identity control for the hook itself.
    qwen_modeling.eager_attention_forward = ORIGINAL_EAGER_ATTENTION
    baseline_logits, baseline_seconds = run_continuation(
        model,
        continuation,
        copy.deepcopy(pristine_cache),
        Intervention("exact"),
    )
    qwen_modeling.eager_attention_forward = algebra_eager_attention_forward

    CONTROLLER.reset_stats()
    variants: dict[str, Any] = {}
    for intervention in interventions:
        cache = copy.deepcopy(pristine_cache)
        name = intervention_name(intervention)
        candidate_logits, elapsed = run_continuation(
            model,
            continuation,
            cache,
            intervention,
            capture=intervention.family == "exact",
        )
        metrics = distribution_metrics(baseline_logits, candidate_logits, targets)
        metrics["continuation_seconds_oracle_kernel"] = elapsed
        if intervention.k is not None:
            metrics["selected_k"] = intervention.k
            metrics["ideal_selected_fraction_at_prompt_boundary"] = (
                intervention.k / prompt_length
            )
        variants[name] = metrics
        del cache, candidate_logits

    result = {
        "cache_ledger": {**asdict(ledger), "total_bytes": ledger.total_bytes},
        "prefill_seconds_reference_kernel": prefill_seconds,
        "baseline_continuation_seconds_reference_kernel": baseline_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "attention_stats": CONTROLLER.summarized_stats(),
        "variants": variants,
    }
    del baseline_logits, pristine_cache, prefill
    torch.cuda.empty_cache()
    return result


def panel_name(case_name: str) -> str:
    if case_name.startswith("heldout_"):
        return "natural_heldout"
    if case_name.startswith("shuffled_"):
        return "block_shuffled"
    if case_name in {"code", "bindings", "periodic"}:
        return "structured"
    return "other"


def aggregate_results(cases: dict[str, Any]) -> dict[str, Any]:
    names = next(iter(cases.values()))["variants"].keys()
    panels = sorted({panel_name(name) for name in cases}) + ["all"]
    aggregate: dict[str, Any] = {}
    for variant in names:
        aggregate[variant] = {}
        for panel in panels:
            selected_cases = [
                value
                for name, value in cases.items()
                if panel == "all" or panel_name(name) == panel
            ]
            if not selected_cases:
                continue
            metrics = [case["variants"][variant] for case in selected_cases]
            kl = [value for item in metrics for value in item["kl_by_position"]]
            delta_nll = [
                value for item in metrics for value in item["nll_delta_by_position"]
            ]
            agreement = [
                value
                for item in metrics
                for value in item["top1_agreement_by_position"]
            ]
            mean_delta = statistics.fmean(delta_nll)
            standard_error = (
                statistics.stdev(delta_nll) / math.sqrt(len(delta_nll))
                if len(delta_nll) > 1
                else 0.0
            )
            aggregate[variant][panel] = {
                "tokens": len(kl),
                "kl_nats_mean": statistics.fmean(kl),
                "kl_nats_p95": float(
                    torch.quantile(torch.tensor(kl, dtype=torch.float64), 0.95).item()
                ),
                "maximum_case_kl_p95": max(item["kl_nats_p95"] for item in metrics),
                "top1_agreement": statistics.fmean(agreement),
                "nll_delta_mean": mean_delta,
                "nll_delta_upper_95_one_sided": mean_delta + 1.645 * standard_error,
            }
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--continuation-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--suite", choices=("handcrafted", "heldout", "both"), default="heldout"
    )
    parser.add_argument("--heldout-parquet", type=Path)
    parser.add_argument("--heldout-windows", type=int, default=8)
    parser.add_argument("--shuffle-block", type=int, default=64)
    parser.add_argument(
        "--interventions",
        default=(
            "exact,topk1,topk4,topk16,topk64,topk256,"
            "tailmean1,tailmean4,tailmean16,tailmean64,"
            "uniform,firstorder,recent64,bottom64,zero"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    interventions = [
        parse_intervention(name.strip())
        for name in args.interventions.split(",")
        if name.strip()
    ]
    diagnostic_ks = sorted(
        {
            intervention.k
            for intervention in interventions
            if intervention.k is not None and intervention.family in {"topk", "tailmean"}
        }
    )
    CONTROLLER.diagnostic_ks = diagnostic_ks

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    qwen_modeling.eager_attention_forward = algebra_eager_attention_forward

    total_length = args.prompt_length + args.continuation_length + 1
    cases: dict[str, torch.Tensor] = {}
    if args.suite in {"handcrafted", "both"}:
        handcrafted = build_cases(tokenizer, total_length, args.seed)
        cases.update(
            {
                name: ids
                for name, ids in handcrafted.items()
                if name in {"code", "bindings", "periodic"}
            }
        )
    if args.suite in {"heldout", "both"}:
        if args.heldout_parquet is None:
            raise RuntimeError("--suite heldout/both requires --heldout-parquet")
        cases.update(
            build_heldout_cases(
                tokenizer,
                args.heldout_parquet,
                total_length,
                args.seed,
                args.heldout_windows,
                args.shuffle_block,
            )
        )

    payload: dict[str, Any] = {
        "model": args.model,
        "seed": args.seed,
        "prompt_length": args.prompt_length,
        "continuation_length": args.continuation_length,
        "suite": args.suite,
        "interventions": [
            intervention.family
            if intervention.k is None
            else f"{intervention.family}{intervention.k}"
            for intervention in interventions
        ],
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "warning": (
            "Oracle frozen-model anatomy only: every variant still computes the full exact "
            "score matrix, top-k selection is not indexed, tail mass is exact, and no byte "
            "or latency saving is achieved by this harness."
        ),
        "cases": {},
    }
    for name, ids in cases.items():
        print(f"running {name}", flush=True)
        payload["cases"][name] = run_case(
            model,
            ids,
            args.prompt_length,
            args.continuation_length,
            interventions,
        )
    payload["aggregate"] = aggregate_results(payload["cases"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": list(cases)}, indent=2))


if __name__ == "__main__":
    main()
