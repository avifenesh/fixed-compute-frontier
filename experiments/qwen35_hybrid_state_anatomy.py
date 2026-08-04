#!/usr/bin/env python3
"""Read-only anatomy probe for Qwen3.5 Gated DeltaNet recurrent state.

This does not train or modify the checkpoint.  It asks whether the dense
per-request DeltaNet matrices are causally compressible at a prompt boundary:

1. prefill several deterministic sequence families;
2. measure the singular spectrum of every recurrent state matrix;
3. replace each matrix by its best rank-r approximation;
4. teacher-force the same continuation and compare it with the exact cache.

The SVD is an oracle diagnostic, not a proposed serving implementation.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CacheLedger:
    recurrent_bytes: int = 0
    convolution_bytes: int = 0
    attention_kv_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.recurrent_bytes + self.convolution_bytes + self.attention_kv_bytes


def tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def first_tensor(mapping: dict[Any, Any]) -> torch.Tensor:
    tensors = [value for value in mapping.values() if torch.is_tensor(value)]
    if len(tensors) != 1:
        raise RuntimeError(f"expected one tensor in cache mapping, found {len(tensors)}")
    return tensors[0]


def recurrent_tensor(layer: Any) -> torch.Tensor | None:
    mapping = getattr(layer, "recurrent_states", None)
    if not isinstance(mapping, dict) or not mapping:
        return None
    return first_tensor(mapping)


def cache_ledger(cache: Any) -> CacheLedger:
    ledger = CacheLedger()
    for layer in cache.layers:
        recurrent = getattr(layer, "recurrent_states", None)
        if isinstance(recurrent, dict):
            ledger.recurrent_bytes += sum(
                tensor_bytes(value) for value in recurrent.values() if torch.is_tensor(value)
            )
        convolution = getattr(layer, "conv_states", None)
        if isinstance(convolution, dict):
            ledger.convolution_bytes += sum(
                tensor_bytes(value) for value in convolution.values() if torch.is_tensor(value)
            )
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if torch.is_tensor(keys):
            ledger.attention_kv_bytes += tensor_bytes(keys)
        if torch.is_tensor(values):
            ledger.attention_kv_bytes += tensor_bytes(values)
    return ledger


def repeat_tokens(tokenizer: Any, text: str, length: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise RuntimeError("tokenizer produced no tokens")
    repeats = (length + len(ids) - 1) // len(ids)
    return torch.tensor((ids * repeats)[:length], dtype=torch.long)


def build_cases(tokenizer: Any, length: int, seed: int) -> dict[str, torch.Tensor]:
    natural_parts = []
    for index in range(96):
        natural_parts.append(
            "A systems researcher compares exact attention with a bounded recurrent state. "
            f"Trial {index} stores marker {1009 + 37 * index}, checks the accounting, and "
            "records whether the next operation preserves the earlier relation. "
        )
    natural = repeat_tokens(tokenizer, "".join(natural_parts), length)

    code_parts = []
    for index in range(96):
        modulus = 17 + index % 13
        code_parts.append(
            f"def transition_{index}(state, value):\n"
            f"    mixed = (state * {index % 7 + 2} + value) % {modulus}\n"
            "    return mixed if mixed != 0 else value\n\n"
        )
    code = repeat_tokens(tokenizer, "".join(code_parts), length)

    rng = random.Random(seed)
    binding_parts = []
    for index in range(256):
        key = rng.randrange(10_000, 99_999)
        value = rng.randrange(10_000, 99_999)
        binding_parts.append(f"slot_{index}_{key} maps to value_{value}; ")
        if index % 8 == 7:
            binding_parts.append(
                f"the checksum for group_{index // 8} is {(key ^ value) % 100_003}. "
            )
    bindings = repeat_tokens(tokenizer, "".join(binding_parts), length)

    motif = repeat_tokens(
        tokenizer,
        "red triangle then blue circle then green square; advance, flip, hold, query; ",
        length,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # Avoid the multimodal/special-token tail of the vocabulary.
    random_ids = torch.randint(0, 200_000, (length,), generator=generator)

    return {
        "natural": natural,
        "code": code,
        "bindings": bindings,
        "periodic": motif,
        "random_ids": random_ids,
    }


def build_heldout_cases(
    tokenizer: Any,
    parquet_path: Path,
    length: int,
    seed: int,
    windows: int,
    shuffle_block: int,
) -> dict[str, torch.Tensor]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("--heldout-parquet requires pyarrow") from error

    table = parquet.read_table(parquet_path, columns=["text"])
    text = "\n".join(value or "" for value in table.column("text").to_pylist())
    all_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    available = all_ids.numel() - length
    if available < 0:
        raise RuntimeError(
            f"held-out text has {all_ids.numel()} tokens, fewer than requested {length}"
        )

    cases: dict[str, torch.Tensor] = {}
    if windows == 1:
        starts = [available // 2]
    else:
        starts = [round(index * available / (windows - 1)) for index in range(windows)]
    for index, start in enumerate(starts):
        window = all_ids[start : start + length].clone()
        cases[f"heldout_{index:02d}"] = window

        shuffled = window.clone()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 10_000 + index)
        for block_start in range(0, length, shuffle_block):
            block_end = min(block_start + shuffle_block, length)
            order = torch.randperm(block_end - block_start, generator=generator)
            shuffled[block_start:block_end] = window[block_start:block_end][order]
        cases[f"shuffled_{index:02d}"] = shuffled
    return cases


def decompose_recurrent(cache: Any) -> tuple[dict[int, tuple[torch.Tensor, ...]], list[dict[str, float]]]:
    decompositions: dict[int, tuple[torch.Tensor, ...]] = {}
    spectra: list[dict[str, float]] = []
    for layer_index, layer in enumerate(cache.layers):
        state = recurrent_tensor(layer)
        if state is None:
            continue
        if state.ndim != 4:
            raise RuntimeError(f"unexpected recurrent state shape at layer {layer_index}: {state.shape}")
        u, singular, vh = torch.linalg.svd(state.float(), full_matrices=False)
        decompositions[layer_index] = (u, singular, vh)

        energy = singular.square()
        normalized = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        cumulative = normalized.cumsum(dim=-1)
        effective_rank = torch.exp(
            -(normalized * normalized.clamp_min(1e-30).log()).sum(dim=-1)
        )
        stable_rank = energy.sum(dim=-1) / energy[..., 0].clamp_min(1e-30)
        for batch_index in range(state.shape[0]):
            for head_index in range(state.shape[1]):
                item: dict[str, float] = {
                    "layer": float(layer_index),
                    "batch": float(batch_index),
                    "head": float(head_index),
                    "effective_rank": float(effective_rank[batch_index, head_index].item()),
                    "stable_rank": float(stable_rank[batch_index, head_index].item()),
                }
                for threshold in (0.90, 0.95, 0.99, 0.999):
                    rank = int(
                        torch.searchsorted(
                            cumulative[batch_index, head_index],
                            torch.tensor(threshold, device=cumulative.device),
                        ).item()
                    ) + 1
                    item[f"rank_{str(threshold).replace('.', '_')}"] = float(rank)
                spectra.append(item)
    return decompositions, spectra


def apply_rank(
    cache: Any,
    decompositions: dict[int, tuple[torch.Tensor, ...]],
    rank: int,
) -> float:
    discarded_energy = 0.0
    total_energy = 0.0
    for layer_index, (u, singular, vh) in decompositions.items():
        state = recurrent_tensor(cache.layers[layer_index])
        if state is None:
            raise RuntimeError("cache structure changed while applying rank")
        retained = min(rank, singular.shape[-1])
        energy = singular.square()
        total_energy += float(energy.sum().item())
        discarded_energy += float(energy[..., retained:].sum().item())
        if retained == singular.shape[-1]:
            # The dense state is already the exact rank-d representation.  Do not
            # introduce a full-SVD reconstruction roundoff into the control.
            continue
        if retained == 0:
            approximation = torch.zeros_like(state)
        else:
            approximation = (u[..., :retained] * singular[..., :retained].unsqueeze(-2)) @ vh[
                ..., :retained, :
            ]
        state.copy_(approximation.to(dtype=state.dtype))
    return math.sqrt(discarded_energy / max(total_energy, 1e-30))


def factorized_recurrent_bytes(cache: Any, rank: int) -> int:
    total = 0
    for layer in cache.layers:
        state = recurrent_tensor(layer)
        if state is None:
            continue
        batch, heads, key_width, value_width = state.shape
        retained = min(rank, key_width, value_width)
        # U, singular values, and V.  This is an optimistic storage estimate:
        # it excludes metadata, alignment, and update workspace.
        elements = batch * heads * retained * (key_width + value_width + 1)
        total += elements * state.element_size()
    return total


def allocate_oracle_ranks(
    decompositions: dict[int, tuple[torch.Tensor, ...]],
    average_rank: int,
    cap: int,
    step: int,
) -> dict[int, torch.Tensor]:
    """Allocate a fixed rank budget by singular energy in hardware-sized chunks.

    This allocation sees the current sequence state and is therefore an oracle
    diagnostic, not a deployable routing rule.
    """
    ranks: dict[int, torch.Tensor] = {}
    heap: list[tuple[float, int, int, int]] = []
    heads_total = 0
    for layer_index, (_, singular, _) in decompositions.items():
        if singular.shape[0] != 1:
            raise RuntimeError("oracle allocator currently expects batch size one")
        layer_ranks = torch.zeros(singular.shape[:2], dtype=torch.int64)
        ranks[layer_index] = layer_ranks
        heads_total += singular.shape[1]
        for head_index in range(singular.shape[1]):
            gain = singular[0, head_index, :step].square().sum().item()
            heapq.heappush(heap, (-gain, layer_index, head_index, 0))

    chunks = average_rank * heads_total // step
    for _ in range(chunks):
        if not heap:
            raise RuntimeError("rank cap is too small for the requested average budget")
        _, layer_index, head_index, start = heapq.heappop(heap)
        next_rank = min(start + step, cap)
        ranks[layer_index][0, head_index] = next_rank
        singular = decompositions[layer_index][1]
        if next_rank < min(cap, singular.shape[-1]):
            end = min(next_rank + step, cap, singular.shape[-1])
            gain = singular[0, head_index, next_rank:end].square().sum().item()
            heapq.heappush(heap, (-gain, layer_index, head_index, next_rank))
    return ranks


def apply_rank_map(
    cache: Any,
    decompositions: dict[int, tuple[torch.Tensor, ...]],
    ranks: dict[int, torch.Tensor],
) -> tuple[float, int, list[int]]:
    discarded_energy = 0.0
    total_energy = 0.0
    factor_bytes = 0
    allocated: list[int] = []
    for layer_index, (u, singular, vh) in decompositions.items():
        state = recurrent_tensor(cache.layers[layer_index])
        if state is None:
            raise RuntimeError("cache structure changed while applying heterogeneous ranks")
        for batch_index in range(state.shape[0]):
            for head_index in range(state.shape[1]):
                rank = int(ranks[layer_index][batch_index, head_index].item())
                allocated.append(rank)
                values = singular[batch_index, head_index]
                energy = values.square()
                total_energy += float(energy.sum().item())
                discarded_energy += float(energy[rank:].sum().item())
                factor_bytes += (
                    rank
                    * (state.shape[-2] + state.shape[-1] + 1)
                    * state.element_size()
                )
                if rank == values.shape[-1]:
                    continue
                if rank == 0:
                    approximation = torch.zeros_like(state[batch_index, head_index])
                else:
                    approximation = (
                        u[batch_index, head_index, :, :rank]
                        * values[:rank].unsqueeze(0)
                    ) @ vh[batch_index, head_index, :rank, :]
                state[batch_index, head_index].copy_(
                    approximation.to(dtype=state.dtype)
                )
    relative_error = math.sqrt(discarded_energy / max(total_energy, 1e-30))
    return relative_error, factor_bytes, allocated


def distribution_metrics(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    baseline = baseline_logits.float()
    candidate = candidate_logits.float()
    baseline_logp = F.log_softmax(baseline, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    kl = (baseline_logp.exp() * (baseline_logp - candidate_logp)).sum(dim=-1)
    baseline_nll = F.cross_entropy(
        baseline.reshape(-1, baseline.shape[-1]), targets.reshape(-1), reduction="mean"
    )
    candidate_nll = F.cross_entropy(
        candidate.reshape(-1, candidate.shape[-1]), targets.reshape(-1), reduction="mean"
    )
    agreement = (baseline.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean()
    baseline_token_nll = -baseline_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    candidate_token_nll = -candidate_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_nll_delta = candidate_token_nll - baseline_token_nll
    agreement_by_position = (
        baseline.argmax(dim=-1) == candidate.argmax(dim=-1)
    ).float().mean(dim=0)
    kl_by_position = kl.mean(dim=0)
    nll_delta_by_position = token_nll_delta.mean(dim=0)
    prefix = min(4, kl_by_position.numel())
    suffix = min(4, kl_by_position.numel())
    return {
        "kl_nats_mean": float(kl.mean().item()),
        "kl_nats_p95": float(torch.quantile(kl.flatten(), 0.95).item()),
        "top1_agreement": float(agreement.item()),
        "baseline_nll": float(baseline_nll.item()),
        "candidate_nll": float(candidate_nll.item()),
        "nll_delta": float((candidate_nll - baseline_nll).item()),
        "kl_nats_first": float(kl_by_position[0].item()),
        "kl_nats_first4_mean": float(kl_by_position[:prefix].mean().item()),
        "kl_nats_last4_mean": float(kl_by_position[-suffix:].mean().item()),
        "top1_agreement_first4": float(agreement_by_position[:prefix].mean().item()),
        "nll_delta_first4_mean": float(nll_delta_by_position[:prefix].mean().item()),
        "kl_by_position": [float(value) for value in kl_by_position.tolist()],
        "nll_delta_by_position": [
            float(value) for value in nll_delta_by_position.tolist()
        ],
        "top1_agreement_by_position": [
            float(value) for value in agreement_by_position.tolist()
        ],
    }


def percentile(values: list[float], quantile: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, quantile).item())


def summarize_spectra(items: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    keys = [
        "effective_rank",
        "stable_rank",
        "rank_0_9",
        "rank_0_95",
        "rank_0_99",
        "rank_0_999",
    ]
    for key in keys:
        values = [item[key] for item in items]
        result[key] = {
            "mean": sum(values) / len(values),
            "median": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "max": max(values),
        }
    return result


def run_case(
    model: Any,
    ids: torch.Tensor,
    prompt_length: int,
    continuation_length: int,
    ranks: list[int],
    heterogeneous_average_ranks: list[int],
    heterogeneous_cap: int,
    heterogeneous_step: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prompt = ids[:prompt_length].unsqueeze(0).to(device)
    continuation = ids[prompt_length : prompt_length + continuation_length].unsqueeze(0).to(device)
    targets = ids[
        prompt_length + 1 : prompt_length + continuation_length + 1
    ].unsqueeze(0).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prefill = model(input_ids=prompt, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - started
    exact_cache = prefill.past_key_values
    ledger = cache_ledger(exact_cache)
    decompositions, spectra = decompose_recurrent(exact_cache)

    baseline_cache = copy.deepcopy(exact_cache)
    with torch.inference_mode():
        baseline_logits = model(
            input_ids=continuation,
            past_key_values=baseline_cache,
            use_cache=True,
        ).logits.detach()

    variants: dict[str, dict[str, float]] = {}
    for rank in ranks:
        candidate_cache = copy.deepcopy(exact_cache)
        relative_state_error = apply_rank(candidate_cache, decompositions, rank)
        with torch.inference_mode():
            candidate_logits = model(
                input_ids=continuation,
                past_key_values=candidate_cache,
                use_cache=True,
            ).logits.detach()
        metrics = distribution_metrics(baseline_logits, candidate_logits, targets)
        metrics["relative_state_frobenius_error"] = relative_state_error
        factor_bytes = factorized_recurrent_bytes(exact_cache, rank)
        metrics["oracle_factorized_recurrent_bytes"] = factor_bytes
        metrics["best_of_dense_or_factorized_recurrent_bytes"] = min(
            ledger.recurrent_bytes, factor_bytes
        )
        metrics["best_case_total_cache_bytes"] = (
            min(ledger.recurrent_bytes, factor_bytes)
            + ledger.convolution_bytes
            + ledger.attention_kv_bytes
        )
        variants[str(rank)] = metrics
        del candidate_cache, candidate_logits

    for average_rank in heterogeneous_average_ranks:
        name = (
            f"hetero_avg{average_rank}_cap{heterogeneous_cap}_step{heterogeneous_step}"
        )
        rank_map = allocate_oracle_ranks(
            decompositions,
            average_rank,
            heterogeneous_cap,
            heterogeneous_step,
        )
        candidate_cache = copy.deepcopy(exact_cache)
        relative_state_error, factor_bytes, allocated = apply_rank_map(
            candidate_cache, decompositions, rank_map
        )
        with torch.inference_mode():
            candidate_logits = model(
                input_ids=continuation,
                past_key_values=candidate_cache,
                use_cache=True,
            ).logits.detach()
        metrics = distribution_metrics(baseline_logits, candidate_logits, targets)
        metrics["relative_state_frobenius_error"] = relative_state_error
        metrics["oracle_factorized_recurrent_bytes"] = factor_bytes
        metrics["best_of_dense_or_factorized_recurrent_bytes"] = min(
            ledger.recurrent_bytes, factor_bytes
        )
        metrics["best_case_total_cache_bytes"] = (
            min(ledger.recurrent_bytes, factor_bytes)
            + ledger.convolution_bytes
            + ledger.attention_kv_bytes
        )
        metrics["allocated_rank_min"] = min(allocated)
        metrics["allocated_rank_median"] = percentile(
            [float(value) for value in allocated], 0.5
        )
        metrics["allocated_rank_p90"] = percentile(
            [float(value) for value in allocated], 0.9
        )
        metrics["allocated_rank_max"] = max(allocated)
        variants[name] = metrics
        del candidate_cache, candidate_logits

    result = {
        "cache_ledger": {**asdict(ledger), "total_bytes": ledger.total_bytes},
        "prefill_seconds_reference_kernel": prefill_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "spectrum": summarize_spectra(spectra),
        "head_spectra": spectra,
        "rank_variants": variants,
    }
    del baseline_cache, baseline_logits, exact_cache, decompositions, prefill
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--continuation-length", type=int, default=32)
    parser.add_argument("--ranks", default="0,8,16,32,64,96,128")
    parser.add_argument("--heterogeneous-average-ranks", default="")
    parser.add_argument("--heterogeneous-cap", type=int, default=64)
    parser.add_argument("--heterogeneous-step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--suite", choices=("handcrafted", "heldout", "both"), default="handcrafted")
    parser.add_argument("--heldout-parquet", type=Path)
    parser.add_argument("--heldout-windows", type=int, default=4)
    parser.add_argument("--shuffle-block", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = sorted({int(value) for value in args.ranks.split(",")})
    heterogeneous_average_ranks = sorted(
        {int(value) for value in args.heterogeneous_average_ranks.split(",") if value}
    )
    total_length = args.prompt_length + args.continuation_length + 1
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    cases: dict[str, torch.Tensor] = {}
    if args.suite in ("handcrafted", "both"):
        cases.update(build_cases(tokenizer, total_length, args.seed))
    if args.suite in ("heldout", "both"):
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
        "ranks": ranks,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "warning": (
            "Oracle SVD truncation at one prompt boundary; this does not prove that a "
            "bounded-rank recurrent update can be trained or served efficiently."
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
            ranks,
            heterogeneous_average_ranks,
            args.heterogeneous_cap,
            args.heterogeneous_step,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": list(cases)}, indent=2))


if __name__ == "__main__":
    main()
