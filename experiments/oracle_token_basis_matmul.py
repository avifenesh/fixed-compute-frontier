#!/usr/bin/env python3
"""Oracle cross-token rank gate for reordering LLM projection matmuls."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


MODELS = (
    ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
    ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71"),
)
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-103-raw-v1"
ERRORS = (0.01, 0.05, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def selected_layer_starts(layer_count: int) -> list[int]:
    return [
        min(layer_count - 1, max(0, round(fraction * layer_count) - 1))
        for fraction in (0.25, 0.50, 0.75)
    ]


def token_blocks(dataset: Any, tokenizer: Any, tokens: int, sequence_length: int) -> torch.Tensor:
    if tokens % sequence_length:
        raise ValueError("tokens must be divisible by sequence length")
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for row in dataset["validation"]:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos is not None:
            ids.append(eos)
        if len(ids) >= tokens:
            break
    if len(ids) < tokens:
        raise RuntimeError("not enough validation tokens")
    return torch.tensor(ids[:tokens], dtype=torch.long).reshape(-1, sequence_length)


def capture_projection_inputs(
    model: Any,
    blocks: torch.Tensor,
    layers: list[int],
    batch_size: int,
) -> dict[str, dict[int, torch.Tensor]]:
    captured: dict[str, dict[int, list[torch.Tensor]]] = {
        "qkv": {layer: [] for layer in layers},
        "ffn_up": {layer: [] for layer in layers},
    }
    handles = []

    def make_hook(kind: str, layer: int):
        def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
            captured[kind][layer].append(args[0].detach().to(torch.bfloat16).cpu())

        return hook

    for layer in layers:
        handles.append(
            model.layers[layer].self_attn.q_proj.register_forward_pre_hook(
                make_hook("qkv", layer)
            )
        )
        handles.append(
            model.layers[layer].mlp.gate_proj.register_forward_pre_hook(
                make_hook("ffn_up", layer)
            )
        )
    try:
        with torch.inference_mode():
            for offset in range(0, blocks.shape[0], batch_size):
                model(
                    input_ids=blocks[offset : offset + batch_size].to("cuda"),
                    use_cache=False,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    return {
        kind: {layer: torch.cat(chunks, dim=0) for layer, chunks in by_layer.items()}
        for kind, by_layer in captured.items()
    }


def minimum_rank_from_row_energy(row_energy: torch.Tensor, error: float) -> int:
    total = row_energy.sum().clamp_min(1e-30)
    target = (1.0 - error * error) * total
    cumulative = row_energy.cumsum(dim=0)
    return int(torch.searchsorted(cumulative, target).item()) + 1


def ideal_cost_ratio(rank: int, tokens: int, hidden: int) -> float:
    return rank / tokens + rank / hidden


def sequence_rank_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    errors: tuple[float, ...] = ERRORS,
) -> dict[str, Any]:
    x = x.float()
    y = y.float()
    u_x, singular_x, _ = torch.linalg.svd(x, full_matrices=False)
    coefficients = u_x.transpose(0, 1) @ y
    input_basis_row_energy = coefficients.square().sum(dim=1)
    singular_y = torch.linalg.svdvals(y)
    output_oracle_row_energy = singular_y.square()
    hidden = x.shape[1]
    tokens = x.shape[0]
    result: dict[str, Any] = {}
    for error in errors:
        key = f"error_{str(error).replace('.', 'p')}"
        input_rank = minimum_rank_from_row_energy(input_basis_row_energy, error)
        oracle_rank = minimum_rank_from_row_energy(output_oracle_row_energy, error)
        result[key] = {
            "input_basis_rank": input_rank,
            "input_basis_rank_fraction": input_rank / tokens,
            "input_basis_ideal_cost_ratio": ideal_cost_ratio(input_rank, tokens, hidden),
            "output_oracle_rank": oracle_rank,
            "output_oracle_rank_fraction": oracle_rank / tokens,
            "output_oracle_ideal_cost_ratio": ideal_cost_ratio(oracle_rank, tokens, hidden),
        }
    result["input_effective_rank"] = float(
        singular_x.square().sum().square() / singular_x.pow(4).sum().clamp_min(1e-30)
    )
    return result


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "median": float(tensor.median()),
        "p90": float(torch.quantile(tensor, 0.90)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def summarize_sequences(sequences: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for error in ERRORS:
        key = f"error_{str(error).replace('.', 'p')}"
        summary[key] = {}
        for metric in (
            "input_basis_rank_fraction",
            "input_basis_ideal_cost_ratio",
            "output_oracle_rank_fraction",
            "output_oracle_ideal_cost_ratio",
        ):
            summary[key][metric] = quantiles([row[key][metric] for row in sequences])
    summary["input_effective_rank"] = quantiles(
        [row["input_effective_rank"] for row in sequences]
    )
    one_percent = summary["error_0p01"]
    gates = {
        "oracle_median_cost_at_most_0p5": one_percent[
            "output_oracle_ideal_cost_ratio"
        ]["median"]
        <= 0.5,
        "oracle_p90_cost_at_most_0p5": one_percent[
            "output_oracle_ideal_cost_ratio"
        ]["p90"]
        <= 0.5,
        "input_basis_median_cost_at_most_0p5": one_percent[
            "input_basis_ideal_cost_ratio"
        ]["median"]
        <= 0.5,
        "input_basis_p90_cost_at_most_0p5": one_percent[
            "input_basis_ideal_cost_ratio"
        ]["p90"]
        <= 0.5,
    }
    summary["gates"] = gates
    summary["all_cell_gates_pass"] = all(gates.values())
    return summary


def projected_output(layer: Any, kind: str, x: torch.Tensor) -> torch.Tensor:
    x = x.to("cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        if kind == "qkv":
            pieces = (
                layer.self_attn.q_proj(x),
                layer.self_attn.k_proj(x),
                layer.self_attn.v_proj(x),
            )
        elif kind == "ffn_up":
            pieces = (layer.mlp.gate_proj(x), layer.mlp.up_proj(x))
        else:
            raise ValueError(kind)
    return torch.cat(pieces, dim=-1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    corpus = load_dataset(
        DATASET,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )
    model_results = []
    for model_id, revision in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        blocks = token_blocks(corpus, tokenizer, args.validation_tokens, args.sequence_length)
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        model.eval()
        layer_indices = selected_layer_starts(len(model.layers))
        captures = capture_projection_inputs(
            model, blocks, layer_indices, args.batch_size
        )
        cells = []
        for kind, by_layer in captures.items():
            for layer_index, sequence_batch in by_layer.items():
                sequence_metrics = []
                for sequence in sequence_batch:
                    x = sequence.to("cuda")
                    y = projected_output(model.layers[layer_index], kind, x)
                    sequence_metrics.append(sequence_rank_metrics(x, y))
                summary = summarize_sequences(sequence_metrics)
                cells.append(
                    {
                        "kind": kind,
                        "layer": layer_index,
                        "depth_fraction": (layer_index + 1) / len(model.layers),
                        "sequence_count": len(sequence_metrics),
                        "summary": summary,
                        "sequences": sequence_metrics,
                    }
                )
        model_results.append(
            {
                "model_id": model_id,
                "revision": revision,
                "hidden_size": int(model.config.hidden_size),
                "layer_count": len(model.layers),
                "cells": cells,
                "all_cell_gates_pass": all(
                    cell["summary"]["all_cell_gates_pass"] for cell in cells
                ),
            }
        )
        del captures, model
        torch.cuda.empty_cache()

    width_trends = []
    for kind in ("qkv", "ffn_up"):
        for position in range(3):
            cells = [
                next(
                    cell
                    for cell in model["cells"]
                    if cell["kind"] == kind
                    and sorted(c["layer"] for c in model["cells"] if c["kind"] == kind).index(cell["layer"])
                    == position
                )
                for model in model_results
            ]
            oracle_costs = [
                cell["summary"]["error_0p01"]["output_oracle_ideal_cost_ratio"]["median"]
                for cell in cells
            ]
            input_costs = [
                cell["summary"]["error_0p01"]["input_basis_ideal_cost_ratio"]["median"]
                for cell in cells
            ]
            width_trends.append(
                {
                    "kind": kind,
                    "position": position,
                    "oracle_median_cost_ratios": oracle_costs,
                    "input_basis_median_cost_ratios": input_costs,
                    "oracle_non_worsening": all(
                        right <= left + 1e-12 for left, right in zip(oracle_costs, oracle_costs[1:])
                    ),
                    "input_basis_non_worsening": all(
                        right <= left + 1e-12 for left, right in zip(input_costs, input_costs[1:])
                    ),
                }
            )
    all_cells = all(model["all_cell_gates_pass"] for model in model_results)
    width_gate = all(
        trend["oracle_non_worsening"] and trend["input_basis_non_worsening"]
        for trend in width_trends
    )
    root = Path(__file__).resolve().parents[1]
    preregistration = root / "results" / "oracle-token-basis-matmul-preregistration.md"
    return {
        "schema": "oracle-token-basis-matmul-v1",
        "status": "survives" if all_cells and width_gate else "closed_at_oracle_rank_gate",
        "all_gates_pass": all_cells and width_gate,
        "gates": {
            "all_cell_twofold_bounds_pass": all_cells,
            "all_width_trends_non_worsening": width_gate,
        },
        "models": model_results,
        "width_trends": width_trends,
        "data": {
            "dataset": DATASET,
            "configuration": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "validation_tokens": args.validation_tokens,
            "sequence_length": args.sequence_length,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "source_sha256": sha256(Path(__file__).resolve()),
        "preregistration_sha256": sha256(preregistration),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7302027)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/oracle-token-basis-matmul.json"),
    )
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

