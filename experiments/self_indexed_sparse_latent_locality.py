#!/usr/bin/env python3
"""Fatal natural-feature-locality probe for self-indexed sparse latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


MODELS = (
    (
        "HuggingFaceTB/SmolLM2-135M",
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    ),
    (
        "HuggingFaceTB/SmolLM2-360M",
        "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    ),
    (
        "HuggingFaceTB/SmolLM2-1.7B",
        "effd688a12921b4cc83e3312b6feb579f70f9c71",
    ),
)
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONFIG = "wikitext-103-raw-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def selected_layer_starts(layer_count: int) -> list[int]:
    starts = []
    for fraction in (0.25, 0.50, 0.75):
        candidate = min(layer_count - 2, max(0, round(fraction * layer_count) - 1))
        if candidate not in starts:
            starts.append(candidate)
    if len(starts) != 3:
        raise ValueError(f"could not select three layer pairs from {layer_count}")
    return starts


def token_blocks(
    dataset: Any,
    tokenizer: Any,
    split: str,
    needed_tokens: int,
    sequence_length: int,
) -> torch.Tensor:
    usable = (needed_tokens // sequence_length) * sequence_length
    if usable != needed_tokens:
        raise ValueError("token counts must be divisible by sequence length")
    tokens: list[int] = []
    eos = tokenizer.eos_token_id
    for row in dataset[split]:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos is not None:
            tokens.append(eos)
        if len(tokens) >= usable:
            break
    if len(tokens) < usable:
        raise RuntimeError(f"split {split} supplied only {len(tokens)} tokens")
    return torch.tensor(tokens[:usable], dtype=torch.long).reshape(-1, sequence_length)


def capture_events(
    model: Any,
    blocks: torch.Tensor,
    layer_indices: list[int],
    active_k: int,
    batch_size: int,
) -> dict[int, dict[str, torch.Tensor]]:
    records: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {"ids": [], "magnitudes": [], "energy": []}
        for layer in layer_indices
    }
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
            activation = args[0].detach()
            if active_k > activation.shape[-1]:
                raise ValueError("active_k exceeds intermediate width")
            magnitudes, features = activation.abs().topk(active_k, dim=-1)
            signed_values = activation.gather(-1, features)
            addresses = features * 2 + (signed_values >= 0).to(torch.long)
            selected_energy = magnitudes.float().square().sum(dim=-1)
            total_energy = activation.float().square().sum(dim=-1).clamp_min(1e-30)
            records[layer]["ids"].append(addresses.to(torch.int32).cpu())
            records[layer]["magnitudes"].append(magnitudes.to(torch.float16).cpu())
            records[layer]["energy"].append((selected_energy / total_energy).cpu())

        return hook

    for layer in layer_indices:
        down_projection = model.layers[layer].mlp.down_proj
        handles.append(down_projection.register_forward_pre_hook(make_hook(layer)))

    try:
        with torch.inference_mode():
            for offset in range(0, blocks.shape[0], batch_size):
                input_ids = blocks[offset : offset + batch_size].to("cuda")
                model(input_ids=input_ids, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    flattened: dict[int, dict[str, torch.Tensor]] = {}
    for layer, values in records.items():
        flattened[layer] = {
            key: torch.cat(chunks, dim=0).reshape(-1, chunks[0].shape[-1])
            if key != "energy"
            else torch.cat(chunks, dim=0).reshape(-1)
            for key, chunks in values.items()
        }
    return flattened


def compile_edges(
    source_ids: torch.Tensor,
    destination_ids: torch.Tensor,
    address_count: int,
    degree: int,
    chunk_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.zeros(
        address_count * address_count, dtype=torch.int32, device="cuda"
    )
    ones_cache: dict[int, torch.Tensor] = {}
    for offset in range(0, source_ids.shape[0], chunk_tokens):
        source = source_ids[offset : offset + chunk_tokens].to(
            "cuda", dtype=torch.long
        )
        destination = destination_ids[offset : offset + chunk_tokens].to(
            "cuda", dtype=torch.long
        )
        pair_indices = (
            source.unsqueeze(2) * address_count + destination.unsqueeze(1)
        ).reshape(-1)
        amount = pair_indices.numel()
        if amount not in ones_cache:
            ones_cache[amount] = torch.ones(amount, dtype=torch.int32, device="cuda")
        counts.index_add_(0, pair_indices, ones_cache[amount])
    matrix = counts.reshape(address_count, address_count)
    edge_counts, neighbors = matrix.topk(degree, dim=1)
    source_frequency = torch.bincount(
        source_ids.reshape(-1).to(torch.long), minlength=address_count
    ).to("cuda", dtype=torch.float32)
    weights = edge_counts.float() / source_frequency[:, None].clamp_min(1.0)
    destination_frequency = torch.bincount(
        destination_ids.reshape(-1).to(torch.long), minlength=address_count
    )
    return neighbors.to(torch.int32), weights.to(torch.float16), destination_frequency


def recall_at_k(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    matches = (truth.unsqueeze(2) == predicted.unsqueeze(1)).any(dim=2)
    return matches.float().mean(dim=1)


def graph_predict(
    source_ids: torch.Tensor,
    source_magnitudes: torch.Tensor,
    neighbors: torch.Tensor,
    weights: torch.Tensor,
    address_count: int,
    active_k: int,
    chunk_tokens: int = 256,
) -> torch.Tensor:
    predictions = []
    for offset in range(0, source_ids.shape[0], chunk_tokens):
        ids = source_ids[offset : offset + chunk_tokens].to("cuda", dtype=torch.long)
        magnitudes = source_magnitudes[offset : offset + chunk_tokens].to(
            "cuda", dtype=torch.float32
        )
        emitted = neighbors[ids].to(torch.long)
        contribution = weights[ids].float() * magnitudes.unsqueeze(2)
        scores = torch.zeros(
            ids.shape[0], address_count, device="cuda", dtype=torch.float32
        )
        scores.scatter_add_(1, emitted.reshape(ids.shape[0], -1), contribution.reshape(ids.shape[0], -1))
        predictions.append(scores.topk(active_k, dim=1).indices.to(torch.int32).cpu())
    return torch.cat(predictions, dim=0)


def evaluate_pair(
    train_source: dict[str, torch.Tensor],
    train_destination: dict[str, torch.Tensor],
    test_source: dict[str, torch.Tensor],
    test_destination: dict[str, torch.Tensor],
    intermediate_size: int,
    active_k: int,
    degree: int,
    seed: int,
) -> dict[str, Any]:
    address_count = 2 * intermediate_size
    neighbors, weights, destination_frequency = compile_edges(
        train_source["ids"],
        train_destination["ids"],
        address_count,
        degree,
    )
    predicted = graph_predict(
        test_source["ids"],
        test_source["magnitudes"],
        neighbors,
        weights,
        address_count,
        active_k,
    )
    truth = test_destination["ids"].to(torch.int32)
    graph_recall_by_token = recall_at_k(predicted, truth)

    global_prediction = destination_frequency.topk(active_k).indices.to(torch.int32)
    global_predictions = global_prediction.unsqueeze(0).expand(truth.shape[0], -1)
    global_recall_by_token = recall_at_k(global_predictions, truth)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(test_source["ids"].shape[0], generator=generator)
    shuffled_prediction = graph_predict(
        test_source["ids"][permutation],
        test_source["magnitudes"][permutation],
        neighbors,
        weights,
        address_count,
        active_k,
    )
    shuffled_recall_by_token = recall_at_k(shuffled_prediction, truth)

    truth_matches = (truth.unsqueeze(2) == predicted.unsqueeze(1)).any(dim=2)
    captured_selected_energy = (
        test_destination["magnitudes"].float().square() * truth_matches.float()
    ).sum(dim=1)
    selected_energy = test_destination["magnitudes"].float().square().sum(dim=1)
    full_energy = selected_energy / test_destination["energy"].clamp_min(1e-30)
    predicted_full_energy_lower_bound = captured_selected_energy / full_energy

    graph_recall = graph_recall_by_token.mean().item()
    global_recall = global_recall_by_token.mean().item()
    shuffled_recall = shuffled_recall_by_token.mean().item()
    energy_median = test_destination["energy"].median().item()
    gates = {
        "topk_energy_at_least_0p90": energy_median >= 0.90,
        "graph_recall_at_least_0p80": graph_recall >= 0.80,
        "global_lift_at_least_0p20": graph_recall - global_recall >= 0.20,
        "shuffled_lift_at_least_0p20": graph_recall - shuffled_recall >= 0.20,
    }
    return {
        "intermediate_size": intermediate_size,
        "address_count_including_sign": address_count,
        "active_k": active_k,
        "degree": degree,
        "active_edge_records": active_k * degree,
        "validation_tokens": truth.shape[0],
        "topk_energy_median": energy_median,
        "topk_energy_p10": torch.quantile(
            test_destination["energy"], 0.10
        ).item(),
        "graph_recall_at_k": graph_recall,
        "global_frequency_recall_at_k": global_recall,
        "shuffled_source_recall_at_k": shuffled_recall,
        "graph_minus_global_recall": graph_recall - global_recall,
        "graph_minus_shuffled_recall": graph_recall - shuffled_recall,
        "predicted_full_energy_lower_bound_mean": predicted_full_energy_lower_bound.mean().item(),
        "gates": gates,
        "all_pair_gates_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    corpus = load_dataset(
        DATASET,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )
    model_results = []
    for model_index, (model_id, revision) in enumerate(MODELS):
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        train_blocks = token_blocks(
            corpus, tokenizer, "train", args.train_tokens, args.sequence_length
        )
        validation_blocks = token_blocks(
            corpus, tokenizer, "validation", args.validation_tokens, args.sequence_length
        )
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        model.eval()
        layers = model.layers
        starts = selected_layer_starts(len(layers))
        captured_layers = sorted(set(starts + [layer + 1 for layer in starts]))
        train_capture = capture_events(
            model,
            train_blocks,
            captured_layers,
            args.active_k,
            args.batch_size,
        )
        validation_capture = capture_events(
            model,
            validation_blocks,
            captured_layers,
            args.active_k,
            args.batch_size,
        )
        intermediate_size = int(model.config.intermediate_size)
        pair_results = []
        for pair_index, source_layer in enumerate(starts):
            pair = evaluate_pair(
                train_capture[source_layer],
                train_capture[source_layer + 1],
                validation_capture[source_layer],
                validation_capture[source_layer + 1],
                intermediate_size,
                args.active_k,
                args.degree,
                args.seed + 1000 * model_index + pair_index,
            )
            pair["source_layer"] = source_layer
            pair["destination_layer"] = source_layer + 1
            pair["depth_fraction"] = (source_layer + 1) / len(layers)
            pair_results.append(pair)
        model_results.append(
            {
                "model_id": model_id,
                "revision": revision,
                "hidden_size": int(model.config.hidden_size),
                "intermediate_size": intermediate_size,
                "layer_count": len(layers),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "pairs": pair_results,
                "all_pair_gates_pass": all(
                    pair["all_pair_gates_pass"] for pair in pair_results
                ),
            }
        )
        del train_capture, validation_capture, model
        torch.cuda.empty_cache()

    width_trends = []
    for pair_position in range(3):
        recalls = [model["pairs"][pair_position]["graph_recall_at_k"] for model in model_results]
        energies = [model["pairs"][pair_position]["topk_energy_median"] for model in model_results]
        width_trends.append(
            {
                "pair_position": pair_position,
                "graph_recalls": recalls,
                "topk_energy_medians": energies,
                "recall_non_decreasing_with_width": all(
                    right + 1e-12 >= left for left, right in zip(recalls, recalls[1:])
                ),
                "energy_non_decreasing_with_width": all(
                    right + 1e-12 >= left for left, right in zip(energies, energies[1:])
                ),
            }
        )
    width_gate = all(
        trend["recall_non_decreasing_with_width"] for trend in width_trends
    )

    root = Path(__file__).resolve().parents[1]
    preregistration = root / "results" / "self-indexed-sparse-latent-locality-preregistration.md"
    all_pair_gates = all(model["all_pair_gates_pass"] for model in model_results)
    return {
        "schema": "self-indexed-sparse-latent-locality-v1",
        "status": "survives" if all_pair_gates and width_gate else "closed_at_natural_locality_gate",
        "seed": args.seed,
        "models": model_results,
        "width_trends": width_trends,
        "gates": {
            "all_pair_thresholds_pass": all_pair_gates,
            "recall_non_decreasing_at_all_depths": width_gate,
        },
        "all_gates_pass": all_pair_gates and width_gate,
        "data": {
            "dataset": DATASET,
            "configuration": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "train_tokens": args.train_tokens,
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
    parser.add_argument("--seed", type=int, default=7302026)
    parser.add_argument("--active-k", type=int, default=64)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-tokens", type=int, default=8192)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/self-indexed-sparse-latent-locality.json"),
    )
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

