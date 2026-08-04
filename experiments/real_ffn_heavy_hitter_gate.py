#!/usr/bin/env python3
"""Measure whether real SwiGLU FFNs admit sublinear heavy-neuron recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "real-ffn-heavy-hitter-gate.json"
MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
LAYERS = (0, 14, 29)
TOPK = (16, 32, 64, 96, 128, 256)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    data = values.float().flatten()
    points = torch.tensor([0.1, 0.5, 0.9, 0.99], device=data.device)
    result = torch.quantile(data, points).cpu().tolist()
    return {
        "p10": result[0],
        "median": result[1],
        "p90": result[2],
        "p99": result[3],
        "mean": float(data.mean().cpu()),
    }


def relative_tail_by_topk(values: torch.Tensor, topk: int) -> torch.Tensor:
    energy = values.float().square()
    total = energy.sum(dim=-1).clamp_min(1e-30)
    retained = energy.topk(topk, dim=-1).values.sum(dim=-1)
    return (1.0 - retained / total).clamp_min(0).sqrt()


def required_k_for_energy(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    energy = values.float().square()
    ordered = energy.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1)
    threshold = fraction * energy.sum(dim=-1, keepdim=True)
    return (cumulative < threshold).sum(dim=-1) + 1


def selected_output_error(
    activations: torch.Tensor,
    down_weight: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Relative output error after executing only selected FFN neurons."""
    errors = []
    weight_by_neuron = down_weight.float().T.contiguous()
    for start in range(0, activations.shape[0], chunk_size):
        stop = min(start + chunk_size, activations.shape[0])
        current = activations[start:stop].float()
        indices = scores[start:stop].topk(topk, dim=-1).indices
        values = current.gather(1, indices)
        selected_weights = weight_by_neuron[indices]
        approximate = torch.bmm(values[:, None, :], selected_weights).squeeze(1)
        exact = current @ down_weight.float().T
        error = (exact - approximate).norm(dim=-1) / exact.norm(dim=-1).clamp_min(
            1e-30
        )
        errors.append(error.cpu())
    return torch.cat(errors)


class LayerCollector:
    def __init__(self, layer: int, topk: tuple[int, ...]) -> None:
        self.layer = layer
        self.topk = topk
        self.rows: list[dict[str, Any]] = []

    @torch.no_grad()
    def __call__(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _: Any) -> None:
        hidden = inputs[0].detach()
        flat = hidden.reshape(-1, hidden.shape[-1])
        gate = module.gate_proj(flat)
        up = module.up_proj(flat)
        gate_activation = torch.nn.functional.silu(gate)
        activations = gate_activation * up
        down_weight = module.down_proj.weight.detach()
        value_norm = down_weight.float().norm(dim=0)
        oracle_scores = activations.float().abs() * value_norm[None, :]
        gate_scores = gate_activation.float().abs()

        row: dict[str, Any] = {
            "tokens": flat.shape[0],
            "intermediate_size": activations.shape[-1],
            "gate_required_k": {},
            "activation_required_k": {},
            "topk": {},
        }
        for fraction in (0.90, 0.95, 0.99):
            label = str(fraction)
            row["gate_required_k"][label] = required_k_for_energy(
                gate_activation, fraction
            ).cpu()
            row["activation_required_k"][label] = required_k_for_energy(
                activations, fraction
            ).cpu()

        for count in self.topk:
            row["topk"][str(count)] = {
                "gate_relative_l2_tail": relative_tail_by_topk(
                    gate_activation, count
                ).cpu(),
                "activation_relative_l2_tail": relative_tail_by_topk(
                    activations, count
                ).cpu(),
                "oracle_output_relative_error": selected_output_error(
                    activations, down_weight, oracle_scores, count
                ),
                "gate_selected_output_relative_error": selected_output_error(
                    activations, down_weight, gate_scores, count
                ),
            }
        self.rows.append(row)

    def summarize(self) -> dict[str, Any]:
        if not self.rows:
            raise RuntimeError(f"layer {self.layer} collected no activations")
        tokens = sum(row["tokens"] for row in self.rows)
        intermediate_size = self.rows[0]["intermediate_size"]
        summary: dict[str, Any] = {
            "layer": self.layer,
            "tokens": tokens,
            "intermediate_size": intermediate_size,
            "required_k": {"gate": {}, "activation": {}},
            "topk": {},
        }
        for fraction in (0.90, 0.95, 0.99):
            label = str(fraction)
            for kind, key in (
                ("gate", "gate_required_k"),
                ("activation", "activation_required_k"),
            ):
                values = torch.cat([row[key][label] for row in self.rows])
                stats = quantiles(values)
                stats["median_fraction"] = stats["median"] / intermediate_size
                stats["p90_fraction"] = stats["p90"] / intermediate_size
                summary["required_k"][kind][label] = stats

        for count in self.topk:
            label = str(count)
            summary["topk"][label] = {}
            for metric in (
                "gate_relative_l2_tail",
                "activation_relative_l2_tail",
                "oracle_output_relative_error",
                "gate_selected_output_relative_error",
            ):
                values = torch.cat(
                    [row["topk"][label][metric] for row in self.rows]
                )
                summary["topk"][label][metric] = quantiles(values)
        return summary


def sample_token_batches(
    tokenizer: Any,
    *,
    token_count: int,
    sequence_length: int,
    batch_size: int,
) -> list[torch.Tensor]:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )
    token_ids: list[int] = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        token_ids.append(tokenizer.eos_token_id)
        if len(token_ids) >= token_count:
            break
    usable = min(len(token_ids), token_count)
    usable -= usable % sequence_length
    tensor = torch.tensor(token_ids[:usable], dtype=torch.long).reshape(
        -1, sequence_length
    )
    return list(tensor.split(batch_size))


def build_report(
    *,
    token_count: int,
    sequence_length: int,
    batch_size: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("this frozen gate requires CUDA")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    collectors = {layer: LayerCollector(layer, TOPK) for layer in LAYERS}
    handles = [
        model.model.layers[layer].mlp.register_forward_hook(collectors[layer])
        for layer in LAYERS
    ]
    batches = sample_token_batches(
        tokenizer,
        token_count=token_count,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    try:
        with torch.inference_mode():
            for index, batch in enumerate(batches):
                model.model(input_ids=batch.to(device), use_cache=False)
                print(
                    json.dumps(
                        {
                            "completed_batch": index + 1,
                            "total_batches": len(batches),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    layers = [collectors[layer].summarize() for layer in LAYERS]
    gate_count = 96
    layer_gates = {}
    for layer in layers:
        metrics = layer["topk"][str(gate_count)]
        layer_gates[str(layer["layer"])] = {
            "oracle_median_output_error_le_5_percent": (
                metrics["oracle_output_relative_error"]["median"] <= 0.05
            ),
            "oracle_p90_output_error_le_10_percent": (
                metrics["oracle_output_relative_error"]["p90"] <= 0.10
            ),
            "gate_selected_median_output_error_le_7_5_percent": (
                metrics["gate_selected_output_relative_error"]["median"]
                <= 0.075
            ),
            "gate_selected_p90_output_error_le_15_percent": (
                metrics["gate_selected_output_relative_error"]["p90"] <= 0.15
            ),
        }
    all_gates = [value for layer in layer_gates.values() for value in layer.values()]
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "candidate": "linear-sketch heavy-hitter recovery for giant sparse FFNs",
        "model": MODEL,
        "revision": REVISION,
        "source_sha256": sha256(Path(__file__)),
        "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train stream",
        "token_count_requested": token_count,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "layers": layers,
        "frozen_prerequisite": {
            "active_neurons": gate_count,
            "active_fraction": gate_count / layers[0]["intermediate_size"],
            "layer_gates": layer_gates,
            "passed": all(all_gates),
            "interpretation": (
                "Failure rejects CountSketch/group-testing implementation work "
                "for native SmolLM2 SwiGLU activations. Passing only permits a "
                "learned-sketch recovery test."
            ),
        },
        "hardware": {
            "gpu_name": properties.name,
            "gpu_total_memory": properties.total_memory,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--token-count", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    arguments = parser.parse_args()
    report = build_report(
        token_count=arguments.token_count,
        sequence_length=arguments.sequence_length,
        batch_size=arguments.batch_size,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["frozen_prerequisite"], indent=2))


if __name__ == "__main__":
    main()
