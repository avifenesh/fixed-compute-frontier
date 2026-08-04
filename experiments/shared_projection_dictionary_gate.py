#!/usr/bin/env python3
"""Test whether real LLM input projections share a compact row dictionary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "runtime" / "checkpoints" / "smollm2-135m"
DEFAULT_OUTPUT = ROOT / "results" / "shared-projection-dictionary-gate.json"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
PROJECTION_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = rows.float()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("rows must be a nonempty matrix")
    norms = values.norm(dim=1)
    if torch.any(norms == 0):
        raise ValueError("zero rows are not supported")
    return values / norms[:, None], norms


def reconstruction_error(
    rows: torch.Tensor,
    atoms: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    unit_rows, _ = normalized_rows(rows)
    unit_atoms, _ = normalized_rows(atoms)
    similarities = unit_rows @ unit_atoms.T
    assignments = similarities.abs().argmax(dim=1)
    selected = unit_atoms[assignments]
    coefficients = (rows.float() * selected).sum(dim=1)
    residual = rows.float() - coefficients[:, None] * selected
    relative = residual.square().sum().sqrt() / rows.float().square().sum().sqrt()
    return float(relative), assignments, coefficients


def fit_line_dictionary(
    rows: torch.Tensor,
    atom_count: int,
    *,
    seed: int,
    iterations: int = 4,
    power_steps: int = 3,
) -> dict[str, Any]:
    """Alternate sign-invariant assignment with per-cluster rank-one fits."""
    unit_rows, _ = normalized_rows(rows)
    row_count, width = unit_rows.shape
    if not 0 < atom_count <= row_count:
        raise ValueError("atom_count must lie in [1, row_count]")
    if iterations <= 0 or power_steps <= 0:
        raise ValueError("iterations and power_steps must be positive")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(row_count, generator=generator)[:atom_count]
    atoms = unit_rows[initial].clone()
    history: list[float] = []

    for _ in range(iterations):
        error, assignments, _ = reconstruction_error(rows, atoms)
        history.append(error)
        counts = torch.bincount(assignments, minlength=atom_count)
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()

        for atom_index in torch.nonzero(counts > 0, as_tuple=False).flatten():
            cluster = rows[assignments == atom_index].float()
            direction = atoms[atom_index]
            for _ in range(power_steps):
                direction = cluster.T @ (cluster @ direction)
                norm = direction.norm()
                if norm == 0:
                    break
                direction = direction / norm
            atoms[atom_index] = direction

        if len(empty):
            selected = atoms[assignments]
            coefficients = (rows.float() * selected).sum(dim=1)
            residual_energy = (
                rows.float() - coefficients[:, None] * selected
            ).square().sum(dim=1)
            replacements = residual_energy.topk(len(empty)).indices
            atoms[empty] = unit_rows[replacements]

    error, assignments, coefficients = reconstruction_error(rows, atoms)
    counts = torch.bincount(assignments, minlength=atom_count)
    return {
        "relative_frobenius_error": error,
        "history": history + [error],
        "empty_atoms": int((counts == 0).sum()),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
        "mean_absolute_coefficient": float(coefficients.abs().mean()),
        "atoms": atoms,
        "assignments": assignments,
    }


def random_control(rows: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    control = torch.randn(rows.shape, generator=generator)
    row_norms = rows.float().norm(dim=1)
    return control / control.norm(dim=1, keepdim=True) * row_norms[:, None]


def byte_ledger(row_count: int, width: int, atom_count: int) -> dict[str, Any]:
    index_bits = max(1, math.ceil(math.log2(atom_count)))
    original_bits = 16 * row_count * width
    dictionary_bits = 16 * atom_count * width
    scale_bits = 16 * row_count
    assignment_bits = index_bits * row_count
    encoded_bits = dictionary_bits + scale_bits + assignment_bits
    baseline_macs = row_count * width
    encoded_macs = atom_count * width + row_count
    return {
        "original_bf16_bits": original_bits,
        "encoded_bits": encoded_bits,
        "encoded_to_original_ratio": encoded_bits / original_bits,
        "index_bits_per_row": index_bits,
        "baseline_projection_macs_per_token": baseline_macs,
        "dictionary_projection_macs_per_token": encoded_macs,
        "ideal_macs_ratio": encoded_macs / baseline_macs,
        "excludes": [
            "gather and index execution cost",
            "output projections",
            "activation-weighted causal error",
            "physical kernel efficiency",
        ],
    }


def load_layer_rows(
    checkpoint_file: Path,
    layer: int,
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    tensors = []
    shapes: dict[str, list[int]] = {}
    with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
        for suffix in PROJECTION_SUFFIXES:
            key = f"model.layers.{layer}.{suffix}"
            tensor = handle.get_tensor(key).float()
            tensors.append(tensor)
            shapes[suffix] = list(tensor.shape)
    return torch.cat(tensors, dim=0), shapes


def build_report(
    checkpoint_dir: Path,
    layers: tuple[int, ...],
    *,
    iterations: int,
) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    checkpoint_file = checkpoint_dir / "model.safetensors"
    config = json.loads(config_path.read_text())
    width = int(config["hidden_size"])
    atom_counts = (width // 2, width, 2 * width)
    results = []

    for layer in layers:
        rows, shapes = load_layer_rows(checkpoint_file, layer)
        control = random_control(rows, seed=20260727 + layer)
        fits = []
        for atom_count in atom_counts:
            real_fit = fit_line_dictionary(
                rows,
                atom_count,
                seed=260727 + 100 * layer + atom_count,
                iterations=iterations,
            )
            random_fit = fit_line_dictionary(
                control,
                atom_count,
                seed=270727 + 100 * layer + atom_count,
                iterations=iterations,
            )
            fits.append(
                {
                    "atom_count": atom_count,
                    "real_relative_frobenius_error": real_fit[
                        "relative_frobenius_error"
                    ],
                    "random_relative_frobenius_error": random_fit[
                        "relative_frobenius_error"
                    ],
                    "real_history": real_fit["history"],
                    "random_history": random_fit["history"],
                    "real_maximum_cluster_size": real_fit[
                        "maximum_cluster_size"
                    ],
                    "random_maximum_cluster_size": random_fit[
                        "maximum_cluster_size"
                    ],
                    "ledger": byte_ledger(rows.shape[0], width, atom_count),
                }
            )
        results.append(
            {
                "layer": layer,
                "row_count": rows.shape[0],
                "width": width,
                "projection_shapes": shapes,
                "fits": fits,
            }
        )
        print(json.dumps({"completed_layer": layer, "fits": fits}, indent=2))

    one_basis_errors = [
        fit["real_relative_frobenius_error"]
        for result in results
        for fit in result["fits"]
        if fit["atom_count"] == width
    ]
    one_basis_random = [
        fit["random_relative_frobenius_error"]
        for result in results
        for fit in result["fits"]
        if fit["atom_count"] == width
    ]
    return {
        "schema_version": 1,
        "candidate": "shared input basis with cheap per-output selection and scaling",
        "model": "HuggingFaceTB/SmolLM2-135M",
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": sha256(checkpoint_file),
        "source_sha256": sha256(Path(__file__)),
        "layers": list(layers),
        "projection_suffixes": list(PROJECTION_SUFFIXES),
        "iterations": iterations,
        "results": results,
        "one_basis_summary": {
            "mean_real_relative_frobenius_error": sum(one_basis_errors)
            / len(one_basis_errors),
            "mean_random_relative_frobenius_error": sum(one_basis_random)
            / len(one_basis_random),
        },
        "decision_rule": {
            "one_basis_weight_error_gate": 0.10,
            "required_real_advantage_over_random": 0.25,
            "note": (
                "Passing is only permission for an activation-weighted causal gate; "
                "weight-space failure rejects this parameterization."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 14, 29))
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    report = build_report(
        arguments.checkpoint,
        tuple(arguments.layers),
        iterations=arguments.iterations,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), **report["one_basis_summary"]}, indent=2))


if __name__ == "__main__":
    main()
