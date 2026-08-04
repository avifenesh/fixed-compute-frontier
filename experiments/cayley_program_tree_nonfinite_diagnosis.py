#!/usr/bin/env python3
"""Locate the first non-finite activation in the frozen Cayley program tree."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers

from experiments.cayley_program_tree_lm_pilot import (
    SEED,
    build_candidate,
    router_auxiliary,
    seed_everything,
)
from experiments.reflex_swiglu_lm_screen import TokenFile, sha256_file, write_payload


MODULE = Path("experiments/cayley_program_tree.py")
PILOT = Path("experiments/cayley_program_tree_lm_pilot.py")
PREREGISTRATION = Path(
    "results/cayley-program-tree-nonfinite-diagnosis-preregistration.md"
)
EXPECTED_MODULE_SHA256 = (
    "166ce56d7e619d1d9a9ea35346d3b474d1f0dd05babbb735f9beff6f6725ff8b"
)
EXPECTED_PILOT_SHA256 = (
    "ace68e73e32bd1efd712b1850ea8425e155edf5d4075afb00f6f84b083463235"
)


def tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    detached = values.detach()
    finite = torch.isfinite(detached)
    count = detached.numel()
    finite_count = int(finite.sum())
    if finite_count:
        finite_values = detached[finite].double()
        rms = float(finite_values.square().mean().sqrt())
        absolute_maximum = float(finite_values.abs().max())
    else:
        rms = None
        absolute_maximum = None
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite_fraction": finite_count / count,
        "rms_of_finite": rms,
        "absolute_maximum_of_finite": absolute_maximum,
    }


def trace_tree(module: torch.nn.Module, source: torch.Tensor) -> list[dict[str, Any]]:
    original_shape = source.shape
    initial = source.reshape(-1, module.width)
    hidden = initial
    node = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
    scale = 1.0 / math.sqrt(module.depth)
    rows = [{"depth": -1, "initial": tensor_stats(initial)}]
    for depth_index in range(module.depth):
        projected = module.bases.apply(hidden, depth_index)
        selector, hard_bit, probabilities = module._route(
            projected, node, depth_index
        )
        if module.training and module.force_bits is None:
            choices = module.payload[node]
            selected = (
                choices[:, 0] * (1.0 - selector[:, None, None])
                + choices[:, 1] * selector[:, None, None]
            )
        else:
            selected = module.payload[node, hard_bit]
        gate, up, down = selected.unbind(dim=1)
        gated = gate * projected
        raised = up * projected
        activation = F.silu(gated) * raised
        local_delta = down * activation
        delta = module.bases.apply(local_delta, depth_index, transpose=True)
        hidden = hidden + scale * delta
        rows.append(
            {
                "depth": depth_index,
                "projected": tensor_stats(projected),
                "route_probability": tensor_stats(probabilities),
                "gate_times_projected": tensor_stats(gated),
                "up_times_projected": tensor_stats(raised),
                "activation": tensor_stats(activation),
                "local_delta": tensor_stats(local_delta),
                "delta": tensor_stats(delta),
                "hidden": tensor_stats(hidden),
            }
        )
        node = 2 * node + 1 + hard_bit
    expected = (hidden - initial).reshape(original_shape)
    rows.append({"depth": module.depth, "returned": tensor_stats(expected)})
    return rows


@torch.no_grad()
def run_case(
    model: torch.nn.Module,
    modules: list[torch.nn.Module],
    token_file: TokenFile,
    device: torch.device,
    *,
    name: str,
    batch_size: int,
    train_mode: bool,
    bf16: bool,
) -> dict[str, Any]:
    seed_everything(SEED)
    model.train(train_mode)
    captured_inputs: list[torch.Tensor] = []
    mlp_outputs: list[dict[str, Any]] = []
    layer_outputs: list[dict[str, Any]] = []
    handles = []
    for module in modules:
        handles.append(
            module.register_forward_pre_hook(
                lambda _module, arguments: captured_inputs.append(
                    arguments[0].detach()
                )
            )
        )
        handles.append(
            module.register_forward_hook(
                lambda _module, _arguments, output: mlp_outputs.append(
                    tensor_stats(output)
                )
            )
        )
    for layer in model.model.layers:
        handles.append(
            layer.register_forward_hook(
                lambda _module, _arguments, output: layer_outputs.append(
                    tensor_stats(output[0] if isinstance(output, tuple) else output)
                )
            )
        )
    inputs, targets = token_file.batch(0, batch_size, device)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    )
    torch.cuda.reset_peak_memory_stats()
    failure = None
    logits = None
    loss = None
    try:
        with context:
            logits = model(input_ids=inputs, use_cache=False).logits
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    except Exception as error:  # Preserve diagnostic evidence on failure.
        failure = f"{type(error).__name__}: {error}"
    finally:
        for handle in handles:
            handle.remove()
    tree_traces = []
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    )
    with context:
        for index, (module, source) in enumerate(zip(modules, captured_inputs)):
            tree_traces.append(
                {"layer": index, "depths": trace_tree(module, source)}
            )
    auxiliary = router_auxiliary(modules) if captured_inputs else None
    torch.cuda.synchronize()
    return {
        "name": name,
        "batch_size": batch_size,
        "train_mode": train_mode,
        "precision": "bf16_autocast" if bf16 else "fp32",
        "failure": failure,
        "loss": None if loss is None or not torch.isfinite(loss) else float(loss),
        "loss_finite": bool(loss is not None and torch.isfinite(loss)),
        "router_auxiliary": (
            None
            if auxiliary is None or not torch.isfinite(auxiliary)
            else float(auxiliary)
        ),
        "router_auxiliary_finite": bool(
            auxiliary is not None and torch.isfinite(auxiliary)
        ),
        "logits": None if logits is None else tensor_stats(logits),
        "mlp_inputs": [tensor_stats(value) for value in captured_inputs],
        "mlp_outputs": mlp_outputs,
        "layer_outputs": layer_outputs,
        "tree_traces": tree_traces,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def first_nonfinite(payload: dict[str, Any]) -> dict[str, Any] | None:
    for case in payload["cases"]:
        for layer, stats in enumerate(case["mlp_inputs"]):
            if stats["finite_fraction"] < 1.0:
                return {
                    "case": case["name"],
                    "location": "mlp_input",
                    "layer": layer,
                }
        for trace in case["tree_traces"]:
            for row in trace["depths"]:
                for location, stats in row.items():
                    if location == "depth" or not isinstance(stats, dict):
                        continue
                    if stats.get("finite_fraction", 1.0) < 1.0:
                        return {
                            "case": case["name"],
                            "location": location,
                            "layer": trace["layer"],
                            "depth": row["depth"],
                        }
        for layer, stats in enumerate(case["layer_outputs"]):
            if stats["finite_fraction"] < 1.0:
                return {
                    "case": case["name"],
                    "location": "layer_output",
                    "layer": layer,
                }
        if case["logits"] and case["logits"]["finite_fraction"] < 1.0:
            return {"case": case["name"], "location": "logits"}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cayley-program-tree-nonfinite-diagnosis.json"),
    )
    args = parser.parse_args()
    module_sha256 = sha256_file(MODULE)
    pilot_sha256 = sha256_file(PILOT)
    if module_sha256 != EXPECTED_MODULE_SHA256:
        raise RuntimeError(f"frozen module hash mismatch: {module_sha256}")
    if pilot_sha256 != EXPECTED_PILOT_SHA256:
        raise RuntimeError(f"frozen pilot hash mismatch: {pilot_sha256}")

    device = torch.device("cuda")
    model, modules = build_candidate(device, SEED)
    model.config.use_cache = False
    tokens = TokenFile(args.train_file, args.sequence_length)
    cases = []
    specifications = (
        ("bf16_train_batch1", 1, True, True),
        ("bf16_eval_batch1", 1, False, True),
        ("fp32_eval_batch1", 1, False, False),
        ("bf16_train_batch8", 8, True, True),
    )
    for name, batch_size, train_mode, bf16 in specifications:
        print(json.dumps({"status": "starting_case", "case": name}), flush=True)
        cases.append(
            run_case(
                model,
                modules,
                tokens,
                device,
                name=name,
                batch_size=batch_size,
                train_mode=train_mode,
                bf16=bf16,
            )
        )
        torch.cuda.empty_cache()
    payload = {
        "schema": "cayley-program-tree-nonfinite-diagnosis-v1",
        "source_sha256": sha256_file(Path(__file__)),
        "module_sha256": module_sha256,
        "pilot_sha256": pilot_sha256,
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "data_sha256": sha256_file(args.train_file),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "cases": cases,
    }
    payload["first_nonfinite"] = first_nonfinite(payload)
    write_payload(args.output, payload)
    print(json.dumps({"first_nonfinite": payload["first_nonfinite"]}, indent=2))


if __name__ == "__main__":
    main()
