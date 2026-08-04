#!/usr/bin/env python3
"""One-seed falsifier separating TVE forward capacity from its backward signal."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import types
from typing import Any

import numpy as np
import torch
import transformers
import triton

import experiments.triangular_value_encoding_lm_pilot as pilot


OUTPUT = Path("results/triangular-value-encoding-forward-mechanism.json")
PREREGISTRATION = Path(
    "results/triangular-value-encoding-forward-mechanism-preregistration.md"
)
MANIFEST = Path(
    "results/triangular-value-encoding-forward-mechanism-integrity-manifest.json"
)
EXPORT_DIR = Path("results/triangular-value-encoding-forward-mechanism-exports")
TRAIN_FILE = Path("data/self-product-ffn-scale/train.uint16.bin")
VALIDATION_FILE = Path("data/self-product-ffn-scale/validation.uint16.bin")
DATA_MANIFEST = Path("results/self-product-ffn-scale-data-manifest.json")
SEED = 21017
ARMS = ("canonical_baseline", "backward_only_tve", "full_tve")
EVAL_STEPS = (61, 152, 305)
STEPS = 305

DEPENDENCIES = {
    "mechanism_source": Path(__file__),
    "preregistration": PREREGISTRATION,
    "pilot_source": Path("experiments/triangular_value_encoding_lm_pilot.py"),
    "attention_source": Path("experiments/triangular_value_encoding_attention.py"),
    "attention_test": Path("tests/test_triangular_value_encoding_attention.py"),
    "data_manifest": DATA_MANIFEST,
    "direct_native_preregistration": Path(
        "results/triangular-value-encoding-direct-native-preregistration.md"
    ),
    "direct_native_manifest": Path(
        "results/triangular-value-encoding-direct-native-integrity-manifest.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrity() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checks = {
        name: manifest.get(name + "_sha256") == sha256_file(path)
        for name, path in DEPENDENCIES.items()
    }
    checks.update({
        "python_version": manifest.get("python_version") == platform.python_version(),
        "numpy_version": manifest.get("numpy_version") == np.__version__,
        "torch_version": manifest.get("torch_version") == torch.__version__,
        "cuda_version": manifest.get("cuda_version") == torch.version.cuda,
        "transformers_version": (
            manifest.get("transformers_version") == transformers.__version__
        ),
        "triton_version": manifest.get("triton_version") == triton.__version__,
    })
    if not all(checks.values()):
        raise ValueError(f"invalid forward-mechanism integrity: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "manifest_sha256": sha256_file(MANIFEST),
    }


def validate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = pilot.scratch_config()
    expected = {
        "device_contains": "H100",
        "train_file": str(TRAIN_FILE),
        "validation_file": str(VALIDATION_FILE),
        "data_manifest": str(DATA_MANIFEST),
        "output": str(OUTPUT),
        "export_dir": str(EXPORT_DIR),
        "seed": SEED,
        "arms": list(ARMS),
        "sequence_length": 512,
        "micro_batch_size": 32,
        "gradient_accumulation": 2,
        "eval_batch_size": 32,
        "eval_batches": 64,
        "steps": STEPS,
        "eval_steps": list(EVAL_STEPS),
        "warmup_steps": 50,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "prediction_tokens": 9_994_240,
        "formal": True,
        "hidden_size": 384,
        "layers": 12,
        "query_heads": 6,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate_size": 1024,
    }
    actual = {
        "device_contains": torch.cuda.get_device_name(0),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "data_manifest": str(args.data_manifest),
        "output": str(args.output),
        "export_dir": str(args.export_dir),
        "seed": args.seed,
        "arms": list(ARMS),
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "eval_batch_size": args.eval_batch_size,
        "eval_batches": args.eval_batches,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "prediction_tokens": (
            args.steps * args.gradient_accumulation * args.micro_batch_size
            * args.sequence_length
        ),
        "formal": args.formal,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
    }
    checks = {
        key: (value in actual[key] if key == "device_contains" else actual[key] == value)
        for key, value in expected.items()
    }
    if not all(checks.values()):
        raise ValueError(f"invalid forward-mechanism protocol: {checks}")
    return {"valid": True, "checks": checks, "expected": expected, "actual": actual}


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _backward_only_value_encoding(self, values: torch.Tensor) -> torch.Tensor:
    if self.arm != "triangular_value_encoding":
        return values
    differentiable = self._torch_value_encoding(values)
    # Forward is bit-exact `values`; VJP is exactly the reference TVE VJP.
    return values.detach() + (differentiable - differentiable.detach())


def build_model(device: torch.device, arm: str) -> tuple[torch.nn.Module, list[Any]]:
    architecture_arm = (
        "canonical_value_control"
        if arm == "canonical_baseline" else "triangular_value_encoding"
    )
    model, modules = pilot.build_model(device, architecture_arm)
    if arm == "backward_only_tve":
        for module in modules:
            module.apply_value_encoding = types.MethodType(
                _backward_only_value_encoding, module
            )
    return model, modules


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().contiguous().cpu().reshape(-1)
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def first_backward_probe(
    train_file: Any, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    reset_seed(args.seed)
    backward_model, backward_modules = build_model(device, "backward_only_tve")
    reset_seed(args.seed)
    full_model, full_modules = build_model(device, "full_tve")
    reset_seed(args.seed)
    canonical_model, canonical_modules = build_model(device, "canonical_baseline")
    backward_hash = state_hash(backward_model)
    full_hash = state_hash(full_model)
    canonical_hash = state_hash(canonical_model)
    inputs, targets = train_file.batch(0, args.micro_batch_size, device)
    backward_model.train()
    full_model.train()
    canonical_model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        backward_logits = backward_model(input_ids=inputs, use_cache=False).logits
        full_logits = full_model(input_ids=inputs, use_cache=False).logits
        canonical_logits = canonical_model(input_ids=inputs, use_cache=False).logits
    backward_loss = torch.nn.functional.cross_entropy(
        backward_logits.float().reshape(-1, backward_logits.shape[-1]),
        targets.reshape(-1),
    )
    full_loss = torch.nn.functional.cross_entropy(
        full_logits.float().reshape(-1, full_logits.shape[-1]), targets.reshape(-1)
    )
    canonical_loss = torch.nn.functional.cross_entropy(
        canonical_logits.float().reshape(-1, canonical_logits.shape[-1]),
        targets.reshape(-1),
    )
    backward_loss.backward()
    full_loss.backward()
    canonical_loss.backward()
    gradient_exact = True
    finite = True
    values = 0
    active_coefficient_gradient_by_layer = []
    outside_active_output_gradient_exact_by_layer = []
    for backward_parameter, full_parameter in zip(
        backward_model.parameters(), full_model.parameters()
    ):
        if (backward_parameter.grad is None) != (full_parameter.grad is None):
            gradient_exact = False
            continue
        if backward_parameter.grad is None:
            continue
        gradient_exact = gradient_exact and torch.equal(
            backward_parameter.grad, full_parameter.grad
        )
        finite = finite and torch.isfinite(backward_parameter.grad).all().item()
        values += backward_parameter.grad.numel()
    for module, canonical_module in zip(backward_modules, canonical_modules):
        mask = torch.zeros_like(module.output_weight, dtype=torch.bool)
        layer_nonzero = False
        for kv_head in range(module.kv_heads):
            representative = (
                kv_head * module.num_key_value_groups * module.head_dim
            )
            for start in range(0, module.head_dim, module.block_size):
                for target in range(1, module.block_size):
                    row = start + target
                    columns = slice(
                        representative + start,
                        representative + start + target,
                    )
                    mask[row, columns] = True
                    layer_nonzero = (
                        layer_nonzero
                        or bool((
                            module.output_weight.grad[row, columns]
                            - canonical_module.output_weight.grad[row, columns]
                        ).count_nonzero())
                    )
        active_coefficient_gradient_by_layer.append(layer_nonzero)
        outside_active_output_gradient_exact_by_layer.append(torch.equal(
            module.output_weight.grad[~mask],
            canonical_module.output_weight.grad[~mask],
        ))
    backward_model.zero_grad(set_to_none=True)
    full_model.zero_grad(set_to_none=True)
    backward_module = backward_modules[0]
    full_module = full_modules[0]
    with torch.no_grad():
        backward_module.output_weight[1, 0] = 0.015625
        full_module.output_weight[1, 0] = 0.015625
    generator = torch.Generator(device=device).manual_seed(args.seed + 91)
    base_values = torch.randn(
        2, backward_module.kv_heads, 65, backward_module.head_dim,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    upstream = torch.randn(
        base_values.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    backward_values = base_values.detach().clone().requires_grad_(True)
    full_values = base_values.detach().clone().requires_grad_(True)
    backward_output = backward_module.apply_value_encoding(backward_values)
    full_output = full_module.apply_value_encoding(full_values)
    (backward_output.float() * upstream.float()).sum().backward()
    (full_output.float() * upstream.float()).sum().backward()
    nonzero_coefficient_probe = {
        "physical_coefficient": 0.015625,
        "backward_only_forward_bit_exact_input": torch.equal(
            backward_output, base_values
        ),
        "full_forward_differs_from_input": not torch.equal(full_output, base_values),
        "value_vjp_bit_exact_full": torch.equal(
            backward_values.grad, full_values.grad
        ),
        "output_weight_vjp_bit_exact_full": torch.equal(
            backward_module.output_weight.grad,
            full_module.output_weight.grad,
        ),
        "active_coefficient_vjp_nonzero": bool(
            backward_module.output_weight.grad[1, 0] != 0
        ),
    }
    gates = {
        "initial_state_bit_exact": backward_hash == full_hash == canonical_hash,
        "first_logits_bit_exact": (
            torch.equal(backward_logits, full_logits)
            and torch.equal(backward_logits, canonical_logits)
        ),
        "first_loss_bit_exact": (
            float(backward_loss) == float(full_loss) == float(canonical_loss)
        ),
        "first_full_model_gradients_bit_exact": gradient_exact,
        "all_gradients_finite": finite,
        "gradient_element_count_positive": values > 0,
        "end_to_end_active_coefficient_gradient_nonzero_all_layers": all(
            active_coefficient_gradient_by_layer
        ),
        "outside_active_output_gradient_matches_canonical_all_layers": all(
            outside_active_output_gradient_exact_by_layer
        ),
        "nonzero_coefficient_forward_vjp_probe_pass": all(
            value for key, value in nonzero_coefficient_probe.items()
            if key != "physical_coefficient"
        ),
    }
    result = {
        "state_sha256": {
            "backward_only_tve": backward_hash,
            "full_tve": full_hash,
            "canonical_baseline": canonical_hash,
        },
        "first_loss": {
            "backward_only_tve": float(backward_loss),
            "full_tve": float(full_loss),
            "canonical_baseline": float(canonical_loss),
        },
        "gradient_element_count": values,
        "active_coefficient_gradient_by_layer": active_coefficient_gradient_by_layer,
        "outside_active_output_gradient_exact_by_layer": (
            outside_active_output_gradient_exact_by_layer
        ),
        "nonzero_coefficient_probe": nonzero_coefficient_probe,
        "gates": gates,
        "pass": all(gates.values()),
    }
    del backward_model, full_model, canonical_model
    del backward_modules, full_modules, canonical_modules
    del backward_logits, full_logits, canonical_logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def save_export(
    modules: list[Any], arm: str, args: argparse.Namespace
) -> dict[str, Any]:
    args.export_dir.mkdir(parents=True, exist_ok=True)
    path = args.export_dir / f"seed-{args.seed}-{arm}-bf16-attention.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save({
        "seed": args.seed,
        "arm": arm,
        "forward_semantics": (
            "ordinary_canonical" if arm == "backward_only_tve" else "tve"
        ),
        "layers": [
            {
                key: (
                    value.detach().to(torch.bfloat16).cpu()
                    if isinstance(value, torch.Tensor) else value
                )
                for key, value in module.export_dense().items()
            }
            for module in modules
        ],
    }, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def train_arm(
    arm: str,
    args: argparse.Namespace,
    train_file: Any,
    validation_file: Any,
    device: torch.device,
) -> dict[str, Any]:
    reset_seed(args.seed)
    model, modules = build_model(device, arm)
    initial_state = state_hash(model)
    optimizer = pilot.optimizer_for(model, args.learning_rate, args.weight_decay)
    evaluations = {
        "0": pilot.evaluate(
            model, validation_file, args.eval_batches, args.eval_batch_size, device
        )
    }
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    durations = []
    norms = []
    for step in range(args.steps):
        model.train()
        multiplier = pilot.lr_multiplier(step, args.steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        started = time.perf_counter()
        accumulated = 0.0
        for micro in range(args.gradient_accumulation):
            inputs, targets = train_file.batch(
                step * args.gradient_accumulation + micro,
                args.micro_batch_size,
                device,
            )
            loss = pilot.causal_loss(model, inputs, targets)
            (loss / args.gradient_accumulation).backward()
            accumulated += float(loss.detach()) / args.gradient_accumulation
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip))
        if not math.isfinite(accumulated) or not math.isfinite(norm):
            raise RuntimeError(f"non-finite {arm} at step {step + 1}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        losses.append(accumulated)
        durations.append(time.perf_counter() - started)
        norms.append(norm)
        if step + 1 in args.eval_steps:
            evaluations[str(step + 1)] = pilot.evaluate(
                model, validation_file, args.eval_batches, args.eval_batch_size, device
            )
            print(json.dumps({
                "forward_mechanism_arm": arm,
                "step": step + 1,
                "train_loss": accumulated,
                "validation_loss": evaluations[str(step + 1)]["loss"],
                "gradient_norm": norm,
            }), flush=True)
    serving = pilot.coefficient_diagnostics(modules)
    bridge = (
        pilot.terminal_serving_bridge(modules)
        if arm == "full_tve"
        else {
            "forward_semantics": "ordinary_canonical",
            "tve_cache_encoding_required": False,
        }
    )
    artifact = save_export(modules, arm, args) if arm != "canonical_baseline" else None
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    state = model.state_dict()
    result = {
        "arm": arm,
        "initial_state_sha256": initial_state,
        "total_parameters": sum(value.numel() for value in parameters),
        "parameter_tensors": len(parameters),
        "model_buffer_values": sum(value.numel() for value in buffers),
        "model_buffer_tensors": len(buffers),
        "state_dict_values": sum(value.numel() for value in state.values()),
        "state_dict_bytes": sum(
            value.numel() * value.element_size() for value in state.values()
        ),
        "optimizer_state_bytes": pilot.optimizer_state_bytes(optimizer),
        "metadata_bits": 0,
        "evaluations": evaluations,
        "serving_diagnostics": serving,
        "terminal_serving_bridge": bridge,
        "terminal_artifact": artifact,
        "train": {
            "prediction_tokens": 9_994_240,
            "step_losses": losses,
            "elapsed_seconds": sum(durations),
            "tokens_per_second": 9_994_240 / sum(durations),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_gradient_norm": max(norms),
            "nonfinite": False,
        },
    }
    del optimizer, modules, model, parameters, buffers, state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def paired_interval(left: dict[str, Any], right: dict[str, Any], step: str) -> dict[str, Any]:
    return pilot.paired_loss_interval(left, right, step)


def decide(results: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    terminal = str(STEPS)
    canonical = results["canonical_baseline"]
    backward = results["backward_only_tve"]
    full = results["full_tve"]
    losses = {
        arm: results[arm]["evaluations"][terminal]["loss"] for arm in ARMS
    }
    full_gain = losses["canonical_baseline"] - losses["full_tve"]
    backward_gain = losses["canonical_baseline"] - losses["backward_only_tve"]
    forward_margin = losses["backward_only_tve"] - losses["full_tve"]
    capture = backward_gain / full_gain if full_gain > 0 else float("nan")
    full_relative_gain = full_gain / losses["canonical_baseline"]
    forward_relative_margin = forward_margin / losses["backward_only_tve"]
    full_below_backward_all_post = all(
        full["evaluations"][str(step)]["loss"]
        < backward["evaluations"][str(step)]["loss"]
        for step in EVAL_STEPS
    )
    forward_path_required = (
        full_gain > 0
        and full_relative_gain >= 0.0005
        and forward_relative_margin >= 0.0005
        and capture <= 0.50
        and full_below_backward_all_post
    )
    backward_signal_captures_gain = (
        full_relative_gain >= 0.0005
        and capture >= 0.80
        and forward_relative_margin < 0.00025
    )
    equal_fields = (
        "total_parameters", "parameter_tensors", "model_buffer_values",
        "model_buffer_tensors", "state_dict_values", "state_dict_bytes",
        "optimizer_state_bytes", "metadata_bits",
    )
    structural = all(
        results[arm][field] == canonical[field]
        for arm in ARMS for field in equal_fields
    ) and all(results[arm]["arm"] == arm for arm in ARMS) \
    and canonical["total_parameters"] == 37_758_336 and len({
        results[arm]["initial_state_sha256"] for arm in ARMS
    }) == 1 and all(
        results[arm]["evaluations"]["0"]["per_batch_loss"]
        == canonical["evaluations"]["0"]["per_batch_loss"]
        for arm in ARMS
    ) and full["terminal_serving_bridge"] == {
        "layers": 12, "all_layers_bit_exact": True, "max_abs": 0.0,
    } and backward["terminal_serving_bridge"] == {
        "forward_semantics": "ordinary_canonical",
        "tve_cache_encoding_required": False,
    } and all(
        results[arm]["serving_diagnostics"]["coefficient_bf16_nonzero_fraction"] > 0
        and results[arm]["serving_diagnostics"]["coefficient_bf16_abs_max"] <= 0.5
        and results[arm]["serving_diagnostics"]["bf16_export_coefficient_bit_exact"]
        for arm in ("backward_only_tve", "full_tve")
    )
    artifact_valid = True
    for arm in ("backward_only_tve", "full_tve"):
        artifact = results[arm]["terminal_artifact"]
        path = Path(artifact["path"])
        expected_path = (
            EXPORT_DIR / f"seed-{SEED}-{arm}-bf16-attention.pt"
        )
        artifact_valid = artifact_valid and (
            results[arm]["arm"] == arm
            and path == expected_path
            and path.exists() and path.stat().st_size == artifact["bytes"]
            and sha256_file(path) == artifact["sha256"]
        )
    screen_valid = probe["pass"] and structural and artifact_valid
    if not screen_valid:
        classification = "invalid_screen"
    elif forward_path_required:
        classification = "forward_path_required_for_gain_replication"
    elif backward_signal_captures_gain:
        classification = "backward_signal_captures_gain_this_seed"
    else:
        classification = "mixed_or_inconclusive"
    return {
        "terminal_losses": losses,
        "full_gain": full_gain,
        "backward_gain": backward_gain,
        "forward_margin": forward_margin,
        "backward_capture_fraction": capture,
        "relative_full_gain": full_relative_gain,
        "relative_forward_margin": forward_relative_margin,
        "full_below_backward_all_post_evaluations": full_below_backward_all_post,
        "paired_terminal_diagnostics": {
            "full_vs_canonical": paired_interval(full, canonical, terminal),
            "backward_vs_canonical": paired_interval(backward, canonical, terminal),
            "full_vs_backward": paired_interval(full, backward, terminal),
        },
        "classification": classification,
        "probe_pass": probe["pass"],
        "structural_equal": structural,
        "artifacts_valid": artifact_valid,
        "screen_valid": screen_valid,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    expected_exports = [
        args.export_dir / f"seed-{args.seed}-{arm}-bf16-attention.pt"
        for arm in ("backward_only_tve", "full_tve")
    ]
    if any(path.exists() for path in expected_exports):
        raise FileExistsError([str(path) for path in expected_exports if path.exists()])
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("H100 required")
    integrity = validate_integrity()
    protocol = validate_protocol(args)
    ledger = pilot.validate_data_ledger(
        args.data_manifest, args.train_file, args.validation_file, args.sequence_length
    )
    train_file = pilot.TokenFile(args.train_file, args.sequence_length)
    validation_file = pilot.TokenFile(args.validation_file, args.sequence_length)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    probe = first_backward_probe(train_file, args, device)
    if not probe["pass"]:
        raise RuntimeError(f"invalid backward-only implementation: {probe['gates']}")
    payload = {
        "schema": "triangular-value-encoding-forward-mechanism-screen-v1",
        "scope": "one-seed forward-path necessity versus backward-signal falsifier",
        "source_sha256": sha256_file(Path(__file__)),
        "formal_integrity": integrity,
        "experiment_protocol": protocol,
        "data_ledger": ledger,
        "device": torch.cuda.get_device_name(0),
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "triton": triton.__version__,
        },
        "first_backward_probe": probe,
        "arms": {},
        "decision": {"complete": False},
    }
    for arm in ARMS:
        print(json.dumps({"starting_forward_mechanism_arm": arm}), flush=True)
        payload["arms"][arm] = train_arm(
            arm, args, train_file, validation_file, device
        )
        pilot.write_payload(args.output, payload)
    payload["decision"] = decide(payload["arms"], probe)
    pilot.write_payload(args.output, payload)
    return payload


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    parser.add_argument("--data-manifest", type=Path, default=DATA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-steps", type=parse_ints, default=list(EVAL_STEPS))
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--formal", action="store_true", default=True)
    payload = run(parser.parse_args())
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
