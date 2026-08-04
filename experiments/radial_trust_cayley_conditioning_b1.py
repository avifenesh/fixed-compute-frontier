#!/usr/bin/env python3
"""Frozen CPU-only B1 conditioning census for the radial-trust Cayley tree."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import re
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.radial_trust_cayley_program_tree import (
    RadialTrustCayleyProgramTreeMLP,
)


PREREGISTRATION = Path(
    "results/radial-trust-cayley-conditioning-preregistration.md"
)
AUDIT = Path(
    "results/radial-trust-cayley-conditioning-preregistration-audit.md"
)
CANDIDATE_SOURCE = Path("experiments/radial_trust_cayley_program_tree.py")
BASE_SOURCE = Path("experiments/cayley_program_tree.py")
TEST_SOURCE = Path("tests/test_radial_trust_cayley_conditioning_b1.py")
MANIFEST = Path("results/radial-trust-cayley-conditioning-b1-manifest.json")
IMPLEMENTATION_AUDIT = Path(
    "results/radial-trust-cayley-conditioning-b1-implementation-audit.md"
)

PREREGISTRATION_SHA256 = (
    "5519f3bbb9622a28e48acf8f25d9a7bf7b629b1c8706669b213830c7b76a9bc7"
)
SEEDS = (815, 6703, 6997, 7307, 7949)
MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
SPARSE_SUPPORTS = (1, 2, 4, 16, 64, 384)
RADII = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    6.8966100057915725,
    8.0,
    math.sqrt(127.0),
    16.0,
    64.0,
)
EPSILON_BF16 = 2.0**-7
R_AMP = 6.8966100057915725
R_GRAD = math.sqrt(127.0)
WIDTH = 384
DEPTH = 9

FAMILY_CODES = {
    "gaussian": 0,
    "rademacher": 1,
    "sparse_k1": 2,
    "sparse_k2": 3,
    "sparse_k4": 4,
    "sparse_k16": 5,
    "sparse_k64": 6,
    "sparse_k384": 7,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(values: torch.Tensor) -> torch.Tensor:
    return values.double().square().mean(dim=-1).sqrt()


def normalize_rms_one(values: torch.Tensor) -> torch.Tensor:
    return values.double() / rms(values).unsqueeze(-1)


def radial_fp64(values: torch.Tensor) -> torch.Tensor:
    values = values.double()
    return values / torch.sqrt(1.0 + values.square().mean(dim=-1, keepdim=True))


def radial_bf16(values: torch.Tensor) -> torch.Tensor:
    source = values.to(torch.bfloat16)
    denominator = torch.sqrt(1.0 + source.float().square().mean(dim=-1, keepdim=True))
    return source / denominator.to(torch.bfloat16)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval needs a positive sample count")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - radius, center + radius


def resource_ceiling_ok(
    elapsed_seconds: float,
    peak_rss_gib: float,
    json_bytes: int,
    arrays_bytes: int,
) -> bool:
    return bool(
        elapsed_seconds <= 1800.0
        and peak_rss_gib <= 8.0
        and json_bytes + arrays_bytes <= 256 * 1024 * 1024
    )


def stable_receipt_text(result: dict[str, Any], arrays_bytes: int) -> str:
    """Materialize exact self-reported JSON and combined byte counts."""
    for _ in range(16):
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        json_bytes = len(text.encode("utf-8"))
        combined_bytes = arrays_bytes + json_bytes
        resource_row = result["resource"]
        if (
            resource_row.get("json_bytes") == json_bytes
            and resource_row.get("combined_bytes") == combined_bytes
        ):
            return text
        resource_row["json_bytes"] = json_bytes
        resource_row["combined_bytes"] = combined_bytes
    raise RuntimeError("receipt byte counts did not reach a fixed point")


def verify_frozen_manifest() -> tuple[bool, dict[str, Any]]:
    try:
        manifest = json.loads(MANIFEST.read_text())
        manifest_sha256 = sha256_file(MANIFEST)
        audit_text = IMPLEMENTATION_AUDIT.read_text()
        match = re.search(
            r"Audited manifest SHA-256:\s*`([0-9a-f]{64})`", audit_text
        )
        audit_points_to_manifest = bool(
            match and match.group(1) == manifest_sha256
        )
        expected_paths = {
            "preregistration": PREREGISTRATION,
            "paper_audit": AUDIT,
            "executable": Path(__file__),
            "candidate_source": CANDIDATE_SOURCE,
            "base_source": BASE_SOURCE,
            "tests": TEST_SOURCE,
        }
        observed = {
            name: sha256_file(path) for name, path in expected_paths.items()
        }
        expected = manifest.get("sha256", {})
        files_match = observed == expected
        return bool(audit_points_to_manifest and files_match), {
            "manifest_sha256": manifest_sha256,
            "implementation_audit_sha256": sha256_file(IMPLEMENTATION_AUDIT),
            "audit_points_to_manifest": audit_points_to_manifest,
            "files_match": files_match,
            "expected": expected,
            "observed": observed,
        }
    except Exception as error:
        return False, {
            "error_type": type(error).__name__,
            "error": str(error),
        }


def make_inputs(seed: int, width: int = WIDTH) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    gaussian = normalize_rms_one(
        torch.randn(256, width, dtype=torch.float64, generator=generator)
    )
    rademacher = (
        2
        * torch.randint(
            0, 2, (256, width), dtype=torch.int64, generator=generator
        )
        - 1
    ).double()
    rows: dict[str, torch.Tensor] = {
        "gaussian": gaussian.float(),
        "rademacher": rademacher.float(),
    }
    for support in SPARSE_SUPPORTS:
        values = torch.zeros(64, width, dtype=torch.float64)
        magnitude = math.sqrt(width / support)
        for row in range(64):
            positions = torch.randperm(width, generator=generator)[:support]
            signs = (
                2
                * torch.randint(
                    0, 2, (support,), dtype=torch.int64, generator=generator
                )
                - 1
            ).double()
            values[row, positions] = magnitude * signs
        rows[f"sparse_k{support}"] = values.float()
    return rows


def all_forced_bits(depth: int = DEPTH) -> torch.Tensor:
    codes = torch.arange(2**depth, dtype=torch.long)
    shifts = torch.arange(depth - 1, -1, -1, dtype=torch.long)
    return torch.bitwise_and(codes[:, None] >> shifts[None], 1)


@torch.no_grad()
def trace_tree(
    module: RadialTrustCayleyProgramTreeMLP,
    inputs: torch.Tensor,
    multiplier: float,
    *,
    forced_bits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], bool]:
    if inputs.ndim != 2 or inputs.shape[1] != module.width:
        raise ValueError("inputs must be [tokens, module.width]")
    if forced_bits is not None and forced_bits.shape != (inputs.shape[0], module.depth):
        raise ValueError("forced_bits must be [tokens, module.depth]")

    initial = inputs
    hidden = initial
    node = torch.zeros(inputs.shape[0], dtype=torch.long)
    scale = 1.0 / math.sqrt(module.depth)
    rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    bit_history: list[torch.Tensor] = []
    node_reconstruction_ok = True

    for depth_index in range(module.depth):
        if depth_index == 0:
            expected_node = torch.zeros_like(node)
        else:
            prefix = torch.stack(bit_history, dim=1)
            powers = 2 ** torch.arange(
                depth_index - 1, -1, -1, dtype=torch.long
            )
            expected_node = (2**depth_index - 1) + (prefix * powers).sum(dim=1)
        node_reconstruction_ok &= bool(torch.equal(node, expected_node))
        projected = module.bases.apply(hidden, depth_index)
        if forced_bits is None:
            selector, hard_bit, _ = module._route(projected, node, depth_index)
            expected_coordinate = torch.remainder(
                node * module._ROUTE_MULTIPLIER
                + module.route_seed
                + depth_index * module._ROUTE_DEPTH_STRIDE,
                module.width,
            )
            expected_logits = projected.gather(
                1, expected_coordinate[:, None]
            ).squeeze(1) - module.threshold[node]
            expected_hard = (
                torch.sigmoid(expected_logits.float()).to(projected.dtype) >= 0.5
            ).long()
            node_reconstruction_ok &= bool(torch.equal(hard_bit, expected_hard))
            choices = module.payload[node]
            selected = (
                choices[:, 0] * (1.0 - selector[:, None, None])
                + choices[:, 1] * selector[:, None, None]
            )
        else:
            hard_bit = forced_bits[:, depth_index]
            selected = module.payload[node, hard_bit]
        gate, up, down = selected.unbind(dim=1)
        activation = F.silu(gate * projected) * (up * projected)
        local_delta = down * activation
        unscaled_raw = module.bases.apply(
            local_delta, depth_index, transpose=True
        )
        raw = float(multiplier) * unscaled_raw
        trusted = module.radial_trust_region(raw)
        trusted64 = radial_fp64(raw)
        trusted_b = radial_bf16(raw)
        doubled_b = radial_bf16(2.0 * raw)
        radius = rms(raw)
        c = torch.rsqrt(1.0 + radius.square())

        rows["r"].append(radius)
        rows["c"].append(c)
        rows["c3"].append(c.pow(3))
        rows["kappa"].append(1.0 + radius.square())
        rows["raw_rms"].append(radius)
        rows["trusted_rms"].append(rms(trusted))
        rows["trusted_fp64_rms"].append(rms(trusted64))
        rows["trusted_bf16_rms"].append(rms(trusted_b.float()))
        rows["raw_abs_max"].append(raw.double().abs().amax(dim=-1))
        rows["bf16_double_equal_fraction"].append(
            (trusted_b == doubled_b).double().mean(dim=-1)
        )
        rows["node"].append(node.clone())
        rows["bit"].append(hard_bit.clone())
        rows["finite"].append(
            torch.isfinite(raw).all(dim=-1)
            & torch.isfinite(trusted).all(dim=-1)
            & torch.isfinite(trusted_b).all(dim=-1)
        )

        hidden = hidden + scale * trusted
        bit_history.append(hard_bit.clone())
        node = 2 * node + 1 + hard_bit

    trace = {key: torch.stack(value, dim=1) for key, value in rows.items()}
    trace["hidden_finite"] = torch.isfinite(hidden).all(dim=-1)
    return hidden - initial, trace, node_reconstruction_ok


def summarize_trace(trace: dict[str, torch.Tensor], natural: bool) -> dict[str, Any]:
    amp_path = (trace["r"] >= R_AMP).any(dim=1)
    grad_path = (trace["r"] >= R_GRAD).any(dim=1)
    r_flat = trace["r"].flatten().numpy()
    summary: dict[str, Any] = {
        "paths": int(trace["r"].shape[0]),
        "edges": int(trace["r"].numel()),
        "r": {
            "median": float(np.quantile(r_flat, 0.5)),
            "p90": float(np.quantile(r_flat, 0.9)),
            "p95": float(np.quantile(r_flat, 0.95)),
            "p99": float(np.quantile(r_flat, 0.99)),
            "maximum": float(r_flat.max()),
        },
        "path_maximum_r": {
            "median": float(np.quantile(trace["r"].amax(dim=1).numpy(), 0.5)),
            "p95": float(np.quantile(trace["r"].amax(dim=1).numpy(), 0.95)),
            "maximum": float(trace["r"].max()),
        },
        "F_amp": float(amp_path.double().mean()),
        "F_grad": float(grad_path.double().mean()),
        "minimum_tangential_gain": float(trace["c"].min()),
        "minimum_radial_gain": float(trace["c3"].min()),
        "maximum_condition_number": float(trace["kappa"].max()),
        "maximum_trusted_fp64_rms": float(trace["trusted_fp64_rms"].max()),
        "maximum_trusted_bf16_rms": float(trace["trusted_bf16_rms"].max()),
        "minimum_bf16_double_equal_fraction": float(
            trace["bf16_double_equal_fraction"].min()
        ),
        "maximum_bf16_double_equal_fraction": float(
            trace["bf16_double_equal_fraction"].max()
        ),
        "all_finite": bool(trace["finite"].all() and trace["hidden_finite"].all()),
        "depth": {},
    }
    for depth_index in range(trace["r"].shape[1]):
        depth_r = trace["r"][:, depth_index].numpy()
        summary["depth"][str(depth_index)] = {
            "count": int(depth_r.size),
            "median": float(np.quantile(depth_r, 0.5)),
            "p90": float(np.quantile(depth_r, 0.9)),
            "p95": float(np.quantile(depth_r, 0.95)),
            "p99": float(np.quantile(depth_r, 0.99)),
            "maximum": float(depth_r.max()),
            "minimum_tangential_gain": float(trace["c"][:, depth_index].min()),
            "minimum_radial_gain": float(trace["c3"][:, depth_index].min()),
            "maximum_condition_number": float(
                trace["kappa"][:, depth_index].max()
            ),
            "ecdf_sorted_r": np.sort(depth_r).tolist(),
        }
    if natural:
        for name, events in (("amp", amp_path), ("grad", grad_path)):
            successes = int(events.sum())
            low, high = wilson_interval(successes, events.numel())
            summary[f"F_{name}_wilson95"] = [low, high]
    return summary


def append_trace_arrays(
    buckets: dict[str, list[np.ndarray]],
    trace: dict[str, torch.Tensor],
    *,
    seed_index: int,
    family: str,
    multiplier: float,
    route_mode: int,
    input_indices: torch.Tensor,
    path_codes: torch.Tensor,
) -> None:
    tokens, depth = trace["r"].shape
    edge_count = tokens * depth
    buckets["edge_seed_index"].append(np.full(edge_count, seed_index, np.int16))
    buckets["edge_family"].append(
        np.full(edge_count, FAMILY_CODES[family], np.int8)
    )
    buckets["edge_multiplier"].append(np.full(edge_count, multiplier, np.float32))
    buckets["edge_route_mode"].append(np.full(edge_count, route_mode, np.int8))
    buckets["edge_input_index"].append(
        input_indices[:, None].expand(tokens, depth).reshape(-1).numpy().astype(np.int32)
    )
    buckets["edge_path_code"].append(
        path_codes[:, None].expand(tokens, depth).reshape(-1).numpy().astype(np.int16)
    )
    buckets["edge_depth"].append(
        torch.arange(depth)[None].expand(tokens, depth).reshape(-1).numpy().astype(np.int8)
    )
    for key in (
        "r",
        "c",
        "c3",
        "kappa",
        "raw_rms",
        "trusted_rms",
        "trusted_fp64_rms",
        "trusted_bf16_rms",
        "raw_abs_max",
        "bf16_double_equal_fraction",
    ):
        buckets[f"edge_{key}"].append(trace[key].reshape(-1).numpy())
    buckets["edge_node"].append(trace["node"].reshape(-1).numpy().astype(np.int32))
    buckets["edge_bit"].append(trace["bit"].reshape(-1).numpy().astype(np.int8))
    buckets["edge_amp_event"].append((trace["r"] >= R_AMP).reshape(-1).numpy())
    buckets["edge_grad_event"].append((trace["r"] >= R_GRAD).reshape(-1).numpy())
    buckets["edge_finite"].append(trace["finite"].reshape(-1).numpy())

    buckets["path_seed_index"].append(np.full(tokens, seed_index, np.int16))
    buckets["path_family"].append(np.full(tokens, FAMILY_CODES[family], np.int8))
    buckets["path_multiplier"].append(np.full(tokens, multiplier, np.float32))
    buckets["path_route_mode"].append(np.full(tokens, route_mode, np.int8))
    buckets["path_input_index"].append(input_indices.numpy().astype(np.int32))
    buckets["path_code"].append(path_codes.numpy().astype(np.int16))
    buckets["path_max_r"].append(trace["r"].amax(dim=1).numpy())
    buckets["path_min_c"].append(trace["c"].amin(dim=1).numpy())
    buckets["path_min_c3"].append(trace["c3"].amin(dim=1).numpy())
    buckets["path_max_kappa"].append(trace["kappa"].amax(dim=1).numpy())
    buckets["path_amp_event"].append((trace["r"] >= R_AMP).any(dim=1).numpy())
    buckets["path_grad_event"].append((trace["r"] >= R_GRAD).any(dim=1).numpy())


def exact_small_jacobians() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    direction_generator = torch.Generator(device="cpu").manual_seed(4_000_815)
    probe_generator = torch.Generator(device="cpu").manual_seed(5_000_815)
    directions = (
        2
        * torch.randint(
            0, 2, (16, 8), dtype=torch.int64, generator=direction_generator
        )
        - 1
    ).double()

    radii_rows = []
    direction_rows = []
    singular_rows = []
    analytic_rows = []
    max_abs_error = 0.0
    max_relative_error = 0.0
    max_jvp_relative_error = 0.0
    max_vjp_relative_error = 0.0
    probe_count = 0
    all_values_finite = True

    def function(value: torch.Tensor) -> torch.Tensor:
        return radial_fp64(value[None])[0]

    for radius in RADII:
        c = 1.0 / math.sqrt(1.0 + radius * radius)
        expected_singular = torch.tensor(
            sorted([c**3] + [c] * 7, reverse=True), dtype=torch.float64
        )
        for direction_index, direction in enumerate(directions):
            point = radius * direction
            jacobian = torch.autograd.functional.jacobian(function, point)
            singular = torch.linalg.svdvals(jacobian)
            absolute = (singular - expected_singular).abs()
            relative = absolute / expected_singular
            all_values_finite &= bool(
                torch.isfinite(jacobian).all()
                and torch.isfinite(singular).all()
                and torch.isfinite(expected_singular).all()
                and torch.isfinite(absolute).all()
                and torch.isfinite(relative).all()
            )
            max_abs_error = max(max_abs_error, float(absolute.max()))
            max_relative_error = max(max_relative_error, float(relative.max()))
            radii_rows.append(radius)
            direction_rows.append(direction_index)
            singular_rows.append(singular.numpy())
            analytic_rows.append(expected_singular.numpy())

            for _ in range(16):
                probe = (
                    2
                    * torch.randint(
                        0,
                        2,
                        (8,),
                        dtype=torch.int64,
                        generator=probe_generator,
                    )
                    - 1
                ).double()
                radial_unit = direction / torch.linalg.vector_norm(direction)
                analytic = c * probe + (c**3 - c) * radial_unit * torch.dot(
                    radial_unit, probe
                )
                _, jvp = torch.autograd.functional.jvp(function, point, probe)
                _, vjp = torch.autograd.functional.vjp(function, point, probe)
                all_values_finite &= bool(
                    torch.isfinite(probe).all()
                    and torch.isfinite(analytic).all()
                    and torch.isfinite(jvp).all()
                    and torch.isfinite(vjp).all()
                )
                denominator = max(float(torch.linalg.vector_norm(analytic)), 1e-30)
                max_jvp_relative_error = max(
                    max_jvp_relative_error,
                    float(torch.linalg.vector_norm(jvp - analytic)) / denominator,
                )
                max_vjp_relative_error = max(
                    max_vjp_relative_error,
                    float(torch.linalg.vector_norm(vjp - analytic)) / denominator,
                )
                probe_count += 1

    result = {
        "direction_cells": len(radii_rows),
        "jvp_probes": probe_count,
        "vjp_probes": probe_count,
        "maximum_singular_absolute_error": max_abs_error,
        "maximum_singular_relative_error": max_relative_error,
        "maximum_jvp_relative_error": max_jvp_relative_error,
        "maximum_vjp_relative_error": max_vjp_relative_error,
        "all_values_finite": all_values_finite,
    }
    arrays = {
        "d8_radius": np.asarray(radii_rows, dtype=np.float64),
        "d8_direction_index": np.asarray(direction_rows, dtype=np.int16),
        "d8_singular_values": np.asarray(singular_rows, dtype=np.float64),
        "d8_analytic_singular_values": np.asarray(analytic_rows, dtype=np.float64),
    }
    return result, arrays


def target_shape_reference() -> tuple[dict[str, Any], dict[str, np.ndarray], bool]:
    rows: dict[str, list[Any]] = defaultdict(list)
    deterministic_bf16 = True
    maximum_abs_error = 0.0
    maximum_relative_error = 0.0
    maximum_bf16_rms = 0.0
    maximum_fp64_rms = 0.0
    all_values_finite = True

    for seed_index, seed in enumerate(SEEDS):
        generator = torch.Generator(device="cpu").manual_seed(seed + 3_000_003)
        gaussian = normalize_rms_one(
            torch.randn(256, 4096, dtype=torch.float64, generator=generator)
        )
        rademacher = (
            2
            * torch.randint(
                0, 2, (256, 4096), dtype=torch.int64, generator=generator
            )
            - 1
        ).double()
        for family_code, directions in enumerate((gaussian, rademacher)):
            for radius_index, radius in enumerate(RADII):
                raw = radius * directions
                actual = radial_fp64(raw)
                c = 1.0 / math.sqrt(1.0 + radius * radius)
                expected = c * raw
                absolute = (actual - expected).abs().amax(dim=1)
                denominator = expected.abs().amax(dim=1).clamp_min(1e-30)
                relative = absolute / denominator
                trusted_b = radial_bf16(raw)
                trusted_b_again = radial_bf16(raw)
                deterministic_bf16 &= bool(torch.equal(trusted_b, trusted_b_again))
                doubled_b = radial_bf16(2.0 * raw)
                bf16_rms = rms(trusted_b.float())
                equality = (trusted_b == doubled_b).double().mean(dim=1)

                all_values_finite &= bool(
                    torch.isfinite(raw).all()
                    and torch.isfinite(actual).all()
                    and torch.isfinite(expected).all()
                    and torch.isfinite(absolute).all()
                    and torch.isfinite(relative).all()
                    and torch.isfinite(trusted_b).all()
                    and torch.isfinite(bf16_rms).all()
                )

                maximum_abs_error = max(maximum_abs_error, float(absolute.max()))
                maximum_relative_error = max(
                    maximum_relative_error, float(relative.max())
                )
                maximum_bf16_rms = max(maximum_bf16_rms, float(bf16_rms.max()))
                maximum_fp64_rms = max(maximum_fp64_rms, float(rms(actual).max()))
                count = directions.shape[0]
                rows["d4096_seed_index"].extend([seed_index] * count)
                rows["d4096_family"].extend([family_code] * count)
                rows["d4096_radius_index"].extend([radius_index] * count)
                rows["d4096_absolute_error"].extend(absolute.numpy())
                rows["d4096_relative_error"].extend(relative.numpy())
                rows["d4096_bf16_rms"].extend(bf16_rms.numpy())
                rows["d4096_bf16_double_equal_fraction"].extend(equality.numpy())

    arrays = {
        key: np.asarray(
            values,
            dtype=(
                np.int16
                if key.endswith("seed_index") or key.endswith("radius_index")
                else np.int8
                if key.endswith("family")
                else np.float64
            ),
        )
        for key, values in rows.items()
    }
    result = {
        "rows": len(rows["d4096_seed_index"]),
        "maximum_absolute_error": maximum_abs_error,
        "maximum_relative_error": maximum_relative_error,
        "maximum_bf16_rms": maximum_bf16_rms,
        "maximum_fp64_rms": maximum_fp64_rms,
        "deterministic_bf16": deterministic_bf16,
        "all_values_finite": all_values_finite,
    }
    return result, arrays, deterministic_bf16


def concatenate_buckets(buckets: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(values, axis=0) for key, values in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/radial-trust-cayley-conditioning-b1.json"),
    )
    parser.add_argument(
        "--arrays",
        type=Path,
        default=Path("results/radial-trust-cayley-conditioning-b1-arrays.npz"),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if args.output.exists() or args.arrays.exists():
        raise FileExistsError("B1 outputs already exist; the frozen run is one-shot")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run with CUDA_VISIBLE_DEVICES='' for the frozen CPU-only protocol")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible in a CPU-only B1 process")
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration hash mismatch")
    protocol_integrity, manifest_report = verify_frozen_manifest()
    if not protocol_integrity:
        receipt = {
            "schema": "radial-trust-cayley-conditioning-b1-v1",
            "outcome": "inconclusive",
            "reason": "protocol_integrity_failed_before_census",
            "manifest_verification": manifest_report,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    small_result, small_arrays = exact_small_jacobians()
    target_result, target_arrays, deterministic_bf16 = target_shape_reference()

    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    natural_summaries: dict[str, Any] = {}
    forced_summaries: dict[str, Any] = {}
    node_reconstruction_ok = True
    instrumentation_output_match = True
    all_finite = True
    maximum_trusted_fp64_rms = target_result["maximum_fp64_rms"]
    maximum_trusted_bf16_rms = target_result["maximum_bf16_rms"]

    forced_codes = all_forced_bits()
    for seed_index, seed in enumerate(SEEDS):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        module = RadialTrustCayleyProgramTreeMLP(
            width=WIDTH, depth=DEPTH, trust_rms=1.0, seed=seed
        )
        module.train()
        inputs = make_inputs(seed)
        natural_summaries[str(seed)] = {}
        forced_summaries[str(seed)] = {}

        for family, values in inputs.items():
            natural_summaries[str(seed)][family] = {}
            for multiplier in MULTIPLIERS:
                output, trace, nodes_ok = trace_tree(module, values, multiplier)
                node_reconstruction_ok &= nodes_ok
                summary = summarize_trace(trace, natural=True)
                natural_summaries[str(seed)][family][str(multiplier)] = summary
                all_finite &= summary["all_finite"]
                maximum_trusted_fp64_rms = max(
                    maximum_trusted_fp64_rms,
                    summary["maximum_trusted_fp64_rms"],
                )
                maximum_trusted_bf16_rms = max(
                    maximum_trusted_bf16_rms,
                    summary["maximum_trusted_bf16_rms"],
                )
                if multiplier == 1.0:
                    with torch.no_grad():
                        reference = module(values)
                    instrumentation_output_match &= bool(torch.equal(output, reference))
                append_trace_arrays(
                    buckets,
                    trace,
                    seed_index=seed_index,
                    family=family,
                    multiplier=multiplier,
                    route_mode=0,
                    input_indices=torch.arange(values.shape[0]),
                    path_codes=torch.full((values.shape[0],), -1, dtype=torch.long),
                )

        for family in ("gaussian", "rademacher"):
            base = inputs[family][:8]
            repeated = base.repeat_interleave(2**DEPTH, dim=0)
            bits = forced_codes.repeat(8, 1)
            output, trace, nodes_ok = trace_tree(
                module, repeated, 1.0, forced_bits=bits
            )
            node_reconstruction_ok &= nodes_ok
            all_finite &= bool(trace["finite"].all() and trace["hidden_finite"].all())
            maximum_trusted_fp64_rms = max(
                maximum_trusted_fp64_rms, float(trace["trusted_fp64_rms"].max())
            )
            maximum_trusted_bf16_rms = max(
                maximum_trusted_bf16_rms, float(trace["trusted_bf16_rms"].max())
            )
            per_input = []
            for input_index in range(8):
                start = input_index * 2**DEPTH
                end = start + 2**DEPTH
                subset = {
                    key: value[start:end]
                    for key, value in trace.items()
                }
                per_input.append(summarize_trace(subset, natural=False))
            forced_summaries[str(seed)][family] = {"per_input": per_input}

            for path_code in (0, 2**DEPTH - 1):
                module.force_bits = tuple(int(v) for v in forced_codes[path_code])
                with torch.no_grad():
                    reference = module(base)
                selected = output[
                    torch.arange(8) * (2**DEPTH) + path_code
                ]
                instrumentation_output_match &= bool(torch.equal(selected, reference))
            module.force_bits = None

            append_trace_arrays(
                buckets,
                trace,
                seed_index=seed_index,
                family=family,
                multiplier=1.0,
                route_mode=1,
                input_indices=torch.arange(8).repeat_interleave(2**DEPTH),
                path_codes=torch.arange(2**DEPTH).repeat(8),
            )

    arrays = concatenate_buckets(buckets)
    arrays.update(small_arrays)
    arrays.update(target_arrays)

    expected_edge_rows = 570_240
    expected_path_rows = 63_360
    expected_d4096_rows = 28_160
    expected_d8_cells = 176
    expected_probe_rows = 2_816
    actual_counts = {
        "edge_rows": int(arrays["edge_r"].shape[0]),
        "path_rows": int(arrays["path_max_r"].shape[0]),
        "d4096_rows": int(arrays["d4096_seed_index"].shape[0]),
        "d8_direction_cells": int(arrays["d8_radius"].shape[0]),
        "d8_jvp_probes": small_result["jvp_probes"],
        "d8_vjp_probes": small_result["vjp_probes"],
    }
    count_gate = actual_counts == {
        "edge_rows": expected_edge_rows,
        "path_rows": expected_path_rows,
        "d4096_rows": expected_d4096_rows,
        "d8_direction_cells": expected_d8_cells,
        "d8_jvp_probes": expected_probe_rows,
        "d8_vjp_probes": expected_probe_rows,
    }

    ordinary_natural_gate = all(
        natural_summaries[str(seed)][family]["1.0"]["F_amp"] < 0.5
        and natural_summaries[str(seed)][family]["1.0"]["F_grad"] < 0.5
        for seed in SEEDS
        for family in ("gaussian", "rademacher")
    )
    ordinary_forced_gate = all(
        row["F_amp"] < 0.5 and row["F_grad"] < 0.5
        for seed in SEEDS
        for family in ("gaussian", "rademacher")
        for row in forced_summaries[str(seed)][family]["per_input"]
    )

    validity_gates = {
        "protocol_integrity": protocol_integrity,
        "small_jacobian_algebra": bool(
            small_result["all_values_finite"]
            and
            small_result["maximum_singular_absolute_error"] <= 1e-11
            and small_result["maximum_singular_relative_error"] <= 1e-9
            and small_result["maximum_jvp_relative_error"] <= 1e-9
            and small_result["maximum_vjp_relative_error"] <= 1e-9
        ),
        "deterministic_target_shape_numeric_reference": bool(
            target_result["maximum_absolute_error"] <= 1e-11
            and target_result["maximum_relative_error"] <= 1e-9
            and deterministic_bf16
            and target_result["all_values_finite"]
        ),
        "instrumentation_accounting": bool(
            count_gate and node_reconstruction_ok and instrumentation_output_match
        ),
    }
    scientific_gates = {
        "numerical_stability": bool(
            all_finite
            and maximum_trusted_fp64_rms < 1.0
            and maximum_trusted_bf16_rms <= 1.0 + EPSILON_BF16
        ),
        "ordinary_natural_paths": ordinary_natural_gate,
        "ordinary_forced_paths": ordinary_forced_gate,
    }
    validity_pass = all(validity_gates.values())
    if not validity_pass:
        outcome = "inconclusive"
    elif all(scientific_gates.values()):
        outcome = "pass"
    else:
        outcome = "fail"

    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    arrays_sha256 = sha256_file(args.arrays)

    result = {
        "schema": "radial-trust-cayley-conditioning-b1-v1",
        "outcome": outcome,
        "constants": {
            "seeds": SEEDS,
            "multipliers": MULTIPLIERS,
            "sparse_supports": SPARSE_SUPPORTS,
            "radii": RADII,
            "epsilon_bf16": EPSILON_BF16,
            "r_amp": R_AMP,
            "r_grad": R_GRAD,
            "width": WIDTH,
            "depth": DEPTH,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
        },
        "hashes": {
            "preregistration": sha256_file(PREREGISTRATION),
            "audit": sha256_file(AUDIT),
            "executable": sha256_file(Path(__file__)),
            "candidate_source": sha256_file(CANDIDATE_SOURCE),
            "base_source": sha256_file(BASE_SOURCE),
            "tests": sha256_file(TEST_SOURCE),
            "arrays": arrays_sha256,
            "manifest": sha256_file(MANIFEST),
            "implementation_audit": sha256_file(IMPLEMENTATION_AUDIT),
        },
        "manifest_verification": manifest_report,
        "resource": {},
        "counts": actual_counts,
        "small_jacobians": small_result,
        "target_shape_reference": target_result,
        "natural": natural_summaries,
        "forced": forced_summaries,
        "global_measurements": {
            "all_finite": all_finite,
            "maximum_trusted_fp64_rms": maximum_trusted_fp64_rms,
            "maximum_trusted_bf16_rms": maximum_trusted_bf16_rms,
            "node_reconstruction_ok": node_reconstruction_ok,
            "instrumentation_output_match": instrumentation_output_match,
        },
        "validity_gates": validity_gates,
        "scientific_gates": scientific_gates,
    }
    # Materialize the complete ECDF-heavy receipt before measuring the frozen
    # wall/RSS/file ceilings.  Five seconds are reserved for the single final
    # write.  The exact persisted artifact is checked after that write and is
    # never mutated if it passes.
    _ = json.dumps(result, indent=2, sort_keys=True) + "\n"
    elapsed_after_materialization = time.perf_counter() - started
    rss_after_materialization = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    )
    arrays_bytes = args.arrays.stat().st_size
    result["resource"] = {
        "elapsed_after_materialization_seconds": elapsed_after_materialization,
        "reserved_final_write_seconds": 5.0,
        "peak_rss_after_materialization_gib": rss_after_materialization,
        "arrays_bytes": arrays_bytes,
        "combined_limit_bytes": 256 * 1024 * 1024,
        "post_write_policy": "failure_replaces_with_terminal_inconclusive_receipt",
    }
    full_text = stable_receipt_text(result, arrays_bytes)
    prewrite_json_bytes = len(full_text.encode("utf-8"))
    prewrite_ceiling_valid = resource_ceiling_ok(
        elapsed_after_materialization + 5.0,
        rss_after_materialization,
        prewrite_json_bytes,
        arrays_bytes,
    )
    validity_gates["runtime_resource_ceiling"] = prewrite_ceiling_valid
    if not prewrite_ceiling_valid:
        outcome = "inconclusive"
    result["outcome"] = outcome
    result["validity_gates"] = validity_gates
    full_text = stable_receipt_text(result, arrays_bytes)
    args.output.write_text(full_text)

    final_elapsed = time.perf_counter() - started
    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    final_ceiling_valid = resource_ceiling_ok(
        final_elapsed,
        final_rss,
        args.output.stat().st_size,
        args.arrays.stat().st_size,
    )
    if not final_ceiling_valid:
        receipt = {
            "schema": "radial-trust-cayley-conditioning-b1-v1",
            "outcome": "inconclusive",
            "reason": "final_persisted_resource_ceiling_failed",
            "invalid_full_receipt_sha256": sha256_file(args.output),
            "arrays_sha256": arrays_sha256,
            "final_resource_measurement": {
                "elapsed_seconds": final_elapsed,
                "peak_rss_gib": final_rss,
                "json_bytes": args.output.stat().st_size,
                "arrays_bytes": args.arrays.stat().st_size,
                "combined_bytes": args.output.stat().st_size
                + args.arrays.stat().st_size,
            },
        }
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return

    print(json.dumps({
        "outcome": outcome,
        "validity_gates": validity_gates,
        "scientific_gates": scientific_gates,
        "resource": {
            **result["resource"],
            "post_write_elapsed_seconds": final_elapsed,
            "post_write_peak_rss_gib": final_rss,
            "post_write_json_bytes": args.output.stat().st_size,
        },
        "global_measurements": result["global_measurements"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
