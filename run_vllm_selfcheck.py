"""Run exact-token streaming smoke cells against a local vLLM server."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


CELLS = [
    {"name": "short-interactive-c1", "input_tokens": 128, "output_tokens": 128, "concurrency": 1},
    {"name": "short-interactive-c8", "input_tokens": 128, "output_tokens": 128, "concurrency": 8},
    {"name": "normal-chat-c1", "input_tokens": 512, "output_tokens": 128, "concurrency": 1},
    {"name": "normal-chat-c8", "input_tokens": 512, "output_tokens": 128, "concurrency": 8},
    {"name": "prefill-heavy-c1", "input_tokens": 2048, "output_tokens": 128, "concurrency": 1},
    {"name": "prefill-heavy-c8", "input_tokens": 2048, "output_tokens": 128, "concurrency": 8},
    {"name": "decode-heavy-c1", "input_tokens": 512, "output_tokens": 512, "concurrency": 1},
    {"name": "decode-heavy-c8", "input_tokens": 512, "output_tokens": 512, "concurrency": 8},
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def request_once(url: str, model: str, cell: dict[str, Any], prompt_token_id: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": [prompt_token_id] * cell["input_tokens"],
        "max_tokens": cell["output_tokens"],
        "min_tokens": cell["output_tokens"],
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    token_times: list[float] = []
    usage: dict[str, Any] | None = None
    with requests.post(url, json=payload, stream=True, timeout=180) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:]
            if data == b"[DONE]":
                continue
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                token_times.append(time.perf_counter())
    finished = time.perf_counter()
    if usage is None:
        raise RuntimeError("stream ended without usage")
    if usage["prompt_tokens"] != cell["input_tokens"]:
        raise RuntimeError(f"prompt-token mismatch: {usage}")
    if usage["completion_tokens"] != cell["output_tokens"]:
        raise RuntimeError(f"completion-token mismatch: {usage}")
    if not token_times:
        raise RuntimeError("stream produced no token chunks")
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "ttft_ms": (token_times[0] - started) * 1000,
        "e2e_ms": (finished - started) * 1000,
        "token_chunk_count": len(token_times),
        "tpot_ms": [
            (right - left) * 1000 for left, right in zip(token_times, token_times[1:])
        ],
    }


def run_cell(url: str, model: str, cell: dict[str, Any], prompt_token_id: int) -> dict[str, Any]:
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cell["concurrency"]) as executor:
        futures = [
            executor.submit(request_once, url, model, cell, prompt_token_id)
            for _ in range(cell["concurrency"])
        ]
        requests_out = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - wall_start
    tpots = [value for request in requests_out for value in request["tpot_ms"]]
    return {
        **cell,
        "request_count": len(requests_out),
        "wall_seconds": wall_seconds,
        "output_tokens_per_second": (
            cell["output_tokens"] * len(requests_out) / wall_seconds
        ),
        "ttft_ms": summary([request["ttft_ms"] for request in requests_out]),
        "e2e_ms": summary([request["e2e_ms"] for request in requests_out]),
        "tpot_ms": summary(tpots),
        "token_chunk_counts": [request["token_chunk_count"] for request in requests_out],
    }


def energy_snapshot() -> dict[str, int]:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            "total_energy_mj": pynvml.nvmlDeviceGetTotalEnergyConsumption(handle),
            "power_mw": pynvml.nvmlDeviceGetPowerUsage(handle),
            "temperature_c": pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            ),
        }
    finally:
        pynvml.nvmlShutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="smollm2-360m")
    parser.add_argument("--prompt-token-id", type=int, default=42)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = {
        "schema_version": 1,
        "model": args.model,
        "prompt_token_id": args.prompt_token_id,
        "prompt_tokens": "prompt_token_id repeated input_tokens times",
        "decoding": {
            "temperature": 0,
            "ignore_eos": True,
            "forced_output_length": True,
            "stream": True,
        },
        "cells": CELLS,
    }
    trace_bytes = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_bytes(trace_bytes + b"\n")

    warmup = {"name": "warmup", "input_tokens": 16, "output_tokens": 8, "concurrency": 1}
    request_once(
        f"{args.base_url}/v1/completions", args.model, warmup, args.prompt_token_id
    )
    before = energy_snapshot()
    started = time.time()
    results = [
        run_cell(
            f"{args.base_url}/v1/completions", args.model, cell, args.prompt_token_id
        )
        for cell in CELLS
    ]
    after = energy_snapshot()
    completed_output_tokens = sum(
        cell["output_tokens"] * cell["request_count"] for cell in results
    )
    output = {
        "schema_version": 1,
        "kind": "instrumentation-selfcheck-not-benchmark-evidence",
        "started_unix": started,
        "trace_sha256": trace_sha256,
        "cells": results,
        "telemetry": {
            "before": before,
            "after": after,
            "gross_energy_j": (after["total_energy_mj"] - before["total_energy_mj"]) / 1000,
            "completed_output_tokens": completed_output_tokens,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
