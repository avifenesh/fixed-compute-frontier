#!/usr/bin/env python3
"""Separately sealed, rate-limited transport retry for the frozen T38 census."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping

from experiments import equivariant_template_engram_t38_census as base


ROOT = Path(__file__).resolve().parents[1]
RETRY_DATA_DIR = ROOT / "data/equivariant-template-engram-t38r"
RETRY_CORPUS_PATH = RETRY_DATA_DIR / "corpus.jsonl"
RETRY_MANIFEST_PATH = RETRY_DATA_DIR / "manifest.json"
RETRY_OUTPUT_PATH = (
    ROOT / "results/equivariant-template-engram-t38r-cpu-census.json"
)
RETRY_OPENING_PATH = (
    ROOT / "results/equivariant-template-engram-t38r-adjudication-opened.json"
)
RETRY_FINALIZED_PATH = (
    ROOT / "results/equivariant-template-engram-t38r-adjudication-finalized.json"
)
RETRY_PREREGISTRATION = (
    ROOT
    / "results/equivariant-template-engram-t38r-transport-retry-preregistration.md"
)
ORIGINAL_FAILURE_SEAL = (
    ROOT / "results/equivariant-template-engram-t38-adjudication-opened.json"
)
ORIGINAL_FAILURE_REPORT = (
    ROOT / "results/equivariant-template-engram-t38-fetch-failure.md"
)

ORIGINAL_IMPLEMENTATION_SHA256 = (
    "3935bed4683fc280ef534e45fda5d4b6fc911e80a8c2413497c144dbf3a29261"
)
ORIGINAL_PREREGISTRATION_SHA256 = (
    "6e1167fe160ce7d4dec764ead38654f9bc472cc0af6061f8d50332462ecd86e3"
)
ORIGINAL_TESTS_SHA256 = (
    "9a4b466467de8770a7c7480a46e44a46edaa98bb80dc01ca8ce756d57a19ce3e"
)
ORIGINAL_FAILURE_SEAL_SHA256 = (
    "5674a876b1a95023e886afe59f60c8a9ea7bc9eaa5fd5c292aa2d8457ceb1934"
)
ORIGINAL_FAILURE_REPORT_SHA256 = (
    "a9b97d71b44b670a40def70730cee99a5bc9d938036b75b10c4cef767b531def"
)
VIEWER_INTERVAL_SECONDS = 3.1
MAX_ATTEMPTS = 8
MAX_BACKOFF_SECONDS = 120.0

_last_viewer_request = 0.0


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_base() -> None:
    checks = {
        Path(base.__file__).resolve(): ORIGINAL_IMPLEMENTATION_SHA256,
        ROOT
        / "results/equivariant-template-engram-t38-cpu-census-preregistration.md": ORIGINAL_PREREGISTRATION_SHA256,
        ROOT
        / "tests/test_equivariant_template_engram_t38_census.py": ORIGINAL_TESTS_SHA256,
        ORIGINAL_FAILURE_SEAL: ORIGINAL_FAILURE_SEAL_SHA256,
        ORIGINAL_FAILURE_REPORT: ORIGINAL_FAILURE_REPORT_SHA256,
    }
    for path, expected in checks.items():
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"T38R frozen dependency changed: {path}: {actual} != {expected}"
            )
    forbidden_original_artifacts = (
        ROOT / "data/equivariant-template-engram-t38/corpus.jsonl",
        ROOT / "data/equivariant-template-engram-t38/manifest.json",
        ROOT
        / "results/equivariant-template-engram-t38-adjudication-finalized.json",
        ROOT / "results/equivariant-template-engram-t38-cpu-census.json",
    )
    present = [str(path) for path in forbidden_original_artifacts if path.exists()]
    if present:
        raise RuntimeError(
            f"T38R precondition violated; original held-out artifacts exist: {present}"
        )


def retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if value is None:
        return None
    try:
        return min(MAX_BACKOFF_SECONDS, max(0.0, float(value)))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return min(MAX_BACKOFF_SECONDS, max(0.0, delay))


def rate_limited_http_json(url: str, attempts: int = MAX_ATTEMPTS) -> Any:
    global _last_viewer_request
    is_viewer = url.startswith("https://datasets-server.huggingface.co/rows?")
    last_error: Exception | None = None
    for attempt in range(min(attempts, MAX_ATTEMPTS)):
        if is_viewer:
            remaining = VIEWER_INTERVAL_SECONDS - (
                time.monotonic() - _last_viewer_request
            )
            if remaining > 0:
                time.sleep(remaining)
            _last_viewer_request = time.monotonic()
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "fixed-compute-frontier-t38r/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code != 429 and not 500 <= error.code < 600:
                raise
            explicit = retry_after_seconds(error)
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            explicit = None
        if attempt + 1 >= min(attempts, MAX_ATTEMPTS):
            break
        delay = (
            explicit
            if explicit is not None
            else min(MAX_BACKOFF_SECONDS, float(2**attempt))
        )
        time.sleep(delay)
    assert last_error is not None
    raise last_error


def write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def create_retry_opening_seal() -> dict[str, Any]:
    payload = {
        "schema": "equivariant-template-engram-t38r-adjudication-open-v1",
        "opened_unix_seconds": time.time(),
        "original_implementation_sha256": ORIGINAL_IMPLEMENTATION_SHA256,
        "original_preregistration_sha256": ORIGINAL_PREREGISTRATION_SHA256,
        "original_tests_sha256": ORIGINAL_TESTS_SHA256,
        "original_failure_seal_sha256": ORIGINAL_FAILURE_SEAL_SHA256,
        "original_failure_report_sha256": ORIGINAL_FAILURE_REPORT_SHA256,
        "retry_preregistration_sha256": file_sha256(RETRY_PREREGISTRATION),
        "retry_wrapper_sha256": file_sha256(Path(__file__).resolve()),
        "tokenizer_sha256": base.TOKENIZER_SHA256,
        "viewer_interval_seconds": VIEWER_INTERVAL_SECONDS,
        "max_attempts": MAX_ATTEMPTS,
    }
    write_exclusive(RETRY_OPENING_PATH, payload)
    return payload


def finalize_retry_seal(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "equivariant-template-engram-t38r-adjudication-final-v1",
        "opening_seal_sha256": file_sha256(RETRY_OPENING_PATH),
        "corpus_sha256": manifest["corpus_sha256"],
        "manifest_sha256": file_sha256(RETRY_MANIFEST_PATH),
        "finalized_unix_seconds": time.time(),
    }
    write_exclusive(RETRY_FINALIZED_PATH, payload)
    return payload


def configure_retry() -> None:
    base.DATA_DIR = RETRY_DATA_DIR
    base.CORPUS_PATH = RETRY_CORPUS_PATH
    base.MANIFEST_PATH = RETRY_MANIFEST_PATH
    base.OUTPUT_PATH = RETRY_OUTPUT_PATH
    base.ADJUDICATION_OPENED_PATH = RETRY_OPENING_PATH
    base.ADJUDICATION_FINALIZED_PATH = RETRY_FINALIZED_PATH
    base.http_json = rate_limited_http_json
    base.create_adjudication_seal = create_retry_opening_seal
    base.finalize_adjudication_seal = finalize_retry_seal


def main() -> None:
    verify_frozen_base()
    configure_retry()
    result = base.run_census()
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_gates_pass": result["all_gates_pass"],
                "failed_gates": [
                    name for name, passed in result["gates"].items() if not passed
                ],
                "output": str(RETRY_OUTPUT_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
