#!/usr/bin/env python3
"""Audited stage controller for T84 effective v4.

There is intentionally no ungated discovery mode.  Every subcommand validates
the immutable protocol and an independent authorization receipt before work.
Artifacts are create-only inside the frozen run directory.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from experiments import shared_score_heterogeneous_attention_t84 as core


FIXTURE_DIR = core.RUN_ROOT / "fixtures"
TIMING_DIR = core.RUN_ROOT / "timing"
STAGE0_DIR = core.RUN_ROOT / "stage0"
PROBE_DIR = core.RUN_ROOT / "probes"
TRAIN_DIR = core.RUN_ROOT / "training"
LEDGER_DIR = core.RUN_ROOT / "ledger"
DECISION_DIR = core.RUN_ROOT / "decisions"
AUTH_DIR = core.RUN_ROOT / "worker-authorizations"
MAX_ACTIVE_NS = 8 * 60 * 60 * 1_000_000_000
MAX_BYTES = 2 * 1024**3
FINAL_CLOSE_RESERVE_BYTES = 1 << 20
TERMINAL_RESERVE_BYTES = 64 * 1024**2
TERMINAL_RESERVE_NS = 15 * 60 * 1_000_000_000
TERMINAL_VMHWM_MARGIN_BYTES = 256 * 1024**2
UNADJUSTED_T95 = 2.131449545559323
CONTROL_ORDER = (
    "finite_beta_8",
    "all_max_shared",
    "hybrid_independent",
    "soft_narrow_grouped",
    "tropical_official",
    "deepsets_conditioned",
    "hybrid_whole_head",
    "soft_grouped",
    "soft_standard",
    "soft_narrow",
    "hard_shared",
    "soft_time_matched",
)
SOFT_PROTECTED = ("soft_standard", "soft_grouped", "soft_narrow_grouped")
SLICE_NAMES = ("id", "l16", "l32", "v2", "l32v2")
OOD_NAMES = ("l16", "l32", "v2", "l32v2")
SLICE_IDS = {"l16": 1, "l32": 2, "v2": 3, "l32v2": 4}
FROZEN_PYTHON = core.ROOT / "runtime/cpkv-topk-001-venv/bin/python"
ENTRYPOINT_MODULE = "experiments.shared_score_heterogeneous_attention_t84_entrypoint"


def save_npz_new(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)


def atomic_text_new(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def canonical_tensor_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        header = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": array.shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def worker_authorization(
    kind: str,
    receipt: Path,
    *,
    attempt: int | None = None,
    round_id: int | None = None,
    variant_id: int | None = None,
    index: int | None = None,
    architecture: str | None = None,
) -> Path:
    receipt = receipt.resolve()
    try:
        receipt_relative = receipt.relative_to(core.ROOT)
    except ValueError as error:
        raise RuntimeError("authorization receipt must live inside the frozen project") from error
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_stage = receipt_payload.get("authorization")
    if not isinstance(receipt_stage, str):
        raise RuntimeError("authorization receipt has no stage")
    core.require_receipt(receipt_stage, receipt, validate_prior_artifacts=False)
    fields = {
        "kind": kind,
        "attempt": attempt,
        "round": round_id,
        "variant": variant_id,
        "index": index,
        "architecture": architecture,
        "parent_pid": os.getpid(),
        "receipt_path": str(receipt_relative),
        "receipt_authorization": receipt_stage,
        "receipt_sha256": core.sha256_file(receipt),
        "source_manifest_sha256": core.sha256_file(core.SOURCE_MANIFEST),
    }
    identity = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = AUTH_DIR / f"{kind}-{identity}.json"
    core.atomic_json(path, fields)
    return path


def require_worker_authorization(
    path: Path,
    kind: str,
    *,
    attempt: int | None = None,
    round_id: int | None = None,
    variant_id: int | None = None,
    index: int | None = None,
    architecture: str | None = None,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": kind,
        "attempt": attempt,
        "round": round_id,
        "variant": variant_id,
        "index": index,
        "architecture": architecture,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("worker authorization mismatch")
    if not isinstance(payload.get("parent_pid"), int):
        raise RuntimeError("worker parent PID missing")
    if os.getppid() != payload["parent_pid"]:
        raise RuntimeError("worker was not launched by its authorizing controller")
    receipt_relative = payload.get("receipt_path")
    receipt_stage = payload.get("receipt_authorization")
    if not isinstance(receipt_relative, str) or not isinstance(receipt_stage, str):
        raise RuntimeError("worker authorization receipt binding missing")
    receipt_path = core.ROOT / receipt_relative
    if payload.get("receipt_sha256") != core.sha256_file(receipt_path):
        raise RuntimeError("worker authorization receipt hash mismatch")
    core.require_receipt(
        receipt_stage, receipt_path, validate_prior_artifacts=False
    )
    core.assert_audited_sources()
    if payload.get("source_manifest_sha256") != core.sha256_file(core.SOURCE_MANIFEST):
        raise RuntimeError("worker source manifest mismatch")
    return payload


def frozen_subprocess_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX"}
    }
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def run_reported_subprocess(
    command: Sequence[str],
    report_path: Path,
    *,
    capture_output: bool = False,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    before_wall = time.monotonic_ns()
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        completed = subprocess.run(
            list(command),
            cwd=core.ROOT,
            check=False,
            env=frozen_subprocess_environment(),
            capture_output=capture_output,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        core.atomic_json(
            report_path,
            {
                "command": list(command),
                "returncode": "TIMEOUT",
                "timeout_seconds": timeout_seconds,
                "parent_observed_lifetime_ns": time.monotonic_ns() - before_wall,
                "diagnostic_child_user_ns": round(
                    (after_usage.ru_utime - before_usage.ru_utime) * 1_000_000_000
                ),
                "diagnostic_child_system_ns": round(
                    (after_usage.ru_stime - before_usage.ru_stime) * 1_000_000_000
                ),
                "child_vmhwm_bytes_conservative_upper_bound": int(
                    after_usage.ru_maxrss
                )
                * 1024,
            },
        )
        raise error
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "command": list(command),
        "returncode": completed.returncode,
        "parent_observed_lifetime_ns": time.monotonic_ns() - before_wall,
        "diagnostic_child_user_ns": round(
            (after_usage.ru_utime - before_usage.ru_utime) * 1_000_000_000
        ),
        "diagnostic_child_system_ns": round(
            (after_usage.ru_stime - before_usage.ru_stime) * 1_000_000_000
        ),
        "child_vmhwm_bytes_conservative_upper_bound": int(after_usage.ru_maxrss) * 1024,
    }
    core.atomic_json(report_path, payload)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def worker_command(*arguments: str) -> list[str]:
    return [
        "taskset",
        "-c",
        "16",
        str(FROZEN_PYTHON),
        "-B",
        "-m",
        ENTRYPOINT_MODULE,
        *arguments,
    ]


def self_cpu_diagnostics() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return round(usage.ru_utime * 1_000_000_000), round(
        usage.ru_stime * 1_000_000_000
    )


def directory_usage(path: Path) -> tuple[int, int]:
    allocated = logical = 0
    if not path.exists():
        return 0, 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise RuntimeError("links forbidden")
        if item.is_file():
            stat = item.stat()
            if stat.st_nlink != 1:
                raise RuntimeError("hard-linked artifact forbidden")
            if stat.st_blocks * 512 < stat.st_size:
                raise RuntimeError("sparse artifact forbidden")
            allocated += stat.st_blocks * 512
            logical += stat.st_size
    return allocated, logical


def directory_allocated_bytes(path: Path) -> int:
    return directory_usage(path)[0]


LEDGER_STATUSES = {
    "CONTINUE",
    "STOP-REJECT",
    "CONTROL FLOOR NOT ESTABLISHED",
    "RESOURCE-INCONCLUSIVE",
    "PROTOCOL INVALID",
}


def ledger_state(
    *, allow_open: bool = False, current_early_marker: Path | None = None
) -> tuple[list[tuple[Path, dict[str, Any]]], tuple[Path, dict[str, Any]] | None]:
    """Validate the exact open/close chain; an abandoned open is terminal."""
    if not LEDGER_DIR.exists():
        return [], None
    paths = sorted(LEDGER_DIR.glob("*.json"))
    recognized: set[Path] = set()
    closes: list[tuple[Path, dict[str, Any]]] = []
    pending: tuple[Path, dict[str, Any]] | None = None
    previous_close_hash = "0" * 64
    index = 0
    while True:
        open_path = LEDGER_DIR / f"{index:04d}-open.json"
        prepare_path = LEDGER_DIR / f"{index:04d}-prepare.json"
        close_path = LEDGER_DIR / f"{index:04d}-close.json"
        if not open_path.exists():
            if close_path.exists():
                raise RuntimeError("ledger close exists without matching open")
            break
        recognized.add(open_path)
        opening = json.loads(open_path.read_text(encoding="utf-8"))
        early_value = opening.get("early_marker_path")
        receipt_value = opening.get("authorization_receipt_path")
        if not isinstance(early_value, str) or not isinstance(receipt_value, str):
            raise RuntimeError("ledger opening lacks marker or receipt binding")
        early_path = core.RUN_ROOT / early_value
        receipt_path = core.ROOT / receipt_value
        try:
            early_path.relative_to(LEDGER_DIR)
            receipt_path.relative_to(core.ROOT)
        except ValueError as error:
            raise RuntimeError("ledger marker or receipt escaped its root") from error
        if (
            opening.get("index") != index
            or opening.get("previous_close_hash") != previous_close_hash
            or not isinstance(opening.get("stage"), str)
            or not isinstance(opening.get("invocation_started_ns"), int)
            or not early_path.is_file()
            or opening.get("early_marker_sha256") != core.sha256_file(early_path)
            or receipt_path.is_relative_to(core.RUN_ROOT)
            or not receipt_path.is_file()
            or opening.get("authorization_receipt_sha256")
            != core.sha256_file(receipt_path)
        ):
            raise RuntimeError("invalid ledger opening receipt")
        recognized.add(early_path)
        if prepare_path.exists():
            recognized.add(prepare_path)
        if close_path.exists():
            if not prepare_path.exists():
                raise RuntimeError("ledger close exists without its fsynced prepare")
            recognized.add(close_path)
            closing = json.loads(close_path.read_text(encoding="utf-8"))
            if (
                closing.get("index") != index
                or closing.get("stage") != opening["stage"]
                or closing.get("open_sha256") != core.sha256_file(open_path)
                or closing.get("prepare_sha256") != core.sha256_file(prepare_path)
                or closing.get("previous_close_hash") != previous_close_hash
                or closing.get("status") not in LEDGER_STATUSES
                or not isinstance(closing.get("active_ns"), int)
                or closing["active_ns"] < 0
            ):
                raise RuntimeError("invalid ledger closing receipt")
            if closes and closes[-1][1]["status"] != "CONTINUE":
                raise RuntimeError("ledger continued after a terminal receipt")
            closes.append((close_path, closing))
            previous_close_hash = core.sha256_file(close_path)
        else:
            pending = (open_path, opening)
            index += 1
            break
        index += 1
    if current_early_marker is not None:
        if current_early_marker.parent != LEDGER_DIR or not current_early_marker.is_file():
            raise RuntimeError("current pre-import marker is invalid")
        recognized.add(current_early_marker)
    if set(paths) != recognized:
        raise RuntimeError("unexpected ledger file or non-contiguous ledger index")
    if pending is not None and not allow_open:
        raise RuntimeError("unclosed ledger opening is terminal PROTOCOL INVALID")
    return closes, pending


def ledger_entries(*, allow_open: bool = False) -> list[dict[str, Any]]:
    closes, _ = ledger_state(allow_open=allow_open)
    return [payload for _, payload in closes]


def ledger_totals(*, allow_open: bool = False) -> tuple[int, int]:
    entries = ledger_entries(allow_open=allow_open)
    active = sum(int(entry["active_ns"]) for entry in entries)
    return active, directory_allocated_bytes(core.RUN_ROOT)


def begin_ledger_attempt(stage: str, started_ns: int, receipt: Path) -> Path:
    marker_value = os.environ.get("T84_EARLY_MARKER")
    if marker_value is None:
        raise RuntimeError("top-level execution lacks its pre-import marker")
    marker = Path(marker_value).absolute()
    closes, pending = ledger_state(
        allow_open=True, current_early_marker=marker
    )
    if pending is not None:
        raise RuntimeError("unclosed prior invocation is terminal PROTOCOL INVALID")
    if closes and closes[-1][1]["status"] != "CONTINUE":
        raise RuntimeError("a terminal ledger receipt forbids replay or continuation")
    prior_active = sum(int(payload["active_ns"]) for _, payload in closes)
    allocated, _ = directory_usage(core.RUN_ROOT)
    if prior_active > MAX_ACTIVE_NS or allocated > MAX_BYTES:
        raise RuntimeError("RESOURCE-INCONCLUSIVE")
    index = len(closes)
    previous_hash = "0" * 64 if not closes else core.sha256_file(closes[-1][0])
    receipt = receipt.resolve()
    try:
        receipt_relative = receipt.relative_to(core.ROOT)
    except ValueError as error:
        raise RuntimeError("authorization receipt must live inside the project") from error
    if receipt.is_relative_to(core.RUN_ROOT):
        raise RuntimeError("authorization receipt cannot live in the run directory")
    path = LEDGER_DIR / f"{index:04d}-open.json"
    core.atomic_json(
        path,
        {
            "index": index,
            "stage": stage,
            "invocation_started_ns": started_ns,
            "previous_close_hash": previous_hash,
            "early_marker_path": str(marker.relative_to(core.RUN_ROOT)),
            "early_marker_sha256": core.sha256_file(marker),
            "authorization_receipt_path": str(receipt_relative),
            "authorization_receipt_sha256": core.sha256_file(receipt),
            "source_manifest_sha256": core.sha256_file(core.SOURCE_MANIFEST),
            "accounting_rule": "timer starts before heavy imports; final close is bookkeeping after a fsynced prepare and final artifact scan/hash",
        },
    )
    return path


def decision_binding(stage_key: str) -> dict[str, Any]:
    manifest = DECISION_DIR / f"{stage_key}-complete-artifacts.json"
    word = DECISION_DIR / f"{stage_key}-word.txt"
    terminal_paths = {
        "decoder_envelope": DECISION_DIR / f"{stage_key}-decoder-envelope.json",
        "decoder_input": DECISION_DIR / f"{stage_key}-input.json",
        "decoder_vmhwm": DECISION_DIR / f"{stage_key}-decoder-vmhwm.json",
        "decoder_worker": DECISION_DIR / f"{stage_key}-decoder-worker.json",
        "decision_word": word,
    }
    if not manifest.exists() or not all(path.exists() for path in terminal_paths.values()):
        raise RuntimeError("authorized stage lacks its exact sealed artifacts or word")
    binding = {
        "complete_artifact_manifest_path": str(manifest.relative_to(core.ROOT)),
        "complete_artifact_manifest_sha256": core.sha256_file(manifest),
        "decision_word_path": str(word.relative_to(core.ROOT)),
        "decision_word_sha256": core.sha256_file(word),
    }
    binding["terminal_artifacts"] = {
            name: {
                "path": str(path.relative_to(core.ROOT)),
                "sha256": core.sha256_file(path),
            }
            for name, path in terminal_paths.items()
        }
    return binding


def close_ledger_attempt(
    open_path: Path,
    supplied_status: str,
    decision_key: str | None,
) -> str:
    if supplied_status not in LEDGER_STATUSES:
        supplied_status = "PROTOCOL INVALID"
    closes, pending = ledger_state(allow_open=True)
    if pending is None or pending[0] != open_path:
        raise RuntimeError("closing receipt does not match the open invocation")
    opening = pending[1]
    binding: dict[str, Any] = {
        "complete_artifact_manifest_path": None,
        "complete_artifact_manifest_sha256": None,
        "decision_word_path": None,
        "decision_word_sha256": None,
        "terminal_artifacts": None,
    }
    if decision_key is not None:
        binding.update(decision_binding(decision_key))
    previous_hash = "0" * 64 if not closes else core.sha256_file(closes[-1][0])
    preprepare_allocated, preprepare_logical = directory_usage(core.RUN_ROOT)
    preprepare_count = sum(
        path.is_file() for path in core.RUN_ROOT.rglob("*")
    )
    prepare_path = LEDGER_DIR / f"{opening['index']:04d}-prepare.json"
    core.atomic_json(
        prepare_path,
        {
            "index": opening["index"],
            "stage": opening["stage"],
            "supplied_status": supplied_status,
            "previous_close_hash": previous_hash,
            "open_sha256": core.sha256_file(open_path),
            "authorization_receipt_path": opening["authorization_receipt_path"],
            "authorization_receipt_sha256": opening[
                "authorization_receipt_sha256"
            ],
            "source_manifest_sha256": core.sha256_file(core.SOURCE_MANIFEST),
            "artifact_count_before_prepare": preprepare_count,
            "allocated_bytes_before_prepare": preprepare_allocated,
            "logical_bytes_before_prepare": preprepare_logical,
            **binding,
        },
    )
    prepare_hash = core.sha256_file(prepare_path)
    allocated, logical = directory_usage(core.RUN_ROOT)
    artifact_count = sum(path.is_file() for path in core.RUN_ROOT.rglob("*"))
    maximum_vmhwm = max_reported_vmhwm()
    ended_ns = time.monotonic_ns()
    active_ns = ended_ns - int(opening["invocation_started_ns"])
    prior_active = sum(int(payload["active_ns"]) for _, payload in closes)
    payload: dict[str, Any] = {
        "index": opening["index"],
        "stage": opening["stage"],
        "supplied_status": supplied_status,
        "status": supplied_status,
        "active_ns": active_ns,
        "previous_close_hash": previous_hash,
        "open_sha256": core.sha256_file(open_path),
        "prepare_sha256": prepare_hash,
        "authorization_receipt_path": opening["authorization_receipt_path"],
        "authorization_receipt_sha256": opening[
            "authorization_receipt_sha256"
        ],
        "source_manifest_sha256": core.sha256_file(core.SOURCE_MANIFEST),
        "artifact_count_after_prepare_before_final_close": artifact_count,
        "allocated_bytes_after_prepare_before_final_close": allocated,
        "logical_bytes_after_prepare_before_final_close": logical,
        "allocated_bytes_conservative_after_final_close": allocated
        + FINAL_CLOSE_RESERVE_BYTES,
        "maximum_reported_vmhwm_bytes": maximum_vmhwm,
        "accounting_boundary": "active_ns ends after prepare fsync and final artifact hash; final close is fixed-small accounting bookkeeping covered by a one-MiB allocation reserve",
        **binding,
    }
    resource_crossed = (
        prior_active + active_ns > MAX_ACTIVE_NS
        or payload["allocated_bytes_conservative_after_final_close"] > MAX_BYTES
        or maximum_vmhwm > MAX_BYTES
    )
    if resource_crossed and supplied_status != "PROTOCOL INVALID":
        payload["status"] = "RESOURCE-INCONCLUSIVE"
    encoded_size = len(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ) + 1
    if encoded_size > FINAL_CLOSE_RESERVE_BYTES:
        raise RuntimeError("final accounting receipt exceeded its fixed reserve")
    close_path = LEDGER_DIR / f"{opening['index']:04d}-close.json"
    core.atomic_json(close_path, payload)
    return str(payload["status"])


def arrays_from_examples(examples: Sequence[core.Example]) -> dict[str, np.ndarray]:
    raw = []
    for item in examples:
        n = len(item.k)
        tokens = np.zeros((n + 1, 14), dtype="<f4")
        tokens[:n, 0] = 1.0
        tokens[:n, 2:6] = item.k
        tokens[:n, 6:8] = item.v
        tokens[:n, 8:10] = item.u
        tokens[n, 1] = 1.0
        tokens[n, 10:14] = item.q
        raw.append(tokens)
    return {
        "raw_tokens": np.stack(raw).astype("<f4", copy=False),
        "tokens": np.ascontiguousarray(np.stack([item.tokens for item in examples]).astype("<f4", copy=False)),
        "ys": np.asarray([item.ys for item in examples], dtype="<i8"),
        "ym": np.asarray([item.ym for item in examples], dtype="<i8"),
        "y": np.asarray([item.y for item in examples], dtype="<i8"),
        "q": np.stack([item.q for item in examples]).astype("<f8", copy=False),
        "k": np.stack([item.k for item in examples]).astype("<f8", copy=False),
        "v": np.stack([item.v for item in examples]).astype("<f8", copy=False),
        "u": np.stack([item.u for item in examples]).astype("<f8", copy=False),
        "scores": np.stack([item.scores for item in examples]).astype("<f8", copy=False),
    }


def generate_fixtures() -> dict[str, str]:
    specs = {
        "id": (841000, 8, 1.0),
        "l16": (841001, 16, 1.0),
        "l32": (841002, 32, 1.0),
        "v2": (841003, 8, 2.0),
        "l32v2": (841004, 32, 2.0),
    }
    outputs: dict[str, str] = {}
    for name, (seed, n, a) in specs.items():
        rng = np.random.Generator(np.random.PCG64(seed))
        path = FIXTURE_DIR / f"eval-{name}.npz"
        examples, permutation = core.balanced_examples_with_permutation(rng, n, a, 2_500)
        save_npz_new(path, **arrays_from_examples(examples), dataset_permutation=permutation)
        outputs[name] = core.sha256_file(path)

    dev_rng = np.random.Generator(np.random.PCG64(841100))
    for n in range(4, 9):
        path = FIXTURE_DIR / f"dev-n{n}.npz"
        examples, permutation = core.balanced_examples_with_permutation(dev_rng, n, 1.0, 500)
        save_npz_new(path, **arrays_from_examples(examples), dataset_permutation=permutation)
        outputs[f"dev_n{n}"] = core.sha256_file(path)

    probe_rng = np.random.Generator(np.random.PCG64(840190))
    for n in range(4, 9):
        path = FIXTURE_DIR / f"probe-n{n}.npz"
        examples, permutation = core.balanced_examples_with_permutation(probe_rng, n, 1.0, 5_000)
        save_npz_new(path, **arrays_from_examples(examples), dataset_permutation=permutation)
        outputs[f"probe_n{n}"] = core.sha256_file(path)

    for name, seed, n in (("len9", 843010, 8), ("len17", 843011, 16), ("len33", 843012, 32)):
        rng = np.random.Generator(np.random.PCG64(seed))
        examples, permutation = core.balanced_examples_with_permutation(rng, n, 1.0, 32)
        path = FIXTURE_DIR / f"timing-{name}.npz"
        save_npz_new(
            path,
            **arrays_from_examples(examples),
            dataset_permutation=permutation,
            mask=core.causal_mask(n).numpy(),
        )
        outputs[f"timing_{name}"] = core.sha256_file(path)

    core.atomic_json(FIXTURE_DIR / "manifest.json", {"hashes": outputs})
    return outputs


def load_fixture(name: str) -> dict[str, np.ndarray]:
    return load_npz(FIXTURE_DIR / name)


def timing_variant(variant_id: int) -> tuple[str, int, int]:
    if variant_id == 0:
        return "candidate", 1, 1
    if not 1 <= variant_id <= 16:
        raise ValueError(variant_id)
    offset = variant_id - 1
    return "soft_time_matched", offset // 4 + 1, offset % 4 + 1


def timing_worker(attempt: int, round_id: int, variant_id: int, worker_auth: Path) -> None:
    require_worker_authorization(
        worker_auth,
        "timing",
        attempt=attempt,
        round_id=round_id,
        variant_id=variant_id,
    )
    core.verify_runtime()
    architecture, r_attention, r_ffn = timing_variant(variant_id)
    model = core.build_model(
        architecture, 843001, r_attention=r_attention, r_ffn=r_ffn
    ).eval()
    samples: dict[str, np.ndarray] = {}
    names = ("len9", "len17", "len33")
    for length_id in ((round_id + variant_id + k) % 3 for k in range(3)):
        name = names[length_id]
        fixture = load_fixture(f"timing-{name}.npz")
        tokens = torch.from_numpy(fixture["tokens"])
        mask = torch.from_numpy(fixture["mask"])
        with torch.inference_mode():
            for _ in range(10):
                returned = model(tokens, mask)
            measured = np.empty(50, dtype="<i8")
            for index in range(50):
                before = time.perf_counter_ns()
                returned = model(tokens, mask)
                measured[index] = time.perf_counter_ns() - before
            if returned is None:
                raise RuntimeError("unmaterialized timing result")
        samples[name] = measured
    user_ns, system_ns = self_cpu_diagnostics()
    save_npz_new(
        TIMING_DIR / f"attempt-{attempt}" / f"round-{round_id:02d}-variant-{variant_id:02d}.npz",
        **samples,
        vmhwm=np.asarray([core.linux_vmhwm_bytes()], dtype="<i8"),
        diagnostic_cpu_user_ns=np.asarray([user_ns], dtype="<i8"),
        diagnostic_cpu_system_ns=np.asarray([system_ns], dtype="<i8"),
    )


def run_one_timing_attempt(receipt: Path, attempt: int) -> dict[str, Any]:
    for round_id in range(20):
        for offset in range(17):
            variant_id = (round_id + offset) % 17
            worker_auth = worker_authorization(
                "timing",
                receipt,
                attempt=attempt,
                round_id=round_id,
                variant_id=variant_id,
            )
            run_reported_subprocess(
                worker_command(
                    "timing-worker",
                    "--attempt",
                    str(attempt),
                    "--round",
                    str(round_id),
                    "--variant",
                    str(variant_id),
                    "--worker-auth",
                    str(worker_auth),
                ),
                TIMING_DIR
                / f"attempt-{attempt}"
                / f"round-{round_id:02d}-variant-{variant_id:02d}-worker.json",
            )
    return summarize_timing(attempt)


def summarize_timing(attempt: int) -> dict[str, Any]:
    metrics: dict[int, dict[str, Any]] = {}
    for variant_id in range(17):
        by_length: dict[str, list[int]] = {name: [] for name in ("len9", "len17", "len33")}
        vmhwm: list[int] = []
        for round_id in range(20):
            fixture = load_npz(
                TIMING_DIR
                / f"attempt-{attempt}"
                / f"round-{round_id:02d}-variant-{variant_id:02d}.npz"
            )
            for name in by_length:
                by_length[name].extend(int(value) for value in fixture[name])
            vmhwm.append(int(fixture["vmhwm"][0]))
        cell: dict[str, Any] = {"vmhwm": vmhwm, "maximum_vmhwm": max(vmhwm)}
        for name, values in by_length.items():
            array = np.asarray(values, dtype=np.float64)
            median = float(np.median(array))
            cell[name] = {
                "p50": median,
                "p95": float(np.quantile(array, 0.95)),
                "mean": float(array.mean()),
                "sd": float(array.std(ddof=1)),
                "mad_ratio": float(np.median(np.abs(array - median)) / median),
            }
        metrics[variant_id] = cell
    candidate = metrics[0]
    stable_matrix = all(
        metrics[variant_id][name]["mad_ratio"] <= 0.05
        for variant_id in range(17)
        for name in ("len9", "len17", "len33")
    )
    eligible: list[tuple[int, int, int]] = []
    for variant_id in range(1, 17):
        cell = metrics[variant_id]
        ratios: list[float] = []
        for percentile in ("p50", "p95"):
            weighted = 0.25 * cell["len9"][percentile] + 0.25 * cell["len17"][percentile] + 0.5 * cell["len33"][percentile]
            target = 0.25 * candidate["len9"][percentile] + 0.25 * candidate["len17"][percentile] + 0.5 * candidate["len33"][percentile]
            ratios.extend((weighted / target, cell["len33"][percentile] / candidate["len33"][percentile]))
        architecture, ra, rf = timing_variant(variant_id)
        if stable_matrix and all(0.95 <= ratio <= 1.05 for ratio in ratios):
            eligible.append((ra + rf, ra, rf))
    selected = max(eligible) if eligible else None
    payload = {
        "attempt": attempt,
        "metrics": metrics,
        "stable_matrix": stable_matrix,
        "selected": selected,
        "maximum_vmhwm": max(
            max(int(value) for value in metrics[variant]["vmhwm"])
            for variant in metrics
        ),
        "status": (
            "TIMING UNSTABLE"
            if not stable_matrix
            else "eligible"
            if selected
            else "TIME MATCH UNAVAILABLE"
        ),
    }
    core.atomic_json(
        TIMING_DIR / f"summary-attempt-{attempt}.json",
        payload,
    )
    return payload


def run_timing_matrix(receipt: Path) -> None:
    core.require_receipt("FIXTURES AND TIMING", receipt)
    core.verify_runtime()
    if not (FIXTURE_DIR / "manifest.json").exists():
        raise RuntimeError("fixtures must precede timing")
    summary = run_one_timing_attempt(receipt, 0)
    if not summary["stable_matrix"]:
        summary = run_one_timing_attempt(receipt, 1)
    final = {
        "authoritative_attempt": summary["attempt"],
        "stable_matrix": summary["stable_matrix"],
        "selected": summary["selected"],
        "status": summary["status"],
        "attempt_summary_sha256": core.sha256_file(
            TIMING_DIR / f"summary-attempt-{summary['attempt']}.json"
        ),
    }
    core.atomic_json(TIMING_DIR / "summary.json", final)
    seal_prerequisite_decision("fixtures-timing")


def masked_operator(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    u: np.ndarray,
    valid: np.ndarray,
    *,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tq = torch.tensor(q, dtype=dtype)
    tk = torch.tensor(k, dtype=dtype)
    tv = torch.tensor(v, dtype=dtype)
    tu = torch.tensor(u, dtype=dtype)
    mask = torch.tensor(valid, dtype=torch.bool)
    scores = core.scaled_masked_scores(
        tq[:, None, None, :],
        tk[:, None, :, :],
        mask[None, None, None, :],
        4,
    )
    soft = core.soft_reduce(scores, tv[:, None, :, :])
    maximum, winners = core.maxplus_reduce(scores, tu[:, None, :, :])
    beta8 = core.finite_beta_reduce(scores, tu[:, None, :, :], 8.0)
    return (
        scores[:, 0, 0].detach().numpy(),
        soft[:, 0, 0].detach().numpy(),
        maximum[:, 0, 0].detach().numpy(),
        winners[:, 0, 0].detach().numpy(),
        beta8[:, 0, 0].detach().numpy(),
    )


def gradient_cases(
    rng: np.random.Generator, n: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    q_list: list[np.ndarray] = []
    k_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    u_list: list[np.ndarray] = []
    while len(scores_list) < 1024:
        q = rng.uniform(-2, 2, size=(4,))
        k = rng.uniform(-2, 2, size=(n, 4))
        v = rng.uniform(-2, 2, size=(n, 2))
        u = rng.uniform(-2, 2, size=(n, 2))
        scores = (k @ q) / 2.0
        combined = scores[:, None] + u
        if n == 1 or bool(np.all(np.partition(combined, -2, axis=0)[-1] - np.partition(combined, -2, axis=0)[-2] >= 1e-6)):
            q_list.append(q)
            k_list.append(k)
            v_list.append(v)
            scores_list.append(scores)
            u_list.append(u)
    scores_array = np.asarray(scores_list, dtype="<f8")
    u_array = np.asarray(u_list, dtype="<f8")
    tf_scores = torch.tensor(scores_array, dtype=torch.float64, requires_grad=True)
    tf_u = torch.tensor(u_array, dtype=torch.float64, requires_grad=True)
    maximum, winners = core.maxplus_reduce(
        tf_scores[:, None, None, :], tf_u[:, None, :, :]
    )
    maximum = maximum[:, 0, 0]
    winners = winners[:, 0, 0]
    maximum.sum().backward()
    if tf_scores.grad is None or tf_u.grad is None:
        raise RuntimeError("missing Stage 0 gradients")
    return (
        np.asarray(q_list, dtype="<f8"),
        np.asarray(k_list, dtype="<f8"),
        np.asarray(v_list, dtype="<f8"),
        scores_array,
        u_array,
        winners.detach().numpy().astype("<i8", copy=False),
        tf_scores.grad.detach().numpy().astype("<f8", copy=False),
        tf_u.grad.detach().numpy().astype("<f8", copy=False),
    )


def stage0_payload(seed: int = 840000) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    outputs: dict[str, np.ndarray] = {"seed": np.asarray([seed], dtype="<i8")}
    for n in (1, 4, 8, 16, 32):
        q = rng.uniform(-2, 2, size=(1024, 4))
        k = rng.uniform(-2, 2, size=(1024, n, 4))
        v = rng.uniform(-2, 2, size=(1024, n, 2))
        u = rng.uniform(-2, 2, size=(1024, n, 2))
        k_pad = np.zeros((1024, 32, 4), dtype=np.float64)
        v_pad = np.zeros((1024, 32, 2), dtype=np.float64)
        u_pad = np.zeros((1024, 32, 2), dtype=np.float64)
        k_pad[:, :n], v_pad[:, :n], u_pad[:, :n] = k, v, u
        valid = np.arange(32) < n
        scores = np.einsum("br,bnr->bn", q, k) / 2.0
        shifted = scores - scores.max(axis=1, keepdims=True)
        probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        soft = np.einsum("bn,bnc->bc", probability, v)
        combined = scores[..., None] + u
        maximum = combined.max(axis=1)
        winners = combined.argmax(axis=1)
        b8 = maximum + np.log(np.exp(8.0 * (combined - maximum[:, None])).sum(axis=1)) / 8.0
        score64, soft64, max64, winner64, beta64 = masked_operator(
            q, k_pad, v_pad, u_pad, valid, dtype=torch.float64
        )
        score32, soft32, max32, winner32, beta32 = masked_operator(
            q.astype(np.float32),
            k_pad.astype(np.float32),
            v_pad.astype(np.float32),
            u_pad.astype(np.float32),
            valid,
            dtype=torch.float32,
        )
        changed_k, changed_v, changed_u = k_pad.copy(), v_pad.copy(), u_pad.copy()
        changed_k[:, n:] = rng.uniform(-100, 100, size=changed_k[:, n:].shape)
        changed_v[:, n:] = rng.uniform(-100, 100, size=changed_v[:, n:].shape)
        changed_u[:, n:] = rng.uniform(-100, 100, size=changed_u[:, n:].shape)
        _, pad_soft32, pad_max32, pad_winner32, pad_beta32 = masked_operator(
            q.astype(np.float32),
            changed_k.astype(np.float32),
            changed_v.astype(np.float32),
            changed_u.astype(np.float32),
            valid,
            dtype=torch.float32,
        )
        (
            grad_q,
            grad_k,
            grad_v,
            grad_scores,
            grad_u,
            grad_winners,
            dscore,
            du,
        ) = gradient_cases(rng, n)
        full_mask = core.causal_mask(n).numpy().astype("|u1", copy=False)
        prefix = f"n{n}"
        outputs.update(
            {
                f"{prefix}_q": q.astype("<f8"),
                f"{prefix}_k": k_pad.astype("<f8"),
                f"{prefix}_v": v_pad.astype("<f8"),
                f"{prefix}_u": u_pad.astype("<f8"),
                f"{prefix}_valid": valid.astype("|u1"),
                f"{prefix}_full_mask": full_mask,
                f"{prefix}_scores_ref": scores.astype("<f8"),
                f"{prefix}_soft_ref": soft.astype("<f8"),
                f"{prefix}_max_ref": maximum.astype("<f8"),
                f"{prefix}_winner_ref": winners.astype("<i8"),
                f"{prefix}_beta8_ref": b8.astype("<f8"),
                f"{prefix}_score64": score64.astype("<f8"),
                f"{prefix}_soft64": soft64.astype("<f8"),
                f"{prefix}_max64": max64.astype("<f8"),
                f"{prefix}_winner64": winner64.astype("<i8"),
                f"{prefix}_beta64": beta64.astype("<f8"),
                f"{prefix}_score32": score32.astype("<f4"),
                f"{prefix}_soft32": soft32.astype("<f4"),
                f"{prefix}_max32": max32.astype("<f4"),
                f"{prefix}_winner32": winner32.astype("<i8"),
                f"{prefix}_beta32": beta32.astype("<f4"),
                f"{prefix}_pad_soft32": pad_soft32.astype("<f4"),
                f"{prefix}_pad_max32": pad_max32.astype("<f4"),
                f"{prefix}_pad_winner32": pad_winner32.astype("<i8"),
                f"{prefix}_pad_beta32": pad_beta32.astype("<f4"),
                f"{prefix}_grad_q": grad_q,
                f"{prefix}_grad_k": grad_k,
                f"{prefix}_grad_v": grad_v,
                f"{prefix}_grad_scores": grad_scores,
                f"{prefix}_grad_u": grad_u,
                f"{prefix}_grad_winners": grad_winners,
                f"{prefix}_dscore": dscore,
                f"{prefix}_du": du,
            }
        )
    return outputs


def stage0_worker(index: int, worker_auth: Path) -> None:
    require_worker_authorization(worker_auth, "stage0", index=index)
    core.verify_runtime()
    payload = stage0_payload()
    user_ns, system_ns = self_cpu_diagnostics()
    save_npz_new(
        STAGE0_DIR / f"replay-{index}.npz",
        **payload,
        vmhwm=np.asarray([core.linux_vmhwm_bytes()], dtype="<i8"),
        diagnostic_cpu_user_ns=np.asarray([user_ns], dtype="<i8"),
        diagnostic_cpu_system_ns=np.asarray([system_ns], dtype="<i8"),
    )


def close_enough(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> bool:
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    return bool(np.all((absolute <= tolerance) | (relative <= tolerance)))


def verify_stage0() -> None:
    first, second = STAGE0_DIR / "replay-0.npz", STAGE0_DIR / "replay-1.npz"
    first_payload, second_payload = load_npz(first), load_npz(second)
    for diagnostic in (
        "vmhwm",
        "diagnostic_cpu_user_ns",
        "diagnostic_cpu_system_ns",
    ):
        first_payload.pop(diagnostic)
        second_payload.pop(diagnostic)
    replay_hash = canonical_tensor_hash(first_payload)
    if replay_hash != canonical_tensor_hash(second_payload):
        raise RuntimeError("bitwise canonical tensor replay mismatch")
    checks: dict[str, Any] = {"canonical_replay_hash": replay_hash}
    for n in (1, 4, 8, 16, 32):
        prefix = f"n{n}"
        scores = first_payload[f"{prefix}_scores_ref"]
        soft_ref = first_payload[f"{prefix}_soft_ref"]
        max_ref = first_payload[f"{prefix}_max_ref"]
        beta8 = first_payload[f"{prefix}_beta8_ref"]
        score64 = first_payload[f"{prefix}_score64"][:, :n]
        if not close_enough(score64, scores, 1e-10):
            raise RuntimeError("float64 score reference mismatch")
        if not close_enough(first_payload[f"{prefix}_soft64"], soft_ref, 1e-10):
            raise RuntimeError("float64 soft reference mismatch")
        if not close_enough(first_payload[f"{prefix}_max64"], max_ref, 1e-10):
            raise RuntimeError("float64 max reference mismatch")
        if not close_enough(first_payload[f"{prefix}_beta64"], beta8, 1e-10):
            raise RuntimeError("float64 finite-beta production mismatch")
        if not close_enough(first_payload[f"{prefix}_soft32"].astype(np.float64), soft_ref, 2e-5):
            raise RuntimeError("float32 soft mismatch")
        if not close_enough(first_payload[f"{prefix}_max32"].astype(np.float64), max_ref, 2e-5):
            raise RuntimeError("float32 max mismatch")
        if not close_enough(first_payload[f"{prefix}_beta32"].astype(np.float64), beta8, 2e-5):
            raise RuntimeError("float32 finite-beta mismatch")
        if not np.array_equal(first_payload[f"{prefix}_winner64"], first_payload[f"{prefix}_winner_ref"]):
            raise RuntimeError("float64 winner mismatch")
        if not np.array_equal(first_payload[f"{prefix}_soft32"], first_payload[f"{prefix}_pad_soft32"]):
            raise RuntimeError("padding changed soft output")
        if not np.array_equal(first_payload[f"{prefix}_max32"], first_payload[f"{prefix}_pad_max32"]):
            raise RuntimeError("padding changed max output")
        if not np.array_equal(first_payload[f"{prefix}_winner32"], first_payload[f"{prefix}_pad_winner32"]):
            raise RuntimeError("padding changed winner")
        if not np.array_equal(first_payload[f"{prefix}_beta32"], first_payload[f"{prefix}_pad_beta32"]):
            raise RuntimeError("padding changed finite-beta output")
        delta = beta8 - max_ref
        if delta.min() < -1e-12 or delta.max() > math.log(n) / 8.0 + 1e-12:
            raise RuntimeError("finite-beta bound")
        grad_winners = first_payload[f"{prefix}_grad_winners"]
        expected_du = np.zeros_like(first_payload[f"{prefix}_du"])
        rows = np.arange(1024)[:, None]
        channels = np.arange(2)[None, :]
        expected_du[rows, grad_winners, channels] = 1.0
        expected_dscore = expected_du.sum(axis=-1)
        if not np.array_equal(first_payload[f"{prefix}_du"], expected_du):
            raise RuntimeError("dL/du is not the winner indicator")
        if not np.array_equal(first_payload[f"{prefix}_dscore"], expected_dscore):
            raise RuntimeError("dL/dS is not channels-won")
        checks[prefix] = {
            "beta_bound_min": float(delta.min()),
            "beta_bound_max": float(delta.max()),
        }
    hard_copy = float(np.asarray([0.0, 100.0])[np.argmax(np.asarray([10.0, 0.0]))])
    tropical = float(np.max(np.asarray([10.0, 0.0]) + np.asarray([0.0, 100.0])))
    if hard_copy != 0.0 or tropical != 100.0:
        raise RuntimeError("hard-copy/max-plus counterexample failed")
    core.atomic_json(STAGE0_DIR / "decision.json", {"status": "pass", "checks": checks})


def run_stage0(receipt: Path) -> None:
    core.require_receipt("RUN STAGE 0", receipt)
    core.verify_runtime()
    for index in (0, 1):
        worker_auth = worker_authorization("stage0", receipt, index=index)
        run_reported_subprocess(
            worker_command(
                "stage0-worker",
                "--index",
                str(index),
                "--worker-auth",
                str(worker_auth),
            ),
            STAGE0_DIR / f"worker-{index}.json",
        )
    verify_stage0()
    seal_prerequisite_decision("stage0")


def raw_tokens(fixture: Mapping[str, np.ndarray]) -> np.ndarray:
    raw = fixture.get("raw_tokens")
    if raw is None or raw.dtype != np.dtype("<f4"):
        raise RuntimeError("stored little-endian float32 raw tokens missing")
    return raw


def surface_features(tokens: np.ndarray) -> np.ndarray:
    records, query = tokens[:, :-1], tokens[:, -1]
    n = records.shape[1]
    parts = [np.full((tokens.shape[0], 1), n / 32.0)]
    parts.extend(
        function(records, axis=1)
        for function in (
            np.mean,
            lambda x, axis: np.std(x, axis=axis, ddof=0),
            np.min,
            np.max,
            np.sum,
            lambda x, axis: np.sum(np.abs(x), axis=axis),
            lambda x, axis: np.sum(np.square(x), axis=axis),
        )
    )
    parts.extend((records[:, 0], records[:, -1], query))
    return np.concatenate(parts, axis=1).astype(np.float32)


class SurfaceLinear(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, 2, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class ProbeMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, 64, bias=True)
        self.fc2 = nn.Linear(64, 2, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x), approximate="none"))


def infinite_batches(size: int, batch: int, seed: int) -> Iterator[np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    current = rng.permutation(size)
    cursor = 0
    while True:
        pieces: list[np.ndarray] = []
        needed = batch
        while needed:
            available = size - cursor
            take = min(available, needed)
            pieces.append(current[cursor : cursor + take])
            cursor += take
            needed -= take
            if cursor == size:
                current = rng.permutation(size)
                cursor = 0
        yield np.concatenate(pieces)


def load_probe_corpus(view: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    feature_parts: list[np.ndarray] = []
    labels = {name: [] for name in ("ys", "ym", "y")}
    for n in range(4, 9):
        fixture = load_fixture(f"probe-n{n}.npz")
        tokens = raw_tokens(fixture) if view == "raw" else fixture["tokens"]
        feature_parts.append(surface_features(tokens))
        for name in labels:
            labels[name].append(fixture[name])
    return np.concatenate(feature_parts), {name: np.concatenate(parts) for name, parts in labels.items()}


def standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    sd = train.std(axis=0, dtype=np.float64)
    return ((train - mean) / (sd + 1e-6)).astype(np.float32), mean, sd


def eval_probe(model: nn.Module, features: np.ndarray, labels: np.ndarray) -> float:
    mistakes = 0
    with torch.inference_mode():
        for start in range(0, len(features), 4096):
            prediction = model(torch.from_numpy(features[start : start + 4096])).argmax(dim=-1).numpy()
            mistakes += int(np.count_nonzero(prediction != labels[start : start + 4096]))
    return 1.0 - mistakes / len(features)


def train_probe_model(
    model: nn.Module,
    architecture_id: str,
    seed: int,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> nn.Module:
    core.initialize_model(model, architecture_id, seed)
    optimizer = core.optimizer_for(model, learning_rate, weight_decay=0.0)
    batches = infinite_batches(len(features), batch_size, seed)
    for _ in range(steps):
        indices = next(batches)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(torch.from_numpy(features[indices])), torch.from_numpy(labels[indices]).long()
        )
        loss.backward()
        optimizer.step()
    return model.eval()


def model_tensor_hash(model: nn.Module) -> str:
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in sorted(model.state_dict().items())
    }
    return canonical_tensor_hash(arrays)


def train_probe_twice(
    factory: Any,
    architecture_id: str,
    seed: int,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[nn.Module, str]:
    models = [
        train_probe_model(
            factory(),
            architecture_id,
            seed,
            features,
            labels,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        for _ in range(2)
    ]
    hashes = [model_tensor_hash(model) for model in models]
    if hashes[0] != hashes[1]:
        raise RuntimeError(f"probe replay mismatch for {architecture_id}")
    return models[0], hashes[0]


def fixed_probe_slices(view: str) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    output = {}
    for name in ("id", "l16", "l32", "v2", "l32v2"):
        fixture = load_fixture(f"eval-{name}.npz")
        tokens = raw_tokens(fixture) if view == "raw" else fixture["tokens"]
        output[name] = (surface_features(tokens), {label: fixture[label] for label in ("ys", "ym", "y")})
    return output


def run_surface_probes() -> tuple[dict[str, Any], dict[str, str]]:
    results: dict[str, Any] = {}
    replay_hashes: dict[str, str] = {}
    for view_id, view in enumerate(("raw", "mixed")):
        features, labels = load_probe_corpus(view)
        standardized, mean, sd = standardize(features)
        slices = fixed_probe_slices(view)
        for label_id, label in enumerate(("ys", "ym", "y")):
            seed = 840200 + 10 * view_id + label_id
            model, replay_hash = train_probe_twice(
                lambda: SurfaceLinear(standardized.shape[1]),
                f"probe.surface.{view}.{label}",
                seed,
                standardized,
                labels[label],
                steps=2000,
                batch_size=4096,
                learning_rate=1e-2,
            )
            key = f"surface_{view}_{label}"
            cell = {}
            for slice_name, (slice_features, slice_labels) in slices.items():
                transformed = ((slice_features - mean) / (sd + 1e-6)).astype(np.float32)
                cell[slice_name] = eval_probe(model, transformed, slice_labels[label])
            results[key] = cell
            replay_hashes[key] = replay_hash
    return results, replay_hashes


def selected_record_features(
    tokens: np.ndarray, start_index: int
) -> tuple[np.ndarray, int]:
    n = tokens.shape[1] - 1
    selected = np.empty((len(tokens), 29), dtype=np.float32)
    for local in range(len(tokens)):
        rng = np.random.Generator(np.random.PCG64(840300 + start_index + local))
        record_index = int(rng.integers(0, n))
        selected[local] = np.concatenate(
            (tokens[local, -1], tokens[local, record_index], [n / 32.0])
        )
    return selected, start_index + len(tokens)


def query_and_record_probe_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_parts, record_parts, labels = [], [], []
    example_index = 0
    for n in range(4, 9):
        fixture = load_fixture(f"probe-n{n}.npz")
        tokens = fixture["tokens"]
        query_parts.append(tokens[:, -1])
        selected, example_index = selected_record_features(tokens, example_index)
        record_parts.append(selected)
        labels.append(fixture["y"])
    return np.concatenate(query_parts), np.concatenate(record_parts), np.concatenate(labels)


def run_shortcut_probes() -> tuple[dict[str, Any], dict[str, str]]:
    query, one_record, labels = query_and_record_probe_data()
    results: dict[str, Any] = {}
    replay_hashes: dict[str, str] = {}
    for name, features, seed in (
        ("query_only", query, 840301),
        ("one_record", one_record, 840302),
    ):
        model, replay_hash = train_probe_twice(
            lambda: ProbeMLP(features.shape[1]),
            f"probe.{name}.y",
            seed,
            features,
            labels,
            steps=1500,
            batch_size=128,
            learning_rate=1e-3,
        )
        cell = {}
        example_index = 0
        for slice_name in SLICE_NAMES:
            fixture = load_fixture(f"eval-{slice_name}.npz")
            tokens = fixture["tokens"]
            if name == "query_only":
                eval_features = tokens[:, -1]
            else:
                eval_features, example_index = selected_record_features(tokens, example_index)
            cell[slice_name] = eval_probe(model, eval_features, fixture["y"])
        results[name] = cell
        replay_hashes[name] = replay_hash
    return results, replay_hashes


def run_probes(receipt: Path) -> None:
    core.require_receipt("RUN PROBES", receipt)
    core.verify_runtime()
    fixture_integrity = verify_fixture_integrity()
    surface, surface_hashes = run_surface_probes()
    shortcut, shortcut_hashes = run_shortcut_probes()
    results = {**surface, **shortcut}
    passed = all(value <= 0.55 for cell in results.values() for value in cell.values())
    core.atomic_json(
        PROBE_DIR / "results.json",
        {
            "passed": passed,
            "accuracies": results,
            "bitwise_replay_hashes": {**surface_hashes, **shortcut_hashes},
            "evaluation_example_index_scope": "global across fixed slices in frozen order",
            "fixture_integrity": fixture_integrity,
        },
    )
    seal_prerequisite_decision("probes")


def verify_fixture_integrity() -> Mapping[str, Any]:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    expected_files = {
        **{name: f"eval-{name}.npz" for name in SLICE_NAMES},
        **{f"dev_n{n}": f"dev-n{n}.npz" for n in range(4, 9)},
        **{f"probe_n{n}": f"probe-n{n}.npz" for n in range(4, 9)},
        **{name: f"timing-{name.removeprefix('timing_')}.npz" for name in ("timing_len9", "timing_len17", "timing_len33")},
    }
    if set(manifest.get("hashes", {})) != set(expected_files):
        raise RuntimeError("fixture manifest key mismatch")
    for key, filename in expected_files.items():
        if core.sha256_file(FIXTURE_DIR / filename) != manifest["hashes"][key]:
            raise RuntimeError(f"fixture hash mismatch {filename}")
    specifications = {
        **{f"eval-{name}.npz": 10_000 for name in SLICE_NAMES},
        **{f"dev-n{n}.npz": 2_000 for n in range(4, 9)},
        **{f"probe-n{n}.npz": 20_000 for n in range(4, 9)},
        "timing-len9.npz": 32,
        "timing-len17.npz": 32,
        "timing-len33.npz": 32,
    }
    checked = {}
    for filename, expected_count in specifications.items():
        fixture = load_fixture(filename)
        count = len(fixture["tokens"])
        if count != expected_count or fixture["tokens"].shape != fixture["raw_tokens"].shape:
            raise RuntimeError(f"fixture size mismatch {filename}")
        if not np.array_equal(
            np.sort(fixture["dataset_permutation"]), np.arange(expected_count)
        ):
            raise RuntimeError(f"stored dataset permutation invalid {filename}")
        quadrants = fixture["ys"] * 2 + fixture["ym"]
        expected_per_quadrant = expected_count // 4
        if np.bincount(quadrants, minlength=4).tolist() != [expected_per_quadrant] * 4:
            raise RuntimeError(f"fixture is not quadrant balanced {filename}")
        if not np.array_equal(fixture["y"], np.bitwise_xor(fixture["ys"], fixture["ym"])):
            raise RuntimeError(f"XOR labels inconsistent {filename}")
        scores = np.einsum("bnr,br->bn", fixture["k"], fixture["q"]) / 2.0
        shifted = scores - scores.max(axis=1, keepdims=True)
        probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        soft = np.einsum("bn,bnc->bc", probability, fixture["v"])
        maximum = np.max(scores[..., None] + fixture["u"], axis=1)
        recomputed_ys = (soft[:, 0] > soft[:, 1]).astype("<i8")
        recomputed_ym = (maximum[:, 0] > maximum[:, 1]).astype("<i8")
        if not np.allclose(scores, fixture["scores"], rtol=0.0, atol=2e-15):
            raise RuntimeError(f"stored QK scores inconsistent {filename}")
        if not np.array_equal(recomputed_ys, fixture["ys"]):
            raise RuntimeError(f"stored soft labels inconsistent {filename}")
        if not np.array_equal(recomputed_ym, fixture["ym"]):
            raise RuntimeError(f"stored max-plus labels inconsistent {filename}")
        reconstructed = np.asarray(
            fixture["raw_tokens"].astype(np.float64) @ core.MIXER,
            dtype=np.float32,
        )
        if not np.allclose(reconstructed, fixture["tokens"], rtol=2e-6, atol=2e-6):
            raise RuntimeError(f"stored mixer output inconsistent {filename}")
        n = fixture["tokens"].shape[1] - 1
        if not (
            np.all(fixture["raw_tokens"][:, :n, 0] == 1.0)
            and np.all(fixture["raw_tokens"][:, :n, 1] == 0.0)
            and np.all(fixture["raw_tokens"][:, n, 0] == 0.0)
            and np.all(fixture["raw_tokens"][:, n, 1] == 1.0)
        ):
            raise RuntimeError(f"record/query tags inconsistent {filename}")
        if not (
            np.array_equal(fixture["raw_tokens"][:, :n, 2:6], fixture["k"].astype("<f4"))
            and np.array_equal(fixture["raw_tokens"][:, :n, 6:8], fixture["v"].astype("<f4"))
            and np.array_equal(fixture["raw_tokens"][:, :n, 8:10], fixture["u"].astype("<f4"))
            and np.array_equal(fixture["raw_tokens"][:, n, 10:14], fixture["q"].astype("<f4"))
        ):
            raise RuntimeError(f"raw token semantic fields inconsistent {filename}")
        checked[filename] = {
            "count": count,
            "n": n,
            "quadrants": [expected_per_quadrant] * 4,
        }
    return {"passed": True, "checked": checked}


def labels_from_fixture(fixture: Mapping[str, np.ndarray], start: int, stop: int) -> dict[str, Tensor]:
    return {name: torch.from_numpy(fixture[name][start:stop]).long() for name in ("ys", "ym", "y")}


def evaluate_fixture_counts(
    model: nn.Module,
    fixture: Mapping[str, np.ndarray],
    *,
    lesion: Literal["soft", "max"] | None = None,
    batch_size: int = 128,
) -> tuple[dict[str, float], dict[str, int], int]:
    mistakes = {name: 0 for name in ("ys", "ym", "y")}
    mask = core.causal_mask(fixture["tokens"].shape[1] - 1)
    with torch.inference_mode():
        for start in range(0, len(fixture["tokens"]), batch_size):
            stop = min(start + batch_size, len(fixture["tokens"]))
            tokens = torch.from_numpy(fixture["tokens"][start:stop])
            if isinstance(model, core.TinyTransformer):
                logits = model(tokens, mask, lesion=lesion)
            else:
                if lesion:
                    raise ValueError("lesion only candidate")
                logits = model(tokens, mask)
            for name in mistakes:
                predicted = logits[name].argmax(dim=-1).numpy()
                mistakes[name] += int(np.count_nonzero(predicted != fixture[name][start:stop]))
    total = len(fixture["tokens"])
    return ({name: mistakes[name] / total for name in mistakes}, mistakes, total)


def evaluate_fixture(
    model: nn.Module,
    fixture: Mapping[str, np.ndarray],
    *,
    lesion: Literal["soft", "max"] | None = None,
    batch_size: int = 128,
) -> dict[str, float]:
    return evaluate_fixture_counts(
        model, fixture, lesion=lesion, batch_size=batch_size
    )[0]


def evaluate_dev(model: nn.Module) -> float:
    mistakes = total = 0
    for n in range(4, 9):
        fixture = load_fixture(f"dev-n{n}.npz")
        _, counts, count = evaluate_fixture_counts(model, fixture)
        mistakes += counts["y"]
        total += count
    return mistakes / total


def train_model(
    architecture: str,
    seed: int,
    learning_rate: float,
    *,
    r_attention: int = 1,
    r_ffn: int = 1,
) -> nn.Module:
    model = core.build_model(
        architecture, seed, r_attention=r_attention, r_ffn=r_ffn
    ).train()
    optimizer = core.optimizer_for(model, learning_rate)
    for step in range(1500):
        tokens, labels, n = core.training_batch(seed, step)
        core.train_step(model, optimizer, tokens, labels, core.causal_mask(n))
    return model.eval()


def save_model_new(path: Path, model: nn.Module) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("xb") as handle:
        torch.save(model.state_dict(), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)


def selected_time_repeats() -> tuple[int, int]:
    summary = json.loads((TIMING_DIR / "summary.json").read_text(encoding="utf-8"))
    if summary["selected"] is None:
        raise RuntimeError("TIME MATCH UNAVAILABLE")
    _, ra, rf = summary["selected"]
    return int(ra), int(rf)


def tune_architecture(architecture: str) -> tuple[float, int, int]:
    started = time.monotonic_ns()
    ra, rf = selected_time_repeats() if architecture == "soft_time_matched" else (1, 1)
    scores = {}
    for learning_rate in (3e-4, 1e-3):
        model = train_model(architecture, 842000, learning_rate, r_attention=ra, r_ffn=rf)
        scores[str(learning_rate)] = evaluate_dev(model)
    chosen = 3e-4 if scores[str(3e-4)] <= scores[str(1e-3)] else 1e-3
    core.atomic_json(
        TRAIN_DIR / architecture / "tuning.json",
        {
            "scores": scores,
            "chosen": chosen,
            "ra": ra,
            "rf": rf,
            "active_ns": time.monotonic_ns() - started,
            "vmhwm_bytes": core.linux_vmhwm_bytes(),
        },
    )
    return chosen, ra, rf


def run_architecture(architecture: str, receipt: Path) -> None:
    core.require_receipt(f"RUN {architecture.upper()}", receipt)
    core.verify_runtime()
    learning_rate, ra, rf = tune_architecture(architecture)
    for seed in core.EVAL_SEEDS:
        seed_started = time.monotonic_ns()
        model = train_model(architecture, seed, learning_rate, r_attention=ra, r_ffn=rf)
        metrics = {}
        for name in SLICE_NAMES:
            metrics[name] = evaluate_fixture(model, load_fixture(f"eval-{name}.npz"))
        diagnostics = candidate_diagnostics(model, seed) if architecture == "candidate" else None
        model_path = TRAIN_DIR / architecture / f"seed-{seed}.pt"
        save_model_new(model_path, model)
        core.atomic_json(
            TRAIN_DIR / architecture / f"seed-{seed}.json",
            {
                "seed": seed,
                "learning_rate": learning_rate,
                "metrics": metrics,
                "diagnostics": diagnostics,
                "model_sha256": core.sha256_file(model_path),
                "active_ns": time.monotonic_ns() - seed_started,
                "vmhwm_bytes": core.linux_vmhwm_bytes(),
            },
        )
    run_trained_timing(architecture, receipt)
    seal_stage_decision(architecture)


def require_candidate_model(model: nn.Module) -> core.TinyTransformer:
    if not isinstance(model, core.TinyTransformer) or model.architecture != "candidate":
        raise TypeError("candidate diagnostics require the candidate model")
    return model


def array_hash(array: np.ndarray, name: str) -> str:
    return canonical_tensor_hash({name: np.ascontiguousarray(array)})


def recursive_route_shuffle(
    model: nn.Module,
    fixture: Mapping[str, np.ndarray],
    seed: int,
    slice_name: str,
    branch: Literal["soft", "max"],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    candidate = require_candidate_model(model)
    if slice_name not in SLICE_IDS:
        raise ValueError(slice_name)
    branch_id = 0 if branch == "soft" else 1
    permutation_seed = 850000 + 100 * seed + SLICE_IDS[slice_name] + 10 * branch_id
    permutation = core.exact_sattolo(permutation_seed, len(fixture["tokens"]))
    permutation_path = (
        TRAIN_DIR
        / "candidate"
        / "perturbations"
        / f"seed-{seed}-{slice_name}-{branch}-derangement.npz"
    )
    save_npz_new(
        permutation_path,
        permutation=permutation.astype("<i8", copy=False),
        seed=np.asarray([permutation_seed], dtype="<i8"),
    )
    mask = core.causal_mask(fixture["tokens"].shape[1] - 1)
    score1_parts: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(fixture["tokens"]), batch_size):
            tokens = torch.from_numpy(fixture["tokens"][start : start + batch_size])
            hidden = candidate.input(tokens)
            _, trace = candidate.block[0].attn(
                candidate.block[0].norm_attn(hidden), mask, collect_trace=True
            )
            if trace is None:
                raise RuntimeError("block-one route trace missing")
            score1_parts.append(trace.scores[:, :, -1, :].cpu())
    score1 = torch.cat(score1_parts, dim=0)
    score1_array = score1.numpy()
    score1_hash = array_hash(score1_array, "block1_query_scores")

    hidden1_parts: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(fixture["tokens"]), batch_size):
            stop = min(start + batch_size, len(fixture["tokens"]))
            tokens = torch.from_numpy(fixture["tokens"][start:stop])
            hidden = candidate.input(tokens)
            donor = score1[torch.from_numpy(permutation[start:stop])]
            hidden, _ = candidate.block[0](
                hidden,
                mask,
                overrides=[{branch: donor}],
            )
            hidden1_parts.append(hidden.cpu())
    hidden1 = torch.cat(hidden1_parts, dim=0)
    hidden1_array = hidden1.numpy()
    hidden1_hash = array_hash(hidden1_array, "block1_hidden")

    score2_parts: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(hidden1), batch_size):
            hidden = hidden1[start : start + batch_size]
            _, trace = candidate.block[1].attn(
                candidate.block[1].norm_attn(hidden), mask, collect_trace=True
            )
            if trace is None:
                raise RuntimeError("block-two route trace missing")
            score2_parts.append(trace.scores[:, :, -1, :].cpu())
    score2 = torch.cat(score2_parts, dim=0)
    score2_array = score2.numpy()
    score2_hash = array_hash(score2_array, "block2_query_scores")

    logits_parts = {name: [] for name in ("ys", "ym", "y")}
    with torch.inference_mode():
        for start in range(0, len(hidden1), batch_size):
            stop = min(start + batch_size, len(hidden1))
            donor = score2[torch.from_numpy(permutation[start:stop])]
            hidden, _ = candidate.block[1](
                hidden1[start:stop],
                mask,
                overrides=[{branch: donor}],
            )
            query = candidate.final_norm(hidden[:, -1])
            for name, head in candidate.readout.items():
                logits_parts[name].append(head(query).cpu())
    logits = {name: torch.cat(parts).numpy() for name, parts in logits_parts.items()}
    mistakes = {
        name: int(np.count_nonzero(values.argmax(axis=-1) != fixture[name]))
        for name, values in logits.items()
    }
    count = len(fixture["tokens"])
    return {
        "errors": {name: mistakes[name] / count for name in mistakes},
        "hashes": {
            "derangement": core.sha256_file(permutation_path),
            "block1_scores": score1_hash,
            "block1_hidden": hidden1_hash,
            "block2_scores": score2_hash,
            "final_logits": canonical_tensor_hash(logits),
        },
    }


def candidate_branch_health(
    model: nn.Module,
    seed: int,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    candidate = require_candidate_model(model)
    id_fixture = load_fixture("eval-id.npz")
    mask = core.causal_mask(id_fixture["tokens"].shape[1] - 1)
    raw_parts: list[list[np.ndarray]] = [[], []]
    gradient_squared = np.zeros((2, 4, 4), dtype=np.float64)
    for start in range(0, 1024, batch_size):
        stop = start + batch_size
        candidate.zero_grad(set_to_none=True)
        tokens = torch.from_numpy(id_fixture["tokens"][start:stop])
        labels = labels_from_fixture(id_fixture, start, stop)
        logits, traces = candidate.trace_forward(tokens, mask)
        for trace in traces:
            if trace.raw_max is None:
                raise RuntimeError("candidate max trace missing")
            trace.raw_max.retain_grad()
        core.total_loss(logits, labels).backward()
        for layer, trace in enumerate(traces):
            if trace.raw_max is None or trace.raw_max.grad is None:
                raise RuntimeError("candidate max gradient missing")
            raw_parts[layer].append(trace.raw_max[:, :, -1, :].detach().numpy())
            gradient = trace.raw_max.grad[:, :, -1, :].detach().numpy().astype(np.float64)
            gradient_squared[layer] += np.square(gradient).sum(axis=0)
    raw = np.stack([np.concatenate(parts, axis=0) for parts in raw_parts])
    population_sd = raw.std(axis=1, ddof=0)
    gradient_norm = np.sqrt(gradient_squared)
    active = (population_sd > 1e-3) & (gradient_norm > 1e-8)
    active_count = int(active.sum())
    active_pass = active_count >= 24

    diversity: dict[str, Any] = {}
    diversity_pass = True
    contribution: dict[str, Any] = {}
    contribution_pass = True
    candidate.zero_grad(set_to_none=True)
    for slice_name in SLICE_NAMES:
        fixture = load_fixture(f"eval-{slice_name}.npz")
        slice_mask = core.causal_mask(fixture["tokens"].shape[1] - 1)
        winner_parts: list[list[np.ndarray]] = [[], []]
        contribution_sums = {
            (layer, branch): [0.0, 0.0, 0]
            for layer in range(2)
            for branch in ("soft", "max")
        }
        with torch.inference_mode():
            for start in range(0, len(fixture["tokens"]), batch_size):
                tokens = torch.from_numpy(fixture["tokens"][start : start + batch_size])
                _, traces = candidate.trace_forward(tokens, slice_mask)
                for layer, trace in enumerate(traces):
                    if trace.winners is None or trace.soft_pre_o is None or trace.max_pre_o is None:
                        raise RuntimeError("candidate branch trace missing")
                    winner_parts[layer].append(trace.winners[:, :, -1, :].cpu().numpy())
                    attention = candidate.block[layer].attn
                    if not isinstance(attention, core.MixedAttention):
                        raise RuntimeError("candidate attention type changed")
                    intact = attention.o(trace.pre_o[:, -1])
                    denominator = torch.linalg.vector_norm(intact, dim=-1)
                    for branch, branch_pre in (
                        ("soft", trace.soft_pre_o),
                        ("max", trace.max_pre_o),
                    ):
                        numerator = torch.linalg.vector_norm(
                            attention.o(branch_pre[:, -1]), dim=-1
                        )
                        cell = contribution_sums[(layer, branch)]
                        cell[0] += float(numerator.sum())
                        cell[1] += float(denominator.sum())
                        cell[2] += len(tokens)
        n = fixture["tokens"].shape[1] - 1
        slice_diversity: dict[str, Any] = {}
        for layer, parts in enumerate(winner_parts):
            winners = np.concatenate(parts, axis=0)
            for head in range(4):
                for channel in range(4):
                    if not active[layer, head, channel]:
                        continue
                    channel_winners = winners[:, head, channel]
                    if np.any(channel_winners < 0) or np.any(channel_winners >= n):
                        raise RuntimeError("illegal query winner position")
                    counts = np.bincount(channel_winners, minlength=n)
                    probabilities = counts / counts.sum()
                    nonzero = probabilities[probabilities > 0]
                    entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(n))
                    passed = entropy >= 0.85 and bool(np.all(counts > 0))
                    diversity_pass = diversity_pass and passed
                    slice_diversity[f"layer{layer}.head{head}.channel{channel}"] = {
                        "entropy": entropy,
                        "minimum_wins": int(counts.min()),
                        "passed": passed,
                    }
        diversity[slice_name] = slice_diversity
        for (layer, branch), (numerator_sum, denominator_sum, count) in contribution_sums.items():
            mean_numerator = numerator_sum / count
            mean_denominator = denominator_sum / count
            ratio = mean_numerator / mean_denominator if mean_denominator else math.inf
            passed = ratio >= 0.10 and math.isfinite(ratio)
            contribution_pass = contribution_pass and passed
            contribution[f"{slice_name}.layer{layer}.{branch}"] = {
                "ratio": ratio,
                "passed": passed,
            }
    return {
        "seed": seed,
        "active_count": active_count,
        "active_fraction": active_count / 32.0,
        "population_sd": population_sd.tolist(),
        "gradient_norm": gradient_norm.tolist(),
        "active": active.tolist(),
        "diversity": diversity,
        "contribution": contribution,
        "passed": active_pass and diversity_pass and contribution_pass,
        "component_passes": {
            "active": active_pass,
            "diversity": diversity_pass,
            "contribution": contribution_pass,
        },
    }


def candidate_diagnostics(model: nn.Module, seed: int) -> dict[str, Any]:
    candidate = require_candidate_model(model)
    lesions: dict[str, Any] = {branch: {} for branch in ("soft", "max")}
    shuffles: dict[str, Any] = {branch: {} for branch in ("soft", "max")}
    for branch in ("soft", "max"):
        for slice_name in OOD_NAMES:
            fixture = load_fixture(f"eval-{slice_name}.npz")
            lesions[branch][slice_name] = evaluate_fixture(
                candidate, fixture, lesion=branch
            )
            shuffles[branch][slice_name] = recursive_route_shuffle(
                candidate, fixture, seed, slice_name, branch
            )
    return {
        "lesions": lesions,
        "route_shuffles": shuffles,
        "health": candidate_branch_health(candidate, seed),
    }


def load_trained_model(architecture: str, seed: int = 842100) -> nn.Module:
    tuning = json.loads(
        (TRAIN_DIR / architecture / "tuning.json").read_text(encoding="utf-8")
    )
    model = core.build_model(
        architecture,
        seed,
        r_attention=int(tuning["ra"]),
        r_ffn=int(tuning["rf"]),
    )
    model_path = TRAIN_DIR / architecture / f"seed-{seed}.pt"
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def trained_timing_worker(
    architecture: str, round_id: int, worker_auth: Path
) -> None:
    require_worker_authorization(
        worker_auth,
        "trained-timing",
        round_id=round_id,
        architecture=architecture,
    )
    core.verify_runtime()
    model = load_trained_model(architecture)
    names = ("len9", "len17", "len33")
    architecture_id = core.ARCHITECTURES.index(architecture)
    samples: dict[str, np.ndarray] = {}
    for length_id in (
        (round_id + architecture_id + offset) % 3 for offset in range(3)
    ):
        name = names[length_id]
        fixture = load_fixture(f"timing-{name}.npz")
        tokens = torch.from_numpy(fixture["tokens"])
        mask = torch.from_numpy(fixture["mask"])
        with torch.inference_mode():
            for _ in range(10):
                returned = model(tokens, mask)
            measured = np.empty(50, dtype="<i8")
            for index in range(50):
                before = time.perf_counter_ns()
                returned = model(tokens, mask)
                measured[index] = time.perf_counter_ns() - before
            if returned is None:
                raise RuntimeError("unmaterialized trained timing result")
        samples[name] = measured
    user_ns, system_ns = self_cpu_diagnostics()
    save_npz_new(
        TRAIN_DIR / architecture / "timing" / f"round-{round_id:02d}.npz",
        **samples,
        vmhwm=np.asarray([core.linux_vmhwm_bytes()], dtype="<i8"),
        diagnostic_cpu_user_ns=np.asarray([user_ns], dtype="<i8"),
        diagnostic_cpu_system_ns=np.asarray([system_ns], dtype="<i8"),
    )


def summarize_trained_timing(architecture: str) -> None:
    by_length: dict[str, list[int]] = {
        name: [] for name in ("len9", "len17", "len33")
    }
    vmhwm: list[int] = []
    for round_id in range(20):
        payload = load_npz(
            TRAIN_DIR / architecture / "timing" / f"round-{round_id:02d}.npz"
        )
        for name in by_length:
            by_length[name].extend(int(value) for value in payload[name])
        vmhwm.append(int(payload["vmhwm"][0]))
    metrics: dict[str, Any] = {"vmhwm": vmhwm, "maximum_vmhwm": max(vmhwm)}
    for name, values in by_length.items():
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        metrics[name] = {
            "p50": median,
            "p95": float(np.quantile(array, 0.95)),
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
            "mad_ratio": float(np.median(np.abs(array - median)) / median),
        }
    core.atomic_json(TRAIN_DIR / architecture / "timing-summary.json", metrics)


def run_trained_timing(architecture: str, receipt: Path) -> None:
    for round_id in range(20):
        worker_auth = worker_authorization(
            "trained-timing",
            receipt,
            round_id=round_id,
            architecture=architecture,
        )
        run_reported_subprocess(
            worker_command(
                "trained-timing-worker",
                "--architecture",
                architecture,
                "--round",
                str(round_id),
                "--worker-auth",
                str(worker_auth),
            ),
            TRAIN_DIR
            / architecture
            / "timing"
            / f"round-{round_id:02d}-worker.json",
        )
    summarize_trained_timing(architecture)


def macro_error(record: Mapping[str, Any], label: str) -> float:
    return float(np.mean([record["metrics"][name][label] for name in OOD_NAMES]))


def diagnostic_macro(
    record: Mapping[str, Any],
    kind: Literal["lesions", "route_shuffles"],
    branch: Literal["soft", "max"],
    label: str,
) -> float:
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("candidate diagnostics missing")
    values = []
    for name in OOD_NAMES:
        cell = diagnostics[kind][branch][name]
        values.append(float(cell[label] if kind == "lesions" else cell["errors"][label]))
    return float(np.mean(values))


def architecture_records(architecture: str) -> list[dict[str, Any]]:
    records = []
    for seed in core.EVAL_SEEDS:
        path = TRAIN_DIR / architecture / f"seed-{seed}.json"
        if not path.exists():
            raise RuntimeError(f"incomplete architecture contrast {architecture}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("seed") != seed:
            raise RuntimeError("training seed receipt mismatch")
        records.append(record)
    return records


def gate_summary(
    name: str,
    values: Sequence[float],
    *,
    direction: Literal["lower", "upper"],
    threshold: float,
    family: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"invalid frozen seed vector for {name}")
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    radius = core.TCRIT * sd / 4.0
    unadjusted_radius = UNADJUSTED_T95 * sd / 4.0
    lower, upper = mean - radius, mean + radius
    passed = lower >= threshold if direction == "lower" else upper <= threshold
    result: dict[str, Any] = {
        "name": name,
        "family": family,
        "direction": direction,
        "threshold": threshold,
        "values": [float(value) for value in values],
        "mean": mean,
        "sample_sd": sd,
        "simultaneous_lower": lower,
        "simultaneous_upper": upper,
        "unadjusted_95_two_sided_lower": mean - unadjusted_radius,
        "unadjusted_95_two_sided_upper": mean + unadjusted_radius,
        "passed": bool(passed),
    }
    if extra:
        result.update(extra)
    return result


def candidate_gates(candidate: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for label in ("ys", "ym"):
        values = [macro_error(record, label) for record in candidate]
        gates.append(
            gate_summary(
                f"candidate.competence.{label}",
                values,
                direction="upper",
                threshold=0.20,
                family="competence",
            )
        )
    intact = {
        label: [macro_error(record, label) for record in candidate]
        for label in ("ys", "ym", "y")
    }
    lesion_specs = (
        ("soft", "ys", 1.2),
        ("soft", "y", 1.1),
        ("max", "ym", 1.2),
        ("max", "y", 1.1),
    )
    for branch, label, factor in lesion_specs:
        perturbed = [
            diagnostic_macro(record, "lesions", branch, label) for record in candidate
        ]
        relative_values = [
            changed - factor * base for changed, base in zip(perturbed, intact[label])
        ]
        absolute_values = [
            changed - base for changed, base in zip(perturbed, intact[label])
        ]
        gates.append(
            gate_summary(
                f"causal.lesion.{branch}.{label}.relative",
                relative_values,
                direction="lower",
                threshold=0.0,
                family="causal",
            )
        )
        gates.append(
            gate_summary(
                f"causal.lesion.{branch}.{label}.absolute",
                absolute_values,
                direction="lower",
                threshold=0.01,
                family="causal",
            )
        )
    for branch, label in (("soft", "ys"), ("max", "ym")):
        perturbed = [
            diagnostic_macro(record, "route_shuffles", branch, label)
            for record in candidate
        ]
        gates.append(
            gate_summary(
                f"causal.route_shuffle.{branch}.{label}.relative",
                [changed - 1.2 * base for changed, base in zip(perturbed, intact[label])],
                direction="lower",
                threshold=0.0,
                family="causal",
            )
        )
        gates.append(
            gate_summary(
                f"causal.route_shuffle.{branch}.{label}.absolute",
                [changed - base for changed, base in zip(perturbed, intact[label])],
                direction="lower",
                threshold=0.01,
                family="causal",
            )
        )
    soft_match = [
        diagnostic_macro(record, "lesions", "soft", "ys") - macro_error(record, "ys")
        for record in candidate
    ]
    soft_other = [
        diagnostic_macro(record, "lesions", "soft", "ym") - macro_error(record, "ym")
        for record in candidate
    ]
    max_match = [
        diagnostic_macro(record, "lesions", "max", "ym") - macro_error(record, "ym")
        for record in candidate
    ]
    max_other = [
        diagnostic_macro(record, "lesions", "max", "ys") - macro_error(record, "ys")
        for record in candidate
    ]
    gates.append(
        gate_summary(
            "causal.double_dissociation.soft",
            [match - other for match, other in zip(soft_match, soft_other)],
            direction="lower",
            threshold=0.01,
            family="double_dissociation",
        )
    )
    gates.append(
        gate_summary(
            "causal.double_dissociation.max",
            [match - other for match, other in zip(max_match, max_other)],
            direction="lower",
            threshold=0.01,
            family="double_dissociation",
        )
    )
    if len(gates) != 16:
        raise RuntimeError("candidate gate cardinality changed")
    return gates


def control_gates(
    name: str,
    candidate: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for slice_name in (*OOD_NAMES, "macro"):
        if slice_name == "macro":
            candidate_errors = [macro_error(record, "y") for record in candidate]
            control_errors = [macro_error(record, "y") for record in control]
        else:
            candidate_errors = [float(record["metrics"][slice_name]["y"]) for record in candidate]
            control_errors = [float(record["metrics"][slice_name]["y"]) for record in control]
        control_sum = float(sum(control_errors))
        candidate_sum = float(sum(candidate_errors))
        relative_point = None if control_sum == 0.0 else 1.0 - candidate_sum / control_sum
        absolute_point = float(np.mean(control_errors) - np.mean(candidate_errors))
        gates.append(
            gate_summary(
                f"capability.{name}.{slice_name}.relative_20pct",
                [0.8 * base - changed for base, changed in zip(control_errors, candidate_errors)],
                direction="lower",
                threshold=0.0,
                family="relative_capability",
                extra={
                    "candidate_errors": candidate_errors,
                    "control_errors": control_errors,
                    "candidate_pooled_error": candidate_sum / len(candidate_errors),
                    "control_pooled_error": control_sum / len(control_errors),
                    "relative_point_effect": relative_point,
                    "absolute_point_effect": absolute_point,
                    "headroom_consumed": relative_point,
                    "headroom_definition": "fraction of control error removed",
                },
            )
        )
    candidate_macro = [macro_error(record, "y") for record in candidate]
    control_macro = [macro_error(record, "y") for record in control]
    gates.append(
        gate_summary(
            f"capability.{name}.macro.absolute_1pp",
            [base - changed for base, changed in zip(control_macro, candidate_macro)],
            direction="lower",
            threshold=0.01,
            family="absolute_capability",
        )
    )
    gates.append(
        gate_summary(
            f"control_floor.{name}.macro_5pct",
            control_macro,
            direction="lower",
            threshold=0.05,
            family="control_floor",
        )
    )
    if name in SOFT_PROTECTED:
        candidate_soft = [macro_error(record, "ys") for record in candidate]
        control_soft = [macro_error(record, "ys") for record in control]
        gates.append(
            gate_summary(
                f"noninferiority.{name}.soft_macro",
                [changed - base for changed, base in zip(candidate_soft, control_soft)],
                direction="upper",
                threshold=0.01,
                family="soft_noninferiority",
            )
        )
        candidate_id = [float(record["metrics"]["id"]["y"]) for record in candidate]
        control_id = [float(record["metrics"]["id"]["y"]) for record in control]
        gates.append(
            gate_summary(
                f"noninferiority.{name}.id_final",
                [changed - base for changed, base in zip(candidate_id, control_id)],
                direction="upper",
                threshold=0.01,
                family="id_noninferiority",
            )
        )
    return gates


def max_reported_vmhwm() -> int:
    values = [core.linux_vmhwm_bytes()]

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif "vmhwm" in key.lower() and isinstance(value, (int, float)):
            values.append(int(value))

    for path in core.RUN_ROOT.rglob("*.json") if core.RUN_ROOT.exists() else ():
        visit(json.loads(path.read_text(encoding="utf-8")))
    for path in STAGE0_DIR.glob("replay-*.npz") if STAGE0_DIR.exists() else ():
        payload = load_npz(path)
        if "vmhwm" in payload:
            values.append(int(payload["vmhwm"][0]))
    return max(values)


INVOCATION_STARTED_NS: int | None = None


def resource_valid_now() -> bool:
    active, allocated = ledger_totals(allow_open=True)
    pending = 0 if INVOCATION_STARTED_NS is None else time.monotonic_ns() - INVOCATION_STARTED_NS
    return (
        active + pending <= MAX_ACTIVE_NS
        and allocated <= MAX_BYTES
        and max_reported_vmhwm() <= MAX_BYTES
    )


def terminal_resource_state() -> dict[str, Any]:
    active, allocated = ledger_totals(allow_open=True)
    now = time.monotonic_ns()
    pending = 0 if INVOCATION_STARTED_NS is None else now - INVOCATION_STARTED_NS
    maximum_vmhwm = max_reported_vmhwm()
    return {
        "measured_monotonic_ns": now,
        "closed_active_ns": active,
        "pending_active_ns": pending,
        "allocated_bytes": allocated,
        "maximum_reported_vmhwm_bytes": maximum_vmhwm,
        "reserved_active_ns": TERMINAL_RESERVE_NS,
        "reserved_allocated_bytes": TERMINAL_RESERVE_BYTES,
        "reserved_vmhwm_margin_bytes": TERMINAL_VMHWM_MARGIN_BYTES,
        "valid": (
            active + pending + TERMINAL_RESERVE_NS <= MAX_ACTIVE_NS
            and allocated + TERMINAL_RESERVE_BYTES <= MAX_BYTES
            and maximum_vmhwm + TERMINAL_VMHWM_MARGIN_BYTES <= MAX_BYTES
        ),
    }


def snapshot_run_artifacts(stage_key: str) -> tuple[Path, str]:
    manifest_path = DECISION_DIR / f"{stage_key}-complete-artifacts.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    files: dict[str, Any] = {}
    if core.RUN_ROOT.exists():
        for path in sorted(core.RUN_ROOT.rglob("*")):
            if path.is_symlink():
                raise RuntimeError("artifact symlink forbidden")
            if not path.is_file():
                continue
            relative = str(path.relative_to(core.RUN_ROOT))
            if relative.endswith(".incomplete"):
                raise RuntimeError("incomplete artifact present before decode")
            files[relative] = {
                "sha256": core.sha256_file(path),
                "logical_bytes": path.stat().st_size,
                "allocated_bytes": path.stat().st_blocks * 512,
            }
    core.atomic_json(
        manifest_path,
        {
            "stage": stage_key,
            "artifact_count": len(files),
            "files": files,
        },
    )
    return manifest_path, core.sha256_file(manifest_path)


def run_decoder(stage_key: str, gate_input: Mapping[str, bool]) -> str:
    artifact_manifest, artifact_manifest_hash = snapshot_run_artifacts(stage_key)
    terminal = terminal_resource_state()
    sealed_gate = dict(gate_input)
    sealed_gate["resource_valid"] = bool(
        sealed_gate["resource_valid"] and terminal["valid"]
    )
    core.atomic_json(
        DECISION_DIR / f"{stage_key}-decoder-envelope.json",
        {
            "complete_artifact_manifest": str(
                artifact_manifest.relative_to(core.RUN_ROOT)
            ),
            "complete_artifact_manifest_sha256": artifact_manifest_hash,
            "terminal_resource_state": terminal,
        },
    )
    input_path = DECISION_DIR / f"{stage_key}-input.json"
    core.atomic_json(input_path, sealed_gate)
    decoder = core.ROOT / "experiments/shared_score_heterogeneous_attention_t84_decoder.py"
    completed = run_reported_subprocess(
        [
            str(FROZEN_PYTHON),
            "-B",
            str(decoder),
            str(input_path),
            str(DECISION_DIR / f"{stage_key}-decoder-vmhwm.json"),
        ],
        DECISION_DIR / f"{stage_key}-decoder-worker.json",
        capture_output=True,
        timeout_seconds=60,
    )
    if completed.stderr:
        raise RuntimeError("sealed decoder emitted stderr")
    word = completed.stdout.strip()
    allowed = {
        "CONTINUE",
        "STOP-REJECT",
        "CONTROL FLOOR NOT ESTABLISHED",
        "RESOURCE-INCONCLUSIVE",
        "PROTOCOL INVALID",
    }
    if word not in allowed or completed.stdout != word + "\n":
        raise RuntimeError("sealed decoder emitted an invalid word")
    atomic_text_new(DECISION_DIR / f"{stage_key}-word.txt", word + "\n")
    return word


def prerequisite_integrity() -> bool:
    try:
        timing = json.loads((TIMING_DIR / "summary.json").read_text(encoding="utf-8"))
        stage0 = json.loads((STAGE0_DIR / "decision.json").read_text(encoding="utf-8"))
        probes = json.loads((PROBE_DIR / "results.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return bool(
        timing.get("stable_matrix")
        and timing.get("selected") is not None
        and stage0.get("status") == "pass"
        and probes.get("passed")
    )


def decision_word(stage_key: str) -> str | None:
    path = DECISION_DIR / f"{stage_key}-word.txt"
    return None if not path.exists() else path.read_text(encoding="utf-8").strip()


def seal_prerequisite_decision(stage_key: str) -> str:
    if stage_key == "fixtures-timing":
        timing = json.loads((TIMING_DIR / "summary.json").read_text(encoding="utf-8"))
        integrity_valid = bool(timing.get("stable_matrix"))
        required_control_valid = timing.get("selected") is not None
        all_gates_pass = True
    elif stage_key == "stage0":
        stage0 = json.loads((STAGE0_DIR / "decision.json").read_text(encoding="utf-8"))
        integrity_valid = decision_word("fixtures-timing") == "CONTINUE"
        required_control_valid = True
        all_gates_pass = stage0.get("status") == "pass"
    elif stage_key == "probes":
        probes = json.loads((PROBE_DIR / "results.json").read_text(encoding="utf-8"))
        integrity_valid = decision_word("stage0") == "CONTINUE"
        required_control_valid = True
        all_gates_pass = bool(probes.get("passed"))
    else:
        raise ValueError(stage_key)
    gate_input = {
        "integrity_valid": integrity_valid,
        "required_control_valid": required_control_valid,
        "resource_valid": resource_valid_now(),
        "control_floor_established": True,
        "all_decidable_gates_pass": all_gates_pass,
    }
    return run_decoder(stage_key, gate_input)


def seal_stage_decision(architecture: str) -> str:
    candidate = architecture_records("candidate")
    gates = candidate_gates(candidate)
    completed_controls = []
    for control in CONTROL_ORDER:
        path = TRAIN_DIR / control / f"seed-{core.EVAL_SEEDS[-1]}.json"
        if not path.exists():
            break
        completed_controls.append(control)
        gates.extend(control_gates(control, candidate, architecture_records(control)))
    expected_current = "candidate" if not completed_controls else completed_controls[-1]
    if architecture != expected_current:
        raise RuntimeError("architecture execution order changed")
    expected_gate_count = 16 + 7 * len(completed_controls) + 2 * sum(
        control in SOFT_PROTECTED for control in completed_controls
    )
    if len(gates) != expected_gate_count or len({gate["name"] for gate in gates}) != len(gates):
        raise RuntimeError("gate family cardinality changed")
    if len(completed_controls) == len(CONTROL_ORDER) and len(gates) != core.FAMILY_SIZE:
        raise RuntimeError("complete gate family is not 106")
    health_valid = all(
        bool(record["diagnostics"]["health"]["passed"]) for record in candidate
    )
    floor_gates = [gate for gate in gates if gate["family"] == "control_floor"]
    nonfloor_gates = [gate for gate in gates if gate["family"] != "control_floor"]
    stage_key = f"after-{architecture}"
    sealed_payload = {
        "family_size": core.FAMILY_SIZE,
        "currently_decidable": len(gates),
        "completed_controls": completed_controls,
        "health_valid": health_valid,
        "gates": gates,
    }
    core.atomic_json(DECISION_DIR / f"{stage_key}-sealed-gates.json", sealed_payload)
    gate_input = {
        "integrity_valid": prerequisite_integrity(),
        "required_control_valid": True,
        "resource_valid": resource_valid_now(),
        "control_floor_established": all(bool(gate["passed"]) for gate in floor_gates),
        "all_decidable_gates_pass": health_valid
        and all(bool(gate["passed"]) for gate in nonfloor_gates),
    }
    return run_decoder(stage_key, gate_input)


def required_previous_word(architecture: str) -> tuple[str, str]:
    if architecture == "candidate":
        return "probes", "CONTINUE"
    index = CONTROL_ORDER.index(architecture)
    previous = "candidate" if index == 0 else CONTROL_ORDER[index - 1]
    return f"after-{previous}", "CONTINUE"


def run_fixtures_and_timing(receipt: Path) -> None:
    core.require_receipt("FIXTURES AND TIMING", receipt)
    capture_runtime_environment()
    generate_fixtures()
    run_timing_matrix(receipt)


def capture_runtime_environment() -> None:
    core.verify_runtime()
    expected = json.loads(core.ENVIRONMENT_EXPECTATION.read_text(encoding="utf-8"))
    distributions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }
    payload = {
        "expected": expected,
        "platform": platform.platform(),
        "python_build": platform.python_build(),
        "python_compiler": platform.python_compiler(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_build_config": torch.__config__.show(),
        "numpy_build_config": np.__config__.CONFIG,
        "environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "package_versions": dict(sorted(distributions.items())),
        "source_hashes": {
            str(path): core.sha256_file(core.ROOT / path)
            for path in core.AUDITED_SOURCE_PATHS
        },
    }
    core.atomic_json(core.RUN_ROOT / "environment.json", payload)


def ensure_previous_stage(stage_key: str, expected: str = "CONTINUE") -> None:
    actual = decision_word(stage_key)
    if actual != expected:
        raise RuntimeError(
            f"previous sealed stage {stage_key} is {actual!r}, expected {expected!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "fixtures-timing",
            "timing-worker",
            "stage0",
            "stage0-worker",
            "probes",
            "train",
            "trained-timing-worker",
        ),
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--worker-auth", type=Path)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--variant", type=int)
    parser.add_argument("--index", type=int)
    parser.add_argument("--architecture", choices=core.ARCHITECTURES)
    return parser.parse_args()


def require_argument(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"{name} is required")
    return value


def configure_runtime() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def run_top_level(stage: str, decision_key: str, receipt: Path, action: Any) -> None:
    global INVOCATION_STARTED_NS
    raw_start = os.environ.get("T84_PROCESS_START_NS")
    if raw_start is None:
        raise RuntimeError("top-level execution must enter through the audited timer entrypoint")
    INVOCATION_STARTED_NS = int(raw_start)
    if INVOCATION_STARTED_NS > time.monotonic_ns():
        raise RuntimeError("invalid audited timer origin")
    open_path = begin_ledger_attempt(stage, INVOCATION_STARTED_NS, receipt)
    try:
        action()
    except BaseException as error:
        status = (
            "RESOURCE-INCONCLUSIVE"
            if isinstance(error, RuntimeError) and str(error) == "RESOURCE-INCONCLUSIVE"
            else "PROTOCOL INVALID"
        )
        close_ledger_attempt(open_path, status, None)
        raise
    else:
        word_path = DECISION_DIR / f"{decision_key}-word.txt"
        if not word_path.exists():
            close_ledger_attempt(open_path, "PROTOCOL INVALID", None)
            raise RuntimeError("authorized stage produced no exact sealed decoder word")
        status = word_path.read_text(encoding="utf-8").strip()
        effective = close_ledger_attempt(open_path, status, decision_key)
        if effective != status:
            raise RuntimeError(effective)


def main() -> None:
    args = parse_args()
    configure_runtime()
    core.assert_frozen_inputs()
    if args.stage == "timing-worker":
        timing_worker(
            require_argument(args.attempt, "--attempt"),
            require_argument(args.round, "--round"),
            require_argument(args.variant, "--variant"),
            require_argument(args.worker_auth, "--worker-auth"),
        )
        return
    if args.stage == "stage0-worker":
        stage0_worker(
            require_argument(args.index, "--index"),
            require_argument(args.worker_auth, "--worker-auth"),
        )
        return
    if args.stage == "trained-timing-worker":
        trained_timing_worker(
            require_argument(args.architecture, "--architecture"),
            require_argument(args.round, "--round"),
            require_argument(args.worker_auth, "--worker-auth"),
        )
        return
    receipt = require_argument(args.receipt, "--receipt")
    if args.stage == "fixtures-timing":
        run_top_level(
            args.stage,
            "fixtures-timing",
            receipt,
            lambda: run_fixtures_and_timing(receipt),
        )
    elif args.stage == "stage0":
        run_top_level(
            args.stage,
            "stage0",
            receipt,
            lambda: (ensure_previous_stage("fixtures-timing"), run_stage0(receipt)),
        )
    elif args.stage == "probes":
        run_top_level(
            args.stage,
            "probes",
            receipt,
            lambda: (ensure_previous_stage("stage0"), run_probes(receipt)),
        )
    elif args.stage == "train":
        architecture = require_argument(args.architecture, "--architecture")
        previous, expected = required_previous_word(architecture)
        run_top_level(
            f"train-{architecture}",
            f"after-{architecture}",
            receipt,
            lambda: (
                ensure_previous_stage(previous, expected),
                run_architecture(architecture, receipt),
            ),
        )
    else:
        raise RuntimeError("unreachable stage")


if __name__ == "__main__":
    main()
