#!/usr/bin/env python3
"""CPU-testable core of the frozen B2 runtime manifest.

This module deliberately contains no CUDA or NVML entrypoint.  It defines the
canonical byte representation, secret-safe environment records, exact-errno
path records, unified-cgroup-v2 resolution, and snapshot equality used by the
later PID-1 auditor.  Integers are represented as decimal strings, so the
supported JSON subset has no numeric serialization ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import errno
import hashlib
import json
import os
from pathlib import PurePosixPath
from typing import Callable, Mapping, Sequence


CONTAINER_SCHEMA = "b2-container-runtime-v1"
PROCESS_SCHEMA = "b2-process-runtime-v1"

RAW_ENV_EXACT = (b"LD_LIBRARY_PATH", b"LD_PRELOAD")
RAW_ENV_PREFIXES = (
    b"CUDA_",
    b"NVIDIA_",
    b"CUBLAS_",
    b"CUDNN_",
    b"NVTE_",
    b"OMP_",
    b"MKL_",
    b"OPENBLAS_",
    b"KMP_",
    b"NUMEXPR_",
    b"TORCH_",
    b"PYTORCH_",
)
SECRET_NAME_FRAGMENTS = (b"KEY", b"TOKEN", b"SECRET", b"PASSWORD")

CGROUP_NAMED_FILES = (
    b"cgroup.controllers",
    b"cgroup.subtree_control",
    b"cgroup.type",
    b"cpu.max",
    b"cpu.max.burst",
    b"cpu.weight",
    b"cpu.uclamp.min",
    b"cpu.uclamp.max",
    b"cpuset.cpus",
    b"cpuset.cpus.effective",
    b"cpuset.mems",
    b"cpuset.mems.effective",
    b"cpuset.cpus.partition",
    b"memory.min",
    b"memory.low",
    b"memory.high",
    b"memory.max",
    b"memory.swap.max",
    b"pids.max",
    b"rdma.max",
)


class ManifestError(ValueError):
    """The runtime record cannot satisfy the frozen manifest contract."""


def b64url(raw: bytes) -> str:
    """RFC 4648 base64url without padding."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def decimal_integer(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError("manifest integers must originate as Python int")
    return str(value)


def _validate_canonical_tree(value: object, *, path: str = "$") -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if not value.isascii():
            raise ManifestError(f"manifest string at {path} must be ASCII text")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ManifestError(f"lone surrogate at {path}")
        return
    if isinstance(value, (int, float)):
        raise ManifestError(f"JSON number forbidden at {path}; use a decimal string")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_canonical_tree(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise ManifestError(f"schema key at {path} must be ASCII text")
            _validate_canonical_tree(item, path=f"{path}.{key}")
        return
    raise ManifestError(f"unsupported canonical JSON value at {path}: {type(value)!r}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize the B2 RFC-8785 subset to canonical UTF-8 bytes.

    B2 schema keys and values are ASCII strings, booleans, nulls, arrays, and
    objects.  All machine integers are decimal strings.  On this restricted
    subset, compact JSON with lexicographically sorted ASCII keys is exactly
    RFC 8785: there are no binary64-number or UTF-16 key-order edge cases.
    """
    _validate_canonical_tree(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class ManifestBlob:
    schema: str
    canonical: bytes
    sha256: str


def build_manifest(schema: str, payload: Mapping[str, object]) -> ManifestBlob:
    if schema not in (CONTAINER_SCHEMA, PROCESS_SCHEMA):
        raise ManifestError("unknown B2 runtime schema")
    document = {"payload": dict(payload), "schema": schema}
    canonical = canonical_json_bytes(document)
    return ManifestBlob(schema, canonical, sha256_bytes(canonical))


def require_same_manifest(before: ManifestBlob, after: ManifestBlob) -> None:
    _validate_manifest_blob(before)
    _validate_manifest_blob(after)
    if before.schema != after.schema:
        raise ManifestError("runtime schema changed across the boundary")
    if before.canonical != after.canonical:
        raise ManifestError(
            "runtime invariant changed: "
            f"{before.sha256} != {after.sha256}"
        )


def _validate_manifest_blob(blob: ManifestBlob) -> None:
    if blob.schema not in (CONTAINER_SCHEMA, PROCESS_SCHEMA):
        raise ManifestError("unknown B2 runtime schema in manifest blob")
    if sha256_bytes(blob.canonical) != blob.sha256:
        raise ManifestError("manifest cached SHA-256 does not match canonical bytes")
    try:
        document = json.loads(blob.canonical)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError("manifest canonical bytes are not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"payload", "schema"}:
        raise ManifestError("manifest envelope has the wrong fields")
    if document["schema"] != blob.schema:
        raise ManifestError("manifest envelope schema disagrees with the blob")
    if not isinstance(document["payload"], dict):
        raise ManifestError("manifest payload must be an object")
    if canonical_json_bytes(document) != blob.canonical:
        raise ManifestError("manifest bytes are not canonical")


def _read_all(path: bytes) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def path_bytes_record(
    path: bytes,
    *,
    reader: Callable[[bytes], bytes] = _read_all,
) -> dict[str, str]:
    """Read one named path or preserve its exact positive errno."""
    try:
        raw = reader(path)
    except OSError as error:
        if error.errno is None or error.errno <= 0:
            raise ManifestError("path failure did not provide a positive errno") from error
        return {
            "errno": decimal_integer(error.errno),
            "path_b64": b64url(path),
            "status": "error",
        }
    if not isinstance(raw, bytes):
        raise ManifestError("path reader must return bytes")
    return {
        "bytes_b64": b64url(raw),
        "path_b64": b64url(path),
        "sha256": sha256_bytes(raw),
        "status": "ok",
    }


def parse_proc_environ(raw: bytes) -> tuple[tuple[bytes, bytes], ...]:
    if not isinstance(raw, bytes):
        raise ManifestError("environment image must be bytes")
    entries: list[tuple[bytes, bytes]] = []
    names: set[bytes] = set()
    fields = raw.split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    for field in fields:
        name, separator, value = field.partition(b"=")
        if not separator or not name or b"\0" in name:
            raise ManifestError("malformed /proc environment entry")
        if name in names:
            raise ManifestError("duplicate environment-variable name")
        names.add(name)
        entries.append((name, value))
    entries.sort(key=lambda item: item[0])
    return tuple(entries)


def _environment_value_is_publishable(name: bytes) -> bool:
    upper = name.upper()
    if any(fragment in upper for fragment in SECRET_NAME_FRAGMENTS):
        return False
    return name in RAW_ENV_EXACT or any(name.startswith(prefix) for prefix in RAW_ENV_PREFIXES)


def environment_records(raw: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name, value in parse_proc_environ(raw):
        record = {
            "name_b64": b64url(name),
            "value_sha256": sha256_bytes(value),
        }
        if _environment_value_is_publishable(name):
            record["value_b64"] = b64url(value)
        records.append(record)
    return records


def parse_unified_cgroup_path(raw: bytes) -> bytes:
    """Return the single PID cgroup path and reject legacy/hybrid records."""
    lines = raw.splitlines()
    if len(lines) != 1:
        raise ManifestError("expected exactly one unified cgroup-v2 record")
    parts = lines[0].split(b":", 2)
    if len(parts) != 3:
        raise ManifestError("malformed /proc cgroup record")
    hierarchy, controllers, path = parts
    if hierarchy != b"0" or controllers != b"" or not path.startswith(b"/"):
        raise ManifestError("process is not in a unified cgroup-v2 record")
    components = path.split(b"/")
    if (
        b"\0" in path
        or (
            path != b"/"
            and (
                components[0] != b""
                or any(part in (b"", b".", b"..") for part in components[1:])
            )
        )
    ):
        raise ManifestError("unsafe cgroup path")
    return path


def _join_cgroup_root(root: bytes, relative: bytes) -> bytes:
    normalized_root = root.rstrip(b"/") or b"/"
    if relative == b"/":
        return normalized_root
    if normalized_root == b"/":
        return b"/" + relative.lstrip(b"/")
    return normalized_root + b"/" + relative.lstrip(b"/")


def _join_path(directory: bytes, name: bytes) -> bytes:
    if directory == b"/":
        return b"/" + name
    return directory.rstrip(b"/") + b"/" + name


def collect_cgroup_records(
    proc_cgroup_raw: bytes,
    *,
    root: bytes = b"/sys/fs/cgroup",
    reader: Callable[[bytes], bytes] = _read_all,
    listdir: Callable[[bytes], Sequence[bytes | str]] = os.listdir,
) -> dict[str, object]:
    relative = parse_unified_cgroup_path(proc_cgroup_raw)
    directory = _join_cgroup_root(root, relative)
    try:
        direct_entries = listdir(directory)
    except OSError as error:
        if error.errno is None or error.errno <= 0:
            raise ManifestError("cgroup listing failed without positive errno") from error
        raise ManifestError(f"cannot list resolved cgroup: errno {error.errno}") from error
    entry_bytes = [os.fsencode(entry) for entry in direct_entries]
    hugepage_names = sorted(
        name
        for name in entry_bytes
        if name.startswith(b"hugetlb.") and name.endswith(b".max")
    )
    names = sorted((*CGROUP_NAMED_FILES, *hugepage_names))
    records = [
        path_bytes_record(_join_path(directory, name), reader=reader)
        for name in names
    ]
    return {
        "directory_b64": b64url(directory),
        "files": records,
        "path_b64": b64url(relative),
    }


def validate_decimal_string(value: str) -> int:
    if not isinstance(value, str) or not value:
        raise ManifestError("integer field is not a nonempty decimal string")
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if not digits.isascii() or not digits.isdigit():
        raise ManifestError("integer field is not canonical decimal")
    if len(digits) > 1 and digits.startswith("0"):
        raise ManifestError("integer field has a leading zero")
    if negative and digits == "0":
        raise ManifestError("negative zero is forbidden")
    return int(value, 10)


def validate_absolute_posix_path(path: bytes) -> None:
    if not path.startswith(b"/") or b"\0" in path:
        raise ManifestError("path must be absolute bytes without NUL")
    decoded = path.decode("utf-8", "surrogateescape")
    if ".." in PurePosixPath(decoded).parts:
        raise ManifestError("path traversal is forbidden")


__all__ = [
    "CGROUP_NAMED_FILES",
    "CONTAINER_SCHEMA",
    "ManifestBlob",
    "ManifestError",
    "PROCESS_SCHEMA",
    "b64url",
    "build_manifest",
    "canonical_json_bytes",
    "collect_cgroup_records",
    "decimal_integer",
    "environment_records",
    "parse_proc_environ",
    "parse_unified_cgroup_path",
    "path_bytes_record",
    "require_same_manifest",
    "sha256_bytes",
    "validate_absolute_posix_path",
    "validate_decimal_string",
]
