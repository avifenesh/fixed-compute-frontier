#!/usr/bin/env python3
"""Pure result-schema evaluator for the frozen B2 NVML invariant calls.

This module neither loads NVML nor touches a GPU.  It converts already captured,
zero-initialized ABI results into the exact canonical-tree records used by the
later PID-1 auditor.  PCI identity and variable-length clock enumeration remain
separate audited slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from experiments.radial_trust_cayley_b2_runtime_manifest import (
    ManifestError,
    b64url,
    decimal_integer,
)


NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3

NVML_DEVICE_UUID_V2_BUFFER_SIZE = 96
NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE = 32
NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE = 16
NVML_GSP_FIRMWARE_VERSION_BUF_SIZE = 64

STRING_QUERY_SIZES = MappingProxyType({
    "gsp_firmware_version": NVML_GSP_FIRMWARE_VERSION_BUF_SIZE,
    "inforom_image_version": NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE,
    "uuid": NVML_DEVICE_UUID_V2_BUFFER_SIZE,
    "vbios_version": NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE,
})

UINT_QUERY_ALLOWED_VALUES = MappingProxyType({
    "applications_clock_mem": None,
    "applications_clock_sm": None,
    "compute_mode": frozenset((0, 1, 2, 3)),
    "persistence_mode": frozenset((0, 1)),
    "power_management_limit": None,
})

PAIR_QUERY_SPECS = MappingProxyType({
    "ecc_mode": ("current", "pending", frozenset((0, 1))),
    "mig_mode": ("current", "pending", frozenset((0, 1))),
    "power_management_limit_constraints": ("minimum", "maximum", None),
})


@dataclass(frozen=True)
class StringQueryResult:
    query: str
    return_code: int
    buffer: bytes | None


@dataclass(frozen=True)
class UintQueryResult:
    query: str
    return_code: int
    value: int | None


@dataclass(frozen=True)
class PairQueryResult:
    query: str
    return_code: int
    first: int | None
    second: int | None


def _uint32(value: int, *, field: str) -> int:
    if type(value) is not int:
        raise ManifestError(f"{field} must be an exact integer")
    if not 0 <= value <= 0xFFFFFFFF:
        raise ManifestError(f"{field} is outside uint32")
    return value


def _return_code(value: int) -> int:
    if type(value) is not int or not 0 <= value <= 0x7FFFFFFF:
        raise ManifestError("NVML return code must be a nonnegative exact int")
    return value


def _query_name(value: str, *, allowed: Iterable[str]) -> str:
    if type(value) is not str or value not in allowed:
        raise ManifestError("unknown or non-exact NVML invariant query name")
    return value


def _unsupported_record(query: str, return_code: int, payloads: tuple[object, ...]) -> dict[str, str]:
    if return_code != NVML_ERROR_NOT_SUPPORTED:
        raise ManifestError(f"{query} returned disallowed NVML code {return_code}")
    if any(payload is not None for payload in payloads):
        raise ManifestError(f"{query} not-supported result retained stale payload")
    return {
        "query": query,
        "return_code": decimal_integer(return_code),
        "status": "not_supported",
    }


def evaluate_string_result(result: StringQueryResult) -> dict[str, str]:
    if type(result) is not StringQueryResult:
        raise ManifestError("string result must be an exact StringQueryResult")
    query = _query_name(result.query, allowed=STRING_QUERY_SIZES)
    code = _return_code(result.return_code)
    if code != NVML_SUCCESS:
        return _unsupported_record(query, code, (result.buffer,))
    size = STRING_QUERY_SIZES[query]
    if type(result.buffer) is not bytes or len(result.buffer) != size:
        raise ManifestError(f"{query} must provide its complete fixed-size buffer")
    terminator = result.buffer.find(b"\0")
    if terminator <= 0:
        raise ManifestError(f"{query} lacks a nonempty NUL-terminated value")
    logical = result.buffer[:terminator]
    return {
        "query": query,
        "return_code": "0",
        "status": "ok",
        "value_b64": b64url(logical),
    }


def evaluate_uint_result(result: UintQueryResult) -> dict[str, str]:
    if type(result) is not UintQueryResult:
        raise ManifestError("uint result must be an exact UintQueryResult")
    query = _query_name(result.query, allowed=UINT_QUERY_ALLOWED_VALUES)
    code = _return_code(result.return_code)
    if code != NVML_SUCCESS:
        return _unsupported_record(query, code, (result.value,))
    if result.value is None:
        raise ManifestError(f"{query} success omitted its value")
    value = _uint32(result.value, field=query)
    allowed = UINT_QUERY_ALLOWED_VALUES[query]
    if allowed is not None and value not in allowed:
        raise ManifestError(f"{query} returned an unknown enum value")
    return {
        "query": query,
        "return_code": "0",
        "status": "ok",
        "value": decimal_integer(value),
    }


def evaluate_pair_result(result: PairQueryResult) -> dict[str, str]:
    if type(result) is not PairQueryResult:
        raise ManifestError("pair result must be an exact PairQueryResult")
    query = _query_name(result.query, allowed=PAIR_QUERY_SPECS)
    code = _return_code(result.return_code)
    if code != NVML_SUCCESS:
        return _unsupported_record(query, code, (result.first, result.second))
    if result.first is None or result.second is None:
        raise ManifestError(f"{query} success omitted a paired value")
    first = _uint32(result.first, field=f"{query}.first")
    second = _uint32(result.second, field=f"{query}.second")
    first_name, second_name, allowed = PAIR_QUERY_SPECS[query]
    if allowed is not None and (first not in allowed or second not in allowed):
        raise ManifestError(f"{query} returned an unknown enum value")
    if query == "power_management_limit_constraints" and first > second:
        raise ManifestError("power limit minimum exceeds maximum")
    return {
        first_name: decimal_integer(first),
        "query": query,
        "return_code": "0",
        second_name: decimal_integer(second),
        "status": "ok",
    }


def compose_scalar_invariant_records(
    string_results: tuple[StringQueryResult, ...],
    uint_results: tuple[UintQueryResult, ...],
    pair_results: tuple[PairQueryResult, ...],
) -> list[dict[str, str]]:
    """Require every frozen non-PCI, non-clock-list query exactly once."""
    if (
        type(string_results) is not tuple
        or type(uint_results) is not tuple
        or type(pair_results) is not tuple
    ):
        raise ManifestError("NVML result groups must be exact tuples")
    records = [
        *(evaluate_string_result(result) for result in string_results),
        *(evaluate_uint_result(result) for result in uint_results),
        *(evaluate_pair_result(result) for result in pair_results),
    ]
    expected = set(STRING_QUERY_SIZES) | set(UINT_QUERY_ALLOWED_VALUES) | set(PAIR_QUERY_SPECS)
    observed = [record["query"] for record in records]
    if len(observed) != len(set(observed)):
        raise ManifestError("duplicate NVML invariant query result")
    if set(observed) != expected:
        raise ManifestError("missing or extra NVML invariant query result")

    by_query = {record["query"]: record for record in records}
    limit = by_query["power_management_limit"]
    constraints = by_query["power_management_limit_constraints"]
    if limit["status"] == "ok" and constraints["status"] == "ok":
        value = int(limit["value"], 10)
        minimum = int(constraints["minimum"], 10)
        maximum = int(constraints["maximum"], 10)
        if not minimum <= value <= maximum:
            raise ManifestError("power management limit lies outside its constraints")
    return [by_query[query] for query in sorted(by_query, key=lambda item: item.encode("ascii"))]


__all__ = [
    "NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE",
    "NVML_DEVICE_UUID_V2_BUFFER_SIZE",
    "NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE",
    "NVML_ERROR_NOT_SUPPORTED",
    "NVML_GSP_FIRMWARE_VERSION_BUF_SIZE",
    "NVML_SUCCESS",
    "PAIR_QUERY_SPECS",
    "PairQueryResult",
    "STRING_QUERY_SIZES",
    "StringQueryResult",
    "UINT_QUERY_ALLOWED_VALUES",
    "UintQueryResult",
    "compose_scalar_invariant_records",
    "evaluate_pair_result",
    "evaluate_string_result",
    "evaluate_uint_result",
]
