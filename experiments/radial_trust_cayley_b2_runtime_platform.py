#!/usr/bin/env python3
"""Pure CPU platform-identity rules for B2 Amendment 1.

This module evaluates the mandatory nvmlDeviceGetPciInfo_v3 outcome and derives
the one Linux-canonical BDF allowed for later sysfs/procfs collection.  It does
not load NVML, inspect a GPU, or touch any BDF-dependent path.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.radial_trust_cayley_b2_runtime_manifest import (
    ManifestError,
    b64url,
    decimal_integer,
)


NVML_SUCCESS = 0
NVML_BUS_ID_BUFFER_SIZE = 32
NVML_BUS_ID_LEGACY_BUFFER_SIZE = 16


@dataclass(frozen=True)
class RawPciInfo:
    domain: int
    bus: int
    device: int
    pci_device_id: int
    pci_subsystem_id: int
    bus_id: bytes
    bus_id_legacy: bytes


@dataclass(frozen=True)
class PciIdentity:
    domain: int
    bus: int
    device: int
    pci_device_id: int
    pci_subsystem_id: int
    bus_id: bytes
    bus_id_legacy: bytes
    path_bdf: bytes

    def record(self) -> dict[str, str]:
        _validate_identity(self)
        return {
            "bus": decimal_integer(self.bus),
            "bus_id_b64": b64url(self.bus_id),
            "bus_id_legacy_b64": b64url(self.bus_id_legacy),
            "device": decimal_integer(self.device),
            "domain": decimal_integer(self.domain),
            "path_bdf_b64": b64url(self.path_bdf),
            "pci_device_id": decimal_integer(self.pci_device_id),
            "pci_subsystem_id": decimal_integer(self.pci_subsystem_id),
            "return_code": "0",
            "status": "valid",
        }


@dataclass(frozen=True)
class PciDecision:
    record: dict[str, str]
    identity: PciIdentity | None

    @property
    def valid(self) -> bool:
        return self.identity is not None


def _bounded_uint(value: int, *, maximum: int, field: str) -> int:
    if type(value) is not int:
        raise ManifestError(f"{field} must be an integer")
    if not 0 <= value <= maximum:
        raise ManifestError(f"{field} is outside its frozen unsigned range")
    return value


def _logical_c_string(raw: bytes, *, size: int, field: str) -> bytes:
    if type(raw) is not bytes or len(raw) != size:
        raise ManifestError(f"{field} must be its complete fixed-size byte array")
    terminator = raw.find(b"\0")
    if terminator < 0:
        raise ManifestError(f"{field} lacks a NUL terminator")
    return raw[:terminator]


def derive_pci_identity(raw: RawPciInfo) -> PciIdentity:
    if type(raw) is not RawPciInfo:
        raise ManifestError("PCI result must be RawPciInfo")
    domain = _bounded_uint(raw.domain, maximum=0xFFFF, field="domain")
    bus = _bounded_uint(raw.bus, maximum=0xFF, field="bus")
    device = _bounded_uint(raw.device, maximum=0x1F, field="device")
    pci_device_id = _bounded_uint(
        raw.pci_device_id,
        maximum=0xFFFFFFFF,
        field="pci_device_id",
    )
    pci_subsystem_id = _bounded_uint(
        raw.pci_subsystem_id,
        maximum=0xFFFFFFFF,
        field="pci_subsystem_id",
    )
    bus_id = _logical_c_string(
        raw.bus_id,
        size=NVML_BUS_ID_BUFFER_SIZE,
        field="bus_id",
    )
    bus_id_legacy = _logical_c_string(
        raw.bus_id_legacy,
        size=NVML_BUS_ID_LEGACY_BUFFER_SIZE,
        field="bus_id_legacy",
    )
    expected_bus_id = f"{domain:08X}:{bus:02X}:{device:02X}.0".encode("ascii")
    expected_legacy = f"{domain:04X}:{bus:02X}:{device:02X}.0".encode("ascii")
    if bus_id != expected_bus_id:
        raise ManifestError("bus_id disagrees with the numeric PCI tuple")
    if bus_id_legacy != expected_legacy:
        raise ManifestError("bus_id_legacy disagrees with the numeric PCI tuple")
    path_bdf = f"{domain:04x}:{bus:02x}:{device:02x}.0".encode("ascii")
    return PciIdentity(
        domain,
        bus,
        device,
        pci_device_id,
        pci_subsystem_id,
        bus_id,
        bus_id_legacy,
        path_bdf,
    )


def _validate_identity(identity: PciIdentity) -> None:
    if type(identity) is not PciIdentity:
        raise ManifestError("valid PCI identity is required")
    domain = _bounded_uint(identity.domain, maximum=0xFFFF, field="domain")
    bus = _bounded_uint(identity.bus, maximum=0xFF, field="bus")
    device = _bounded_uint(identity.device, maximum=0x1F, field="device")
    _bounded_uint(
        identity.pci_device_id,
        maximum=0xFFFFFFFF,
        field="pci_device_id",
    )
    _bounded_uint(
        identity.pci_subsystem_id,
        maximum=0xFFFFFFFF,
        field="pci_subsystem_id",
    )
    expected_bus_id = f"{domain:08X}:{bus:02X}:{device:02X}.0".encode("ascii")
    expected_legacy = f"{domain:04X}:{bus:02X}:{device:02X}.0".encode("ascii")
    expected_path = f"{domain:04x}:{bus:02x}:{device:02x}.0".encode("ascii")
    if (
        type(identity.bus_id) is not bytes
        or type(identity.bus_id_legacy) is not bytes
        or type(identity.path_bdf) is not bytes
    ):
        raise ManifestError("trusted PCI strings must be exact bytes")
    if identity.bus_id != expected_bus_id:
        raise ManifestError("trusted bus_id disagrees with the numeric PCI tuple")
    if identity.bus_id_legacy != expected_legacy:
        raise ManifestError("trusted bus_id_legacy disagrees with the numeric PCI tuple")
    if identity.path_bdf != expected_path:
        raise ManifestError("trusted path_bdf disagrees with the numeric PCI tuple")


def evaluate_pci_query(
    return_code: int,
    raw: RawPciInfo | None,
) -> PciDecision:
    code = _bounded_uint(return_code, maximum=0x7FFFFFFF, field="return_code")
    if code != NVML_SUCCESS:
        return PciDecision(
            {
                "return_code": decimal_integer(code),
                "status": "invalid_nvml_return",
            },
            None,
        )
    if type(raw) is not RawPciInfo:
        return PciDecision(
            {
                "return_code": decimal_integer(code),
                "status": "invalid_missing_pci_result",
            },
            None,
        )
    try:
        identity = derive_pci_identity(raw)
    except ManifestError:
        return PciDecision(
            {
                "return_code": decimal_integer(code),
                "status": "invalid_pci_result",
            },
            None,
        )
    return PciDecision(identity.record(), identity)


def bdf_dependent_paths(identity: PciIdentity) -> tuple[bytes, bytes]:
    _validate_identity(identity)
    return (
        b"/sys/bus/pci/devices/" + identity.path_bdf,
        b"/proc/driver/nvidia/gpus/" + identity.path_bdf + b"/information",
    )


__all__ = [
    "NVML_BUS_ID_BUFFER_SIZE",
    "NVML_BUS_ID_LEGACY_BUFFER_SIZE",
    "NVML_SUCCESS",
    "PciDecision",
    "PciIdentity",
    "RawPciInfo",
    "bdf_dependent_paths",
    "derive_pci_identity",
    "evaluate_pci_query",
]
