import base64
from dataclasses import replace

import pytest

import experiments.radial_trust_cayley_b2_runtime_platform as platform
from experiments.radial_trust_cayley_b2_runtime_manifest import ManifestError, b64url


class _DeceptiveBytes(bytes):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _DeceptiveInt(int):
    def __le__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True


class _RawPciInfoSubclass(platform.RawPciInfo):
    pass


class _PciIdentitySubclass(platform.PciIdentity):
    pass


def _array(value: bytes, size: int, *, tail: bytes = b"") -> bytes:
    if len(value) + 1 + len(tail) > size:
        raise ValueError("fixture does not fit")
    return value + b"\0" + tail + bytes(size - len(value) - 1 - len(tail))


def _raw(
    *,
    domain: int = 0xABCD,
    bus: int = 0xAF,
    device: int = 0x1B,
    pci_device_id: int = 0x233010DE,
    pci_subsystem_id: int = 0x16A110DE,
    bus_id: bytes | None = None,
    legacy: bytes | None = None,
    bus_tail: bytes = b"",
    legacy_tail: bytes = b"",
) -> platform.RawPciInfo:
    if bus_id is None:
        bus_id = f"{domain:08X}:{bus:02X}:{device:02X}.0".encode("ascii")
    if legacy is None:
        legacy = f"{domain:04X}:{bus:02X}:{device:02X}.0".encode("ascii")
    return platform.RawPciInfo(
        domain=domain,
        bus=bus,
        device=device,
        pci_device_id=pci_device_id,
        pci_subsystem_id=pci_subsystem_id,
        bus_id=_array(
            bus_id,
            platform.NVML_BUS_ID_BUFFER_SIZE,
            tail=bus_tail,
        ),
        bus_id_legacy=_array(
            legacy,
            platform.NVML_BUS_ID_LEGACY_BUFFER_SIZE,
            tail=legacy_tail,
        ),
    )


def test_nvml_constants_match_the_pinned_header_literals() -> None:
    assert platform.NVML_SUCCESS == 0
    assert platform.NVML_BUS_ID_BUFFER_SIZE == 32
    assert platform.NVML_BUS_ID_LEGACY_BUFFER_SIZE == 16


def test_numeric_tuple_validates_uppercase_strings_and_derives_lowercase_path() -> None:
    identity = platform.derive_pci_identity(
        _raw(bus_tail=b"\xff\x01", legacy_tail=b"\xfe")
    )
    assert identity.bus_id == b"0000ABCD:AF:1B.0"
    assert identity.bus_id_legacy == b"ABCD:AF:1B.0"
    assert identity.path_bdf == b"abcd:af:1b.0"
    assert platform.bdf_dependent_paths(identity) == (
        b"/sys/bus/pci/devices/abcd:af:1b.0",
        b"/proc/driver/nvidia/gpus/abcd:af:1b.0/information",
    )


def test_valid_decision_records_complete_tuple_and_logical_strings() -> None:
    decision = platform.evaluate_pci_query(0, _raw())
    assert decision.valid
    assert decision.identity is not None
    assert decision.record == {
        "bus": "175",
        "bus_id_b64": b64url(b"0000ABCD:AF:1B.0"),
        "bus_id_legacy_b64": b64url(b"ABCD:AF:1B.0"),
        "device": "27",
        "domain": "43981",
        "path_bdf_b64": b64url(b"abcd:af:1b.0"),
        "pci_device_id": str(0x233010DE),
        "pci_subsystem_id": str(0x16A110DE),
        "return_code": "0",
        "status": "valid",
    }
    assert base64.urlsafe_b64decode(
        decision.record["path_bdf_b64"] + "===="
    ) == b"abcd:af:1b.0"


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (_raw(bus_id=b"0000abcd:af:1b.0"), "bus_id disagrees"),
        (_raw(legacy=b"abcd:af:1b.0"), "bus_id_legacy disagrees"),
        (_raw(bus_id=b"0000ABCD:AF:1B.1"), "bus_id disagrees"),
        (_raw(legacy=b"ABCD:AF:1B.1"), "bus_id_legacy disagrees"),
        (_raw(domain=0x10000), "domain"),
        (_raw(bus=0x100), "bus"),
        (_raw(device=0x20), "device"),
    ),
)
def test_mismatched_strings_function_and_numeric_ranges_are_invalid(
    raw: platform.RawPciInfo,
    message: str,
) -> None:
    with pytest.raises(ManifestError, match=message):
        platform.derive_pci_identity(raw)
    decision = platform.evaluate_pci_query(0, raw)
    assert not decision.valid
    assert decision.identity is None
    assert decision.record == {
        "return_code": "0",
        "status": "invalid_pci_result",
    }


def test_fixed_arrays_require_exact_size_and_first_nul() -> None:
    valid = _raw()
    cases = (
        platform.RawPciInfo(**{**valid.__dict__, "bus_id": valid.bus_id[:-1]}),
        platform.RawPciInfo(
            **{**valid.__dict__, "bus_id_legacy": b"A" * platform.NVML_BUS_ID_LEGACY_BUFFER_SIZE}
        ),
    )
    for raw in cases:
        with pytest.raises(ManifestError):
            platform.derive_pci_identity(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("domain", True),
        ("bus", -1),
        ("device", 1.0),
        ("pci_device_id", 0x1_0000_0000),
        ("pci_subsystem_id", -1),
    ),
)
def test_every_numeric_field_is_strict_unsigned_integer(field: str, value: object) -> None:
    valid = _raw()
    raw = platform.RawPciInfo(**{**valid.__dict__, field: value})
    with pytest.raises(ManifestError):
        platform.derive_pci_identity(raw)


def test_numeric_subclass_cannot_override_range_or_format_validation() -> None:
    raw = _raw()
    forged = platform.RawPciInfo(
        **{**raw.__dict__, "domain": _DeceptiveInt(0x10000)}
    )
    with pytest.raises(ManifestError, match="domain must be an integer"):
        platform.derive_pci_identity(forged)


@pytest.mark.parametrize("field", ("bus_id", "bus_id_legacy"))
def test_raw_fixed_arrays_require_exact_bytes(field: str) -> None:
    raw = _raw()
    forged = platform.RawPciInfo(
        **{**raw.__dict__, field: _DeceptiveBytes(getattr(raw, field))}
    )
    with pytest.raises(ManifestError, match="complete fixed-size byte array"):
        platform.derive_pci_identity(forged)


@pytest.mark.parametrize(
    ("field", "maximum"),
    (
        ("domain", 0xFFFF),
        ("bus", 0xFF),
        ("device", 0x1F),
        ("pci_device_id", 0xFFFFFFFF),
        ("pci_subsystem_id", 0xFFFFFFFF),
    ),
)
def test_every_numeric_field_accepts_exact_zero_and_exact_maximum(
    field: str,
    maximum: int,
) -> None:
    for value in (0, maximum):
        identity = platform.derive_pci_identity(_raw(**{field: value}))
        assert getattr(identity, field) == value


def test_non_success_is_total_invalid_and_cannot_produce_paths() -> None:
    decision = platform.evaluate_pci_query(3, _raw())
    assert not decision.valid
    assert decision.identity is None
    assert decision.record == {
        "return_code": "3",
        "status": "invalid_nvml_return",
    }
    with pytest.raises(ManifestError, match="valid PCI identity"):
        platform.bdf_dependent_paths(decision.identity)  # type: ignore[arg-type]


def test_success_without_struct_is_invalid_and_does_not_derive() -> None:
    decision = platform.evaluate_pci_query(0, None)
    assert not decision.valid
    assert decision.identity is None
    assert decision.record == {
        "return_code": "0",
        "status": "invalid_missing_pci_result",
    }


def test_success_with_wrong_object_type_is_total_invalid() -> None:
    decision = platform.evaluate_pci_query(0, object())  # type: ignore[arg-type]
    assert not decision.valid
    assert decision.identity is None
    assert decision.record == {
        "return_code": "0",
        "status": "invalid_missing_pci_result",
    }


def test_raw_record_subclass_is_not_a_pinned_nvml_result() -> None:
    raw = _raw()
    subclass = _RawPciInfoSubclass(**raw.__dict__)
    with pytest.raises(ManifestError, match="must be RawPciInfo"):
        platform.derive_pci_identity(subclass)
    decision = platform.evaluate_pci_query(0, subclass)
    assert not decision.valid
    assert decision.record["status"] == "invalid_missing_pci_result"


def test_identity_subclass_is_rejected_at_both_trust_boundaries() -> None:
    identity = platform.derive_pci_identity(_raw())
    subclass = _PciIdentitySubclass(**identity.__dict__)
    with pytest.raises(ManifestError, match="valid PCI identity"):
        subclass.record()
    with pytest.raises(ManifestError, match="valid PCI identity"):
        platform.bdf_dependent_paths(subclass)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("domain", 0x10000),
        ("bus", 0x100),
        ("device", 0x20),
        ("pci_device_id", 0x1_0000_0000),
        ("pci_subsystem_id", 0x1_0000_0000),
        ("bus_id", b"wrong"),
        ("bus_id_legacy", b"wrong"),
        ("path_bdf", b"../../forbidden"),
    ),
)
def test_forged_identity_is_rejected_at_record_and_path_boundaries(
    field: str,
    value: object,
) -> None:
    identity = platform.derive_pci_identity(_raw())
    forged = replace(identity, **{field: value})
    with pytest.raises(ManifestError):
        forged.record()
    with pytest.raises(ManifestError):
        platform.bdf_dependent_paths(forged)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("bus_id", _DeceptiveBytes(b"wrong")),
        ("bus_id_legacy", _DeceptiveBytes(b"wrong")),
        ("path_bdf", _DeceptiveBytes(b"../../forbidden")),
        ("path_bdf", bytearray(b"abcd:af:1b.0")),
    ),
)
def test_forged_identity_cannot_override_equality_or_use_mutable_bytes(
    field: str,
    value: object,
) -> None:
    identity = platform.derive_pci_identity(_raw())
    forged = replace(identity, **{field: value})
    with pytest.raises(ManifestError, match="exact bytes"):
        forged.record()
    with pytest.raises(ManifestError, match="exact bytes"):
        platform.bdf_dependent_paths(forged)


@pytest.mark.parametrize("code", (True, -1, 0x80000000, 1.0))
def test_return_code_itself_is_strict(code: object) -> None:
    with pytest.raises(ManifestError):
        platform.evaluate_pci_query(code, None)  # type: ignore[arg-type]
