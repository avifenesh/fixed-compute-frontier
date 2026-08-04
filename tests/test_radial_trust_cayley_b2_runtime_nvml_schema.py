import base64
from dataclasses import replace

import pytest

import experiments.radial_trust_cayley_b2_runtime_nvml_schema as schema
from experiments.radial_trust_cayley_b2_runtime_manifest import ManifestError


def _buffer(value: bytes, size: int, tail: bytes = b"") -> bytes:
    if len(value) + len(tail) + 1 > size:
        raise ValueError("fixture does not fit")
    return value + b"\0" + tail + bytes(size - len(value) - len(tail) - 1)


def _strings() -> tuple[schema.StringQueryResult, ...]:
    return tuple(
        schema.StringQueryResult(
            query,
            0,
            _buffer(query[:8].encode("ascii"), size, b"\xff"),
        )
        for query, size in schema.STRING_QUERY_SIZES.items()
    )


def _uints() -> tuple[schema.UintQueryResult, ...]:
    return (
        schema.UintQueryResult("applications_clock_mem", 0, 1593),
        schema.UintQueryResult("applications_clock_sm", 0, 1980),
        schema.UintQueryResult("compute_mode", 0, 0),
        schema.UintQueryResult("persistence_mode", 0, 1),
        schema.UintQueryResult("power_management_limit", 0, 700000),
    )


def _pairs() -> tuple[schema.PairQueryResult, ...]:
    return (
        schema.PairQueryResult("ecc_mode", 0, 1, 1),
        schema.PairQueryResult("mig_mode", 0, 0, 0),
        schema.PairQueryResult(
            "power_management_limit_constraints",
            0,
            300000,
            700000,
        ),
    )


def test_pinned_header_constants_and_exact_query_sets() -> None:
    assert schema.NVML_SUCCESS == 0
    assert schema.NVML_ERROR_NOT_SUPPORTED == 3
    assert schema.NVML_DEVICE_UUID_V2_BUFFER_SIZE == 96
    assert schema.NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE == 32
    assert schema.NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE == 16
    assert schema.NVML_GSP_FIRMWARE_VERSION_BUF_SIZE == 64
    assert dict(schema.STRING_QUERY_SIZES) == {
        "gsp_firmware_version": 64,
        "inforom_image_version": 16,
        "uuid": 96,
        "vbios_version": 32,
    }
    assert set(schema.UINT_QUERY_ALLOWED_VALUES) == {
        "compute_mode",
        "persistence_mode",
        "power_management_limit",
        "applications_clock_mem",
        "applications_clock_sm",
    }
    assert set(schema.PAIR_QUERY_SPECS) == {
        "mig_mode",
        "ecc_mode",
        "power_management_limit_constraints",
    }
    with pytest.raises(TypeError):
        schema.STRING_QUERY_SIZES["uuid"] = 1  # type: ignore[index]


def test_string_success_uses_first_nul_and_ignores_fixed_array_tail() -> None:
    result = schema.StringQueryResult(
        "vbios_version",
        0,
        _buffer(b"96.00.5E.00.01", 32, b"untrusted-tail"),
    )
    record = schema.evaluate_string_result(result)
    assert base64.urlsafe_b64decode(record["value_b64"] + "====") == b"96.00.5E.00.01"
    assert record == {
        "query": "vbios_version",
        "return_code": "0",
        "status": "ok",
        "value_b64": record["value_b64"],
    }


@pytest.mark.parametrize(
    "buffer",
    (
        b"short\0",
        b"x" * schema.NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE,
        bytes(schema.NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE),
        bytearray(schema.NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE),
    ),
)
def test_string_success_requires_exact_complete_nonempty_buffer(buffer: object) -> None:
    with pytest.raises(ManifestError):
        schema.evaluate_string_result(
            schema.StringQueryResult("vbios_version", 0, buffer)  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "result",
    (
        schema.StringQueryResult("uuid", 3, None),
        schema.UintQueryResult("persistence_mode", 3, None),
        schema.PairQueryResult("ecc_mode", 3, None, None),
    ),
)
def test_not_supported_is_the_only_encoded_non_success(result: object) -> None:
    if type(result) is schema.StringQueryResult:
        record = schema.evaluate_string_result(result)
    elif type(result) is schema.UintQueryResult:
        record = schema.evaluate_uint_result(result)
    else:
        record = schema.evaluate_pair_result(result)  # type: ignore[arg-type]
    assert record["return_code"] == "3"
    assert record["status"] == "not_supported"
    assert set(record) == {"query", "return_code", "status"}


@pytest.mark.parametrize("code", (1, 2, 4, 999))
def test_every_other_nvml_failure_invalidates(code: int) -> None:
    with pytest.raises(ManifestError, match="disallowed"):
        schema.evaluate_uint_result(schema.UintQueryResult("compute_mode", code, None))


def test_not_supported_rejects_stale_payloads() -> None:
    with pytest.raises(ManifestError, match="stale"):
        schema.evaluate_string_result(
            schema.StringQueryResult(
                "uuid",
                3,
                bytes(schema.NVML_DEVICE_UUID_V2_BUFFER_SIZE),
            )
        )
    with pytest.raises(ManifestError, match="stale"):
        schema.evaluate_pair_result(schema.PairQueryResult("mig_mode", 3, 0, None))


@pytest.mark.parametrize(
    "result",
    (
        schema.UintQueryResult("compute_mode", 0, 4),
        schema.UintQueryResult("persistence_mode", 0, 2),
        schema.PairQueryResult("mig_mode", 0, 0, 2),
        schema.PairQueryResult("ecc_mode", 0, 3, 1),
    ),
)
def test_enum_queries_reject_unknown_values(result: object) -> None:
    if type(result) is schema.UintQueryResult:
        with pytest.raises(ManifestError, match="enum"):
            schema.evaluate_uint_result(result)
    else:
        with pytest.raises(ManifestError, match="enum"):
            schema.evaluate_pair_result(result)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, -1, 0x1_0000_0000, 1.0))
def test_uint32_payload_is_strict(value: object) -> None:
    with pytest.raises(ManifestError):
        schema.evaluate_uint_result(
            schema.UintQueryResult("power_management_limit", 0, value)  # type: ignore[arg-type]
        )


def test_power_constraint_order_and_current_consistency() -> None:
    with pytest.raises(ManifestError, match="minimum exceeds"):
        schema.evaluate_pair_result(
            schema.PairQueryResult(
                "power_management_limit_constraints",
                0,
                700000,
                300000,
            )
        )
    bad_uints = tuple(
        replace(result, value=700001)
        if result.query == "power_management_limit"
        else result
        for result in _uints()
    )
    with pytest.raises(ManifestError, match="outside"):
        schema.compose_scalar_invariant_records(_strings(), bad_uints, _pairs())


def test_complete_composition_is_exact_unique_and_query_sorted() -> None:
    records = schema.compose_scalar_invariant_records(
        tuple(reversed(_strings())),
        tuple(reversed(_uints())),
        tuple(reversed(_pairs())),
    )
    queries = [record["query"] for record in records]
    assert queries == sorted(queries, key=lambda value: value.encode("ascii"))
    assert len(records) == 12


def test_composition_rejects_missing_duplicate_and_non_tuple_groups() -> None:
    with pytest.raises(ManifestError, match="missing or extra"):
        schema.compose_scalar_invariant_records(_strings()[:-1], _uints(), _pairs())
    with pytest.raises(ManifestError, match="duplicate"):
        schema.compose_scalar_invariant_records(
            _strings() + (_strings()[0],),
            _uints(),
            _pairs(),
        )
    with pytest.raises(ManifestError, match="exact tuples"):
        schema.compose_scalar_invariant_records(list(_strings()), _uints(), _pairs())  # type: ignore[arg-type]


class _StringSubclass(schema.StringQueryResult):
    pass


class _UintSubclass(schema.UintQueryResult):
    pass


class _PairSubclass(schema.PairQueryResult):
    pass


@pytest.mark.parametrize(
    ("forged", "evaluator", "message"),
    (
        (
            _StringSubclass(**_strings()[0].__dict__),
            schema.evaluate_string_result,
            "exact StringQueryResult",
        ),
        (
            _UintSubclass(**_uints()[0].__dict__),
            schema.evaluate_uint_result,
            "exact UintQueryResult",
        ),
        (
            _PairSubclass(**_pairs()[0].__dict__),
            schema.evaluate_pair_result,
            "exact PairQueryResult",
        ),
    ),
)
def test_result_record_subclasses_are_rejected(
    forged: object,
    evaluator: object,
    message: str,
) -> None:
    with pytest.raises(ManifestError, match=message):
        evaluator(forged)  # type: ignore[operator]


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


def test_primitive_subclasses_are_rejected() -> None:
    with pytest.raises(ManifestError, match="return code"):
        schema.evaluate_uint_result(
            schema.UintQueryResult("compute_mode", _IntSubclass(0), 0)
        )
    with pytest.raises(ManifestError, match="exact integer"):
        schema.evaluate_uint_result(
            schema.UintQueryResult("compute_mode", 0, _IntSubclass(0))
        )
    with pytest.raises(ManifestError, match="query name"):
        schema.evaluate_uint_result(
            schema.UintQueryResult(_StrSubclass("compute_mode"), 0, 0)
        )
