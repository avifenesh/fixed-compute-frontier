import base64
import errno
import hashlib

import pytest

import experiments.radial_trust_cayley_b2_runtime_manifest as runtime


def test_frozen_schema_environment_and_cgroup_names_are_literal() -> None:
    assert runtime.CONTAINER_SCHEMA == "b2-container-runtime-v1"
    assert runtime.PROCESS_SCHEMA == "b2-process-runtime-v1"
    assert runtime.RAW_ENV_EXACT == (b"LD_LIBRARY_PATH", b"LD_PRELOAD")
    assert runtime.RAW_ENV_PREFIXES == (
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
    assert runtime.SECRET_NAME_FRAGMENTS == (
        b"KEY",
        b"TOKEN",
        b"SECRET",
        b"PASSWORD",
    )
    assert runtime.CGROUP_NAMED_FILES == (
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


def test_canonical_json_subset_has_frozen_bytes_and_forbids_numbers() -> None:
    value = {
        "z": ["2", True, None],
        "a": {"path_b64": runtime.b64url(b"/tmp/\xff")},
    }
    assert runtime.canonical_json_bytes(value) == (
        b'{"a":{"path_b64":"L3RtcC__"},"z":["2",true,null]}'
    )
    with pytest.raises(runtime.ManifestError, match="JSON number forbidden"):
        runtime.canonical_json_bytes({"bad": 1})
    with pytest.raises(runtime.ManifestError, match="JSON number forbidden"):
        runtime.canonical_json_bytes({"bad": -0.0})
    with pytest.raises(runtime.ManifestError, match="ASCII"):
        runtime.canonical_json_bytes({"non_ascii_é": "value"})
    with pytest.raises(runtime.ManifestError, match="ASCII"):
        runtime.canonical_json_bytes({"value": "é"})
    with pytest.raises(runtime.ManifestError, match="ASCII"):
        runtime.canonical_json_bytes({"value": "\ud800"})


def test_byte_encodings_and_decimal_integer_are_unambiguous() -> None:
    raw = b"\x00\xfb\xff"
    assert runtime.b64url(raw) == "APv_"
    assert "=" not in runtime.b64url(raw)
    assert base64.urlsafe_b64decode(runtime.b64url(raw) + "====") == raw
    assert runtime.sha256_bytes(raw) == hashlib.sha256(raw).hexdigest()
    assert runtime.decimal_integer(-17) == "-17"
    with pytest.raises(runtime.ManifestError):
        runtime.decimal_integer(True)
    for valid, expected in (("0", 0), ("17", 17), ("-17", -17)):
        assert runtime.validate_decimal_string(valid) == expected
    for invalid in ("", "+1", "01", "-0", "1.0", " 1"):
        with pytest.raises(runtime.ManifestError):
            runtime.validate_decimal_string(invalid)


def test_environment_records_sort_bytes_hash_every_value_and_redact_secrets() -> None:
    raw = (
        b"PATH=/bin\0"
        b"CUDA_VISIBLE_DEVICES=0\0"
        b"CUDA_API_TOKEN=do-not-publish\0"
        b"LD_PRELOAD=/opt/a.so\0"
        b"nvidia_lower=not-allowlisted\0"
    )
    records = runtime.environment_records(raw)
    names = [
        base64.urlsafe_b64decode(record["name_b64"] + "====")
        for record in records
    ]
    assert names == sorted(names)
    by_name = dict(zip(names, records, strict=True))
    assert by_name[b"CUDA_VISIBLE_DEVICES"]["value_b64"] == runtime.b64url(b"0")
    assert by_name[b"LD_PRELOAD"]["value_b64"] == runtime.b64url(b"/opt/a.so")
    assert "value_b64" not in by_name[b"CUDA_API_TOKEN"]
    assert "value_b64" not in by_name[b"PATH"]
    assert "value_b64" not in by_name[b"nvidia_lower"]
    assert by_name[b"CUDA_API_TOKEN"]["value_sha256"] == runtime.sha256_bytes(
        b"do-not-publish"
    )


@pytest.mark.parametrize(
    "name",
    (b"CUDA_API_kEy", b"NVIDIA_auth_ToKeN", b"CUBLAS_client_sEcReT", b"PYTORCH_PaSsWoRd"),
)
def test_every_secret_fragment_overrides_raw_allowlist_case_insensitively(
    name: bytes,
) -> None:
    record = runtime.environment_records(name + b"=never-publish\0")[0]
    assert "value_b64" not in record
    assert record["value_sha256"] == runtime.sha256_bytes(b"never-publish")


@pytest.mark.parametrize(
    "raw",
    (b"NO_EQUALS\0", b"=empty-name\0", b"A=1\0A=2\0"),
)
def test_environment_parser_rejects_ambiguous_records(raw: bytes) -> None:
    with pytest.raises(runtime.ManifestError):
        runtime.parse_proc_environ(raw)


def test_path_record_preserves_bytes_or_exact_positive_errno() -> None:
    good = runtime.path_bytes_record(b"/exact", reader=lambda _: b"raw\x00bytes")
    assert good == {
        "bytes_b64": runtime.b64url(b"raw\x00bytes"),
        "path_b64": runtime.b64url(b"/exact"),
        "sha256": runtime.sha256_bytes(b"raw\x00bytes"),
        "status": "ok",
    }

    def missing(_: bytes) -> bytes:
        raise OSError(errno.ENOENT, "absent")

    assert runtime.path_bytes_record(b"/absent", reader=missing) == {
        "errno": str(errno.ENOENT),
        "path_b64": runtime.b64url(b"/absent"),
        "status": "error",
    }


def test_unified_cgroup_parser_rejects_hybrid_and_unsafe_paths() -> None:
    assert runtime.parse_unified_cgroup_path(b"0::/tenant/job\n") == b"/tenant/job"
    for invalid in (
        b"malformed\n",
        b"2:cpu:/tenant\n",
        b"0::/a\n0::/b\n",
        b"0::relative\n",
        b"0::/a/../b\n",
        b"0::/a/./b\n",
        b"0:://a\n",
        b"0::/a//b\n",
        b"0::/a/\n",
    ):
        with pytest.raises(runtime.ManifestError):
            runtime.parse_unified_cgroup_path(invalid)


def test_cgroup_collection_uses_exact_named_files_and_sorted_hugetlb_max() -> None:
    directory = b"/fake/cgroup/tenant/job"
    existing = {
        directory + b"/" + name: b"value:" + name
        for name in runtime.CGROUP_NAMED_FILES
    }
    existing[directory + b"/hugetlb.2MB.max"] = b"2048\n"
    existing[directory + b"/hugetlb.1GB.max"] = b"max\n"

    def reader(path: bytes) -> bytes:
        try:
            return existing[path]
        except KeyError as error:
            raise OSError(errno.ENOENT, "absent") from error

    result = runtime.collect_cgroup_records(
        b"0::/tenant/job\n",
        root=b"/fake/cgroup",
        reader=reader,
        listdir=lambda _: [
            b"hugetlb.2MB.max",
            b"memory.current",
            b"hugetlb.1GB.max",
            b"hugetlb.2MB.current",
        ],
    )
    paths = [
        base64.urlsafe_b64decode(record["path_b64"] + "====")
        for record in result["files"]
    ]
    assert paths == sorted(paths)
    assert directory + b"/hugetlb.1GB.max" in paths
    assert directory + b"/hugetlb.2MB.max" in paths
    assert len(paths) == len(runtime.CGROUP_NAMED_FILES) + 2
    assert all(record["status"] == "ok" for record in result["files"])


def test_missing_named_cgroup_file_is_an_errno_record_not_an_omission() -> None:
    def missing(path: bytes) -> bytes:
        raise OSError(errno.ENOENT, path.decode("ascii"))

    result = runtime.collect_cgroup_records(
        b"0::/\n",
        root=b"/cgroup",
        reader=missing,
        listdir=lambda _: [],
    )
    assert len(result["files"]) == len(runtime.CGROUP_NAMED_FILES)
    assert {record["errno"] for record in result["files"]} == {str(errno.ENOENT)}


def test_cgroup_collection_joins_filesystem_root_without_double_slash() -> None:
    listed: list[bytes] = []
    read: list[bytes] = []

    def reader(path: bytes) -> bytes:
        read.append(path)
        raise OSError(errno.ENOENT, "absent")

    result = runtime.collect_cgroup_records(
        b"0::/tenant\n",
        root=b"/",
        reader=reader,
        listdir=lambda path: listed.append(path) or [],
    )
    assert listed == [b"/tenant"]
    assert read
    assert all(path.startswith(b"/tenant/") for path in read)
    assert all(not path.startswith(b"//") for path in read)
    assert result["directory_b64"] == runtime.b64url(b"/tenant")


def test_manifest_hash_and_boundary_comparison_are_exact() -> None:
    first = runtime.build_manifest(
        runtime.CONTAINER_SCHEMA,
        {"counter": "1", "raw": runtime.b64url(b"same")},
    )
    reordered = runtime.build_manifest(
        runtime.CONTAINER_SCHEMA,
        {"raw": runtime.b64url(b"same"), "counter": "1"},
    )
    assert first.canonical == reordered.canonical
    assert first.sha256 == hashlib.sha256(first.canonical).hexdigest()
    runtime.require_same_manifest(first, reordered)

    changed = runtime.build_manifest(
        runtime.CONTAINER_SCHEMA,
        {"counter": "2", "raw": runtime.b64url(b"same")},
    )
    with pytest.raises(runtime.ManifestError, match="runtime invariant changed"):
        runtime.require_same_manifest(first, changed)
    process = runtime.build_manifest(runtime.PROCESS_SCHEMA, {"counter": "1"})
    with pytest.raises(runtime.ManifestError, match="schema changed"):
        runtime.require_same_manifest(first, process)

    forged_hash = runtime.ManifestBlob(first.schema, first.canonical, "0" * 64)
    with pytest.raises(runtime.ManifestError, match="cached SHA-256"):
        runtime.require_same_manifest(forged_hash, first)
    wrong_embedded_schema = first.canonical.replace(
        b"b2-container-runtime-v1", b"b2-process-runtime-v1"
    )
    forged_schema = runtime.ManifestBlob(
        first.schema,
        wrong_embedded_schema,
        runtime.sha256_bytes(wrong_embedded_schema),
    )
    with pytest.raises(runtime.ManifestError, match="schema disagrees"):
        runtime.require_same_manifest(forged_schema, first)
    noncanonical = b'{"schema":"b2-container-runtime-v1", "payload":{}}'
    forged_canonical = runtime.ManifestBlob(
        first.schema,
        noncanonical,
        runtime.sha256_bytes(noncanonical),
    )
    with pytest.raises(runtime.ManifestError, match="not canonical"):
        runtime.require_same_manifest(forged_canonical, first)

    unknown_document = {"payload": {}, "schema": "unknown-schema"}
    unknown_canonical = runtime.canonical_json_bytes(unknown_document)
    unknown_schema = runtime.ManifestBlob(
        "unknown-schema",
        unknown_canonical,
        runtime.sha256_bytes(unknown_canonical),
    )
    with pytest.raises(runtime.ManifestError, match="unknown B2 runtime schema"):
        runtime.require_same_manifest(unknown_schema, unknown_schema)

    list_document = {"payload": [], "schema": runtime.CONTAINER_SCHEMA}
    list_canonical = runtime.canonical_json_bytes(list_document)
    list_payload = runtime.ManifestBlob(
        runtime.CONTAINER_SCHEMA,
        list_canonical,
        runtime.sha256_bytes(list_canonical),
    )
    with pytest.raises(runtime.ManifestError, match="payload must be an object"):
        runtime.require_same_manifest(list_payload, list_payload)


def test_live_proc_self_environment_can_be_recorded_without_raw_secret_values() -> None:
    with open("/proc/self/environ", "rb") as handle:
        records = runtime.environment_records(handle.read())
    assert records
    for record in records:
        name = base64.urlsafe_b64decode(record["name_b64"] + "====")
        if any(fragment in name.upper() for fragment in runtime.SECRET_NAME_FRAGMENTS):
            assert "value_b64" not in record
