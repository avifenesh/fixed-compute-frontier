from __future__ import annotations

import base64
from dataclasses import replace
import errno
import os
from pathlib import Path

import pytest

import experiments.radial_trust_cayley_b2_runtime_platform as platform
import experiments.radial_trust_cayley_b2_runtime_platform_filesystem as pfs
from experiments.radial_trust_cayley_b2_runtime_manifest import ManifestError


def _array(value: bytes, size: int) -> bytes:
    return value + b"\0" + bytes(size - len(value) - 1)


def _identity() -> platform.PciIdentity:
    return platform.derive_pci_identity(
        platform.RawPciInfo(
            domain=0xABCD,
            bus=0xAF,
            device=0x1B,
            pci_device_id=0x233010DE,
            pci_subsystem_id=0x16A110DE,
            bus_id=_array(
                b"0000ABCD:AF:1B.0",
                platform.NVML_BUS_ID_BUFFER_SIZE,
            ),
            bus_id_legacy=_array(
                b"ABCD:AF:1B.0",
                platform.NVML_BUS_ID_LEGACY_BUFFER_SIZE,
            ),
        )
    )


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "====")


def test_frozen_platform_name_sets_are_independent_literals() -> None:
    assert pfs.BDF_FILE_NAMES == (
        b"vendor",
        b"device",
        b"subsystem_vendor",
        b"subsystem_device",
        b"class",
        b"revision",
        b"numa_node",
        b"current_link_speed",
        b"current_link_width",
        b"max_link_speed",
        b"max_link_width",
        b"local_cpulist",
        b"local_cpus",
    )
    assert pfs.PROC_FILE_SUFFIXES == ((b"version",), (b"params",))
    assert pfs.MODULE_PARAMETER_SUFFIXES == (
        (b"nvidia", b"parameters"),
        (b"nvidia_uvm", b"parameters"),
    )


def test_exact_bdf_and_proc_paths_are_sorted_and_missing_errno_is_preserved(
    tmp_path: Path,
) -> None:
    sys_root = tmp_path / "pci"
    proc_root = tmp_path / "nvidia"
    bdf = sys_root / "abcd:af:1b.0"
    info = proc_root / "gpus" / "abcd:af:1b.0"
    bdf.mkdir(parents=True)
    info.mkdir(parents=True)
    (bdf / "vendor").write_bytes(b"0x10de\n")
    (proc_root / "version").write_bytes(b"NVRM version\n")
    (info / "information").write_bytes(b"Model: H100\n")

    record = pfs.collect_bdf_and_proc_records(
        _identity(),
        sysfs_pci_root=os.fsencode(sys_root),
        proc_nvidia_root=os.fsencode(proc_root),
    )

    assert _decode(record["bdf_directory_b64"]) == os.fsencode(bdf)
    bdf_records = record["bdf_files"]
    assert len(bdf_records) == 13
    bdf_paths = [_decode(item["path_b64"]) for item in bdf_records]
    assert bdf_paths == sorted(bdf_paths)
    vendor = next(item for item in bdf_records if _decode(item["path_b64"]).endswith(b"/vendor"))
    device = next(item for item in bdf_records if _decode(item["path_b64"]).endswith(b"/device"))
    assert _decode(vendor["bytes_b64"]) == b"0x10de\n"
    assert device == {
        "errno": str(errno.ENOENT),
        "path_b64": device["path_b64"],
        "status": "error",
    }

    proc_records = record["proc_files"]
    proc_paths = [_decode(item["path_b64"]) for item in proc_records]
    assert proc_paths == sorted(proc_paths)
    assert len(proc_records) == 3
    assert any(path.endswith(b"/params") for path in proc_paths)
    assert any(path.endswith(b"/gpus/abcd:af:1b.0/information") for path in proc_paths)


def test_forged_bdf_is_rejected_before_any_reader_call() -> None:
    identity = replace(_identity(), path_bdf=b"../../forbidden")

    def forbidden_reader(path: bytes) -> bytes:
        raise AssertionError(f"reader called for {path!r}")

    with pytest.raises(ManifestError, match="path_bdf"):
        pfs.collect_bdf_and_proc_records(identity, reader=forbidden_reader)


def test_verified_but_unreadable_named_file_preserves_exact_errno(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named = tmp_path / "named"
    named.write_bytes(b"value")
    original_open = pfs.os.open

    def deny_procfd(path: bytes, flags: int, *args: object, **kwargs: object) -> int:
        if type(path) is bytes and path.startswith(b"/proc/self/fd/"):
            raise OSError(errno.EACCES, "denied")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(pfs.os, "open", deny_procfd)
    record = pfs.path_bytes_record(
        os.fsencode(named),
        reader=pfs._stable_named_regular_bytes,
    )
    assert record == {
        "errno": str(errno.EACCES),
        "path_b64": record["path_b64"],
        "status": "error",
    }


def test_read_errno_is_not_attributed_after_named_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named = tmp_path / "named"
    replacement = tmp_path / "replacement"
    named.write_bytes(b"old")
    replacement.write_bytes(b"new")
    original_read = pfs.os.read
    fired = False

    def replace_then_fail(descriptor: int, size: int) -> bytes:
        nonlocal fired
        if not fired:
            fired = True
            os.replace(replacement, named)
            raise OSError(errno.EIO, "read failure")
        return original_read(descriptor, size)

    monkeypatch.setattr(pfs.os, "read", replace_then_fail)
    with pytest.raises(ManifestError, match="changed"):
        pfs.path_bytes_record(
            os.fsencode(named),
            reader=pfs._stable_named_regular_bytes,
        )


@pytest.mark.parametrize(
    "root",
    (b"relative", b"/double//slash", b"/traversal/../escape", bytearray(b"/sys")),
)
def test_collection_roots_are_canonical_exact_bytes(root: object) -> None:
    with pytest.raises(ManifestError):
        pfs.collect_bdf_and_proc_records(
            _identity(),
            sysfs_pci_root=root,  # type: ignore[arg-type]
            reader=lambda path: b"",
        )


def test_module_parameters_are_direct_sorted_and_symlinks_are_not_followed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "module"
    parameters = root / "nvidia" / "parameters"
    parameters.mkdir(parents=True)
    (parameters / "zeta").write_bytes(b"1\n")
    (parameters / "alpha").symlink_to(b"../../outside-secret")

    records = pfs.collect_module_parameter_records(sys_module_root=os.fsencode(root))
    assert len(records) == 2
    nvidia = next(
        record
        for record in records
        if _decode(record["path_b64"]).endswith(b"/nvidia/parameters")
    )
    assert nvidia["status"] == "ok"
    assert [_decode(item["path_b64"]).split(b"/")[-1] for item in nvidia["entries"]] == [
        b"alpha",
        b"zeta",
    ]
    alpha, zeta = nvidia["entries"]
    assert alpha == {
        "path_b64": alpha["path_b64"],
        "status": "ok",
        "symlink_b64": alpha["symlink_b64"],
        "type": "symlink",
    }
    assert _decode(alpha["symlink_b64"]) == b"../../outside-secret"
    assert zeta["type"] == "regular"
    assert _decode(zeta["bytes_b64"]) == b"1\n"

    uvm = next(
        record
        for record in records
        if _decode(record["path_b64"]).endswith(b"/nvidia_uvm/parameters")
    )
    assert uvm == {
        "errno": str(errno.ENOENT),
        "path_b64": uvm["path_b64"],
        "status": "error",
    }


def test_module_parameter_collection_rejects_recursive_blind_spot(tmp_path: Path) -> None:
    root = tmp_path / "module"
    (root / "nvidia" / "parameters" / "nested").mkdir(parents=True)
    with pytest.raises(ManifestError, match="non-regular, non-symlink"):
        pfs.collect_module_parameter_records(sys_module_root=os.fsencode(root))


@pytest.mark.parametrize("substitution", ("regular", "symlink"))
def test_module_parameter_directory_structural_substitution_is_invalid(
    tmp_path: Path,
    substitution: str,
) -> None:
    root = tmp_path / "module"
    module = root / "nvidia"
    module.mkdir(parents=True)
    parameters = module / "parameters"
    if substitution == "regular":
        parameters.write_bytes(b"not a directory")
    else:
        target = tmp_path / "target"
        target.mkdir()
        parameters.symlink_to(target, target_is_directory=True)
    with pytest.raises(ManifestError, match="not a directory"):
        pfs.collect_module_parameter_records(sys_module_root=os.fsencode(root))


class _ChangingListingOps(pfs.ModuleParameterOps):
    def __init__(self) -> None:
        object.__setattr__(self, "calls", 0)

    def listdir(self, descriptor: int) -> tuple[bytes, ...]:
        object.__setattr__(self, "calls", self.calls + 1)
        return () if self.calls == 1 else (b"appeared",)


def test_module_directory_listing_race_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "module"
    (root / "nvidia" / "parameters").mkdir(parents=True)
    with pytest.raises(ManifestError, match="names changed"):
        pfs.collect_module_parameter_records(
            sys_module_root=os.fsencode(root),
            ops=_ChangingListingOps(),
        )


class _UnreadableOps(pfs.ModuleParameterOps):
    def open_regular_at(self, directory_fd: int, name: bytes) -> int:
        raise OSError(errno.EACCES, "denied")


def test_unreadable_direct_regular_parameter_records_exact_errno(tmp_path: Path) -> None:
    root = tmp_path / "module"
    parameters = root / "nvidia" / "parameters"
    parameters.mkdir(parents=True)
    (parameters / "locked").write_bytes(b"secret")
    records = pfs.collect_module_parameter_records(
        sys_module_root=os.fsencode(root),
        ops=_UnreadableOps(),
    )
    nvidia = next(record for record in records if record["status"] == "ok")
    assert nvidia["entries"][0] == {
        "errno": str(errno.EACCES),
        "path_b64": nvidia["entries"][0]["path_b64"],
        "status": "error",
        "type": "regular",
    }


class _EnumerableUnsearchableOps(pfs.ModuleParameterOps):
    def stat_at(self, directory_fd: int, name: bytes) -> os.stat_result:
        raise OSError(errno.EACCES, "directory is not searchable")


def test_enumerable_but_unsearchable_child_records_exact_errno(tmp_path: Path) -> None:
    root = tmp_path / "module"
    parameters = root / "nvidia" / "parameters"
    parameters.mkdir(parents=True)
    (parameters / "visible-name").write_bytes(b"hidden-value")
    records = pfs.collect_module_parameter_records(
        sys_module_root=os.fsencode(root),
        ops=_EnumerableUnsearchableOps(),
    )
    nvidia = next(record for record in records if record["status"] == "ok")
    assert nvidia["entries"] == [
        {
            "errno": str(errno.EACCES),
            "path_b64": nvidia["entries"][0]["path_b64"],
            "status": "error",
        }
    ]
    assert _decode(nvidia["entries"][0]["path_b64"]).endswith(b"/visible-name")


class _VanishedAfterListingOps(pfs.ModuleParameterOps):
    def stat_at(self, directory_fd: int, name: bytes) -> os.stat_result:
        raise OSError(errno.ENOENT, "vanished")


def test_listed_child_enoent_is_a_race_not_an_absence_record(tmp_path: Path) -> None:
    root = tmp_path / "module"
    parameters = root / "nvidia" / "parameters"
    parameters.mkdir(parents=True)
    (parameters / "visible-name").write_bytes(b"value")
    with pytest.raises(ManifestError, match="became unstatable"):
        pfs.collect_module_parameter_records(
            sys_module_root=os.fsencode(root),
            ops=_VanishedAfterListingOps(),
        )
