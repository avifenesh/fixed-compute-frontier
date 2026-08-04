#!/usr/bin/env python3
"""Exact BDF and NVIDIA-module filesystem slice for the B2 invariant.

This module is CPU-only.  It consumes an already validated PCI identity, reads
only the frozen sysfs/procfs names, and records only direct regular or symlink
entries in the two frozen NVIDIA module-parameter directories.  It never scans
for a GPU BDF and never follows a module-parameter symlink.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import os
import stat
from typing import Callable, Sequence

from experiments.radial_trust_cayley_b2_runtime_manifest import (
    ManifestError,
    b64url,
    decimal_integer,
    path_bytes_record,
    sha256_bytes,
)
from experiments.radial_trust_cayley_b2_runtime_platform import (
    PciIdentity,
    bdf_dependent_paths,
)


BDF_FILE_NAMES = (
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

PROC_FILE_SUFFIXES = (
    (b"version",),
    (b"params",),
)

MODULE_PARAMETER_SUFFIXES = (
    (b"nvidia", b"parameters"),
    (b"nvidia_uvm", b"parameters"),
)


class _NamedReadError(OSError):
    """An errno caused by reading the named file, rather than recorder state."""


def _canonical_absolute(path: bytes) -> None:
    if type(path) is not bytes or not path.startswith(b"/") or b"\0" in path:
        raise ManifestError("root must be absolute exact bytes without NUL")
    if path == b"/":
        return
    components = path.split(b"/")
    if components[0] != b"" or any(
        component in (b"", b".", b"..") for component in components[1:]
    ):
        raise ManifestError("root must have canonical absolute components")


def _basename(name: bytes) -> None:
    if (
        type(name) is not bytes
        or name in (b"", b".", b"..")
        or b"/" in name
        or b"\0" in name
    ):
        raise ManifestError("entry name must be one exact byte basename")


def _join(root: bytes, *components: bytes) -> bytes:
    _canonical_absolute(root)
    for component in components:
        _basename(component)
    prefix = root.rstrip(b"/")
    return prefix + b"/" + b"/".join(components)


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_rdev,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(descriptor, 1 << 20)
        except OSError as error:
            raise _NamedReadError(error.errno, error.strerror) from error
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _verify_named_path(
    path: bytes,
    path_fd: int,
    expected: os.stat_result,
    *,
    path_flags: int,
) -> None:
    if _identity(os.fstat(path_fd)) != _identity(expected):
        raise ManifestError("named path descriptor changed while reading")
    try:
        reopened = os.open(path, path_flags)
    except OSError as error:
        raise ManifestError(
            f"cannot reopen named path after read: errno {error.errno}"
        ) from error
    try:
        if _identity(os.fstat(reopened)) != _identity(expected):
            raise ManifestError("named path identity changed while reading")
    finally:
        os.close(reopened)


def _stable_named_regular_bytes(path: bytes) -> bytes:
    """Read one exact regular path without following its final component."""
    _canonical_absolute(path)
    if not hasattr(os, "O_PATH") or not hasattr(os, "O_NOFOLLOW"):
        raise ManifestError("O_PATH and O_NOFOLLOW are required")
    path_flags = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        path_fd = os.open(path, path_flags)
    except OSError as error:
        raise _NamedReadError(error.errno, error.strerror, path) from error
    try:
        before = os.fstat(path_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ManifestError("frozen named path is not a regular file")
        procfd = f"/proc/self/fd/{path_fd}".encode("ascii")
        try:
            read_fd = os.open(procfd, os.O_RDONLY | os.O_CLOEXEC)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EPERM):
                _verify_named_path(path, path_fd, before, path_flags=path_flags)
                raise _NamedReadError(error.errno, error.strerror, path) from error
            raise ManifestError(
                f"cannot open verified named inode through procfd: errno {error.errno}"
            ) from error
        try:
            read_before = os.fstat(read_fd)
            if _identity(read_before) != _identity(before):
                raise ManifestError("named read descriptor identity differs")
            try:
                raw = _read_all(read_fd)
            except _NamedReadError:
                if _identity(os.fstat(read_fd)) != _identity(read_before):
                    raise ManifestError("unreadable named inode changed")
                _verify_named_path(path, path_fd, before, path_flags=path_flags)
                raise
            if _identity(os.fstat(read_fd)) != _identity(read_before):
                raise ManifestError("named regular file changed while reading")
        finally:
            os.close(read_fd)
        _verify_named_path(path, path_fd, before, path_flags=path_flags)
        return raw
    finally:
        os.close(path_fd)


def collect_bdf_and_proc_records(
    identity: PciIdentity,
    *,
    sysfs_pci_root: bytes = b"/sys/bus/pci/devices",
    proc_nvidia_root: bytes = b"/proc/driver/nvidia",
    reader: Callable[[bytes], bytes] = _stable_named_regular_bytes,
) -> dict[str, object]:
    """Collect exactly the frozen BDF attributes and NVIDIA proc files."""
    bdf_dependent_paths(identity)
    _canonical_absolute(sysfs_pci_root)
    _canonical_absolute(proc_nvidia_root)
    if not callable(reader):
        raise ManifestError("named path reader must be callable")

    bdf_directory = _join(sysfs_pci_root, identity.path_bdf)
    bdf_paths = [_join(bdf_directory, name) for name in BDF_FILE_NAMES]
    proc_paths = [
        *(_join(proc_nvidia_root, *suffix) for suffix in PROC_FILE_SUFFIXES),
        _join(proc_nvidia_root, b"gpus", identity.path_bdf, b"information"),
    ]

    return {
        "bdf_directory_b64": b64url(bdf_directory),
        "bdf_files": [
            path_bytes_record(path, reader=reader) for path in sorted(bdf_paths)
        ],
        "proc_files": [
            path_bytes_record(path, reader=reader) for path in sorted(proc_paths)
        ],
    }


@dataclass(frozen=True)
class ModuleParameterOps:
    """Descriptor-relative operations for one-level module parameter capture."""

    def open_path(self, path: bytes) -> int:
        if not hasattr(os, "O_PATH") or not hasattr(os, "O_NOFOLLOW"):
            raise ManifestError("O_PATH and O_NOFOLLOW are required")
        return os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)

    def open_directory_from_path(self, path_descriptor: int) -> int:
        procfd = f"/proc/self/fd/{path_descriptor}".encode("ascii")
        return os.open(procfd, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)

    def listdir(self, descriptor: int) -> Sequence[bytes | str]:
        return os.listdir(descriptor)

    def fstat(self, descriptor: int) -> os.stat_result:
        return os.fstat(descriptor)

    def stat_at(self, directory_fd: int, name: bytes) -> os.stat_result:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)

    def open_regular_at(self, directory_fd: int, name: bytes) -> int:
        return os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )

    def readlink_at(self, directory_fd: int, name: bytes) -> bytes:
        target = os.readlink(name, dir_fd=directory_fd)
        if type(target) is not bytes:
            raise ManifestError("bytes symlink path must return exact bytes")
        return target

    def read_all(self, descriptor: int) -> bytes:
        return _read_all(descriptor)

    def close(self, descriptor: int) -> None:
        os.close(descriptor)


def _directory_names(
    ops: ModuleParameterOps,
    descriptor: int,
) -> tuple[bytes, ...]:
    names = tuple(os.fsencode(name) for name in ops.listdir(descriptor))
    for name in names:
        _basename(name)
    if len(names) != len(set(names)):
        raise ManifestError("module parameter listing contains duplicate names")
    return tuple(sorted(names))


def _module_regular_record(
    ops: ModuleParameterOps,
    directory_fd: int,
    name: bytes,
    path: bytes,
    initial: os.stat_result,
) -> dict[str, str]:
    try:
        descriptor = ops.open_regular_at(directory_fd, name)
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM):
            raise ManifestError(
                f"module parameter changed before open: errno {error.errno}"
            ) from error
        if _identity(ops.stat_at(directory_fd, name)) != _identity(initial):
            raise ManifestError("unreadable module parameter changed")
        return {
            "errno": decimal_integer(error.errno),
            "path_b64": b64url(path),
            "status": "error",
            "type": "regular",
        }
    try:
        opened = ops.fstat(descriptor)
        if _identity(opened) != _identity(initial):
            raise ManifestError("opened module parameter identity changed")
        try:
            raw = ops.read_all(descriptor)
        except _NamedReadError as error:
            if error.errno is None or error.errno <= 0:
                raise ManifestError("module read failed without positive errno") from error
            if _identity(ops.fstat(descriptor)) != _identity(opened):
                raise ManifestError("unreadable opened module parameter changed")
            if _identity(ops.stat_at(directory_fd, name)) != _identity(initial):
                raise ManifestError("unreadable named module parameter changed")
            return {
                "errno": decimal_integer(error.errno),
                "path_b64": b64url(path),
                "status": "error",
                "type": "regular",
            }
        if _identity(ops.fstat(descriptor)) != _identity(opened):
            raise ManifestError("module parameter changed while reading")
    finally:
        ops.close(descriptor)
    if _identity(ops.stat_at(directory_fd, name)) != _identity(initial):
        raise ManifestError("named module parameter changed while reading")
    return {
        "bytes_b64": b64url(raw),
        "path_b64": b64url(path),
        "sha256": sha256_bytes(raw),
        "status": "ok",
        "type": "regular",
    }


def _module_symlink_record(
    ops: ModuleParameterOps,
    directory_fd: int,
    name: bytes,
    path: bytes,
    initial: os.stat_result,
) -> dict[str, str]:
    target = ops.readlink_at(directory_fd, name)
    if _identity(ops.stat_at(directory_fd, name)) != _identity(initial):
        raise ManifestError("module parameter symlink changed")
    if ops.readlink_at(directory_fd, name) != target:
        raise ManifestError("module parameter symlink target changed")
    return {
        "path_b64": b64url(path),
        "status": "ok",
        "symlink_b64": b64url(target),
        "type": "symlink",
    }


def _collect_one_module_directory(
    path: bytes,
    *,
    ops: ModuleParameterOps,
) -> dict[str, object]:
    try:
        path_descriptor = ops.open_path(path)
    except OSError as error:
        if error.errno is None or error.errno <= 0:
            raise ManifestError("module directory failed without positive errno") from error
        if error.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
            raise ManifestError(
                f"module directory has an invalid path structure: errno {error.errno}"
            ) from error
        return {
            "errno": decimal_integer(error.errno),
            "path_b64": b64url(path),
            "status": "error",
        }
    try:
        path_before = ops.fstat(path_descriptor)
        if not stat.S_ISDIR(path_before.st_mode):
            raise ManifestError("module parameter path is not a directory")
        try:
            descriptor = ops.open_directory_from_path(path_descriptor)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM):
                raise ManifestError(
                    f"cannot open verified module directory: errno {error.errno}"
                ) from error
            if _identity(ops.fstat(path_descriptor)) != _identity(path_before):
                raise ManifestError("unreadable module directory changed")
            try:
                reopened = ops.open_path(path)
            except OSError as reopen_error:
                raise ManifestError(
                    f"cannot reopen unreadable module directory: errno {reopen_error.errno}"
                ) from reopen_error
            try:
                if _identity(ops.fstat(reopened)) != _identity(path_before):
                    raise ManifestError("unreadable module path identity changed")
            finally:
                ops.close(reopened)
            return {
                "errno": decimal_integer(error.errno),
                "path_b64": b64url(path),
                "status": "error",
            }
        try:
            directory_before = ops.fstat(descriptor)
            if _identity(directory_before) != _identity(path_before):
                raise ManifestError("opened module directory identity changed")
            names_before = _directory_names(ops, descriptor)
            records: list[dict[str, str]] = []
            for name in names_before:
                child_path = _join(path, name)
                try:
                    initial = ops.stat_at(descriptor, name)
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EPERM):
                        raise ManifestError(
                            f"listed module parameter became unstatable: errno {error.errno}"
                        ) from error
                    records.append(
                        {
                            "errno": decimal_integer(error.errno),
                            "path_b64": b64url(child_path),
                            "status": "error",
                        }
                    )
                    continue
                if stat.S_ISREG(initial.st_mode):
                    record = _module_regular_record(
                        ops,
                        descriptor,
                        name,
                        child_path,
                        initial,
                    )
                elif stat.S_ISLNK(initial.st_mode):
                    record = _module_symlink_record(
                        ops,
                        descriptor,
                        name,
                        child_path,
                        initial,
                    )
                else:
                    raise ManifestError(
                        "module parameter directory contains a non-regular, non-symlink entry"
                    )
                records.append(record)
            names_after = _directory_names(ops, descriptor)
            if names_before != names_after:
                raise ManifestError("module parameter names changed during snapshot")
            if _identity(ops.fstat(descriptor)) != _identity(directory_before):
                raise ManifestError("module parameter directory changed during snapshot")
        finally:
            ops.close(descriptor)
        if _identity(ops.fstat(path_descriptor)) != _identity(path_before):
            raise ManifestError("module path descriptor changed during snapshot")
        try:
            reopened = ops.open_path(path)
        except OSError as error:
            raise ManifestError(
                f"cannot reopen module path after snapshot: errno {error.errno}"
            ) from error
        try:
            if _identity(ops.fstat(reopened)) != _identity(path_before):
                raise ManifestError("module path identity changed during snapshot")
        finally:
            ops.close(reopened)
    finally:
        ops.close(path_descriptor)
    return {
        "entries": records,
        "path_b64": b64url(path),
        "status": "ok",
    }


def collect_module_parameter_records(
    *,
    sys_module_root: bytes = b"/sys/module",
    ops: ModuleParameterOps | None = None,
) -> list[dict[str, object]]:
    """Collect the two exact module directories without recursive traversal."""
    _canonical_absolute(sys_module_root)
    active_ops = ops or ModuleParameterOps()
    paths = [
        _join(sys_module_root, *suffix) for suffix in MODULE_PARAMETER_SUFFIXES
    ]
    return [
        _collect_one_module_directory(path, ops=active_ops)
        for path in sorted(paths)
    ]


__all__ = [
    "BDF_FILE_NAMES",
    "MODULE_PARAMETER_SUFFIXES",
    "PROC_FILE_SUFFIXES",
    "ModuleParameterOps",
    "collect_bdf_and_proc_records",
    "collect_module_parameter_records",
]
