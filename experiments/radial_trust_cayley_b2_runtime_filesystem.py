#!/usr/bin/env python3
"""Filesystem collectors for the frozen B2 container-runtime invariant.

The functions are CPU-only and do not inspect or initialize CUDA/NVML.  They
parse raw mountinfo, hash stable regular-file mounts, and recursively lstat the
exact `/dev/nvidia*` roots without following symlinks.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import os
import stat
from typing import Callable, Sequence

from experiments.radial_trust_cayley_b2_runtime_manifest import (
    ManifestError,
    b64url,
    decimal_integer,
)


EXCLUDED_MOUNT_FILESYSTEMS = (
    b"cgroup2",
    b"devpts",
    b"devtmpfs",
    b"mqueue",
    b"proc",
    b"sysfs",
    b"tmpfs",
)
_PATH_ESCAPE = {
    b"011": b"\t",
    b"012": b"\n",
    b"040": b" ",
    b"134": b"\\",
}
_MANGLED_FIELD_ESCAPE = {**_PATH_ESCAPE, b"043": b"#"}


@dataclass(frozen=True)
class MountEntry:
    mount_id: int
    parent_id: int
    root: bytes
    mount_point: bytes
    filesystem: bytes


class _NamedPathAccessError(OSError):
    """An errno attributable to opening or reading the named file itself."""


def _canonical_absolute(path: bytes) -> None:
    if not isinstance(path, bytes) or not path.startswith(b"/") or b"\0" in path:
        raise ManifestError("path must be absolute bytes without NUL")
    if path == b"/":
        return
    parts = path.split(b"/")
    if parts[0] != b"" or any(part in (b"", b".", b"..") for part in parts[1:]):
        raise ManifestError("path must have canonical absolute components")


def _mount_unescape(field: bytes, escapes: dict[bytes, bytes]) -> bytes:
    output = bytearray()
    index = 0
    while index < len(field):
        if field[index] != 0x5C:
            output.append(field[index])
            index += 1
            continue
        code = field[index + 1 : index + 4]
        replacement = escapes.get(code)
        if replacement is None:
            raise ManifestError("invalid mountinfo escape")
        output.extend(replacement)
        index += 4
    return bytes(output)


def _parse_nonnegative_decimal(raw: bytes, *, field: str) -> int:
    if not raw or not raw.isdigit() or (len(raw) > 1 and raw.startswith(b"0")):
        raise ManifestError(f"mountinfo {field} is not canonical decimal")
    return int(raw, 10)


def parse_mountinfo(raw: bytes) -> tuple[MountEntry, ...]:
    if not isinstance(raw, bytes):
        raise ManifestError("mountinfo image must be bytes")
    if not raw.endswith(b"\n"):
        raise ManifestError("mountinfo image lacks its terminal LF")
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines or any(line == b"" for line in lines):
        raise ManifestError("mountinfo must contain nonempty LF-delimited records")
    entries: list[MountEntry] = []
    mount_ids: set[int] = set()
    for line in lines:
        if b"\t" in line or b"\0" in line:
            raise ManifestError("mountinfo contains an unescaped control separator")
        fields = line.split(b" ")
        try:
            separator = fields.index(b"-")
        except ValueError as error:
            raise ManifestError("mountinfo line lacks separator") from error
        if separator < 6 or len(fields) != separator + 4 or any(field == b"" for field in fields):
            raise ManifestError("mountinfo line has the wrong field count")
        mount_id = _parse_nonnegative_decimal(fields[0], field="mount ID")
        parent_id = _parse_nonnegative_decimal(fields[1], field="parent ID")
        if mount_id == 0 or parent_id == 0:
            raise ManifestError("mountinfo IDs must be positive")
        if mount_id in mount_ids:
            raise ManifestError("duplicate mountinfo mount ID")
        mount_ids.add(mount_id)
        device_numbers = fields[2].split(b":")
        if len(device_numbers) != 2:
            raise ManifestError("mountinfo major:minor field is malformed")
        _parse_nonnegative_decimal(device_numbers[0], field="major device")
        _parse_nonnegative_decimal(device_numbers[1], field="minor device")
        root = _mount_unescape(fields[3], _PATH_ESCAPE)
        mount_point = _mount_unescape(fields[4], _PATH_ESCAPE)
        _validate_option_list(fields[5], field="mount options")
        for optional in fields[6:separator]:
            if not optional or b"\0" in optional or b"\t" in optional:
                raise ManifestError("malformed mountinfo optional field")
        filesystem = _mount_unescape(fields[separator + 1], _MANGLED_FIELD_ESCAPE)
        _mount_unescape(fields[separator + 2], _MANGLED_FIELD_ESCAPE)
        _validate_option_list(fields[separator + 3], field="super options")
        _canonical_absolute(root)
        _canonical_absolute(mount_point)
        if not filesystem or b"\0" in filesystem:
            raise ManifestError("invalid mountinfo filesystem name")
        entries.append(
            MountEntry(mount_id, parent_id, root, mount_point, filesystem)
        )
    return tuple(entries)


def _validate_option_list(raw: bytes, *, field: str) -> None:
    if (
        not raw
        or raw.startswith(b",")
        or raw.endswith(b",")
        or b",," in raw
        or b"\t" in raw
        or b"\0" in raw
        or b" " in raw
    ):
        raise ManifestError(f"invalid mountinfo {field}")


def _fd_mount_id(descriptor: int) -> int:
    path = f"/proc/self/fdinfo/{descriptor}".encode("ascii")
    info_fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(info_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(info_fd)
    values = []
    for line in b"".join(chunks).split(b"\n"):
        if line.startswith(b"mnt_id:\t"):
            values.append(line.removeprefix(b"mnt_id:\t"))
    if len(values) != 1:
        raise ManifestError("fdinfo does not contain exactly one mount ID")
    mount_id = _parse_nonnegative_decimal(values[0], field="fdinfo mount ID")
    if mount_id == 0:
        raise ManifestError("fdinfo mount ID must be positive")
    return mount_id


def _open_file_digest(
    path: bytes,
    *,
    expected_mount_id: int | None,
    allow_nonregular: bool,
) -> str | None:
    if not hasattr(os, "O_PATH") or not hasattr(os, "O_NOFOLLOW"):
        raise ManifestError("O_PATH and O_NOFOLLOW are required for file hashing")
    try:
        descriptor = _open_path_descriptor(path)
    except OSError as error:
        raise _NamedPathAccessError(error.errno, error.strerror, path) from error
    try:
        before = os.fstat(descriptor)
        mount_id_before: int | None = None
        if expected_mount_id is not None:
            mount_id_before = _fd_mount_id(descriptor)
            if mount_id_before != expected_mount_id:
                raise ManifestError(
                    f"opened mount ID {mount_id_before} != recorded {expected_mount_id}"
                )
        if not stat.S_ISREG(before.st_mode):
            if not allow_nonregular:
                raise ManifestError("hashed path is not a regular file")
            _verify_reopened_path(path, before, expected_mount_id)
            return None
        proc_descriptor_path = f"/proc/self/fd/{descriptor}".encode("ascii")
        try:
            read_descriptor = _open_read_descriptor(proc_descriptor_path)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EPERM):
                raise _NamedPathAccessError(error.errno, error.strerror, path) from error
            raise ManifestError(
                f"cannot open verified inode through procfd: errno {error.errno}"
            ) from error
        try:
            read_before = os.fstat(read_descriptor)
            if _metadata_identity(read_before) != _metadata_identity(before):
                raise ManifestError("read descriptor identity differs from O_PATH inode")
            read_mount_before: int | None = None
            if expected_mount_id is not None:
                read_mount_before = _fd_mount_id(read_descriptor)
                if read_mount_before != expected_mount_id:
                    raise ManifestError("read descriptor has the wrong mount ID")
            digest = hashlib.sha256()
            while True:
                try:
                    chunk = os.read(read_descriptor, 1 << 20)
                except OSError as error:
                    raise _NamedPathAccessError(
                        error.errno,
                        error.strerror,
                        path,
                    ) from error
                if not chunk:
                    break
                digest.update(chunk)
            read_after = os.fstat(read_descriptor)
            if _metadata_identity(read_after) != _metadata_identity(read_before):
                raise ManifestError("regular file changed while hashing")
            if expected_mount_id is not None:
                read_mount_after = _fd_mount_id(read_descriptor)
                if read_mount_after != read_mount_before:
                    raise ManifestError("read descriptor mount ID changed while hashing")
        finally:
            os.close(read_descriptor)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ManifestError("regular file changed while hashing")
        if expected_mount_id is not None:
            mount_id_after = _fd_mount_id(descriptor)
            if mount_id_before != mount_id_after:
                raise ManifestError("file mount ID changed while hashing")
        _verify_reopened_path(path, before, expected_mount_id)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _open_path_descriptor(path: bytes) -> int:
    return os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)


def _open_read_descriptor(proc_descriptor_path: bytes) -> int:
    return os.open(proc_descriptor_path, os.O_RDONLY | os.O_CLOEXEC)


def _verify_reopened_path(
    path: bytes,
    expected_metadata: os.stat_result,
    expected_mount_id: int | None,
) -> None:
    try:
        descriptor = _open_path_descriptor(path)
    except OSError as error:
        raise ManifestError(f"cannot reopen named path: errno {error.errno}") from error
    try:
        if _metadata_identity(os.fstat(descriptor)) != _metadata_identity(expected_metadata):
            raise ManifestError("reopened path identity changed while hashing")
        if expected_mount_id is not None and _fd_mount_id(descriptor) != expected_mount_id:
            raise ManifestError("reopened path has the wrong mount ID")
    finally:
        os.close(descriptor)


def _stable_regular_file_sha256(path: bytes) -> str:
    digest = _open_file_digest(
        path,
        expected_mount_id=None,
        allow_nonregular=False,
    )
    assert digest is not None
    return digest


def _stable_mounted_regular_sha256(path: bytes, mount_id: int) -> str | None:
    return _open_file_digest(
        path,
        expected_mount_id=mount_id,
        allow_nonregular=True,
    )


def hash_or_errno_record(
    path: bytes,
    *,
    hasher: Callable[[bytes], str] = _stable_regular_file_sha256,
) -> dict[str, str]:
    _canonical_absolute(path)
    try:
        digest = hasher(path)
    except _NamedPathAccessError as error:
        if error.errno is None or error.errno <= 0:
            raise ManifestError("hash failure did not provide a positive errno") from error
        return {
            "errno": decimal_integer(error.errno),
            "path_b64": b64url(path),
            "status": "error",
        }
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ManifestError("file hasher did not return lowercase SHA-256")
    return {"path_b64": b64url(path), "sha256": digest, "status": "ok"}


def regular_file_mount_hashes(
    mountinfo_raw: bytes,
    *,
    hasher: Callable[[bytes, int], str | None] = _stable_mounted_regular_sha256,
) -> list[dict[str, str]]:
    entries = parse_mountinfo(mountinfo_raw)
    mountpoints = [entry.mount_point for entry in entries]
    if len(mountpoints) != len(set(mountpoints)):
        raise ManifestError("stacked or duplicate mountpoints are unsupported")
    selected: dict[bytes, dict[str, str]] = {}
    for entry in entries:
        if entry.filesystem in EXCLUDED_MOUNT_FILESYSTEMS:
            continue
        if entry.mount_point == b"/" and entry.filesystem == b"overlay":
            continue
        try:
            digest = hasher(entry.mount_point, entry.mount_id)
        except OSError as error:
            raise ManifestError(
                f"mountpoint became unreadable during snapshot: errno {error.errno}"
            ) from error
        if digest is None:
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ManifestError("mount hasher did not return lowercase SHA-256")
        selected[entry.mount_point] = {
            "path_b64": b64url(entry.mount_point),
            "sha256": digest,
            "status": "ok",
        }
    return [selected[path] for path in sorted(selected)]


def _entry_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISCHR(mode):
        return "character"
    if stat.S_ISBLK(mode):
        return "block"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    raise ManifestError("unknown lstat file type")


def _validate_entry_name(name: bytes) -> None:
    if not isinstance(name, bytes) or name in (b"", b".", b"..") or b"/" in name or b"\0" in name:
        raise ManifestError("directory entry is not one raw basename")


def _child_path(directory: bytes, name: bytes) -> bytes:
    _validate_entry_name(name)
    if directory == b"/":
        return b"/" + name
    return directory.rstrip(b"/") + b"/" + name


def _metadata_record(
    path: bytes,
    metadata: os.stat_result,
    *,
    symlink_target: bytes | None = None,
) -> dict[str, str | None]:
    entry_type = _entry_type(metadata.st_mode)
    device = entry_type in ("character", "block")
    symlink = entry_type == "symlink"
    if symlink != (symlink_target is not None):
        raise ManifestError("symlink target presence disagrees with lstat type")
    return {
        "gid": decimal_integer(metadata.st_gid),
        "major": decimal_integer(os.major(metadata.st_rdev)) if device else None,
        "minor": decimal_integer(os.minor(metadata.st_rdev)) if device else None,
        "mode": decimal_integer(metadata.st_mode),
        "path_b64": b64url(path),
        "symlink_b64": b64url(symlink_target) if symlink_target is not None else None,
        "type": entry_type,
        "uid": decimal_integer(metadata.st_uid),
    }


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


class DescriptorOps:
    """Descriptor-relative filesystem calls used by the device walker."""

    @staticmethod
    def _directory_flags() -> int:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
        if not hasattr(os, "O_NOFOLLOW"):
            raise ManifestError("O_NOFOLLOW is required for device traversal")
        return flags | os.O_NOFOLLOW

    def open_root(self, path: bytes) -> int:
        return os.open(path, self._directory_flags())

    def open_dir_at(self, parent_fd: int, name: bytes) -> int:
        return os.open(name, self._directory_flags(), dir_fd=parent_fd)

    def listdir(self, descriptor: int) -> Sequence[bytes | str]:
        return os.listdir(descriptor)

    def stat_at(self, parent_fd: int, name: bytes) -> os.stat_result:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)

    def fstat(self, descriptor: int) -> os.stat_result:
        return os.fstat(descriptor)

    def readlink_at(self, parent_fd: int, name: bytes) -> bytes:
        target = os.readlink(name, dir_fd=parent_fd)
        if not isinstance(target, bytes):
            raise ManifestError("bytes path must produce bytes symlink target")
        return target

    def close(self, descriptor: int) -> None:
        os.close(descriptor)


def _directory_names(ops: DescriptorOps, descriptor: int) -> tuple[bytes, ...]:
    names = [os.fsencode(name) for name in ops.listdir(descriptor)]
    for name in names:
        _validate_entry_name(name)
    if len(names) != len(set(names)):
        raise ManifestError("directory listing contains duplicate basenames")
    return tuple(sorted(names))


def nvidia_device_records(
    *,
    dev_root: bytes = b"/dev",
    ops: DescriptorOps | None = None,
) -> list[dict[str, str | None]]:
    _canonical_absolute(dev_root)
    active_ops = ops or DescriptorOps()
    try:
        root_fd = active_ops.open_root(dev_root)
    except OSError as error:
        raise ManifestError(f"cannot open device root safely: errno {error.errno}") from error
    try:
        try:
            direct_before = _directory_names(active_ops, root_fd)
            root_names = tuple(name for name in direct_before if name.startswith(b"nvidia"))
            records: dict[bytes, dict[str, str | None]] = {}
            for name in root_names:
                _walk_device_entry(active_ops, root_fd, name, dev_root, records)
            direct_after = _directory_names(active_ops, root_fd)
            after_names = tuple(name for name in direct_after if name.startswith(b"nvidia"))
            if root_names != after_names:
                raise ManifestError("/dev/nvidia* roots changed during snapshot")
        except OSError as error:
            raise ManifestError(
                f"device tree changed or became unreadable: errno {error.errno}"
            ) from error
    finally:
        active_ops.close(root_fd)
    return [records[path] for path in sorted(records)]


def _walk_device_entry(
    ops: DescriptorOps,
    parent_fd: int,
    name: bytes,
    parent_path: bytes,
    records: dict[bytes, dict[str, str | None]],
) -> None:
    path = _child_path(parent_path, name)
    initial = ops.stat_at(parent_fd, name)
    target = ops.readlink_at(parent_fd, name) if stat.S_ISLNK(initial.st_mode) else None
    records[path] = _metadata_record(path, initial, symlink_target=target)
    if stat.S_ISDIR(initial.st_mode):
        child_fd = ops.open_dir_at(parent_fd, name)
        try:
            if _metadata_identity(ops.fstat(child_fd)) != _metadata_identity(initial):
                raise ManifestError("opened device directory identity changed")
            children_before = _directory_names(ops, child_fd)
            for child in children_before:
                _walk_device_entry(ops, child_fd, child, path, records)
            children_after = _directory_names(ops, child_fd)
            if children_before != children_after:
                raise ManifestError("device directory entries changed during snapshot")
            if _metadata_identity(ops.fstat(child_fd)) != _metadata_identity(initial):
                raise ManifestError("opened device directory metadata changed")
        finally:
            ops.close(child_fd)
    final = ops.stat_at(parent_fd, name)
    if _metadata_identity(final) != _metadata_identity(initial):
        raise ManifestError("device entry identity changed during snapshot")
    if target is not None and ops.readlink_at(parent_fd, name) != target:
        raise ManifestError("device symlink target changed during snapshot")


__all__ = [
    "EXCLUDED_MOUNT_FILESYSTEMS",
    "DescriptorOps",
    "MountEntry",
    "hash_or_errno_record",
    "nvidia_device_records",
    "parse_mountinfo",
    "regular_file_mount_hashes",
]
