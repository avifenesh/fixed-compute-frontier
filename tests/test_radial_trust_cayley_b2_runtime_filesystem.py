import base64
import errno
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

import experiments.radial_trust_cayley_b2_runtime_filesystem as filesystem
from experiments.radial_trust_cayley_b2_runtime_manifest import ManifestError, b64url


def test_frozen_excluded_mount_filesystems_are_literal() -> None:
    assert filesystem.EXCLUDED_MOUNT_FILESYSTEMS == (
        b"cgroup2",
        b"devpts",
        b"devtmpfs",
        b"mqueue",
        b"proc",
        b"sysfs",
        b"tmpfs",
    )


def test_mountinfo_parser_decodes_only_kernel_escape_set() -> None:
    raw = (
        b"31 22 8:1 /root\\134name /mnt/a\\040b\\011c\\012d\\134e rw,nosuid shared:7 future:value - ext4 /dev/sda1 rw\n"
        b"32 22 0:5 / /proc rw - proc proc rw\n"
        b"33 22 0:6 / /mnt\rname rw - ext4 source rw\n"
    )
    entries = filesystem.parse_mountinfo(raw)
    assert entries[0] == filesystem.MountEntry(
        31,
        22,
        b"/root\\name",
        b"/mnt/a b\tc\nd\\e",
        b"ext4",
    )
    assert entries[1].filesystem == b"proc"
    assert entries[2].mount_point == b"/mnt\rname"


def test_mountinfo_parser_decodes_kernel_mangled_hash_in_type_and_source() -> None:
    entry = filesystem.parse_mountinfo(
        b"41 22 0:9 / /mnt rw future:value - fuse\\043type source\\043name rw\n"
    )[0]
    assert entry.filesystem == b"fuse#type"


def test_mountinfo_parser_preserves_raw_carriage_return_inside_options() -> None:
    entry = filesystem.parse_mountinfo(
        b"42 22 0:9 / /mnt rw - ext4 source rw,label=a\rb\n"
    )[0]
    assert entry.mount_point == b"/mnt"


@pytest.mark.parametrize(
    "raw",
    (
        b"31 22 8:1 / /mnt rw ext4 /dev/sda1 rw\n",
        b"31 22 8:1 / /mnt rw - ext4\n",
        b"31 22 8:1 / /mnt\\999 rw - ext4 /dev/sda1 rw\n",
        b"01 22 8:1 / /mnt rw - ext4 /dev/sda1 rw\n",
        b"31 22 8:1 / /mnt/./x rw - ext4 /dev/sda1 rw\n",
        b"31 22 8:1 / /mnt rw - ext4 /dev/sda1 rw\n31 22 8:2 / /x rw - xfs x rw\n",
        b"31 22 8:x / /mnt rw - ext4 source rw\n",
        b"31 22 8:1:2 / /mnt rw - ext4 source rw\n",
        b"31 22 8:1 / /mnt rw - ext4 source rw extra\n",
        b"31 22 8:1 / /mnt\tname rw - ext4 source rw\n",
        b"0 22 8:1 / /mnt rw - ext4 source rw\n",
        b"31 0 8:1 / /mnt rw - ext4 source rw\n",
        b"31 22 8:1 / /mnt rw - ext4 source rw",
        b"",
    ),
)
def test_mountinfo_parser_rejects_ambiguous_records(raw: bytes) -> None:
    with pytest.raises(ManifestError):
        filesystem.parse_mountinfo(raw)


def _mount_line(mount_id: int, mountpoint: bytes, filesystem: bytes) -> bytes:
    encoded = mountpoint.replace(b"\\", b"\\134").replace(b" ", b"\\040")
    return (
        str(mount_id).encode("ascii")
        + b" 1 8:1 / "
        + encoded
        + b" rw - "
        + filesystem
        + b" source rw\n"
    )


def test_regular_file_mount_hashes_select_only_external_regular_mounts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "z mounted"
    second = tmp_path / "a-mounted"
    directory = tmp_path / "directory"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    directory.mkdir()
    raw = b"".join(
        (
            _mount_line(1, b"/", b"overlay"),
            _mount_line(2, os.fsencode(first), b"xfs"),
            _mount_line(3, os.fsencode(second), b"ext4"),
            _mount_line(4, os.fsencode(directory), b"ext4"),
        )
    )
    calls: list[tuple[bytes, int]] = []

    def fake_hasher(path: bytes, mount_id: int) -> str | None:
        calls.append((path, mount_id))
        if path == os.fsencode(directory):
            return None
        return hashlib.sha256(Path(os.fsdecode(path)).read_bytes()).hexdigest()

    records = filesystem.regular_file_mount_hashes(raw, hasher=fake_hasher)
    decoded_paths = [
        base64.urlsafe_b64decode(record["path_b64"] + "====") for record in records
    ]
    assert decoded_paths == sorted((os.fsencode(first), os.fsencode(second)))
    by_path = dict(zip(decoded_paths, records, strict=True))
    assert by_path[os.fsencode(first)]["sha256"] == hashlib.sha256(b"first").hexdigest()
    assert by_path[os.fsencode(second)]["sha256"] == hashlib.sha256(b"second").hexdigest()
    assert all(record["status"] == "ok" for record in records)
    assert calls == [
        (os.fsencode(first), 2),
        (os.fsencode(second), 3),
        (os.fsencode(directory), 4),
    ]


def test_regular_file_mount_hash_verifies_open_fd_mount_id(tmp_path: Path) -> None:
    path = tmp_path / "mounted-file"
    path.write_bytes(b"mounted")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        actual_mount_id = filesystem._fd_mount_id(descriptor)
    finally:
        os.close(descriptor)
    raw = _mount_line(actual_mount_id, os.fsencode(path), b"ext4")
    records = filesystem.regular_file_mount_hashes(raw)
    assert records[0]["sha256"] == hashlib.sha256(b"mounted").hexdigest()

    wrong_id = actual_mount_id + 1
    with pytest.raises(ManifestError, match="opened mount ID"):
        filesystem.regular_file_mount_hashes(
            _mount_line(wrong_id, os.fsencode(path), b"ext4")
        )


def test_stacked_mountpoints_fail_closed_even_when_top_entry_is_excluded(
    tmp_path: Path,
) -> None:
    path = os.fsencode(tmp_path / "same")
    raw = _mount_line(10, path, b"xfs") + _mount_line(11, path, b"tmpfs")
    with pytest.raises(ManifestError, match="stacked or duplicate"):
        filesystem.regular_file_mount_hashes(raw, hasher=lambda _path, _id: None)


def test_hash_or_errno_requires_canonical_path_and_exact_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"artifact")
    record = filesystem.hash_or_errno_record(os.fsencode(path))
    assert record == {
        "path_b64": b64url(os.fsencode(path)),
        "sha256": hashlib.sha256(b"artifact").hexdigest(),
        "status": "ok",
    }
    missing = filesystem.hash_or_errno_record(os.fsencode(tmp_path / "missing"))
    assert missing["status"] == "error"
    assert missing["errno"] == str(errno.ENOENT)
    for invalid in (b"relative", b"/a/./b", b"/a//b", b"/a/../b"):
        with pytest.raises(ManifestError):
            filesystem.hash_or_errno_record(invalid)
    with pytest.raises(ManifestError, match="lowercase SHA-256"):
        filesystem.hash_or_errno_record(b"/x", hasher=lambda _: "A" * 64)


def test_named_file_hash_does_not_depend_on_fdinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "named"
    path.write_bytes(b"named")

    def forbidden(_: int) -> int:
        raise OSError(errno.ENOENT, "fdinfo unavailable")

    monkeypatch.setattr(filesystem, "_fd_mount_id", forbidden)
    record = filesystem.hash_or_errno_record(os.fsencode(path))
    assert record["status"] == "ok"
    assert record["sha256"] == hashlib.sha256(b"named").hexdigest()


def test_procfd_infrastructure_failure_is_not_attributed_to_named_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "named"
    path.write_bytes(b"named")

    def unavailable(_: bytes) -> int:
        raise OSError(errno.ENOENT, "procfd unavailable")

    monkeypatch.setattr(filesystem, "_open_read_descriptor", unavailable)
    with pytest.raises(ManifestError, match="through procfd"):
        filesystem.hash_or_errno_record(os.fsencode(path))


def test_nonregular_fifo_mount_is_classified_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    descriptor = os.open(fifo, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        mount_id = filesystem._fd_mount_id(descriptor)
    finally:
        os.close(descriptor)
    code = (
        "import os,sys; "
        "from experiments.radial_trust_cayley_b2_runtime_filesystem import "
        "_stable_mounted_regular_sha256; "
        "print(_stable_mounted_regular_sha256(os.fsencode(sys.argv[1]), int(sys.argv[2])))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, os.fspath(fifo), str(mount_id)],
        check=False,
        capture_output=True,
        timeout=2.0,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stdout == b"None\n"


def test_mount_id_transition_is_causally_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"value")
    real_open = filesystem._open_path_descriptor
    path_descriptors: list[int] = []

    def tracked_open(raw_path: bytes) -> int:
        descriptor = real_open(raw_path)
        path_descriptors.append(descriptor)
        return descriptor

    def mount_id(descriptor: int) -> int:
        if len(path_descriptors) == 2 and descriptor == path_descriptors[-1]:
            return 8
        return 7

    monkeypatch.setattr(filesystem, "_open_path_descriptor", tracked_open)
    monkeypatch.setattr(filesystem, "_fd_mount_id", mount_id)
    with pytest.raises(ManifestError, match="reopened path has the wrong mount ID"):
        filesystem._stable_mounted_regular_sha256(os.fsencode(path), 7)


def test_nvidia_device_records_are_recursive_sorted_and_do_not_follow_links(
    tmp_path: Path,
) -> None:
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "unrelated").write_bytes(b"ignore")
    (dev / "nvidia0").write_bytes(b"device-placeholder")
    caps = dev / "nvidia-caps"
    caps.mkdir()
    (caps / "cap2").write_bytes(b"two")
    (caps / "cap1").write_bytes(b"one")
    os.symlink("nvidia0", dev / "nvidia-link")

    records = filesystem.nvidia_device_records(dev_root=os.fsencode(dev))
    decoded = [
        base64.urlsafe_b64decode(record["path_b64"] + "====") for record in records
    ]
    assert decoded == sorted(
        (
            os.fsencode(dev / "nvidia-link"),
            os.fsencode(dev / "nvidia0"),
            os.fsencode(caps),
            os.fsencode(caps / "cap1"),
            os.fsencode(caps / "cap2"),
        )
    )
    by_path = dict(zip(decoded, records, strict=True))
    assert by_path[os.fsencode(dev / "nvidia-link")]["type"] == "symlink"
    assert by_path[os.fsencode(dev / "nvidia-link")]["symlink_b64"] == b64url(
        b"nvidia0"
    )
    assert by_path[os.fsencode(caps)]["type"] == "directory"
    assert by_path[os.fsencode(caps / "cap1")]["type"] == "regular"
    assert all(record["major"] is None for record in records)
    assert all(record["minor"] is None for record in records)


def test_lstat_device_record_uses_decimal_major_minor() -> None:
    metadata = SimpleNamespace(
        st_mode=stat.S_IFCHR | 0o660,
        st_uid=123,
        st_gid=456,
        st_rdev=os.makedev(195, 7),
    )
    record = filesystem._metadata_record(b"/dev/nvidia7", metadata)
    assert record["type"] == "character"
    assert record["major"] == "195"
    assert record["minor"] == "7"
    assert record["mode"] == str(stat.S_IFCHR | 0o660)
    assert record["uid"] == "123"
    assert record["gid"] == "456"


def test_device_walker_rejects_fake_non_basename_entries(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="one raw basename"):
        filesystem._validate_entry_name(b"nvidia/escape")


def test_device_walker_fails_closed_when_directory_is_swapped_to_symlink(
    tmp_path: Path,
) -> None:
    dev = tmp_path / "dev"
    dev.mkdir()
    caps = dev / "nvidia-caps"
    caps.mkdir()
    (caps / "inside").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escaped").write_bytes(b"must-not-be-recorded")
    displaced = tmp_path / "displaced-caps"

    class SwappingOps(filesystem.DescriptorOps):
        swapped = False

        def open_dir_at(self, parent_fd: int, name: bytes) -> int:
            if name == b"nvidia-caps" and not self.swapped:
                self.swapped = True
                os.rename(caps, displaced)
                os.symlink(outside, caps)
            return super().open_dir_at(parent_fd, name)

    with pytest.raises(ManifestError, match="device tree changed or became unreadable"):
        filesystem.nvidia_device_records(
            dev_root=os.fsencode(dev),
            ops=SwappingOps(),
        )


def test_device_walker_detects_directory_identity_mismatch(tmp_path: Path) -> None:
    dev = tmp_path / "dev"
    dev.mkdir()
    caps = dev / "nvidia-caps"
    caps.mkdir()

    class WrongIdentityOps(filesystem.DescriptorOps):
        changed = False

        def fstat(self, descriptor: int):
            metadata = super().fstat(descriptor)
            if not self.changed and stat.S_ISDIR(metadata.st_mode) and descriptor != -1:
                self.changed = True
                values = {
                    name: getattr(metadata, name)
                    for name in (
                        "st_dev",
                        "st_ino",
                        "st_mode",
                        "st_nlink",
                        "st_uid",
                        "st_gid",
                        "st_rdev",
                        "st_size",
                        "st_mtime_ns",
                        "st_ctime_ns",
                    )
                }
                values["st_ino"] += 1
                return SimpleNamespace(**values)
            return metadata

    with pytest.raises(ManifestError, match="opened device directory identity changed"):
        filesystem.nvidia_device_records(
            dev_root=os.fsencode(dev),
            ops=WrongIdentityOps(),
        )


def test_device_walker_detects_directory_listing_mutation(tmp_path: Path) -> None:
    dev = tmp_path / "dev"
    dev.mkdir()
    caps = dev / "nvidia-caps"
    caps.mkdir()

    class ListingMutationOps(filesystem.DescriptorOps):
        calls: dict[int, int] = {}

        def listdir(self, descriptor: int):
            result = list(super().listdir(descriptor))
            self.calls[descriptor] = self.calls.get(descriptor, 0) + 1
            if self.calls[descriptor] == 2 and not result:
                return [b"appeared"]
            return result

    with pytest.raises(ManifestError, match="device directory entries changed"):
        filesystem.nvidia_device_records(
            dev_root=os.fsencode(dev),
            ops=ListingMutationOps(),
        )


def test_device_walker_detects_symlink_target_mutation(tmp_path: Path) -> None:
    dev = tmp_path / "dev"
    dev.mkdir()
    link = dev / "nvidia-link"
    os.symlink("first", link)

    class TargetMutationOps(filesystem.DescriptorOps):
        calls = 0

        def readlink_at(self, parent_fd: int, name: bytes) -> bytes:
            self.calls += 1
            if self.calls == 2:
                return b"second"
            return super().readlink_at(parent_fd, name)

    with pytest.raises(ManifestError, match="symlink target changed"):
        filesystem.nvidia_device_records(
            dev_root=os.fsencode(dev),
            ops=TargetMutationOps(),
        )
