from __future__ import annotations

import hashlib
from argparse import Namespace

import pytest

from experiments import prepare_raw_prose_equality_plane_t10_data as data


def test_frozen_sizes_match_sequence_contract() -> None:
    assert data.EXPECTED["train"]["bytes"] == (
        data.TRAIN_SEQUENCES * (data.SEQUENCE_LENGTH + 1) * 2
    )
    assert data.EXPECTED["validation"]["bytes"] == (
        data.VALIDATION_SEQUENCES * (data.SEQUENCE_LENGTH + 1) * 2
    )


def test_verify_file_checks_size_and_hash(tmp_path) -> None:
    payload = b"fixed T10 corpus"
    path = tmp_path / "stream.bin"
    path.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()

    result = data.verify_file(path, len(payload), expected_hash)
    assert result["bytes"] == len(payload)
    assert result["sha256"] == expected_hash

    with pytest.raises(RuntimeError, match="bytes"):
        data.verify_file(path, len(payload) + 1, expected_hash)
    with pytest.raises(RuntimeError, match="sha256"):
        data.verify_file(path, len(payload), "0" * 64)


def test_split_is_stable_and_binary() -> None:
    observed = {data.split_for(f"document-{index}") for index in range(100)}
    assert observed == {"train", "validation"}
    assert data.split_for("document-17") == data.split_for("document-17")


def test_run_refuses_invalid_existing_stream_without_explicit_replace(tmp_path) -> None:
    output_dir = tmp_path / "data"
    output_dir.mkdir()
    (output_dir / "train.uint16.bin").write_bytes(b"invalid")
    arguments = Namespace(
        output_dir=output_dir,
        manifest=tmp_path / "manifest.json",
        tokenizer_batch=64,
        verify_only=False,
        replace_invalid=False,
    )
    with pytest.raises(RuntimeError, match="refusing to replace"):
        data.run(arguments)
