#!/usr/bin/env python3
"""Rebuild and verify the exact natural-token streams required by T10."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/block-algebra-scratch-v4576"
MANIFEST = ROOT / "results/raw-prose-equality-plane-t10-data-manifest.json"

MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATASET = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"
DATASET_REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"

SEQUENCE_LENGTH = 512
TRAIN_SEQUENCES = 97_656
VALIDATION_SEQUENCES = 4_096
EXPECTED = {
    "train": {
        "bytes": 100_195_056,
        "sha256": "1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a",
    },
    "validation": {
        "bytes": 4_202_496,
        "sha256": "889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(document_id: str) -> str:
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 10 == 0 else "train"


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes}"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{path} sha256 is {actual_sha256}; expected {expected_sha256}"
        )
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def verify_data(output_dir: Path) -> dict[str, dict[str, object]]:
    return {
        name: verify_file(
            output_dir / f"{name}.uint16.bin",
            int(expectation["bytes"]),
            str(expectation["sha256"]),
        )
        for name, expectation in EXPECTED.items()
    }


def generate(output_dir: Path, tokenizer_batch: int) -> dict[str, object]:
    try:
        import datasets
        import tokenizers
        import transformers
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "generation requires `datasets` and `transformers`; verification does not"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / f"{name}.uint16.bin" for name in EXPECTED
    }
    partial = {
        name: path.with_suffix(path.suffix + ".partial") for name, path in paths.items()
    }
    targets = {
        "train": TRAIN_SEQUENCES * (SEQUENCE_LENGTH + 1),
        "validation": VALIDATION_SEQUENCES * (SEQUENCE_LENGTH + 1),
    }
    arrays = {
        name: np.memmap(path, dtype="<u2", mode="w+", shape=(targets[name],))
        for name, path in partial.items()
    }
    positions = {name: 0 for name in EXPECTED}
    documents = {name: 0 for name in EXPECTED}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.model_max_length = 10**12
    if len(tokenizer) > np.iinfo(np.uint16).max:
        raise RuntimeError(f"tokenizer length {len(tokenizer)} does not fit uint16")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no EOS token")
    stream = load_dataset(
        DATASET,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    )

    pending: list[dict[str, object]] = []
    scanned_documents = 0

    def consume(batch: list[dict[str, object]]) -> None:
        encoded = tokenizer(
            [str(row["text"]) for row in batch],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        for row, token_ids in zip(batch, encoded, strict=True):
            split = split_for(str(row["id"]))
            if positions[split] >= targets[split]:
                continue
            values = np.asarray([*token_ids, tokenizer.eos_token_id], dtype="<u2")
            take = min(len(values), targets[split] - positions[split])
            arrays[split][positions[split] : positions[split] + take] = values[:take]
            positions[split] += take
            documents[split] += 1

    for row in stream:
        pending.append(row)
        scanned_documents += 1
        if len(pending) >= tokenizer_batch:
            consume(pending)
            pending.clear()
        if positions == targets:
            break
    if pending and positions != targets:
        consume(pending)
    if positions != targets:
        raise RuntimeError(f"stream ended before targets were filled: {positions}")

    for array in arrays.values():
        array.flush()
    del arrays

    for name in EXPECTED:
        verify_file(
            partial[name],
            int(EXPECTED[name]["bytes"]),
            str(EXPECTED[name]["sha256"]),
        )
    for name, path in paths.items():
        partial[name].replace(path)
    return {
        "files": verify_data(output_dir),
        "documents_contributing": documents,
        "documents_scanned": scanned_documents,
        "tokenizer_length": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "datasets_version": datasets.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    generated = False
    generation: dict[str, object] | None = None
    try:
        files = verify_data(args.output_dir)
    except (FileNotFoundError, RuntimeError):
        if args.verify_only:
            raise
        existing = [
            args.output_dir / f"{name}.uint16.bin"
            for name in EXPECTED
            if (args.output_dir / f"{name}.uint16.bin").exists()
        ]
        if existing and not args.replace_invalid:
            names = ", ".join(str(path) for path in existing)
            raise RuntimeError(
                f"refusing to replace invalid existing data: {names}; "
                "pass --replace-invalid after inspecting it"
            )
        generation = generate(args.output_dir, args.tokenizer_batch)
        generated = True
        files = verify_data(args.output_dir)

    payload = {
        "schema": "raw-prose-equality-plane-t10-data-v1",
        "status": "pass",
        "generated": generated,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "split_rule": "validation iff uint64_be(sha256(id)[:8]) mod 10 == 0",
        "dtype": "uint16 little-endian",
        "sequence_length": SEQUENCE_LENGTH,
        "train_sequences": TRAIN_SEQUENCES,
        "validation_sequences": VALIDATION_SEQUENCES,
        "files": files,
        "generation": generation,
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": file_sha256(Path(__file__)),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(args.manifest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--tokenizer-batch", type=int, default=64)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--replace-invalid", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
