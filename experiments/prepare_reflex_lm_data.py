#!/usr/bin/env python3
"""Prepare a frozen document-disjoint token stream for the Reflex LM screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATASET = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"
DATASET_REVISION = "3ba9d605774198c5868892d7a8deda78031a781f"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(document_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(document_id.encode("utf-8")).digest()[:8], "big")
    return "validation" if value % 10 == 0 else "train"


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.vocab_limit > np.iinfo(np.uint16).max:
        raise ValueError("uint16 token files require vocab_limit <= 65535")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train.uint16.bin",
        "validation": args.output_dir / "validation.uint16.bin",
    }
    temporary_paths = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    targets = {
        "train": args.train_sequences * (args.sequence_length + 1),
        "validation": args.validation_sequences * (args.sequence_length + 1),
    }
    arrays = {
        name: np.memmap(temporary_paths[name], dtype=np.uint16, mode="w+", shape=(targets[name],))
        for name in paths
    }
    positions = {"train": 0, "validation": 0}
    documents = {"train": 0, "validation": 0}

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.model_max_length = 10**12
    if len(tokenizer) > args.vocab_limit:
        raise ValueError(f"tokenizer has {len(tokenizer)} tokens, above {args.vocab_limit}")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    stream = load_dataset(
        DATASET,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    )
    pending: list[dict[str, object]] = []
    scanned_documents = 0
    started = time.perf_counter()

    def consume(batch: list[dict[str, object]]) -> None:
        texts = [str(row["text"]) for row in batch]
        encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        for row, token_ids in zip(batch, encoded, strict=True):
            name = split_for(str(row["id"]))
            if positions[name] >= targets[name]:
                continue
            values = np.asarray([*token_ids, tokenizer.eos_token_id], dtype=np.uint16)
            take = min(len(values), targets[name] - positions[name])
            arrays[name][positions[name] : positions[name] + take] = values[:take]
            positions[name] += take
            documents[name] += 1

    for row in stream:
        pending.append(row)
        scanned_documents += 1
        if len(pending) >= args.tokenizer_batch:
            consume(pending)
            pending.clear()
        if all(positions[name] >= targets[name] for name in targets):
            break
    if pending and not all(positions[name] >= targets[name] for name in targets):
        consume(pending)
    if positions != targets:
        raise RuntimeError(f"stream ended before targets were filled: {positions} != {targets}")
    for array in arrays.values():
        array.flush()
    del arrays
    for name, path in paths.items():
        temporary_paths[name].replace(path)

    manifest = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "split_rule": "validation iff uint64_be(sha256(id)[:8]) mod 10 == 0",
        "tokenizer_length": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint16",
        "sequence_length": args.sequence_length,
        "train_sequences": args.train_sequences,
        "validation_sequences": args.validation_sequences,
        "target_tokens": targets,
        "documents_contributing": documents,
        "documents_scanned": scanned_documents,
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(args.manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/reflex-lm-screen"))
    parser.add_argument("--manifest", type=Path, default=Path("results/reflex-swiglu-lm-data-manifest.json"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--train-sequences", type=int, default=40_960)
    parser.add_argument("--validation-sequences", type=int, default=4_096)
    parser.add_argument("--tokenizer-batch", type=int, default=64)
    parser.add_argument("--vocab-limit", type=int, default=65_535)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
