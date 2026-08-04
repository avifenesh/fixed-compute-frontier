#!/usr/bin/env python3
"""Prepare pinned peS2o train/validation token files for the 360M TVE gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MODEL = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
DATASET = "allenai/peS2o"
DATASET_CONFIG = "v2"
DATASET_REVISION = "636a503e44a3ca1b58e01fb61eab0825cd574de0"


def shard_urls(split: str) -> list[str]:
    if split == "train":
        names = [f"train-{index:05d}-of-00020.json.gz" for index in range(20)]
    elif split == "validation":
        names = [f"validation-{index:05d}-of-00002.json.gz" for index in range(2)]
    else:
        raise ValueError(split)
    root = (
        f"https://huggingface.co/datasets/{DATASET}/resolve/"
        f"{DATASET_REVISION}/data/{DATASET_CONFIG}"
    )
    return [f"{root}/{name}" for name in names]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.vocab_limit > np.iinfo(np.uint16).max:
        raise ValueError("uint16 token files require vocab_limit <= 65535")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train.uint16.bin",
        "validation": args.output_dir / "validation.uint16.bin",
    }
    existing = [str(path) for path in (*paths.values(), args.manifest) if path.exists()]
    if existing:
        raise FileExistsError(existing)
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()
    }
    targets = {
        "train": args.train_sequences * (args.sequence_length + 1),
        "validation": args.validation_sequences * (args.sequence_length + 1),
    }
    arrays = {
        name: np.memmap(
            temporary_paths[name], dtype=np.uint16, mode="w+", shape=(targets[name],)
        )
        for name in paths
    }
    positions = {name: 0 for name in paths}
    documents = {name: 0 for name in paths}
    scanned = {name: 0 for name in paths}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.model_max_length = 10**12
    if len(tokenizer) > args.vocab_limit:
        raise ValueError(f"tokenizer has {len(tokenizer)} tokens, above limit")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    started = time.perf_counter()

    for split in ("train", "validation"):
        stream = load_dataset(
            "json",
            data_files={split: shard_urls(split)},
            split=split,
            streaming=True,
        )
        pending: list[str] = []

        def consume(texts: list[str]) -> None:
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            for token_ids in encoded:
                if positions[split] >= targets[split]:
                    break
                values = np.asarray([*token_ids, tokenizer.eos_token_id], dtype=np.uint16)
                take = min(len(values), targets[split] - positions[split])
                arrays[split][positions[split]:positions[split] + take] = values[:take]
                positions[split] += take
                documents[split] += 1

        for row in stream:
            pending.append(str(row["text"]))
            scanned[split] += 1
            if len(pending) >= args.tokenizer_batch:
                consume(pending)
                pending.clear()
            if positions[split] >= targets[split]:
                break
        if pending and positions[split] < targets[split]:
            consume(pending)
        if positions[split] != targets[split]:
            raise RuntimeError(
                f"{split} stream ended early: {positions[split]} != {targets[split]}"
            )

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
        "split_rule": "official peS2o v2 train and validation splits",
        "tokenizer_length": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint16",
        "sequence_length": args.sequence_length,
        "train_sequences": args.train_sequences,
        "validation_sequences": args.validation_sequences,
        "target_tokens": targets,
        "documents_contributing": documents,
        "documents_scanned": scanned,
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
    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(args.manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/tve-pes2o-scale"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/tve-pes2o-scale-data-manifest.json"),
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--train-sequences", type=int, default=195_200)
    parser.add_argument("--validation-sequences", type=int, default=4_096)
    parser.add_argument("--tokenizer-batch", type=int, default=64)
    parser.add_argument("--vocab-limit", type=int, default=65_535)
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
