"""Capture a deterministic model-artifact and H100 software manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


TOKENIZER_FILES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
MODEL_METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(paths)
    ]


def inventory_sha256(records: list[dict[str, Any]]) -> str:
    """Hash canonical path, byte count, and content-hash records."""
    digest = hashlib.sha256()
    for record in records:
        line = f"{record['path']}\0{record['bytes']}\0{record['sha256']}\n"
        digest.update(line.encode())
    return digest.hexdigest()


def distribution_record_sha256(name: str) -> str:
    distribution = importlib.metadata.distribution(name)
    record = next(
        file for file in distribution.files or []
        if str(file).endswith(".dist-info/RECORD")
    )
    return file_sha256(Path(distribution.locate_file(record)))


def gpu_identity() -> dict[str, Any]:
    fields = [
        "name",
        "pci.bus_id",
        "memory.total",
        "power.limit",
        "driver_version",
        "temperature.gpu",
        "clocks.current.sm",
        "clocks.current.memory",
        "ecc.mode.current",
        "mig.mode.current",
    ]
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    return dict(zip(fields, values, strict=True))


def capture(model_dir: Path, container_image: str, container_digest: str) -> dict[str, Any]:
    import torch

    core_paths = list(model_dir.glob("*.safetensors"))
    tokenizer_paths = [
        model_dir / name for name in TOKENIZER_FILES if (model_dir / name).is_file()
    ]
    metadata_paths = [
        model_dir / name
        for name in MODEL_METADATA_FILES
        if (model_dir / name).is_file()
    ]
    if not core_paths:
        raise ValueError(f"no safetensors checkpoint found in {model_dir}")
    if not tokenizer_paths:
        raise ValueError(f"no tokenizer files found in {model_dir}")

    core = inventory(core_paths, model_dir)
    tokenizer = inventory(tokenizer_paths, model_dir)
    metadata = inventory(metadata_paths, model_dir)
    core_bytes = sum(record["bytes"] for record in core)
    metadata_bytes = sum(record["bytes"] for record in tokenizer + metadata)
    vllm_version = importlib.metadata.version("vllm")

    return {
        "schema_version": 1,
        "hash_scheme": "sha256(canonical path\\0bytes\\0file_sha256 records)",
        "artifact": {
            "checkpoint_sha256": inventory_sha256(core),
            "tokenizer_sha256": inventory_sha256(tokenizer),
            "learned_bytes": {
                "core": core_bytes,
                "auxiliary": 0,
                "metadata": metadata_bytes,
                "total": core_bytes + metadata_bytes,
            },
            "checkpoint_files": core,
            "tokenizer_files": tokenizer,
            "metadata_files": metadata,
        },
        "hardware": {
            "gpu": gpu_identity(),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cpu": platform.processor(),
            "kernel": platform.release(),
        },
        "software": {
            "container_image": container_image,
            "container_digest": container_digest,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm_version,
            "vllm_record_sha256": distribution_record_sha256("vllm"),
            "transformers": importlib.metadata.version("transformers"),
            "flashinfer_python": importlib.metadata.version("flashinfer-python"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = capture(args.model_dir.resolve(), args.container_image, args.container_digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
