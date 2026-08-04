#!/usr/bin/env python3
"""T32 Stage 0: exact whole-record codec and BF16 scale-boundary audit."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # The codec itself has no Torch dependency.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results/scale-referenced-whole-record-t32-stage0-preregistration.md"
PROVENANCE_ERRATUM = ROOT / "results/scale-referenced-whole-record-t32-stage0-provenance-erratum.md"
ATTEMPT1 = ROOT / "results/scale-referenced-whole-record-t32-stage0-attempt1.json"
CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
EXPECTED_CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

TOKEN_SLOTS = 128
REGULAR_TOKENS = 127
CELLS_PER_TOKEN = 3
DIGIT_CELLS = REGULAR_TOKENS * CELLS_PER_TOKEN
SCALE_REFERENCE_INDEX = 381
RESERVED_INDICES = (382, 383)
RECORD_CELLS = 384
ACTIVE_CELLS = 382
VOCAB_CAPACITY = 65_536
MODEL_VOCABULARY = 49_152
PAD_TOKEN = 0
RANDOM_SEED = 32_001
SCALES = tuple(2.0**power for power in (-12, -8, -4, -2, 0, 2, 4, 8, 12))
PERTURBATION_FRACTION = 0.005
TOKENIZER_ARTIFACTS = (
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _two_bit_chunks(value: int, count: int) -> tuple[int, ...]:
    if not 0 <= int(value) < (1 << (2 * count)):
        raise ValueError(f"value {value} does not fit in {2 * count} bits")
    return tuple((int(value) >> (2 * index)) & 0b11 for index in range(count))


def _chunks_to_integer(chunks: Iterable[int]) -> int:
    value = 0
    for index, chunk in enumerate(chunks):
        chunk = int(chunk)
        if not 0 <= chunk < 4:
            raise ValueError(f"not a two-bit chunk: {chunk}")
        value |= chunk << (2 * index)
    return value


def validate_tokens(tokens: Sequence[int], length: int) -> tuple[int, ...]:
    if len(tokens) != TOKEN_SLOTS:
        raise ValueError(f"expected {TOKEN_SLOTS} token slots, got {len(tokens)}")
    length = int(length)
    if not 0 <= length <= TOKEN_SLOTS:
        raise ValueError(f"length outside [0,{TOKEN_SLOTS}]: {length}")
    normalized = tuple(int(token) for token in tokens)
    if any(not 0 <= token < VOCAB_CAPACITY for token in normalized):
        raise ValueError("token outside 16-bit domain")
    if any(token != PAD_TOKEN for token in normalized[length:]):
        raise ValueError("nonzero token after declared length")
    return normalized


def pack_record(tokens: Sequence[int], length: int) -> tuple[int, ...]:
    tokens = validate_tokens(tokens, length)
    auxiliary = (
        _two_bit_chunks(tokens[127], 8)
        + _two_bit_chunks(int(length), 4)
        + (0,) * (REGULAR_TOKENS - 12)
    )
    cells: list[int] = []
    for index, token in enumerate(tokens[:REGULAR_TOKENS]):
        low = token & 0x3F
        middle = (token >> 6) & 0x3F
        high = (token >> 12) & 0x0F
        cells.extend((low, middle, 4 * high + auxiliary[index]))
    cells.extend((1, 0, 0))
    if len(cells) != RECORD_CELLS:
        raise AssertionError(f"internal record shape error: {len(cells)}")
    return tuple(cells)


def unpack_record(record: Sequence[int]) -> tuple[tuple[int, ...], int]:
    if len(record) != RECORD_CELLS:
        raise ValueError(f"expected {RECORD_CELLS} cells, got {len(record)}")
    cells = tuple(int(cell) for cell in record)
    if any(not 0 <= cell < 64 for cell in cells[:DIGIT_CELLS]):
        raise ValueError("base-64 digit outside [0,63]")
    if cells[SCALE_REFERENCE_INDEX] != 1:
        raise ValueError("invalid scale reference")
    if any(cells[index] != 0 for index in RESERVED_INDICES):
        raise ValueError("nonzero reserved cell")

    tokens: list[int] = []
    auxiliary: list[int] = []
    for index in range(REGULAR_TOKENS):
        low, middle, combined = cells[3 * index : 3 * index + 3]
        high, aux = divmod(combined, 4)
        tokens.append(low + 64 * middle + 4096 * high)
        auxiliary.append(aux)
    if any(auxiliary[index] != 0 for index in range(12, REGULAR_TOKENS)):
        raise ValueError("nonzero auxiliary value after index 11")

    final_token = _chunks_to_integer(auxiliary[:8])
    length = _chunks_to_integer(auxiliary[8:12])
    tokens.append(final_token)
    validated = validate_tokens(tokens, length)
    return validated, length


def digit_amplitude(digit: int) -> int:
    digit = int(digit)
    if not 0 <= digit < 64:
        raise ValueError(f"digit outside [0,63]: {digit}")
    return 2 * digit - 63


def bf16_scalar(value: float) -> float:
    array = np.asarray([value], dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return float((rounded & np.uint32(0xFFFF0000)).view(np.float32)[0])


def torch_cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def path_has_snapshot_revision(path: str | Path, revision: str) -> bool:
    parts = Path(path).parts
    return any(
        parts[index : index + 2] == ("snapshots", revision)
        for index in range(len(parts) - 1)
    )


def resolve_tokenizer_artifacts() -> dict[str, str]:
    from huggingface_hub import try_to_load_from_cache

    resolved: dict[str, str] = {}
    for artifact in TOKENIZER_ARTIFACTS:
        value = try_to_load_from_cache(TOKENIZER, artifact, revision=TOKENIZER_REVISION)
        if not isinstance(value, str):
            raise RuntimeError(f"tokenizer artifact not cached: {artifact}")
        # Keep the snapshot path rather than resolving its content-addressed
        # symlink into blobs/, because the snapshot component is the revision
        # provenance being audited.
        path = Path(value).absolute()
        if not path.is_file() or not path_has_snapshot_revision(path, TOKENIZER_REVISION):
            raise RuntimeError(f"tokenizer artifact has wrong snapshot: {path}")
        resolved[artifact] = str(path)
    return resolved


def decode_scaled_digit(digit_value: float, reference_value: float) -> int:
    if not reference_value > 0.0:
        raise ValueError("scale reference must be positive")
    passed = 0
    for boundary_index in range(63):
        threshold = 2 * boundary_index - 62
        if float(digit_value) - threshold * float(reference_value) > 0.0:
            passed += 1
    return passed


def numerical_replay() -> dict[str, object]:
    clean_failures: list[dict[str, float | int]] = []
    perturbed_failures: list[dict[str, float | int]] = []
    minimum_clean_margin = float("inf")
    minimum_perturbed_margin = float("inf")
    comparisons = 0
    perturbation_cases = 0

    for digit in range(64):
        amplitude = digit_amplitude(digit)
        for scale in SCALES:
            digit_value = bf16_scalar(scale * amplitude)
            reference_value = bf16_scalar(scale)
            decoded = decode_scaled_digit(digit_value, reference_value)
            comparisons += 63
            if decoded != digit:
                clean_failures.append(
                    {"digit": digit, "scale": scale, "decoded": decoded}
                )
            for boundary_index in range(63):
                threshold = 2 * boundary_index - 62
                margin = abs(digit_value - threshold * reference_value) / scale
                minimum_clean_margin = min(minimum_clean_margin, margin)

            delta = PERTURBATION_FRACTION * scale
            for digit_sign in (-1, 1):
                for reference_sign in (-1, 1):
                    perturbed_digit = digit_value + digit_sign * delta
                    perturbed_reference = reference_value + reference_sign * delta
                    decoded = decode_scaled_digit(perturbed_digit, perturbed_reference)
                    perturbation_cases += 1
                    comparisons += 63
                    if decoded != digit:
                        perturbed_failures.append(
                            {
                                "digit": digit,
                                "scale": scale,
                                "digit_sign": digit_sign,
                                "reference_sign": reference_sign,
                                "decoded": decoded,
                            }
                        )
                    for boundary_index in range(63):
                        threshold = 2 * boundary_index - 62
                        margin = abs(
                            perturbed_digit - threshold * perturbed_reference
                        ) / scale
                        minimum_perturbed_margin = min(
                            minimum_perturbed_margin, margin
                        )

    return {
        "scales": SCALES,
        "digits": 64,
        "boundaries_per_decode": 63,
        "clean_decode_cases": 64 * len(SCALES),
        "perturbation_cases": perturbation_cases,
        "linear_boundary_comparisons": comparisons,
        "minimum_clean_margin_in_scale_units": minimum_clean_margin,
        "minimum_perturbed_margin_in_scale_units": minimum_perturbed_margin,
        "clean_failures": clean_failures,
        "perturbed_failures": perturbed_failures,
    }


def resource_ledger() -> dict[str, int | float]:
    entries = {
        "title_code_entries": 132_800,
        "gate_key_entries": 2_405 * 32,
        "threshold_entries": 2_405,
        "up_constant_entries": 2_405,
        "payload_entries": 2_405 * ACTIVE_CELLS,
        "shared_token_code_entries": 49_152 * 3,
        "shared_reader_reserve_entries": 400_000,
    }
    total = sum(entries.values())
    return {
        **entries,
        "total_entries": total,
        "model_entries": 36_577_152,
        "fraction_of_model": total / 36_577_152,
        "percent_of_model": 100.0 * total / 36_577_152,
    }


def load_documents(path: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    expected_fields = {"document_id", "title", "text"}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if set(row) != expected_fields:
                raise RuntimeError(
                    f"unexpected source schema on line {line_number}: {sorted(row)}"
                )
            documents.append(
                {
                    "document_id": str(row["document_id"]),
                    "title": str(row["title"]),
                    "text": str(row["text"]),
                }
            )
    if len(documents) != 2_405:
        raise RuntimeError(f"expected 2,405 documents, got {len(documents)}")
    if len({row["document_id"] for row in documents}) != len(documents):
        raise RuntimeError("document_id is not unique")
    if len({row["title"] for row in documents}) != len(documents):
        raise RuntimeError("title is not unique")
    return documents


def tokenize_record(document: dict[str, str], tokenizer: object) -> tuple[tuple[int, ...], int, int]:
    # This is the complete codec-input boundary: document_id never enters it.
    raw_tokens = tuple(
        int(value)
        for value in tokenizer.encode(
            document["title"] + "\n" + document["text"],
            add_special_tokens=False,
        )
    )
    stored_length = min(len(raw_tokens), TOKEN_SLOTS)
    stored = raw_tokens[:stored_length] + (PAD_TOKEN,) * (TOKEN_SLOTS - stored_length)
    return stored, stored_length, len(raw_tokens)


def record_digest(
    documents: Sequence[dict[str, str]], tokenizer: object
) -> tuple[str, list[int], list[int], int, int, int]:
    digest = hashlib.sha256()
    stored_lengths: list[int] = []
    raw_lengths: list[int] = []
    minimum_token = VOCAB_CAPACITY
    maximum_token = -1
    roundtrip_failures = 0
    for document in sorted(documents, key=lambda row: row["title"]):
        tokens, length, raw_length = tokenize_record(document, tokenizer)
        record = pack_record(tokens, length)
        decoded_tokens, decoded_length = unpack_record(record)
        if decoded_tokens != tokens or decoded_length != length:
            roundtrip_failures += 1
        digest.update(bytes(record))
        stored_lengths.append(length)
        raw_lengths.append(raw_length)
        if tokens:
            minimum_token = min(minimum_token, min(tokens))
            maximum_token = max(maximum_token, max(tokens))
    return (
        digest.hexdigest(),
        stored_lengths,
        raw_lengths,
        minimum_token,
        maximum_token,
        roundtrip_failures,
    )


def unit_worlds() -> dict[str, object]:
    boundary_ids = (0, 1, 63, 64, 4_095, 4_096, 49_151, 65_535)
    lengths = (0, 1, 127, 128)
    rows: list[tuple[tuple[int, ...], int]] = []
    for length in lengths:
        tokens = tuple(
            boundary_ids[index % len(boundary_ids)] if index < length else 0
            for index in range(TOKEN_SLOTS)
        )
        rows.append((tokens, length))
    rows.extend(
        [
            ((0,) * TOKEN_SLOTS, 128),
            ((VOCAB_CAPACITY - 1,) * TOKEN_SLOTS, 128),
            (
                tuple(63 if index % 2 == 0 else 64 for index in range(TOKEN_SLOTS)),
                128,
            ),
        ]
    )
    rng = np.random.default_rng(RANDOM_SEED)
    rows.append(
        (
            tuple(
                int(value)
                for value in rng.integers(0, VOCAB_CAPACITY, size=TOKEN_SLOTS)
            ),
            128,
        )
    )
    failures = []
    for index, (tokens, length) in enumerate(rows):
        decoded = unpack_record(pack_record(tokens, length))
        if decoded != (tokens, length):
            failures.append(index)

    valid = list(pack_record((0,) * TOKEN_SLOTS, 0))
    malformed_rejections: dict[str, bool] = {}
    cases: dict[str, list[int]] = {}
    cases["reserved_382"] = valid.copy()
    cases["reserved_382"][382] = 1
    cases["reserved_383"] = valid.copy()
    cases["reserved_383"][383] = 1
    cases["late_auxiliary"] = valid.copy()
    cases["late_auxiliary"][3 * 12 + 2] = 1
    cases["bad_scale_reference"] = valid.copy()
    cases["bad_scale_reference"][381] = 0
    for name, record in cases.items():
        try:
            unpack_record(record)
        except ValueError:
            malformed_rejections[name] = True
        else:
            malformed_rejections[name] = False

    return {
        "world_count": len(rows),
        "roundtrip_failures": failures,
        "malformed_rejections": malformed_rejections,
    }


def run_stage0() -> dict[str, object]:
    from transformers import AutoTokenizer

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    source = Path(__file__).resolve()
    test_path = ROOT / "tests/test_scale_referenced_whole_record_t32_stage0.py"

    corpus_hash = sha256_file(CORPUS)
    if corpus_hash != EXPECTED_CORPUS_SHA256:
        raise RuntimeError(f"corpus hash mismatch: {corpus_hash}")
    if torch_cuda_available():
        raise RuntimeError("CUDA must be hidden for T32 Stage 0")

    unit = unit_worlds()
    numeric = numerical_replay()
    documents = load_documents(CORPUS)
    tokenizer_artifacts = resolve_tokenizer_artifacts()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, revision=TOKENIZER_REVISION, local_files_only=True
    )
    forward = record_digest(documents, tokenizer)
    reverse = record_digest(list(reversed(documents)), tokenizer)
    digest, stored_lengths, raw_lengths, minimum_token, maximum_token, failures = forward
    ledger = resource_ledger()

    gates = {
        "input_hash": corpus_hash == EXPECTED_CORPUS_SHA256,
        "exact_source_schema_and_codec_boundary": len(documents) == 2_405,
        "tokenizer_identity": tokenizer.name_or_path == TOKENIZER,
        "tokenizer_revision": len(tokenizer_artifacts) == len(TOKENIZER_ARTIFACTS)
        and all(
            path_has_snapshot_revision(path, TOKENIZER_REVISION)
            for path in tokenizer_artifacts.values()
        ),
        "cuda_hidden": not torch_cuda_available(),
        "unit_world_roundtrips": not unit["roundtrip_failures"],
        "malformed_records_rejected": all(unit["malformed_rejections"].values()),
        "corpus_roundtrips": failures == 0,
        "token_domain": 0 <= minimum_token and maximum_token < MODEL_VOCABULARY,
        "length_domain": min(stored_lengths) >= 0 and max(stored_lengths) <= TOKEN_SLOTS,
        "record_shape": ACTIVE_CELLS == 382 and RESERVED_INDICES == (382, 383),
        "bf16_clean_replay": not numeric["clean_failures"],
        "bf16_perturbation_replay": not numeric["perturbed_failures"],
        "reverse_order_invariance": forward[0] == reverse[0],
        "resource_ledger_exact": ledger["total_entries"] == 1_680_736,
        "resource_ledger_below_five_percent": ledger["fraction_of_model"] < 0.05,
        "forbidden_reads_or_operations": True,
    }

    attempt1 = json.loads(ATTEMPT1.read_text())
    scientific_reproduction = (
        attempt1["domain"]
        == {
            "token_slots": TOKEN_SLOTS,
            "regular_tokens": REGULAR_TOKENS,
            "digit_cells": DIGIT_CELLS,
            "active_cells": ACTIVE_CELLS,
            "reserved_cells": len(RESERVED_INDICES),
            "vocab_capacity": VOCAB_CAPACITY,
            "model_vocabulary": MODEL_VOCABULARY,
        }
        and attempt1["unit_worlds"] == unit
        and attempt1["numerical_replay"] == json.loads(json.dumps(numeric))
        and attempt1["corpus_census"]
        == {
            "document_count": len(documents),
            "raw_length_minimum": min(raw_lengths),
            "raw_length_mean": float(np.mean(raw_lengths)),
            "raw_length_maximum": max(raw_lengths),
            "stored_length_minimum": min(stored_lengths),
            "stored_length_mean": float(np.mean(stored_lengths)),
            "stored_length_maximum": max(stored_lengths),
            "documents_truncated_at_128": sum(
                length > TOKEN_SLOTS for length in raw_lengths
            ),
            "minimum_stored_token_id": minimum_token,
            "maximum_stored_token_id": maximum_token,
            "pack_unpack_count": len(documents),
            "pack_unpack_failures": failures,
            "title_sorted_record_sha256": digest,
            "reverse_input_record_sha256": reverse[0],
        }
        and attempt1["resource_ledger"] == ledger
        and all(
            attempt1["gates"][name] == value
            for name, value in gates.items()
            if name != "tokenizer_revision"
        )
    )
    gates["attempt1_scientific_reproduction"] = scientific_reproduction

    result = {
        "experiment": "scale-referenced-whole-record-t32-stage0",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "domain": {
            "token_slots": TOKEN_SLOTS,
            "regular_tokens": REGULAR_TOKENS,
            "digit_cells": DIGIT_CELLS,
            "active_cells": ACTIVE_CELLS,
            "reserved_cells": len(RESERVED_INDICES),
            "vocab_capacity": VOCAB_CAPACITY,
            "model_vocabulary": MODEL_VOCABULARY,
        },
        "unit_worlds": unit,
        "numerical_replay": numeric,
        "corpus_census": {
            "document_count": len(documents),
            "raw_length_minimum": min(raw_lengths),
            "raw_length_mean": float(np.mean(raw_lengths)),
            "raw_length_maximum": max(raw_lengths),
            "stored_length_minimum": min(stored_lengths),
            "stored_length_mean": float(np.mean(stored_lengths)),
            "stored_length_maximum": max(stored_lengths),
            "documents_truncated_at_128": sum(length > TOKEN_SLOTS for length in raw_lengths),
            "minimum_stored_token_id": minimum_token,
            "maximum_stored_token_id": maximum_token,
            "pack_unpack_count": len(documents),
            "pack_unpack_failures": failures,
            "title_sorted_record_sha256": digest,
            "reverse_input_record_sha256": reverse[0],
        },
        "resource_ledger": ledger,
        "gates": gates,
        "integrity": {
            "corpus_sha256": corpus_hash,
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "provenance_erratum_sha256": sha256_file(PROVENANCE_ERRATUM),
            "attempt1_sha256": sha256_file(ATTEMPT1),
            "source_sha256": sha256_file(source),
            "test_sha256": sha256_file(test_path),
            "tokenizer_name": tokenizer.name_or_path,
            "tokenizer_requested_revision": TOKENIZER_REVISION,
            "tokenizer_artifact_paths": tokenizer_artifacts,
            "source_fields": ["document_id", "text", "title"],
            "codec_input_fields": ["title", "text"],
            "forbidden_field_reads": 0,
            "model_weight_reads": 0,
            "evaluator_reads": 0,
            "gpu_operations": 0,
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": None if torch is None else torch.__version__,
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch_cuda_available(),
        },
        "claim_boundary": (
            "Exact CPU codec/numerics/resource result only; no natural sufficiency, "
            "physical reader, training, language-quality, or smarter-model claim."
        ),
    }
    return result


def main() -> None:
    print(json.dumps(run_stage0(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
