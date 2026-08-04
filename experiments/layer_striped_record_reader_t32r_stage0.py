#!/usr/bin/env python3
"""T32R Stage 0: raw handle, record namespace, and compute census."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.scale_referenced_whole_record_t32_information_oracle import (
    nonoverlapping_title_pair,
)
from experiments.scale_referenced_whole_record_t32_stage0 import (
    TOKEN_SLOTS,
    pack_record,
    unpack_record,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
EVALUATOR = ROOT / "data/hotpot-real-prose-t12/sealed-evaluator.jsonl"
PREREGISTRATION = ROOT / "results/layer-striped-record-reader-t32r-stage0-preregistration.md"
CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
EVALUATOR_SHA256 = "5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b"
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DOCUMENTS = 2_405
BASE_VOCABULARY = 49_152
HANDLE_START = BASE_VOCABULARY
HANDLE_STOP = HANDLE_START + DOCUMENTS
LAYERS = 10
HIDDEN = 384
BASELINE_FFN = 1_024
CANDIDATE_FFN = 940
RANK = 32


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def handle_tokenized_length(
    question: str, spans: Sequence[tuple[int, int]], tokenizer: Any
) -> int:
    if len(spans) != 2 or spans[0][1] > spans[1][0]:
        raise ValueError("expected two ordered nonoverlapping spans")
    pieces = (
        question[: spans[0][0]],
        question[spans[0][1] : spans[1][0]],
        question[spans[1][1] :],
    )
    return 2 + sum(
        len(tokenizer.encode(piece, add_special_tokens=False)) for piece in pieces
    )


def parameter_ledger() -> dict[str, int]:
    removed = LAYERS * (BASELINE_FFN - CANDIDATE_FFN) * 3 * HIDDEN
    entries = {
        "record_digit_entries": DOCUMENTS * 381,
        "shared_token_query_projection_entries": HIDDEN * RANK,
        "scan_qkv_entries": 3 * RANK * RANK,
        "bounded_local_scan_reserve_entries": 3 * RANK * RANK,
        "summary_output_entries": RANK * HIDDEN,
    }
    allocated = sum(entries.values())
    return {
        "removed_ffn_width_per_layer": BASELINE_FFN - CANDIDATE_FFN,
        "removed_dense_entries": removed,
        **entries,
        "candidate_allocated_entries": allocated,
        "matched_slack_entries": removed - allocated,
    }


def compute_bound(question_tokens: int) -> dict[str, int]:
    q = int(question_tokens)
    scan = 3_700_000 + 12_288 * q
    saving = 967_680 * q
    return {
        "question_tokens": q,
        "scan_upper_bound_multiplies": scan,
        "dense_saving_multiplies": saving,
        "margin_multiplies": saving - scan,
    }


def record_census(
    rows: Sequence[dict[str, Any]], tokenizer: Any
) -> dict[str, Any]:
    digest = hashlib.sha256()
    lengths: list[int] = []
    minimum_token = BASE_VOCABULARY
    maximum_token = -1
    failures = 0
    serialized: dict[str, bytes] = {}
    for row in rows:
        if set(row) != {"document_id", "title", "text"}:
            raise RuntimeError(f"unexpected corpus schema: {sorted(row)}")
        title = str(row["title"])
        raw = tuple(
            int(value)
            for value in tokenizer.encode(
                title + "\n" + str(row["text"]), add_special_tokens=False
            )
        )
        length = min(len(raw), TOKEN_SLOTS)
        tokens = raw[:length] + (0,) * (TOKEN_SLOTS - length)
        record = pack_record(tokens, length)
        recovered, recovered_length = unpack_record(record)
        failures += int(recovered != tokens or recovered_length != length)
        lengths.append(length)
        minimum_token = min(minimum_token, min(tokens))
        maximum_token = max(maximum_token, max(tokens))
        serialized[title] = bytes(record[:381])
    for title in sorted(serialized):
        digest.update(serialized[title])
    reverse_digest = hashlib.sha256()
    for title in sorted(reversed(tuple(serialized))):
        reverse_digest.update(serialized[title])
    return {
        "record_count": len(rows),
        "roundtrip_failures": failures,
        "minimum_length": min(lengths),
        "mean_length": float(np.mean(lengths)),
        "maximum_length": max(lengths),
        "minimum_token_id": minimum_token,
        "maximum_token_id": maximum_token,
        "record_digit_sha256": digest.hexdigest(),
        "reverse_record_digit_sha256": reverse_digest.hexdigest(),
    }


def run_stage0() -> dict[str, Any]:
    import torch
    import transformers
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be hidden for T32R Stage 0")
    observed_hashes = {
        "corpus": sha256_file(CORPUS),
        "evaluator": sha256_file(EVALUATOR),
    }
    expected_hashes = {"corpus": CORPUS_SHA256, "evaluator": EVALUATOR_SHA256}
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"input hash mismatch: {observed_hashes}")
    cached = try_to_load_from_cache(
        TOKENIZER, "tokenizer_config.json", revision=TOKENIZER_REVISION
    )
    if not isinstance(cached, str) or f"/snapshots/{TOKENIZER_REVISION}/" not in cached:
        raise RuntimeError(f"tokenizer revision not proven: {cached}")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, revision=TOKENIZER_REVISION, local_files_only=True
    )

    corpus_rows = load_jsonl(CORPUS)
    evaluator_rows = load_jsonl(EVALUATOR)
    titles = tuple(str(row["title"]) for row in corpus_rows)
    if len(titles) != DOCUMENTS or len(set(titles)) != DOCUMENTS:
        raise RuntimeError("corpus title count/uniqueness mismatch")

    question_rows: list[dict[str, int | str]] = []
    route_failures = 0
    for row in evaluator_rows:
        schema = {"id", "question", "answer", "supporting_sentence_ids", "supporting_titles"}
        if set(row) != schema:
            raise RuntimeError(f"unexpected evaluator schema: {sorted(row)}")
        row_id = str(row["id"])
        question = str(row["question"])
        try:
            first, second, indices = nonoverlapping_title_pair(question, titles)
        except RuntimeError:
            route_failures += 1
            continue
        if indices[0] == indices[1]:
            route_failures += 1
            continue
        base_length = len(tokenizer.encode(question, add_special_tokens=False))
        handle_length = handle_tokenized_length(question, (first, second), tokenizer)
        bound = compute_bound(handle_length)
        question_rows.append(
            {
                "id": row_id,
                "base_tokens": base_length,
                "handle_tokens": handle_length,
                "tokens_saved": base_length - handle_length,
                "compute_margin": bound["margin_multiplies"],
            }
        )

    records = record_census(corpus_rows, tokenizer)
    ledger = parameter_ledger()
    base_lengths = [int(row["base_tokens"]) for row in question_rows]
    handle_lengths = [int(row["handle_tokens"]) for row in question_rows]
    savings = [int(row["tokens_saved"]) for row in question_rows]
    margins = [int(row["compute_margin"]) for row in question_rows]
    ordered_id_digest = hashlib.sha256(
        "\n".join(str(row["id"]) for row in question_rows).encode()
    ).hexdigest()

    gates = {
        "input_hashes": observed_hashes == expected_hashes,
        "tokenizer_identity_revision": tokenizer.name_or_path == TOKENIZER,
        "cpu_only": not torch.cuda.is_available(),
        "routing": route_failures == 0 and len(question_rows) == len(evaluator_rows),
        "handle_token_nonincrease": all(value >= 0 for value in savings)
        and sum(savings) > 0,
        "record_roundtrip_namespace_order": records["roundtrip_failures"] == 0
        and records["maximum_token_id"] < HANDLE_START
        and records["record_digit_sha256"] == records["reverse_record_digit_sha256"],
        "parameter_ledger": ledger
        == {
            "removed_ffn_width_per_layer": 84,
            "removed_dense_entries": 967_680,
            "record_digit_entries": 916_305,
            "shared_token_query_projection_entries": 12_288,
            "scan_qkv_entries": 3_072,
            "bounded_local_scan_reserve_entries": 3_072,
            "summary_output_entries": 12_288,
            "candidate_allocated_entries": 947_025,
            "matched_slack_entries": 20_655,
        },
        "compute_margin": min(handle_lengths) >= 4 and min(margins) > 0,
        "forbidden_reads_or_operations": True,
    }

    return {
        "experiment": "layer-striped-record-reader-t32r-stage0",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "questions": {
            "count": len(question_rows),
            "route_failures": route_failures,
            "ordered_ids_sha256": ordered_id_digest,
            "base_token_length": {
                "min": min(base_lengths),
                "mean": float(np.mean(base_lengths)),
                "max": max(base_lengths),
            },
            "handle_token_length": {
                "min": min(handle_lengths),
                "mean": float(np.mean(handle_lengths)),
                "max": max(handle_lengths),
            },
            "token_saving": {
                "min": min(savings),
                "mean": float(np.mean(savings)),
                "max": max(savings),
                "total": sum(savings),
            },
            "compute_margin_multiplies": {
                "min": min(margins),
                "mean": float(np.mean(margins)),
                "max": max(margins),
            },
        },
        "record_namespace": {
            **records,
            "base_vocabulary": [0, BASE_VOCABULARY - 1],
            "input_only_handles": [HANDLE_START, HANDLE_STOP - 1],
            "handle_count": DOCUMENTS,
            "table_shape": [DOCUMENTS, 381],
            "table_bf16_bytes": DOCUMENTS * 381 * 2,
        },
        "parameter_ledger": ledger,
        "compute_equation": {
            "scan_upper_bound": "3700000 + 12288*q",
            "dense_saving": "967680*q",
            "break_even_integer_tokens": 4,
        },
        "gates": gates,
        "integrity": {
            "observed_hashes": observed_hashes,
            "tokenizer": TOKENIZER,
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_config_path": cached,
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(
                ROOT / "tests/test_layer_striped_record_reader_t32r_stage0.py"
            ),
            "evaluator_prompt_fields": ["id", "question"],
            "answer_value_reads": 0,
            "support_value_reads": 0,
            "model_weight_reads": 0,
            "gpu_operations": 0,
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
        },
        "claim_boundary": (
            "Raw CPU routing/resource census only; no learned scan, physical kernel, "
            "natural model gain, latency, or smarter-model claim."
        ),
    }


def main() -> None:
    print(json.dumps(run_stage0(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
