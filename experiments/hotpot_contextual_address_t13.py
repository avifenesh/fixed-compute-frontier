#!/usr/bin/env python3
"""T13: fixed-checkpoint contextual semantic-interface development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import hotpot_semantic_address_t12 as t12
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/hotpot-contextual-address-t13-preregistration.md"
OUTPUT = ROOT / "results/hotpot-contextual-address-t13.json"
CHECKPOINT = ROOT / "results/hotpot-semantic-address-t12-shared-base.pt"
T12_RESULT = ROOT / "results/hotpot-semantic-address-t12.json"

CHECKPOINT_BYTES = 146_340_367
CHECKPOINT_SHA256 = "a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b"
T12_RESULT_SHA256 = "e7c260ec7cbf1aef33e1f8e15213a6ee53f58741ab6971b1446e8e6186b0853a"
TERMINAL_STATE_SHA256 = "53b3117727045ad31b52efd719b55fb251a557b2256008c924318c3cc244de57"
INITIAL_STATE_SHA256 = "40d208029720fb9221f5aa74cbf772e5fa214464f27f90de3a5e7edc77a8843c"
CANDIDATE_CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
DATA_MANIFEST_SHA256 = "90e539279f540a7516e3c90c3ee59539d3f8654cc84e88a82dc950ce6fb5036e"
LEXICAL_REFERENCE_ACCURACY = 0.5576923076923077
STATIC_REFERENCE_ACCURACY = 0.6057692307692307
DOCUMENT_TOKENS = 512
CHUNK_TOKENS = 128
ENCODE_BATCH = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_ffn_interface(
    model: t10.core.small.SharedInterpreterLM, tokens: Tensor
) -> Tensor:
    """Return the exact normalized tensor consumed by the final FFN matrices."""
    hidden = model.token(tokens)
    for block in model.blocks[:-1]:
        hidden = block(hidden)
    final = model.blocks[-1]
    hidden = hidden + final.attention(final.attention_norm(hidden))
    return final.ffn_norm(hidden)


@torch.inference_mode()
def encode_sequences(
    model: t10.core.small.SharedInterpreterLM,
    sequences: list[list[int]],
    eos_token_id: int,
    device: torch.device,
    batch_size: int = ENCODE_BATCH,
) -> list[Tensor]:
    """Encode variable-length sequences; return normalized real-token rows on CPU."""
    if any(not sequence or len(sequence) > CHUNK_TOKENS for sequence in sequences):
        raise ValueError("every sequence must contain 1..128 tokens")
    encoded: list[Tensor] = []
    for start in range(0, len(sequences), batch_size):
        rows = sequences[start : start + batch_size]
        maximum = max(len(row) for row in rows)
        batch = torch.full(
            (len(rows), maximum),
            eos_token_id,
            dtype=torch.long,
            device=device,
        )
        for index, row in enumerate(rows):
            batch[index, : len(row)] = torch.tensor(row, device=device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden = final_ffn_interface(model, batch)
        hidden = F.normalize(hidden.float(), dim=-1)
        encoded.extend(hidden[index, : len(row)].cpu() for index, row in enumerate(rows))
    return encoded


def document_token_chunks(identifiers: list[int]) -> list[list[int]]:
    selected = identifiers[:DOCUMENT_TOKENS]
    return [
        selected[start : start + CHUNK_TOKENS]
        for start in range(0, len(selected), CHUNK_TOKENS)
    ]


@torch.inference_mode()
def contextual_scores(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    documents: dict[str, str],
    rows: tuple[dict[str, object], ...],
    device: torch.device,
) -> list[tuple[float, bool]]:
    titles = sorted(
        {title for row in rows for title in t12.data.supporting_titles(row)}
    )
    all_chunks: list[list[int]] = []
    chunk_counts: dict[str, int] = {}
    for title in titles:
        if title not in documents:
            raise RuntimeError(f"supporting title absent from candidate corpus: {title}")
        identifiers = tokenizer.encode(documents[title], add_special_tokens=False)
        chunks = document_token_chunks(identifiers)
        if not chunks:
            raise RuntimeError(f"empty supporting document: {title}")
        all_chunks.extend(chunks)
        chunk_counts[title] = len(chunks)
    encoded_chunks = encode_sequences(
        model, all_chunks, int(tokenizer.eos_token_id), device
    )
    page_vectors: dict[str, Tensor] = {}
    cursor = 0
    for title in titles:
        count = chunk_counts[title]
        page_vectors[title] = torch.cat(encoded_chunks[cursor : cursor + count])
        cursor += count
    if cursor != len(encoded_chunks):
        raise RuntimeError("document chunk accounting mismatch")

    question_tokens = [
        tokenizer.encode(str(row["question"]), add_special_tokens=False)[-CHUNK_TOKENS:]
        for row in rows
    ]
    if any(not identifiers for identifiers in question_tokens):
        raise RuntimeError("empty natural question")
    question_hidden = encode_sequences(
        model, question_tokens, int(tokenizer.eos_token_id), device
    )

    results: list[tuple[float, bool]] = []
    for row, hidden in zip(rows, question_hidden, strict=True):
        query = hidden[-1]
        page_scores = [
            float((page_vectors[title] @ query).max().item())
            for title in t12.data.supporting_titles(row)
        ]
        value = min(page_scores)
        if not math.isfinite(value):
            raise RuntimeError("nonfinite contextual score")
        results.append((value, str(row["answer"]).lower() == "yes"))
    return results


@torch.inference_mode()
def contextual_probe(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    documents: list[dict[str, str]],
    split: t12.data.Split,
    device: torch.device,
) -> dict[str, object]:
    by_title = t12.documents_by_title(documents)
    train = contextual_scores(model, tokenizer, by_title, split.train, device)
    evaluation = contextual_scores(
        model, tokenizer, by_title, split.evaluation, device
    )
    threshold, direction, train_accuracy = t12.fit_threshold(train)
    return {
        "threshold": threshold,
        "greater_equal": direction,
        "train": t12.score_threshold(train, threshold, direction),
        "evaluation": t12.score_threshold(evaluation, threshold, direction),
        "fitted_train_accuracy": train_accuracy,
        "score_summary": {
            "train_min": min(value for value, _ in train),
            "train_max": max(value for value, _ in train),
            "evaluation_min": min(value for value, _ in evaluation),
            "evaluation_max": max(value for value, _ in evaluation),
        },
    }


def verify_inputs(checkpoint_path: Path) -> dict[str, bool]:
    checks = {
        "checkpoint_bytes": checkpoint_path.stat().st_size == CHECKPOINT_BYTES,
        "checkpoint_sha256": sha256_file(checkpoint_path) == CHECKPOINT_SHA256,
        "t12_result_sha256": sha256_file(T12_RESULT) == T12_RESULT_SHA256,
        "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS)
        == CANDIDATE_CORPUS_SHA256,
        "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST)
        == DATA_MANIFEST_SHA256,
        "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET)
        == t12.data.SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"T13 input integrity failed: {checks}")
    return checks


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> t10.core.small.SharedInterpreterLM:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != "hotpot-semantic-address-t12-shared-base-v1"
        or payload.get("model_seed") != t12.MODEL_SEED
        or payload.get("steps") != t12.STEPS
    ):
        raise RuntimeError("unexpected T12 checkpoint metadata")
    model = t10.core.build_model(t12.MODEL_SEED, device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if t10.core.small.state_sha256(model) != TERMINAL_STATE_SHA256:
        raise RuntimeError("T12 terminal model-state hash mismatch")
    return model


def run(device: torch.device, checkpoint_path: Path) -> dict[str, object]:
    checks = verify_inputs(checkpoint_path)
    checkpoint_before = sha256_file(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(
        t12.TOKENIZER, revision=t12.TOKENIZER_REVISION
    )
    documents = t12.load_documents(t12.CANDIDATE_CORPUS)
    split = t12.load_probe_split()

    trained = load_checkpoint(checkpoint_path, device)
    trained_state_before = t10.core.small.state_sha256(trained)
    trained_probe = contextual_probe(trained, tokenizer, documents, split, device)
    trained_state_after = t10.core.small.state_sha256(trained)
    del trained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random_model = t10.core.build_model(t12.MODEL_SEED, device)
    random_state = t10.core.small.state_sha256(random_model)
    if random_state != INITIAL_STATE_SHA256:
        raise RuntimeError("initial model-state hash mismatch")
    random_probe = contextual_probe(
        random_model, tokenizer, documents, split, device
    )
    random_state_after = t10.core.small.state_sha256(random_model)
    del random_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_after = sha256_file(checkpoint_path)
    trained_accuracy = trained_probe["evaluation"]["accuracy"]
    random_accuracy = random_probe["evaluation"]["accuracy"]
    finite = all(
        math.isfinite(value)
        for probe in (trained_probe, random_probe)
        for value in probe["score_summary"].values()
    )
    gates = {
        "input_integrity": all(checks.values()),
        "trained_state_unchanged": trained_state_before
        == TERMINAL_STATE_SHA256
        == trained_state_after,
        "random_state_unchanged": random_state
        == INITIAL_STATE_SHA256
        == random_state_after,
        "checkpoint_byte_identical": checkpoint_before
        == CHECKPOINT_SHA256
        == checkpoint_after,
        "finite_scores": finite,
        "trained_contextual_at_least_70pct": trained_accuracy >= 0.70,
        "trained_contextual_beats_lexical_by_10pp": trained_accuracy
        >= LEXICAL_REFERENCE_ACCURACY + 0.10,
        "trained_contextual_beats_random": trained_accuracy > random_accuracy,
        "trained_contextual_beats_static": trained_accuracy
        > STATIC_REFERENCE_ACCURACY,
        "zero_training_updates": True,
    }
    return {
        "schema": "hotpot-contextual-address-t13-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "development_only": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": t10.core.small.parameter_count(),
            "checkpoint_steps": t12.STEPS,
            "document_tokens": DOCUMENT_TOKENS,
            "chunk_tokens": CHUNK_TOKENS,
            "encode_batch": ENCODE_BATCH,
            "interface": "final_block_pre_ffn_normalized_hidden",
            "question_pooling": "last_real_token",
            "document_pooling": "maximum_token_cosine_then_minimum_page",
            "training_updates": 0,
        },
        "contextual_probe": {
            "claim_boundary": (
                "Privileged gold-page development diagnostic on a previously "
                "observed aggregate evaluator; not fresh held-out evidence."
            ),
            "trained": trained_probe,
            "random_initialization": random_probe,
            "privileged_lexical_reference_accuracy": LEXICAL_REFERENCE_ACCURACY,
            "trained_static_reference_accuracy": STATIC_REFERENCE_ACCURACY,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "trained_state_sha256_before": trained_state_before,
            "trained_state_sha256_after": trained_state_after,
            "random_state_sha256_before": random_state,
            "random_state_sha256_after": random_state_after,
            "t12_result_sha256": sha256_file(T12_RESULT),
            "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS),
            "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST),
            "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET),
        },
        "claim_boundary": (
            "Passing only admits a compiler experiment on a new holdout. It does "
            "not demonstrate autonomous extraction, QA capability, strict Pareto "
            "superiority, or a smarter production model."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device), arguments.checkpoint)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "all_gates_pass": result["all_gates_pass"],
                "failed_gates": sorted(
                    name for name, passed in result["gates"].items() if not passed
                ),
                "trained_contextual_accuracy": result["contextual_probe"]["trained"]["evaluation"]["accuracy"],
                "random_contextual_accuracy": result["contextual_probe"]["random_initialization"]["evaluation"]["accuracy"],
                "static_reference_accuracy": STATIC_REFERENCE_ACCURACY,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
