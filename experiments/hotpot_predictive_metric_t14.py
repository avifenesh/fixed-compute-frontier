#!/usr/bin/env python3
"""T14: fixed output-geometry compiler-metric development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import hotpot_contextual_address_t13 as t13
from experiments import hotpot_semantic_address_t12 as t12
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/hotpot-predictive-metric-t14-preregistration.md"
OUTPUT = ROOT / "results/hotpot-predictive-metric-t14.json"
CHECKPOINT = t13.CHECKPOINT
T13_RESULT = ROOT / "results/hotpot-contextual-address-t13.json"

T13_RESULT_SHA256 = "a09fc367ad52bfe0b357690e9b0aed6e252c5c774b9efaeae8b8f8c906347329"
LEXICAL_REFERENCE_ACCURACY = t13.LEXICAL_REFERENCE_ACCURACY
STATIC_REFERENCE_ACCURACY = t13.STATIC_REFERENCE_ACCURACY
CONTEXTUAL_REFERENCE_ACCURACY = 0.5384615384615384


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_metric(model: t10.core.small.SharedInterpreterLM) -> Tensor:
    embedding = model.token.weight.float()
    return embedding.T @ embedding / embedding.shape[0]


@torch.inference_mode()
def encode_final_sequences(
    model: t10.core.small.SharedInterpreterLM,
    sequences: list[list[int]],
    eos_token_id: int,
    device: torch.device,
    batch_size: int = t13.ENCODE_BATCH,
) -> list[Tensor]:
    if any(not sequence or len(sequence) > t13.CHUNK_TOKENS for sequence in sequences):
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
            hidden = model.hidden(batch)
        hidden = hidden.float()
        encoded.extend(hidden[index, : len(row)].cpu() for index, row in enumerate(rows))
    return encoded


@torch.inference_mode()
def predictive_scores(
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
        chunks = t13.document_token_chunks(identifiers)
        if not chunks:
            raise RuntimeError(f"empty supporting document: {title}")
        all_chunks.extend(chunks)
        chunk_counts[title] = len(chunks)
    encoded_chunks = encode_final_sequences(
        model, all_chunks, int(tokenizer.eos_token_id), device
    )
    metric = output_metric(model).cpu()
    page_keys: dict[str, Tensor] = {}
    cursor = 0
    for title in titles:
        count = chunk_counts[title]
        page_hidden = torch.cat(encoded_chunks[cursor : cursor + count])
        page_keys[title] = page_hidden @ metric
        cursor += count
    if cursor != len(encoded_chunks):
        raise RuntimeError("document chunk accounting mismatch")

    question_tokens = [
        tokenizer.encode(str(row["question"]), add_special_tokens=False)[
            -t13.CHUNK_TOKENS :
        ]
        for row in rows
    ]
    if any(not identifiers for identifiers in question_tokens):
        raise RuntimeError("empty natural question")
    question_hidden = encode_final_sequences(
        model, question_tokens, int(tokenizer.eos_token_id), device
    )

    results: list[tuple[float, bool]] = []
    for row, hidden in zip(rows, question_hidden, strict=True):
        query = hidden[-1]
        page_scores = [
            float((page_keys[title] @ query).max().item())
            for title in t12.data.supporting_titles(row)
        ]
        value = min(page_scores)
        if not math.isfinite(value):
            raise RuntimeError("nonfinite predictive-metric score")
        results.append((value, str(row["answer"]).lower() == "yes"))
    return results


@torch.inference_mode()
def predictive_probe(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    documents: list[dict[str, str]],
    split: t12.data.Split,
    device: torch.device,
) -> dict[str, object]:
    by_title = t12.documents_by_title(documents)
    train = predictive_scores(model, tokenizer, by_title, split.train, device)
    evaluation = predictive_scores(
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


def metric_spectrum(model: t10.core.small.SharedInterpreterLM) -> dict[str, float]:
    eigenvalues = torch.linalg.eigvalsh(output_metric(model).cpu())
    minimum = float(eigenvalues.min().item())
    maximum = float(eigenvalues.max().item())
    return {
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "condition_number": maximum / minimum,
        "trace": float(eigenvalues.sum().item()),
    }


def verify_inputs(checkpoint_path: Path) -> dict[str, bool]:
    checks = {
        "checkpoint_bytes": checkpoint_path.stat().st_size == t13.CHECKPOINT_BYTES,
        "checkpoint_sha256": sha256_file(checkpoint_path) == t13.CHECKPOINT_SHA256,
        "t13_result_sha256": sha256_file(T13_RESULT) == T13_RESULT_SHA256,
        "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS)
        == t13.CANDIDATE_CORPUS_SHA256,
        "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST)
        == t13.DATA_MANIFEST_SHA256,
        "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET)
        == t12.data.SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"T14 input integrity failed: {checks}")
    return checks


def evaluate_model(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    documents: list[dict[str, str]],
    split: t12.data.Split,
    device: torch.device,
) -> dict[str, object]:
    state_before = t10.core.small.state_sha256(model)
    probe = predictive_probe(model, tokenizer, documents, split, device)
    state_after = t10.core.small.state_sha256(model)
    return {
        "probe": probe,
        "metric_spectrum": metric_spectrum(model),
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
    }


def run(device: torch.device, checkpoint_path: Path) -> dict[str, object]:
    checks = verify_inputs(checkpoint_path)
    checkpoint_before = sha256_file(checkpoint_path)
    tokenizer = t12.AutoTokenizer.from_pretrained(
        t12.TOKENIZER, revision=t12.TOKENIZER_REVISION
    )
    documents = t12.load_documents(t12.CANDIDATE_CORPUS)
    split = t12.load_probe_split()

    trained_model = t13.load_checkpoint(checkpoint_path, device)
    trained = evaluate_model(trained_model, tokenizer, documents, split, device)
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random_model = t10.core.build_model(t12.MODEL_SEED, device).eval()
    if t10.core.small.state_sha256(random_model) != t13.INITIAL_STATE_SHA256:
        raise RuntimeError("initial model-state hash mismatch")
    random = evaluate_model(random_model, tokenizer, documents, split, device)
    del random_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_after = sha256_file(checkpoint_path)
    trained_accuracy = trained["probe"]["evaluation"]["accuracy"]
    random_accuracy = random["probe"]["evaluation"]["accuracy"]
    finite = all(
        math.isfinite(value)
        for arm in (trained, random)
        for value in (
            *arm["probe"]["score_summary"].values(),
            *arm["metric_spectrum"].values(),
        )
    )
    gates = {
        "input_integrity": all(checks.values()),
        "trained_state_unchanged": trained["state_sha256_before"]
        == t13.TERMINAL_STATE_SHA256
        == trained["state_sha256_after"],
        "random_state_unchanged": random["state_sha256_before"]
        == t13.INITIAL_STATE_SHA256
        == random["state_sha256_after"],
        "checkpoint_byte_identical": checkpoint_before
        == t13.CHECKPOINT_SHA256
        == checkpoint_after,
        "finite_scores_and_metric": finite,
        "trained_predictive_at_least_70pct": trained_accuracy >= 0.70,
        "trained_predictive_beats_lexical_by_10pp": trained_accuracy
        >= LEXICAL_REFERENCE_ACCURACY + 0.10,
        "trained_predictive_beats_random": trained_accuracy > random_accuracy,
        "trained_predictive_beats_static": trained_accuracy
        > STATIC_REFERENCE_ACCURACY,
        "trained_predictive_beats_direct_contextual": trained_accuracy
        > CONTEXTUAL_REFERENCE_ACCURACY,
        "zero_training_updates": True,
    }
    return {
        "schema": "hotpot-predictive-metric-t14-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "development_only": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": t10.core.small.parameter_count(),
            "metric": "token_weight_transpose_times_token_weight_div_vocab",
            "document_tokens": t13.DOCUMENT_TOKENS,
            "chunk_tokens": t13.CHUNK_TOKENS,
            "question_pooling": "last_real_final_normalized_hidden",
            "document_pooling": "maximum_predictive_dot_then_minimum_page",
            "training_updates": 0,
        },
        "predictive_metric_probe": {
            "claim_boundary": (
                "Privileged gold-page development diagnostic on a previously "
                "observed aggregate evaluator; not fresh held-out evidence."
            ),
            "trained": trained,
            "random_initialization": random,
            "privileged_lexical_reference_accuracy": LEXICAL_REFERENCE_ACCURACY,
            "trained_static_reference_accuracy": STATIC_REFERENCE_ACCURACY,
            "trained_direct_contextual_reference_accuracy": CONTEXTUAL_REFERENCE_ACCURACY,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "t13_result_sha256": sha256_file(T13_RESULT),
            "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS),
            "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST),
            "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET),
        },
        "claim_boundary": (
            "Passing only admits finite-capacity compilation on a new holdout; "
            "it is not autonomous extraction, QA, or a strict Pareto result."
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
                "trained_predictive_accuracy": result["predictive_metric_probe"]["trained"]["probe"]["evaluation"]["accuracy"],
                "random_predictive_accuracy": result["predictive_metric_probe"]["random_initialization"]["probe"]["evaluation"]["accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
