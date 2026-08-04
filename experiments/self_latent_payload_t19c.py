#!/usr/bin/env python3
"""T19c: use the raw-trained model's own hidden states as in-place payloads."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import raw_countsketch_payload_t19b as t19b


PREREGISTRATION = ROOT / "results/self-latent-payload-t19c-preregistration.md"
OUTPUT = ROOT / "results/self-latent-payload-t19c.json"
CHECKPOINT = ROOT / "results/hotpot-semantic-address-t12-shared-base.pt"
CHECKPOINT_SHA256 = "a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b"
CHECKPOINT_SCHEMA = "hotpot-semantic-address-t12-shared-base-v1"
CHECKPOINT_MODEL_SEED = 8_209
CHECKPOINT_STEPS = 20_000
BATCH_SIZE = 32
CONTEXT = 128
PAYLOAD_DIMS = t19b.PAYLOAD_DIMS
T19B_ACCURACY = 0.6057692307692307


@dataclass(frozen=True)
class SelfLatentCorpus:
    payloads: Tensor
    quantized_payloads: Tensor
    minimum_bf16_cosine: float


def sha256_file(path: Path) -> str:
    return t19b.sha256_file(path)


def load_writer(device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    artifact = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "model_seed": CHECKPOINT_MODEL_SEED,
        "steps": CHECKPOINT_STEPS,
    }
    observed = {key: artifact[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"writer metadata drifted: {observed}")
    model = t19b.t19a.t10.core.build_model(CHECKPOINT_MODEL_SEED, device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model, observed


def padded_tokens(
    texts: list[str], tokenizer: object
) -> tuple[Tensor, Tensor]:
    eos = int(tokenizer.eos_token_id)
    rows: list[list[int]] = []
    lengths: list[int] = []
    for text in texts:
        identifiers = [
            int(value)
            for value in tokenizer.encode(text, add_special_tokens=False)[:CONTEXT]
        ]
        if not identifiers:
            raise RuntimeError(f"empty writer text: {text!r}")
        lengths.append(len(identifiers))
        rows.append(identifiers + [eos] * (CONTEXT - len(identifiers)))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


@torch.no_grad()
def encode_document_payloads(
    model: torch.nn.Module,
    documents: list[dict[str, str]],
    tokenizer: object,
    device: torch.device,
) -> SelfLatentCorpus:
    payloads: list[Tensor] = []
    for start in range(0, len(documents), BATCH_SIZE):
        texts = [
            document["title"] + "\n" + document["text"]
            for document in documents[start : start + BATCH_SIZE]
        ]
        tokens, lengths = padded_tokens(texts, tokenizer)
        tokens = tokens.to(device)
        lengths = lengths.to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden = model.hidden(tokens)[..., :PAYLOAD_DIMS]
        positions = torch.arange(CONTEXT, device=device)[None, :]
        mask = positions < lengths[:, None]
        pooled = (
            (hidden.float() * mask[..., None]).sum(dim=1)
            / lengths.float()[:, None]
        )
        payloads.append(F.normalize(pooled, dim=-1).cpu())
    values = torch.cat(payloads)
    quantized = values.to(torch.bfloat16).float()
    cosine = F.cosine_similarity(values, quantized, dim=-1)
    return SelfLatentCorpus(
        payloads=values,
        quantized_payloads=quantized,
        minimum_bf16_cosine=float(cosine.min().item()),
    )


@torch.no_grad()
def encode_property_queries(
    model: torch.nn.Module,
    rows: list[dict[str, object]],
    titles: tuple[str, ...],
    tokenizer: object,
    device: torch.device,
) -> tuple[Tensor, tuple[tuple[int, int], ...]]:
    texts: list[str] = []
    pairs: list[tuple[int, int]] = []
    for row in rows:
        question = str(row["question"])
        first_span, second_span, indices = t19b.nonoverlapping_title_pair(
            question, titles
        )
        texts.append(t19b.remove_spans(question, (first_span, second_span)))
        pairs.append(indices)
    vectors: list[Tensor] = []
    for start in range(0, len(texts), BATCH_SIZE):
        tokens, lengths = padded_tokens(texts[start : start + BATCH_SIZE], tokenizer)
        tokens = tokens.to(device)
        lengths = lengths.to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden = model.hidden(tokens)[..., :PAYLOAD_DIMS]
        selected = hidden[
            torch.arange(len(tokens), device=device), lengths - 1
        ].float()
        vectors.append(F.normalize(selected, dim=-1).cpu())
    return torch.cat(vectors).to(torch.bfloat16).float(), tuple(pairs)


def build_reader_matrix(
    rows: list[dict[str, object]],
    query_vectors: Tensor,
    pairs: tuple[tuple[int, int], ...],
    payloads: Tensor,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    examples: list[Tensor] = []
    labels: list[float] = []
    for row, query, pair in zip(rows, query_vectors, pairs, strict=True):
        examples.append(
            t19b.symmetric_reader_features(
                query, payloads[pair[0]], payloads[pair[1]]
            )
        )
        labels.append(float(str(row["answer"]).lower() == "yes"))
    targets = torch.tensor(labels, dtype=torch.float64)
    return torch.stack(examples), targets, {
        "examples": len(rows),
        "positive_rate": float(targets.mean().item()),
        "unique_title_pairs": len(set(pairs)),
        "minimum_query_norm": float(query_vectors.norm(dim=-1).min().item()),
    }


def verify_inputs() -> dict[str, bool]:
    checks = {
        "checkpoint": sha256_file(CHECKPOINT) == CHECKPOINT_SHA256,
        "candidate_corpus": sha256_file(t19b.t19a.CANDIDATE_CORPUS)
        == t19b.t19a.CANDIDATE_SHA256,
        "train_qa": sha256_file(t19b.TRAIN_QA) == t19b.TRAIN_QA_SHA256,
        "evaluation_qa": sha256_file(t19b.EVALUATION_QA)
        == t19b.t19a.EVALUATOR_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def run(device: torch.device) -> dict[str, object]:
    input_checks = verify_inputs()
    documents = t19b.t19a.load_raw_documents(t19b.t19a.CANDIDATE_CORPUS)
    tokenizer = AutoTokenizer.from_pretrained(
        t19b.t19a.TOKENIZER, revision=t19b.t19a.TOKENIZER_REVISION
    )
    title_data = t19b.t19a.build_title_data(documents, tokenizer)
    writer, writer_metadata = load_writer(device)
    corpus = encode_document_payloads(writer, documents, tokenizer, device)
    train_rows = t19b.load_qa(t19b.TRAIN_QA)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    train_queries, train_pairs = encode_property_queries(
        writer, train_rows, title_data.titles, tokenizer, device
    )
    evaluation_queries, evaluation_pairs = encode_property_queries(
        writer, evaluation_rows, title_data.titles, tokenizer, device
    )
    train_inputs, train_targets, train_manifest = build_reader_matrix(
        train_rows,
        train_queries,
        train_pairs,
        corpus.quantized_payloads,
    )
    evaluation_inputs, evaluation_targets, evaluation_manifest = build_reader_matrix(
        evaluation_rows,
        evaluation_queries,
        evaluation_pairs,
        corpus.quantized_payloads,
    )
    reader = t19b.fit_logistic_reader(
        train_inputs, train_targets, evaluation_inputs, evaluation_targets
    )

    control, _ = load_writer(device)
    candidate, _ = load_writer(device)
    base_hashes_equal = (
        t19b.t19a.t10.core.small.state_sha256(control)
        == t19b.t19a.t10.core.small.state_sha256(candidate)
    )
    control_schema = t19b.state_schema(control)
    physical = t19b.compile_physical_payload(
        candidate, title_data, corpus.quantized_payloads
    )
    candidate_schema = t19b.state_schema(candidate)
    parameter_counts = {
        sum(parameter.numel() for parameter in control.parameters()),
        sum(parameter.numel() for parameter in candidate.parameters()),
    }
    evaluation_accuracy = float(reader["evaluation"]["accuracy"])
    finite_payloads = bool(
        torch.isfinite(corpus.payloads).all()
        and torch.isfinite(corpus.quantized_payloads).all()
        and (corpus.quantized_payloads.norm(dim=-1) > 0).all()
    )
    finite_queries = bool(
        torch.isfinite(train_queries).all()
        and torch.isfinite(evaluation_queries).all()
        and (train_queries.norm(dim=-1) > 0).all()
        and (evaluation_queries.norm(dim=-1) > 0).all()
    )
    compiler_fields = set().union(*(document.keys() for document in documents))
    gates = {
        "input_integrity": all(input_checks.values()),
        "checkpoint_metadata": writer_metadata
        == {
            "schema": CHECKPOINT_SCHEMA,
            "model_seed": CHECKPOINT_MODEL_SEED,
            "steps": CHECKPOINT_STEPS,
        },
        "raw_only_compiler_fields": compiler_fields
        == {"document_id", "title", "text"},
        "raw_only_writer_signature": tuple(
            inspect.signature(encode_document_payloads).parameters
        )
        == ("model", "documents", "tokenizer", "device"),
        "payloads_and_queries_finite_nonzero": finite_payloads and finite_queries,
        "bf16_payload_cosine_at_least_0p9999": corpus.minimum_bf16_cosine
        >= 0.9999,
        "two_titles_every_question": train_manifest["examples"] == 146
        and evaluation_manifest["examples"] == 104,
        "write_count_exact": physical["total_writes"] == t19b.EXPECTED_WRITES,
        "write_budget": physical["total_writes"] <= t19b.WRITE_BUDGET,
        "ordinary_served_structure_identical": base_hashes_equal
        and control_schema == candidate_schema
        and len(parameter_counts) == 1
        and next(iter(parameter_counts))
        == t19b.t19a.t10.core.small.parameter_count()
        and type(control) is type(candidate),
        "reader_evaluation_at_least_75pct": evaluation_accuracy >= 0.75,
        "reader_beats_t12_by_10pp": evaluation_accuracy
        >= t19b.T12_SEMANTIC_ACCURACY + 0.10,
        "reader_beats_t19b_by_10pp": evaluation_accuracy >= T19B_ACCURACY + 0.10,
        "both_classes": {float(value) for value in train_targets.tolist()}
        == {0.0, 1.0}
        and {float(value) for value in evaluation_targets.tolist()} == {0.0, 1.0},
        "finite_no_retry_reader": reader["finite"] is True,
    }
    return {
        "schema": "self-latent-payload-t19c-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "configuration": {
            "documents": len(documents),
            "payload_dimensions": PAYLOAD_DIMS,
            "context": CONTEXT,
            "batch_size": BATCH_SIZE,
            "write_budget": t19b.WRITE_BUDGET,
            "expected_writes": t19b.EXPECTED_WRITES,
            "reader_feature_dimensions": train_inputs.shape[1],
        },
        "writer": {
            "metadata": writer_metadata,
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "minimum_payload_bf16_cosine": corpus.minimum_bf16_cosine,
            "minimum_payload_norm": float(
                corpus.quantized_payloads.norm(dim=-1).min().item()
            ),
        },
        "reader_data": {
            "train": train_manifest,
            "evaluation": evaluation_manifest,
        },
        "reader": reader,
        "physical_write_ledger": physical,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "candidate_corpus_sha256": sha256_file(t19b.t19a.CANDIDATE_CORPUS),
            "train_qa_sha256": sha256_file(t19b.TRAIN_QA),
            "evaluation_qa_sha256": sha256_file(t19b.EVALUATION_QA),
        },
        "claim_boundary": (
            "This tests whether raw-only self-latents are sufficient payloads. It "
            "does not train the in-place writer/reader or establish QA superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device))
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
                "reader": result["reader"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
