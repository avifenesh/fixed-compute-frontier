#!/usr/bin/env python3
"""T19b: raw-only CountSketch payload sufficiency and exact write ledger."""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import raw_title_code_plane_t19a as t19a


PREREGISTRATION = ROOT / "results/raw-countsketch-payload-t19b-preregistration.md"
OUTPUT = ROOT / "results/raw-countsketch-payload-t19b.json"
TRAIN_QA = ROOT / "data/hotpot-real-prose-t12/router-and-control-train.jsonl"
EVALUATION_QA = ROOT / "data/hotpot-real-prose-t12/sealed-evaluator.jsonl"

TRAIN_QA_SHA256 = "edaef010f68f1a919ac9f9e8389444ddde1afde6d2b9d1375156195f8758b0f9"
PAYLOAD_DIMS = 220
PAYLOAD_START = 64
ADDRESS_CONST = 32
WRITE_BUDGET = 743_734
EXPECTED_WRITES = 743_670
MODEL_SEED = 9_209
L2_COEFFICIENT = 0.01
LBFGS_ITERATIONS = 200
T12_SEMANTIC_ACCURACY = 0.6057692307692307


def sha256_file(path: Path) -> str:
    return t19a.sha256_file(path)


def lexical_features(text: str) -> collections.Counter[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    features: collections.Counter[str] = collections.Counter(
        f"u:{word}" for word in words
    )
    features.update(
        f"b:{first}\x1f{second}" for first, second in zip(words, words[1:])
    )
    for word in words:
        if len(word) == 4 and word.isdigit() and 1000 <= int(word) <= 2099:
            year = int(word)
            features[f"y:{year}"] += 1
            features[f"d:{10 * (year // 10)}"] += 1
    return features


def feature_bucket_and_sign(feature: str) -> tuple[int, float]:
    digest = hashlib.sha256(("t19b-feature-" + feature).encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % PAYLOAD_DIMS
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


@dataclass(frozen=True)
class SketchCorpus:
    sketches: Tensor
    quantized_sketches: Tensor
    idf: dict[str, float]
    feature_counts: tuple[collections.Counter[str], ...]
    minimum_bf16_cosine: float


def build_sketch_corpus(documents: list[dict[str, str]]) -> SketchCorpus:
    feature_counts = tuple(lexical_features(document["text"]) for document in documents)
    frequencies: collections.Counter[str] = collections.Counter()
    for counts in feature_counts:
        frequencies.update(counts.keys())
    count = len(documents)
    idf = {
        feature: math.log((count + 1) / (frequency + 1)) + 1.0
        for feature, frequency in frequencies.items()
    }
    rows: list[Tensor] = []
    for counts in feature_counts:
        row = torch.zeros(PAYLOAD_DIMS, dtype=torch.float32)
        for feature, term_count in counts.items():
            bucket, sign = feature_bucket_and_sign(feature)
            row[bucket] += sign * (1.0 + math.log(term_count)) * idf[feature]
        norm = float(row.norm().item())
        if not math.isfinite(norm) or norm == 0.0:
            raise RuntimeError("raw document produced an empty/nonfinite sketch")
        rows.append(row / norm)
    sketches = torch.stack(rows)
    quantized = sketches.to(torch.bfloat16).float()
    cosine = torch.nn.functional.cosine_similarity(sketches, quantized, dim=-1)
    return SketchCorpus(
        sketches=sketches,
        quantized_sketches=quantized,
        idf=idf,
        feature_counts=feature_counts,
        minimum_bf16_cosine=float(cosine.min().item()),
    )


def sketch_query(text: str, idf: dict[str, float], documents: int) -> Tensor:
    row = torch.zeros(PAYLOAD_DIMS, dtype=torch.float32)
    for feature, term_count in lexical_features(text).items():
        bucket, sign = feature_bucket_and_sign(feature)
        weight = idf.get(feature, math.log(documents + 1) + 1.0)
        row[bucket] += sign * (1.0 + math.log(term_count)) * weight
    norm = float(row.norm().item())
    if norm == 0.0 or not math.isfinite(norm):
        raise RuntimeError(f"question produced empty/nonfinite sketch: {text!r}")
    return row / norm


def nonoverlapping_title_pair(
    question: str, titles: tuple[str, ...]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    folded = question.casefold()
    candidates: list[tuple[int, int, int]] = []
    for index, title in enumerate(titles):
        surface = title.casefold()
        start = folded.find(surface)
        if start >= 0:
            candidates.append((start, start + len(surface), index))
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
    selected: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if all(
            candidate[1] <= existing[0] or candidate[0] >= existing[1]
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise RuntimeError(
            f"expected two nonoverlapping raw titles, found {len(selected)}: {question}"
        )
    selected.sort()
    first, second = selected
    return (first[0], first[1]), (second[0], second[1]), (first[2], second[2])


def remove_spans(text: str, spans: tuple[tuple[int, int], tuple[int, int]]) -> str:
    values = list(text)
    for start, end in spans:
        values[start:end] = " " * (end - start)
    return "".join(values)


def symmetric_reader_features(query: Tensor, first: Tensor, second: Tensor) -> Tensor:
    first_match = query * first
    second_match = query * second
    scalars = torch.tensor(
        [
            float(query @ first),
            float(query @ second),
            float(first @ second),
            min(float(query @ first), float(query @ second)),
            max(float(query @ first), float(query @ second)),
        ],
        dtype=torch.float64,
    )
    return torch.cat(
        (
            torch.minimum(first_match, second_match).double(),
            torch.maximum(first_match, second_match).double(),
            (first * second).double(),
            (first - second).abs().double(),
            scalars,
        )
    )


def load_qa(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(str(row["answer"]).lower() not in {"yes", "no"} for row in rows):
        raise RuntimeError("QA rows missing binary labels")
    return rows


def build_reader_matrix(
    rows: list[dict[str, object]],
    titles: tuple[str, ...],
    corpus: SketchCorpus,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    examples: list[Tensor] = []
    labels: list[float] = []
    matched: list[tuple[int, int]] = []
    for row in rows:
        question = str(row["question"])
        first_span, second_span, indices = nonoverlapping_title_pair(question, titles)
        property_text = remove_spans(question, (first_span, second_span))
        query = sketch_query(property_text, corpus.idf, len(titles))
        first = corpus.quantized_sketches[indices[0]]
        second = corpus.quantized_sketches[indices[1]]
        examples.append(symmetric_reader_features(query, first, second))
        labels.append(float(str(row["answer"]).lower() == "yes"))
        matched.append(indices)
    targets = torch.tensor(labels, dtype=torch.float64)
    return torch.stack(examples), targets, {
        "examples": len(rows),
        "positive_rate": float(targets.mean().item()),
        "unique_title_pairs": len(set(matched)),
    }


def fit_logistic_reader(
    train_inputs: Tensor,
    train_targets: Tensor,
    evaluation_inputs: Tensor,
    evaluation_targets: Tensor,
) -> dict[str, object]:
    mean = train_inputs.mean(dim=0)
    standard_deviation = train_inputs.std(dim=0, unbiased=False)
    scale = torch.where(standard_deviation > 1e-12, standard_deviation, torch.ones_like(standard_deviation))
    train = (train_inputs - mean) / scale
    evaluation = (evaluation_inputs - mean) / scale
    weights = torch.zeros(train.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (weights, bias),
        lr=1.0,
        max_iter=LBFGS_ITERATIONS,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> Tensor:
        nonlocal evaluations
        optimizer.zero_grad()
        logits = train @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_targets)
        loss = loss + 0.5 * L2_COEFFICIENT * weights.square().sum()
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        terminal_logits = train @ weights + bias
        terminal_loss = float(
            (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    terminal_logits, train_targets
                )
                + 0.5 * L2_COEFFICIENT * weights.square().sum()
            ).item()
        )

        def score(inputs: Tensor, targets: Tensor) -> dict[str, object]:
            probabilities = torch.sigmoid(inputs @ weights + bias)
            predictions = probabilities >= 0.5
            target_bool = targets.bool()
            return {
                "examples": len(targets),
                "accuracy": float((predictions == target_bool).double().mean().item()),
                "errors": int((predictions != target_bool).sum().item()),
                "predicted_positive_rate": float(predictions.double().mean().item()),
                "positive_rate": float(targets.mean().item()),
            }

        return {
            "terminal_regularized_loss": terminal_loss,
            "closure_evaluations": evaluations,
            "weight_norm": float(weights.norm().item()),
            "finite": bool(torch.isfinite(weights).all() and torch.isfinite(bias)),
            "train": score(train, train_targets),
            "evaluation": score(evaluation, evaluation_targets),
        }


def state_schema(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())


@torch.no_grad()
def compile_physical_payload(
    model: t19a.t10.core.small.SharedInterpreterLM,
    title_data: t19a.TitleData,
    payloads: Tensor,
) -> dict[str, object]:
    token_result = t19a.compile_token_codes(model, title_data)
    writes = int(token_result["written_entries"])
    device = model.token.weight.device
    prototypes = title_data.prototypes.to(device)
    payload_values = payloads.to(device=device, dtype=torch.bfloat16).to(
        model.token.weight.dtype
    )
    for index in range(len(title_data.titles)):
        block = model.blocks[index // t19a.t10.core.FFN_WIDTH]
        channel = index % t19a.t10.core.FFN_WIDTH
        block.gate.weight[channel, : t19a.CODE_DIMS] = prototypes[index]
        block.gate.weight[channel, ADDRESS_CONST] = -0.5
        block.up.weight[channel, ADDRESS_CONST] = 1.0
        block.down.weight[
            PAYLOAD_START : PAYLOAD_START + PAYLOAD_DIMS, channel
        ] = payload_values[index]
        writes += t19a.CODE_DIMS + 1 + 1 + PAYLOAD_DIMS
    return {
        "title_code_writes": int(token_result["written_entries"]),
        "gate_key_writes": len(title_data.titles) * t19a.CODE_DIMS,
        "gate_threshold_writes": len(title_data.titles),
        "up_constant_writes": len(title_data.titles),
        "down_payload_writes": len(title_data.titles) * PAYLOAD_DIMS,
        "total_writes": writes,
        "blocks_used": math.ceil(len(title_data.titles) / t19a.t10.core.FFN_WIDTH),
        "channels_used": len(title_data.titles),
    }


def verify_inputs() -> dict[str, bool]:
    checks = {
        "candidate_corpus": sha256_file(t19a.CANDIDATE_CORPUS) == t19a.CANDIDATE_SHA256,
        "train_qa": sha256_file(TRAIN_QA) == TRAIN_QA_SHA256,
        "evaluation_qa": sha256_file(EVALUATION_QA) == t19a.EVALUATOR_SHA256,
        "natural_train": sha256_file(t19a.t10.TRAIN_FILE) == t19a.NATURAL_TRAIN_SHA256,
        "natural_validation": sha256_file(t19a.t10.VALIDATION_FILE)
        == t19a.NATURAL_VALIDATION_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def run(device: torch.device) -> dict[str, object]:
    input_checks = verify_inputs()
    documents = t19a.load_raw_documents(t19a.CANDIDATE_CORPUS)
    tokenizer = AutoTokenizer.from_pretrained(
        t19a.TOKENIZER, revision=t19a.TOKENIZER_REVISION
    )
    title_data = t19a.build_title_data(documents, tokenizer)
    corpus = build_sketch_corpus(documents)
    train_rows = load_qa(TRAIN_QA)
    evaluation_rows = load_qa(EVALUATION_QA)
    train_inputs, train_targets, train_manifest = build_reader_matrix(
        train_rows, title_data.titles, corpus
    )
    evaluation_inputs, evaluation_targets, evaluation_manifest = build_reader_matrix(
        evaluation_rows, title_data.titles, corpus
    )
    reader = fit_logistic_reader(
        train_inputs, train_targets, evaluation_inputs, evaluation_targets
    )

    control = t19a.t10.core.build_model(MODEL_SEED, device)
    candidate = t19a.t10.core.build_model(MODEL_SEED, device)
    base_hashes_equal = (
        t19a.t10.core.small.state_sha256(control)
        == t19a.t10.core.small.state_sha256(candidate)
    )
    control_schema = state_schema(control)
    physical = compile_physical_payload(
        candidate, title_data, corpus.quantized_sketches
    )
    candidate_schema = state_schema(candidate)
    parameter_counts = {
        sum(parameter.numel() for parameter in control.parameters()),
        sum(parameter.numel() for parameter in candidate.parameters()),
    }
    compiler_fields = set().union(*(document.keys() for document in documents))
    evaluation_accuracy = float(reader["evaluation"]["accuracy"])
    gates = {
        "input_integrity": all(input_checks.values()),
        "raw_only_compiler_fields": compiler_fields
        == {"document_id", "title", "text"},
        "compiler_signature_raw_only": tuple(
            inspect.signature(build_sketch_corpus).parameters
        )
        == ("documents",),
        "two_titles_every_question": train_manifest["examples"] == 146
        and evaluation_manifest["examples"] == 104,
        "write_count_exact": physical["total_writes"] == EXPECTED_WRITES,
        "write_budget": physical["total_writes"] <= WRITE_BUDGET,
        "ordinary_served_structure_identical": base_hashes_equal
        and control_schema == candidate_schema
        and len(parameter_counts) == 1
        and next(iter(parameter_counts)) == t19a.t10.core.small.parameter_count()
        and type(control) is type(candidate),
        "payloads_finite_nonzero": bool(
            torch.isfinite(corpus.sketches).all()
            and torch.isfinite(corpus.quantized_sketches).all()
            and (corpus.quantized_sketches.norm(dim=-1) > 0).all()
        ),
        "bf16_cosine_at_least_0p9999": corpus.minimum_bf16_cosine >= 0.9999,
        "reader_evaluation_at_least_75pct": evaluation_accuracy >= 0.75,
        "reader_beats_t12_semantic_by_10pp": evaluation_accuracy
        >= T12_SEMANTIC_ACCURACY + 0.10,
        "both_classes": {float(value) for value in train_targets.tolist()}
        == {0.0, 1.0}
        and {float(value) for value in evaluation_targets.tolist()} == {0.0, 1.0},
        "finite_no_retry_reader": reader["finite"] is True,
    }
    return {
        "schema": "raw-countsketch-payload-t19b-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "configuration": {
            "documents": len(documents),
            "titles": len(title_data.titles),
            "active_title_token_rows": len(title_data.active_tokens),
            "payload_dimensions": PAYLOAD_DIMS,
            "payload_start": PAYLOAD_START,
            "address_dimensions": t19a.CODE_DIMS,
            "write_budget": WRITE_BUDGET,
            "expected_writes": EXPECTED_WRITES,
            "reader_feature_dimensions": train_inputs.shape[1],
            "l2_coefficient": L2_COEFFICIENT,
            "lbfgs_iterations": LBFGS_ITERATIONS,
        },
        "corpus": {
            "unique_features": len(corpus.idf),
            "minimum_bf16_cosine": corpus.minimum_bf16_cosine,
            "minimum_fp32_norm": float(corpus.sketches.norm(dim=-1).min().item()),
            "minimum_bf16_norm": float(
                corpus.quantized_sketches.norm(dim=-1).min().item()
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
            "candidate_corpus_sha256": sha256_file(t19a.CANDIDATE_CORPUS),
            "train_qa_sha256": sha256_file(TRAIN_QA),
            "evaluation_qa_sha256": sha256_file(EVALUATION_QA),
        },
        "claim_boundary": (
            "This is a raw-payload sufficiency and physical-write screen. It does "
            "not train an LM or demonstrate deployed knowledge/reasoning."
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
                "physical_write_ledger": result["physical_write_ledger"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
