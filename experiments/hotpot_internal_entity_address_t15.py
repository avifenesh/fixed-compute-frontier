#!/usr/bin/env python3
"""T15: internal contextual address gate for compiled entity records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import hotpot_contextual_address_t13 as t13
from experiments import hotpot_predictive_metric_t14 as t14
from experiments import hotpot_semantic_address_t12 as t12
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/hotpot-internal-entity-address-t15-preregistration.md"
OUTPUT = ROOT / "results/hotpot-internal-entity-address-t15.json"
CHECKPOINT = t13.CHECKPOINT
T14_RESULT = ROOT / "results/hotpot-predictive-metric-t14.json"
T14_RESULT_SHA256 = "4e80b58fb9d2d6372b8093792ba57c133eb6ea27b78ec01b63330325cd019839"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def title_sequences(tokenizer: object, titles: list[str]) -> list[list[int]]:
    sequences = [
        tokenizer.encode(" " + title, add_special_tokens=False)[-t13.CHUNK_TOKENS :]
        for title in titles
    ]
    if any(not sequence for sequence in sequences):
        raise RuntimeError("empty canonical title tokenization")
    return sequences


def question_tokens_and_title_indices(
    tokenizer: object,
    row: dict[str, object],
) -> tuple[list[int], dict[str, int]]:
    question = str(row["question"])
    encoded = tokenizer(
        question,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    identifiers = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    if len(identifiers) > t13.CHUNK_TOKENS:
        dropped = len(identifiers) - t13.CHUNK_TOKENS
        identifiers = identifiers[dropped:]
        offsets = offsets[dropped:]
    indices: dict[str, int] = {}
    for title in t12.data.supporting_titles(row):
        match = re.search(re.escape(title), question, flags=re.I)
        if match is None:
            raise RuntimeError(f"title absent from frozen question: {title}")
        overlapping = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < match.end() and end > match.start()
        ]
        if not overlapping:
            raise RuntimeError(f"title span has no token overlap: {title}")
        indices[title] = overlapping[-1]
    return identifiers, indices


@torch.inference_mode()
def canonical_keys(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    titles: list[str],
    device: torch.device,
) -> Tensor:
    encoded = t13.encode_sequences(
        model,
        title_sequences(tokenizer, titles),
        int(tokenizer.eos_token_id),
        device,
    )
    return torch.stack([hidden[-1] for hidden in encoded])


@torch.inference_mode()
def surface_probe(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    titles: list[str],
    rows: tuple[dict[str, object], ...],
    device: torch.device,
) -> dict[str, object]:
    keys = canonical_keys(model, tokenizer, titles, device)
    title_index = {title: index for index, title in enumerate(titles)}
    prepared = [question_tokens_and_title_indices(tokenizer, row) for row in rows]
    question_hidden = t13.encode_sequences(
        model,
        [identifiers for identifiers, _ in prepared],
        int(tokenizer.eos_token_id),
        device,
    )
    correct = 0
    margins: list[float] = []
    surfaces = 0
    for row, hidden, (_, indices) in zip(rows, question_hidden, prepared, strict=True):
        for title in t12.data.supporting_titles(row):
            query = hidden[indices[title]]
            scores = keys @ query
            target = title_index[title]
            prediction = int(scores.argmax().item())
            correct += prediction == target
            competitor = torch.cat((scores[:target], scores[target + 1 :])).max()
            margin = float((scores[target] - competitor).item())
            if not math.isfinite(margin):
                raise RuntimeError("nonfinite entity-address margin")
            margins.append(margin)
            surfaces += 1
    key_similarity = keys @ keys.T
    key_similarity.fill_diagonal_(-2.0)
    canonical_margins = 1.0 - key_similarity.max(dim=1).values
    return {
        "canonical_titles": len(titles),
        "canonical_minimum_margin": float(canonical_margins.min().item()),
        "canonical_collisions": int((canonical_margins <= 0.0).sum().item()),
        "surfaces": surfaces,
        "top1_accuracy": correct / surfaces,
        "correct": correct,
        "minimum_target_margin": min(margins),
        "mean_target_margin": sum(margins) / len(margins),
    }


def verify_inputs(checkpoint_path: Path) -> dict[str, bool]:
    checks = {
        "checkpoint_bytes": checkpoint_path.stat().st_size == t13.CHECKPOINT_BYTES,
        "checkpoint_sha256": sha256_file(checkpoint_path) == t13.CHECKPOINT_SHA256,
        "t14_result_sha256": sha256_file(T14_RESULT) == T14_RESULT_SHA256,
        "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS)
        == t13.CANDIDATE_CORPUS_SHA256,
        "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST)
        == t13.DATA_MANIFEST_SHA256,
        "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET)
        == t12.data.SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"T15 input integrity failed: {checks}")
    return checks


def evaluate_model(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    titles: list[str],
    split: t12.data.Split,
    device: torch.device,
) -> dict[str, object]:
    state_before = t10.core.small.state_sha256(model)
    train = surface_probe(model, tokenizer, titles, split.train, device)
    evaluation = surface_probe(model, tokenizer, titles, split.evaluation, device)
    state_after = t10.core.small.state_sha256(model)
    return {
        "train": train,
        "evaluation": evaluation,
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
    titles = sorted({document["title"] for document in documents})
    if len(titles) != 2_405:
        raise RuntimeError("unexpected canonical title count")
    split = t12.load_probe_split()

    trained_model = t13.load_checkpoint(checkpoint_path, device)
    trained = evaluate_model(trained_model, tokenizer, titles, split, device)
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random_model = t10.core.build_model(t12.MODEL_SEED, device).eval()
    if t10.core.small.state_sha256(random_model) != t13.INITIAL_STATE_SHA256:
        raise RuntimeError("initial model-state hash mismatch")
    random = evaluate_model(random_model, tokenizer, titles, split, device)
    del random_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_after = sha256_file(checkpoint_path)
    gates = {
        "input_integrity": all(checks.values()),
        "trained_canonical_keys_distinct": trained["train"]["canonical_collisions"]
        == 0
        and trained["evaluation"]["canonical_collisions"] == 0
        and trained["train"]["canonical_minimum_margin"] > 0.0
        and trained["evaluation"]["canonical_minimum_margin"] > 0.0,
        "trained_train_top1_exact": trained["train"]["top1_accuracy"] == 1.0,
        "trained_evaluation_top1_exact": trained["evaluation"]["top1_accuracy"]
        == 1.0,
        "trained_train_margin_positive": trained["train"]["minimum_target_margin"]
        > 0.0,
        "trained_evaluation_margin_positive": trained["evaluation"]["minimum_target_margin"]
        > 0.0,
        "trained_state_unchanged": trained["state_sha256_before"]
        == t13.TERMINAL_STATE_SHA256
        == trained["state_sha256_after"],
        "random_state_unchanged": random["state_sha256_before"]
        == t13.INITIAL_STATE_SHA256
        == random["state_sha256_after"],
        "checkpoint_byte_identical": checkpoint_before
        == t13.CHECKPOINT_SHA256
        == checkpoint_after,
        "zero_training_updates": True,
        "zero_answer_label_reads": True,
    }
    return {
        "schema": "hotpot-internal-entity-address-t15-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "development_only": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": t10.core.small.parameter_count(),
            "canonical_titles": len(titles),
            "interface": "final_block_pre_ffn_normalized_hidden",
            "canonical_form": "leading_space_plus_title_last_token",
            "surface_form": "full_question_title_span_last_token",
            "metric": "cosine",
            "training_updates": 0,
            "answer_label_reads": 0,
        },
        "entity_address": {
            "claim_boundary": (
                "Privileged target scoring using supporting-title identities, "
                "with no answer reads; not a QA or capability result."
            ),
            "trained": trained,
            "random_initialization": random,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "t14_result_sha256": sha256_file(T14_RESULT),
            "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS),
            "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST),
            "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET),
        },
        "claim_boundary": (
            "Passing only admits entity-record loading. It does not demonstrate "
            "record extraction, record capacity, QA gain, or strict Pareto gain."
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
                "trained_train_top1": result["entity_address"]["trained"]["train"]["top1_accuracy"],
                "trained_evaluation_top1": result["entity_address"]["trained"]["evaluation"]["top1_accuracy"],
                "random_evaluation_top1": result["entity_address"]["random_initialization"]["evaluation"]["top1_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
