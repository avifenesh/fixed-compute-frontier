#!/usr/bin/env python3
"""T16: raw-prose-only train-time entity projection folded into served FFN keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import hotpot_contextual_address_t13 as t13
from experiments import hotpot_internal_entity_address_t15 as t15
from experiments import hotpot_semantic_address_t12 as t12
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/hotpot-folded-entity-address-t16-preregistration.md"
OUTPUT = ROOT / "results/hotpot-folded-entity-address-t16.json"
PROJECTION = ROOT / "results/hotpot-folded-entity-address-t16-projection.pt"
CHECKPOINT = t13.CHECKPOINT
T15_RESULT = ROOT / "results/hotpot-internal-entity-address-t15.json"

T15_RESULT_SHA256 = "b88c42915fd4b7e594a3fb337d3153c90db25be33fb3fce5a5a538c2044c36a0"
PROJECTION_DIMS = 64
PREFIX_TOKENS = 32
TRAIN_VIEWS = 4
HELD_VIEWS = 1
PROJECTION_SEED = 9_109
PROJECTION_STEPS = 800
PROJECTION_BATCH = 256
PROJECTION_LR = 0.003
TEMPERATURE = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_integer(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def raw_prefix_views(
    tokenizer: object,
    documents: list[dict[str, str]],
    titles: list[str],
) -> tuple[list[list[int]], list[list[list[int]]]]:
    document_tokens = [
        tokenizer.encode(document["text"], add_special_tokens=False)
        for document in documents
    ]
    canonical: list[list[int]] = []
    views: list[list[list[int]]] = []
    for title in titles:
        title_ids = tokenizer.encode(" " + title, add_special_tokens=False)
        if not title_ids:
            raise RuntimeError(f"empty title tokenization: {title}")
        canonical.append(title_ids[-t13.CHUNK_TOKENS :])
        title_views: list[list[int]] = []
        for view in range(TRAIN_VIEWS + HELD_VIEWS):
            salt = f"t16-prefix-{title}-{view}"
            source = stable_integer(salt) % len(documents)
            tokens = document_tokens[source]
            if not tokens:
                raise RuntimeError("empty raw-prose prefix source")
            maximum_start = max(len(tokens) - PREFIX_TOKENS, 0)
            start = (
                stable_integer(salt + "-offset") % (maximum_start + 1)
                if maximum_start
                else 0
            )
            prefix = tokens[start : start + PREFIX_TOKENS]
            title_views.append((prefix + title_ids)[-t13.CHUNK_TOKENS :])
        views.append(title_views)
    return canonical, views


@torch.no_grad()
def encode_last_interfaces(
    model: t10.core.small.SharedInterpreterLM,
    sequences: list[list[int]],
    eos_token_id: int,
    device: torch.device,
    batch_size: int = 64,
) -> Tensor:
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
        lengths = []
        for index, row in enumerate(rows):
            batch[index, : len(row)] = torch.tensor(row, device=device)
            lengths.append(len(row))
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden = t13.final_ffn_interface(model, batch)
        hidden = hidden.float()
        encoded.append(
            torch.stack([hidden[index, length - 1] for index, length in enumerate(lengths)])
        )
    return torch.cat(encoded)


def initialize_projection(device: torch.device) -> nn.Linear:
    torch.manual_seed(PROJECTION_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(PROJECTION_SEED)
    projection = nn.Linear(t10.core.small.HIDDEN, PROJECTION_DIMS, bias=False).to(device)
    nn.init.orthogonal_(projection.weight)
    return projection


def retrieval_metrics(
    vectors: Tensor,
    folded_keys: Tensor,
    targets: Tensor,
    batch_size: int = 512,
) -> dict[str, object]:
    correct = 0
    margins: list[Tensor] = []
    for start in range(0, len(vectors), batch_size):
        rows = vectors[start : start + batch_size]
        expected = targets[start : start + batch_size]
        scores = rows @ folded_keys.T
        predictions = scores.argmax(dim=1)
        correct += int((predictions == expected).sum().item())
        target_scores = scores.gather(1, expected[:, None]).squeeze(1)
        scores.scatter_(1, expected[:, None], float("-inf"))
        competitors = scores.max(dim=1).values
        margins.append((target_scores - competitors).detach().cpu())
    margin = torch.cat(margins)
    return {
        "examples": len(vectors),
        "correct": correct,
        "top1_accuracy": correct / len(vectors),
        "minimum_target_margin": float(margin.min().item()),
        "mean_target_margin": float(margin.mean().item()),
    }


def folded_keys(projection: nn.Linear, canonical: Tensor) -> Tensor:
    addresses = F.normalize(projection(canonical), dim=1)
    return addresses @ projection.weight


def train_projection(
    canonical: Tensor,
    train_views: Tensor,
    device: torch.device,
) -> tuple[nn.Linear, dict[str, object]]:
    projection = initialize_projection(device)
    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=PROJECTION_LR, weight_decay=0.0
    )
    generator = torch.Generator(device=device).manual_seed(PROJECTION_SEED + 1)
    maximum_loss = 0.0
    checkpoints: dict[str, object] = {}
    started = time.perf_counter()
    for step in range(1, PROJECTION_STEPS + 1):
        indices = torch.randint(
            0, len(canonical), (PROJECTION_BATCH,), generator=generator, device=device
        )
        choices = torch.randint(
            0, TRAIN_VIEWS, (PROJECTION_BATCH,), generator=generator, device=device
        )
        contexts = train_views[indices, choices]
        queries = F.normalize(projection(contexts), dim=1)
        keys = F.normalize(projection(canonical), dim=1)
        logits = queries @ keys.T / TEMPERATURE
        loss = F.cross_entropy(logits, indices)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite projection loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step in (100, 400, 800):
            checkpoints[str(step)] = {"loss": float(loss.item())}
    elapsed = time.perf_counter() - started
    if not torch.isfinite(projection.weight).all():
        raise RuntimeError("nonfinite trained projection")
    return projection, {
        "steps": PROJECTION_STEPS,
        "failed_or_retried_updates": 0,
        "maximum_loss": maximum_loss,
        "checkpoints": checkpoints,
        "elapsed_seconds": elapsed,
        "optimizer_state_bytes": sum(
            value.numel() * value.element_size()
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, Tensor)
        ),
    }


def save_projection(projection: nn.Linear, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "hotpot-folded-entity-address-t16-projection-v1",
            "dimensions": PROJECTION_DIMS,
            "seed": PROJECTION_SEED,
            "steps": PROJECTION_STEPS,
            "weight": projection.weight.detach().cpu(),
        },
        path,
    )
    return sha256_file(path)


@torch.inference_mode()
def question_surface_vectors(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    rows: tuple[dict[str, object], ...],
    titles: list[str],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    title_index = {title: index for index, title in enumerate(titles)}
    prepared = [t15.question_tokens_and_title_indices(tokenizer, row) for row in rows]
    matrices = t13.encode_sequences(
        model,
        [identifiers for identifiers, _ in prepared],
        int(tokenizer.eos_token_id),
        device,
    )
    vectors: list[Tensor] = []
    targets: list[int] = []
    for row, hidden, (_, indices) in zip(rows, matrices, prepared, strict=True):
        for title in t12.data.supporting_titles(row):
            vectors.append(hidden[indices[title]])
            targets.append(title_index[title])
    return torch.stack(vectors).to(device), torch.tensor(targets, device=device)


def verify_inputs(checkpoint_path: Path) -> dict[str, bool]:
    checks = {
        "checkpoint_bytes": checkpoint_path.stat().st_size == t13.CHECKPOINT_BYTES,
        "checkpoint_sha256": sha256_file(checkpoint_path) == t13.CHECKPOINT_SHA256,
        "t15_result_sha256": sha256_file(T15_RESULT) == T15_RESULT_SHA256,
        "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS)
        == t13.CANDIDATE_CORPUS_SHA256,
        "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST)
        == t13.DATA_MANIFEST_SHA256,
        "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET)
        == t12.data.SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"T16 input integrity failed: {checks}")
    return checks


def run(
    device: torch.device,
    checkpoint_path: Path,
    projection_path: Path,
) -> dict[str, object]:
    checks = verify_inputs(checkpoint_path)
    checkpoint_before = sha256_file(checkpoint_path)
    tokenizer = t12.AutoTokenizer.from_pretrained(
        t12.TOKENIZER, revision=t12.TOKENIZER_REVISION
    )
    documents = t12.load_documents(t12.CANDIDATE_CORPUS)
    titles = sorted({document["title"] for document in documents})
    if len(titles) != 2_405:
        raise RuntimeError("unexpected canonical title count")
    model = t13.load_checkpoint(checkpoint_path, device)
    model_state_before = t10.core.small.state_sha256(model)

    canonical_sequences, context_sequences = raw_prefix_views(
        tokenizer, documents, titles
    )
    canonical = encode_last_interfaces(
        model, canonical_sequences, int(tokenizer.eos_token_id), device
    )
    flat_contexts = [sequence for views in context_sequences for sequence in views]
    contexts = encode_last_interfaces(
        model, flat_contexts, int(tokenizer.eos_token_id), device
    ).reshape(len(titles), TRAIN_VIEWS + HELD_VIEWS, -1)
    projection, training = train_projection(
        canonical.detach(), contexts[:, :TRAIN_VIEWS].detach(), device
    )
    projection_sha256 = save_projection(projection, projection_path)
    projection_sealed_before_questions = projection_path.exists() and bool(
        projection_sha256
    )

    keys = folded_keys(projection, canonical).detach()
    target_all = torch.arange(len(titles), device=device)
    canonical_metrics = retrieval_metrics(canonical, keys, target_all)
    held_metrics = retrieval_metrics(contexts[:, TRAIN_VIEWS], keys, target_all)
    train_context_metrics = retrieval_metrics(
        contexts[:, :TRAIN_VIEWS].reshape(-1, contexts.shape[-1]),
        keys,
        target_all[:, None].expand(-1, TRAIN_VIEWS).reshape(-1),
    )

    # Question-bearing data is deliberately opened only after P is sealed.
    split = t12.load_probe_split()
    train_vectors, train_targets = question_surface_vectors(
        model, tokenizer, split.train, titles, device
    )
    evaluation_vectors, evaluation_targets = question_surface_vectors(
        model, tokenizer, split.evaluation, titles, device
    )
    train_question_metrics = retrieval_metrics(train_vectors, keys, train_targets)
    evaluation_question_metrics = retrieval_metrics(
        evaluation_vectors, keys, evaluation_targets
    )

    model_state_after = t10.core.small.state_sha256(model)
    checkpoint_after = sha256_file(checkpoint_path)
    algebra_generator = torch.Generator(device=device).manual_seed(PROJECTION_SEED + 2)
    sample = torch.randn(t10.core.small.HIDDEN, generator=algebra_generator, device=device)
    address = F.normalize(projection(canonical[:1]), dim=1)[0]
    folded = projection.weight.T @ address
    fold_error = float(
        torch.abs(sample @ folded - projection(sample) @ address).item()
    )
    finite = all(
        math.isfinite(value)
        for result in (
            canonical_metrics,
            held_metrics,
            train_context_metrics,
            train_question_metrics,
            evaluation_question_metrics,
        )
        for key, value in result.items()
        if key.endswith("margin")
    ) and math.isfinite(training["maximum_loss"])
    gates = {
        "input_integrity": all(checks.values()),
        "fold_algebra_exact": fold_error <= 1e-5,
        "finite_training_and_scores": finite,
        "exact_training_ledger": training["steps"] == PROJECTION_STEPS
        and training["failed_or_retried_updates"] == 0,
        "canonical_top1_exact": canonical_metrics["top1_accuracy"] == 1.0,
        "canonical_margin_positive": canonical_metrics["minimum_target_margin"]
        > 0.0,
        "held_raw_context_top1_exact": held_metrics["top1_accuracy"] == 1.0,
        "held_raw_context_margin_positive": held_metrics["minimum_target_margin"]
        > 0.0,
        "train_question_top1_exact": train_question_metrics["top1_accuracy"]
        == 1.0,
        "train_question_margin_positive": train_question_metrics[
            "minimum_target_margin"
        ]
        > 0.0,
        "evaluation_question_top1_exact": evaluation_question_metrics[
            "top1_accuracy"
        ]
        == 1.0,
        "evaluation_question_margin_positive": evaluation_question_metrics[
            "minimum_target_margin"
        ]
        > 0.0,
        "projection_sealed_before_questions": projection_sealed_before_questions,
        "projection_exists_and_hashed": projection_path.exists()
        and bool(projection_sha256),
        "served_model_state_unchanged": model_state_before
        == t13.TERMINAL_STATE_SHA256
        == model_state_after,
        "checkpoint_byte_identical": checkpoint_before
        == t13.CHECKPOINT_SHA256
        == checkpoint_after,
        "zero_answer_label_reads": True,
        "zero_served_model_updates": True,
    }
    return {
        "schema": "hotpot-folded-entity-address-t16-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "development_only": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "served_parameter_count": t10.core.small.parameter_count(),
            "served_model_updates": 0,
            "answer_label_reads": 0,
            "canonical_titles": len(titles),
            "projection_dimensions": PROJECTION_DIMS,
            "prefix_tokens": PREFIX_TOKENS,
            "train_views_per_title": TRAIN_VIEWS,
            "held_views_per_title": HELD_VIEWS,
            "projection_seed": PROJECTION_SEED,
            "projection_steps": PROJECTION_STEPS,
            "projection_batch": PROJECTION_BATCH,
            "projection_lr": PROJECTION_LR,
            "temperature": TEMPERATURE,
            "served_projection_operations": 0,
            "served_projection_parameters": 0,
        },
        "compiler_training": training,
        "address_results": {
            "canonical": canonical_metrics,
            "training_raw_contexts": train_context_metrics,
            "held_raw_contexts": held_metrics,
            "train_question_surfaces": train_question_metrics,
            "evaluation_question_surfaces": evaluation_question_metrics,
            "t15_identity_reference": {
                "train_question_top1": 0.8561643835616438,
                "evaluation_question_top1": 0.8125,
            },
        },
        "fold_contract": {
            "formula": "h_q^T P^T a_e == (P h_q)^T a_e",
            "absolute_error": fold_error,
        },
        "projection": {
            "path": str(projection_path),
            "bytes": projection_path.stat().st_size,
            "sha256": projection_sha256,
            "sealed_before_questions": projection_sealed_before_questions,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "served_state_sha256_before": model_state_before,
            "served_state_sha256_after": model_state_after,
            "t15_result_sha256": sha256_file(T15_RESULT),
            "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS),
            "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST),
            "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET),
        },
        "claim_boundary": (
            "Passing only establishes a foldable entity address. It does not "
            "compile records, improve QA, or establish a strict Pareto gain."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--projection", type=Path, default=PROJECTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device), arguments.checkpoint, arguments.projection)
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
                "held_raw_top1": result["address_results"]["held_raw_contexts"]["top1_accuracy"],
                "train_question_top1": result["address_results"]["train_question_surfaces"]["top1_accuracy"],
                "evaluation_question_top1": result["address_results"]["evaluation_question_surfaces"]["top1_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
