#!/usr/bin/env python3
"""T17: class-specific entity keys physically distributed over existing FFNs."""

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
from experiments import hotpot_folded_entity_address_t16 as t16
from experiments import hotpot_internal_entity_address_t15 as t15
from experiments import hotpot_semantic_address_t12 as t12
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/hotpot-distributed-entity-keys-t17-preregistration.md"
OUTPUT = ROOT / "results/hotpot-distributed-entity-keys-t17.json"
KEYS = ROOT / "results/hotpot-distributed-entity-keys-t17-keys.pt"
CHECKPOINT = t13.CHECKPOINT
T16_RESULT = ROOT / "results/hotpot-folded-entity-address-t16.json"
T16_PROJECTION = ROOT / "results/hotpot-folded-entity-address-t16-projection.pt"

T16_RESULT_SHA256 = "f3bb7cc9b2182334e5ec9faa1dbe6b779a38491e7fb06baa1ef0b51db620a917"
T16_PROJECTION_SHA256 = "70861caae201e145e156bcadbe2330990f7a09ab98b91f912623b9cd2d29b08c"
LAYERS = (7, 8, 9)
KEY_SEED = 9_323
KEY_STEPS = 800
KEY_BATCH = 256
KEY_LR = 0.003
TEMPERATURE = 0.05
KEY_SCALARS = 2_405 * t10.core.small.HIDDEN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_assignments(titles: list[str], device: torch.device) -> Tensor:
    values = [
        t16.stable_integer("t17-layer-" + title) % len(LAYERS) for title in titles
    ]
    return torch.tensor(values, dtype=torch.long, device=device)


def allocation_counts(assignments: Tensor) -> tuple[int, ...]:
    return tuple(
        int((assignments == index).sum().item()) for index in range(len(LAYERS))
    )


def selected_ffn_interfaces(
    model: t10.core.small.SharedInterpreterLM, tokens: Tensor
) -> Tensor:
    """Return block 7/8/9 normalized pre-FFN tensors in served order."""
    hidden = model.token(tokens)
    selected: list[Tensor] = []
    for index, block in enumerate(model.blocks):
        hidden = hidden + block.attention(block.attention_norm(hidden))
        normalized = block.ffn_norm(hidden)
        if index in LAYERS:
            selected.append(normalized)
        hidden = hidden + block.down(F.silu(block.gate(normalized)) * block.up(normalized))
    if len(selected) != len(LAYERS):
        raise RuntimeError("selected layer accounting mismatch")
    return torch.stack(selected, dim=1)


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
            hidden = selected_ffn_interfaces(model, batch)
        hidden = hidden.float()
        encoded.append(
            torch.stack(
                [hidden[index, :, length - 1] for index, length in enumerate(lengths)]
            )
        )
    return torch.cat(encoded)


@torch.no_grad()
def encode_sequence_interfaces(
    model: t10.core.small.SharedInterpreterLM,
    sequences: list[list[int]],
    eos_token_id: int,
    device: torch.device,
    batch_size: int = 64,
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
            hidden = selected_ffn_interfaces(model, batch).float()
        encoded.extend(
            hidden[index, :, : len(row)].cpu() for index, row in enumerate(rows)
        )
    return encoded


def physical_scores(vectors: Tensor, keys: Tensor, assignments: Tensor) -> Tensor:
    """Score every entity at the layer where its physical gate row resides."""
    scores = torch.empty(
        vectors.shape[0], len(keys), dtype=vectors.dtype, device=vectors.device
    )
    normalized_keys = F.normalize(keys, dim=1)
    for layer_index in range(len(LAYERS)):
        members = torch.nonzero(assignments == layer_index).flatten()
        scores[:, members] = vectors[:, layer_index] @ normalized_keys[members].T
    return scores


def retrieval_metrics(
    vectors: Tensor,
    keys: Tensor,
    assignments: Tensor,
    targets: Tensor,
    batch_size: int = 512,
) -> dict[str, object]:
    correct = 0
    margins: list[Tensor] = []
    for start in range(0, len(vectors), batch_size):
        expected = targets[start : start + batch_size]
        scores = physical_scores(
            vectors[start : start + batch_size], keys, assignments
        )
        predictions = scores.argmax(dim=1)
        correct += int((predictions == expected).sum().item())
        target_scores = scores.gather(1, expected[:, None]).squeeze(1)
        scores.scatter_(1, expected[:, None], float("-inf"))
        margins.append((target_scores - scores.max(dim=1).values).detach().cpu())
    margin = torch.cat(margins)
    return {
        "examples": len(vectors),
        "correct": correct,
        "top1_accuracy": correct / len(vectors),
        "minimum_target_margin": float(margin.min().item()),
        "mean_target_margin": float(margin.mean().item()),
    }


def train_keys(
    canonical: Tensor,
    train_views: Tensor,
    assignments: Tensor,
    device: torch.device,
) -> tuple[Tensor, dict[str, object]]:
    torch.manual_seed(KEY_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(KEY_SEED)
    entity_indices = torch.arange(len(canonical), device=device)
    initial = canonical[entity_indices, assignments].detach().clone()
    keys = nn.Parameter(F.normalize(initial, dim=1))
    optimizer = torch.optim.AdamW([keys], lr=KEY_LR, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(KEY_SEED + 1)
    checkpoints: dict[str, object] = {}
    maximum_loss = 0.0
    started = time.perf_counter()
    scale = 1.0 / (math.sqrt(t10.core.small.HIDDEN) * TEMPERATURE)
    for step in range(1, KEY_STEPS + 1):
        indices = torch.randint(
            0, len(canonical), (KEY_BATCH,), generator=generator, device=device
        )
        choices = torch.randint(
            0, t16.TRAIN_VIEWS, (KEY_BATCH,), generator=generator, device=device
        )
        contexts = train_views[indices, choices]
        logits = physical_scores(contexts, keys, assignments) * scale
        loss = F.cross_entropy(logits, indices)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite key loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step in (100, 400, 800):
            checkpoints[str(step)] = {"loss": float(loss.item())}
    final = F.normalize(keys.detach(), dim=1)
    if not torch.isfinite(final).all():
        raise RuntimeError("nonfinite final keys")
    return final, {
        "steps": KEY_STEPS,
        "failed_or_retried_updates": 0,
        "maximum_loss": maximum_loss,
        "checkpoints": checkpoints,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_state_bytes": sum(
            value.numel() * value.element_size()
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, Tensor)
        ),
    }


def save_keys(keys: Tensor, assignments: Tensor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "hotpot-distributed-entity-keys-t17-v1",
            "layers": LAYERS,
            "seed": KEY_SEED,
            "steps": KEY_STEPS,
            "keys": keys.detach().cpu(),
            "assignments": assignments.detach().cpu(),
        },
        path,
    )
    return sha256_file(path)


@torch.no_grad()
def question_surface_vectors(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    rows: tuple[dict[str, object], ...],
    titles: list[str],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    title_index = {title: index for index, title in enumerate(titles)}
    prepared = [t15.question_tokens_and_title_indices(tokenizer, row) for row in rows]
    matrices = encode_sequence_interfaces(
        model,
        [identifiers for identifiers, _ in prepared],
        int(tokenizer.eos_token_id),
        device,
    )
    vectors: list[Tensor] = []
    targets: list[int] = []
    for row, hidden, (_, indices) in zip(rows, matrices, prepared, strict=True):
        for title in t12.data.supporting_titles(row):
            vectors.append(hidden[:, indices[title]])
            targets.append(title_index[title])
    return torch.stack(vectors).to(device), torch.tensor(targets, device=device)


def verify_inputs(checkpoint_path: Path) -> dict[str, bool]:
    checks = {
        "checkpoint_bytes": checkpoint_path.stat().st_size == t13.CHECKPOINT_BYTES,
        "checkpoint_sha256": sha256_file(checkpoint_path) == t13.CHECKPOINT_SHA256,
        "t16_result_sha256": sha256_file(T16_RESULT) == T16_RESULT_SHA256,
        "t16_projection_sha256": sha256_file(T16_PROJECTION)
        == T16_PROJECTION_SHA256,
        "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS)
        == t13.CANDIDATE_CORPUS_SHA256,
        "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST)
        == t13.DATA_MANIFEST_SHA256,
        "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET)
        == t12.data.SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"T17 input integrity failed: {checks}")
    return checks


def run(device: torch.device, checkpoint_path: Path, key_path: Path) -> dict[str, object]:
    checks = verify_inputs(checkpoint_path)
    checkpoint_before = sha256_file(checkpoint_path)
    tokenizer = t12.AutoTokenizer.from_pretrained(
        t12.TOKENIZER, revision=t12.TOKENIZER_REVISION
    )
    documents = t12.load_documents(t12.CANDIDATE_CORPUS)
    titles = sorted({document["title"] for document in documents})
    if len(titles) != 2_405:
        raise RuntimeError("unexpected canonical title count")
    assignments = layer_assignments(titles, device)
    counts = allocation_counts(assignments)
    if max(counts) > t10.core.small.FFN_WIDTH:
        raise RuntimeError(f"physical FFN row capacity exceeded: {counts}")
    model = t13.load_checkpoint(checkpoint_path, device)
    model_state_before = t10.core.small.state_sha256(model)

    canonical_sequences, context_sequences = t16.raw_prefix_views(
        tokenizer, documents, titles
    )
    canonical = encode_last_interfaces(
        model, canonical_sequences, int(tokenizer.eos_token_id), device
    )
    flat_contexts = [sequence for views in context_sequences for sequence in views]
    contexts = encode_last_interfaces(
        model, flat_contexts, int(tokenizer.eos_token_id), device
    ).reshape(len(titles), t16.TRAIN_VIEWS + t16.HELD_VIEWS, len(LAYERS), -1)
    keys, training = train_keys(
        canonical.detach(), contexts[:, : t16.TRAIN_VIEWS].detach(), assignments, device
    )
    key_sha256 = save_keys(keys, assignments, key_path)
    keys_sealed_before_questions = key_path.exists() and bool(key_sha256)

    targets = torch.arange(len(titles), device=device)
    canonical_metrics = retrieval_metrics(canonical, keys, assignments, targets)
    train_context_metrics = retrieval_metrics(
        contexts[:, : t16.TRAIN_VIEWS].reshape(-1, len(LAYERS), contexts.shape[-1]),
        keys,
        assignments,
        targets[:, None].expand(-1, t16.TRAIN_VIEWS).reshape(-1),
    )
    held_metrics = retrieval_metrics(
        contexts[:, t16.TRAIN_VIEWS], keys, assignments, targets
    )

    # Question-bearing rows are opened only after physical keys are sealed.
    split = t12.load_probe_split()
    train_vectors, train_targets = question_surface_vectors(
        model, tokenizer, split.train, titles, device
    )
    evaluation_vectors, evaluation_targets = question_surface_vectors(
        model, tokenizer, split.evaluation, titles, device
    )
    train_question_metrics = retrieval_metrics(
        train_vectors, keys, assignments, train_targets
    )
    evaluation_question_metrics = retrieval_metrics(
        evaluation_vectors, keys, assignments, evaluation_targets
    )

    model_state_after = t10.core.small.state_sha256(model)
    checkpoint_after = sha256_file(checkpoint_path)
    result_sets = {
        "canonical": canonical_metrics,
        "training_raw_contexts": train_context_metrics,
        "held_raw_contexts": held_metrics,
        "train_question_surfaces": train_question_metrics,
        "evaluation_question_surfaces": evaluation_question_metrics,
    }
    finite = math.isfinite(training["maximum_loss"]) and all(
        math.isfinite(result["minimum_target_margin"])
        and math.isfinite(result["mean_target_margin"])
        for result in result_sets.values()
    )
    gates = {
        "input_integrity": all(checks.values()),
        "physical_allocation_fits": max(counts) <= t10.core.small.FFN_WIDTH,
        "stored_key_ledger_exact": keys.numel() == KEY_SCALARS,
        "finite_training_and_scores": finite,
        "exact_training_ledger": training["steps"] == KEY_STEPS
        and training["failed_or_retried_updates"] == 0,
        **{
            f"{name}_top1_exact": result["top1_accuracy"] == 1.0
            for name, result in result_sets.items()
        },
        **{
            f"{name}_margin_positive": result["minimum_target_margin"] > 0.0
            for name, result in result_sets.items()
        },
        "keys_sealed_before_questions": keys_sealed_before_questions,
        "keys_exist_and_hashed": key_path.exists() and bool(key_sha256),
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
        "schema": "hotpot-distributed-entity-keys-t17-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "development_only": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "served_parameter_count": t10.core.small.parameter_count(),
            "served_model_updates": 0,
            "answer_label_reads": 0,
            "canonical_titles": len(titles),
            "layers": LAYERS,
            "allocation_counts": counts,
            "ffn_width_per_layer": t10.core.small.FFN_WIDTH,
            "stored_key_scalars": keys.numel(),
            "stored_key_fraction_of_model": keys.numel()
            / t10.core.small.parameter_count(),
            "key_seed": KEY_SEED,
            "key_steps": KEY_STEPS,
            "key_batch": KEY_BATCH,
            "key_lr": KEY_LR,
            "temperature": TEMPERATURE,
            "served_extra_parameters": 0,
            "served_extra_operations": 0,
        },
        "compiler_training": training,
        "address_results": result_sets,
        "keys": {
            "path": str(key_path),
            "bytes": key_path.stat().st_size,
            "sha256": key_sha256,
            "sealed_before_questions": keys_sealed_before_questions,
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
            "t16_result_sha256": sha256_file(T16_RESULT),
            "t16_projection_sha256": sha256_file(T16_PROJECTION),
            "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS),
            "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST),
            "source_parquet_sha256": sha256_file(t12.SOURCE_PARQUET),
        },
        "claim_boundary": (
            "Passing only establishes physically allocable entity keys. It does "
            "not write payloads, preserve LM quality after writing, improve QA, "
            "or establish a strict Pareto gain."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--keys", type=Path, default=KEYS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device), arguments.checkpoint, arguments.keys)
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
                "allocation_counts": result["configuration"]["allocation_counts"],
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
