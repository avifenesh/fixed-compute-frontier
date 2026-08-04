#!/usr/bin/env python3
"""T12: from-scratch semantic and compositional-address feasibility gate."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import heterogeneous_family_discovery_t7_scale as t7
from experiments import hotpot_real_prose_t12_data as data
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/hotpot-semantic-address-t12-preregistration.md"
OUTPUT = ROOT / "results/hotpot-semantic-address-t12.json"
CHECKPOINT = ROOT / "results/hotpot-semantic-address-t12-shared-base.pt"
DATA_MANIFEST = ROOT / "results/hotpot-real-prose-t12-data-manifest.json"
CANDIDATE_CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
SOURCE_PARQUET = data.SOURCE

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_SEED = 8_209
STEPS = 20_000
CHECKPOINTS = (5_000, 20_000)
BATCH_SIZE = 16
CONTEXT = 128
HOT_INTERVAL = 20
WARMUP_STEPS = 500
MUON_PEAK_LR = 0.005
ADAMW_PEAK_LR = 0.0003
ADDRESS_DIMS = 32
LEXICAL_REFERENCE_ACCURACY = 0.5576923076923077


def sha256_file(path: Path) -> str:
    return data.sha256_file(path)


@dataclass
class ArrayTokenStream:
    tokens: np.ndarray

    def batch(
        self, seed: int, batch_size: int, length: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        generator = np.random.default_rng(seed)
        starts = generator.integers(0, len(self.tokens) - length - 1, size=batch_size)
        rows = np.stack(
            [
                np.asarray(self.tokens[start : start + length + 1], dtype=np.int64)
                for start in starts
            ]
        )
        values = torch.from_numpy(rows).to(device)
        return values[:, :-1], values[:, 1:]


def load_documents(path: Path) -> list[dict[str, str]]:
    documents = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {"document_id", "title", "text"}
    if not documents or any(set(document) != expected for document in documents):
        raise RuntimeError("candidate corpus contains non-prose fields")
    return documents


def build_document_stream(
    documents: list[dict[str, str]], tokenizer: object
) -> ArrayTokenStream:
    values: list[int] = []
    eos = int(tokenizer.eos_token_id)
    for document in documents:
        values.extend(
            tokenizer.encode(
                document["title"] + "\n" + document["text"],
                add_special_tokens=False,
            )
        )
        values.append(eos)
    tokens = np.asarray(values, dtype="<u2")
    if int(tokens.max()) >= t10.core.small.VOCAB:
        raise RuntimeError("document token outside model vocabulary")
    return ArrayTokenStream(tokens)


def schedule_fraction(step: int) -> float:
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return 0.1 + 0.9 * cosine


def set_learning_rate(optimizer: t7.OptimizerBundle, step: int) -> None:
    fraction = schedule_fraction(step)
    if optimizer.matrix is not None:
        for group in optimizer.matrix.param_groups:
            group["lr"] = MUON_PEAK_LR * fraction
    for group in optimizer.auxiliary.param_groups:
        group["lr"] = ADAMW_PEAK_LR * fraction


@torch.no_grad()
def natural_nll(
    model: t10.core.small.SharedInterpreterLM,
    validation: t10.core.small.TokenStream,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    return t10.core.small.evaluate_natural(
        model, validation, device, seed, batches=32
    )


def train_shared_base(
    natural: t10.core.small.TokenStream,
    documents: ArrayTokenStream,
    validation: t10.core.small.TokenStream,
    device: torch.device,
) -> tuple[t10.core.small.SharedInterpreterLM, dict[str, object]]:
    """Train only on protected natural tokens and label-blind candidate prose."""
    model = t10.core.build_model(MODEL_SEED, device)
    initial_state_hash = t10.core.small.state_sha256(model)
    initial_natural = natural_nll(
        model, validation, device, t10.core.EVALUATION_SEED + MODEL_SEED
    )
    optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=MUON_PEAK_LR
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    maximum_loss = 0.0
    maximum_gradient = 0.0
    failed_or_retried = 0
    natural_tokens = 0
    document_tokens = 0
    checkpoints: dict[str, object] = {}
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad()
        set_learning_rate(optimizer, step)
        stream = documents if step % HOT_INTERVAL == 0 else natural
        inputs, targets = stream.batch(
            MODEL_SEED * 1_000_003 + step, BATCH_SIZE, CONTEXT, device
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
        loss = F.cross_entropy(
            logits.float().reshape(-1, t10.core.small.VOCAB), targets.reshape(-1)
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite shared-base loss at step {step}")
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient):
            raise RuntimeError(f"nonfinite shared-base gradient at step {step}")
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if stream is documents:
            document_tokens += BATCH_SIZE * CONTEXT
        else:
            natural_tokens += BATCH_SIZE * CONTEXT
        if step in CHECKPOINTS:
            checkpoints[str(step)] = natural_nll(
                model,
                validation,
                device,
                t10.core.EVALUATION_SEED + MODEL_SEED + step,
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "natural_nll": checkpoints[str(step)]["nll"],
                        "maximum_loss": maximum_loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    terminal_state_hash = t10.core.small.state_sha256(model)
    return model, {
        "initial_state_sha256": initial_state_hash,
        "terminal_state_sha256": terminal_state_hash,
        "initial_natural": initial_natural,
        "checkpoints": checkpoints,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
        "failed_or_retried_steps": failed_or_retried,
        "natural_token_presentations": natural_tokens,
        "document_token_presentations": document_tokens,
        "total_token_presentations": natural_tokens + document_tokens,
        "elapsed_seconds": elapsed,
        "peak_hbm_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "optimizer_state_bytes": t7.optimizer_state_bytes(optimizer),
    }


def load_probe_split() -> data.Split:
    import pyarrow.parquet as pq

    rows = pq.read_table(SOURCE_PARQUET).to_pylist()
    return data.split_rows(rows)


def documents_by_title(documents: list[dict[str, str]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for document in documents:
        grouped[document["title"]].append(document["text"])
    return {title: "\n".join(sorted(set(texts))) for title, texts in grouped.items()}


def property_words(row: dict[str, object]) -> list[str]:
    question = str(row["question"]).lower()
    for title in sorted(data.supporting_titles(row), key=len, reverse=True):
        question = re.sub(re.escape(title.lower()), " ", question, flags=re.I)
    return [
        word
        for word in data.word_tokens(question)
        if word not in data.STOPWORDS
    ]


@torch.no_grad()
def semantic_scores(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    documents: dict[str, str],
    rows: tuple[dict[str, object], ...],
    device: torch.device,
) -> list[tuple[float, bool]]:
    embedding = F.normalize(model.token.weight.float(), dim=1)
    document_ids: dict[str, Tensor] = {}
    for title in {title for row in rows for title in data.supporting_titles(row)}:
        if title not in documents:
            raise RuntimeError(f"supporting title absent from candidate corpus: {title}")
        identifiers = tokenizer.encode(documents[title], add_special_tokens=False)[:512]
        document_ids[title] = torch.tensor(identifiers, dtype=torch.long, device=device)
    results: list[tuple[float, bool]] = []
    for row in rows:
        words = property_words(row)
        identifiers = tokenizer.encode(" " + " ".join(words), add_special_tokens=False)
        if not identifiers:
            raise RuntimeError(f"empty property phrase for {row['id']}")
        query = embedding[
            torch.tensor(identifiers, dtype=torch.long, device=device)
        ]
        page_scores = []
        for title in data.supporting_titles(row):
            page = embedding[document_ids[title]]
            page_scores.append(float((query @ page.T).max(dim=1).values.mean().item()))
        results.append((min(page_scores), str(row["answer"]).lower() == "yes"))
    return results


def fit_threshold(scores: list[tuple[float, bool]]) -> tuple[float, bool, float]:
    candidates = sorted({-1.0, 1.0, *(value for value, _ in scores)})
    best: tuple[float, float, bool, float] | None = None
    for threshold in candidates:
        for greater_equal in (True, False):
            accuracy = sum(
                ((value >= threshold) if greater_equal else (value <= threshold))
                == target
                for value, target in scores
            ) / len(scores)
            choice = (accuracy, -abs(threshold), greater_equal, threshold)
            if best is None or choice > best:
                best = choice
    assert best is not None
    return best[3], best[2], best[0]


def score_threshold(
    scores: list[tuple[float, bool]], threshold: float, greater_equal: bool
) -> dict[str, object]:
    predictions = [
        ((value >= threshold) if greater_equal else (value <= threshold), target)
        for value, target in scores
    ]
    return {
        "examples": len(predictions),
        "accuracy": sum(prediction == target for prediction, target in predictions)
        / len(predictions),
        "positive_rate": sum(target for _, target in predictions) / len(predictions),
        "predicted_positive_rate": sum(prediction for prediction, _ in predictions)
        / len(predictions),
    }


@torch.no_grad()
def semantic_probe(
    model: t10.core.small.SharedInterpreterLM,
    tokenizer: object,
    documents: list[dict[str, str]],
    split: data.Split,
    device: torch.device,
) -> dict[str, object]:
    by_title = documents_by_title(documents)
    train = semantic_scores(model, tokenizer, by_title, split.train, device)
    evaluation = semantic_scores(
        model, tokenizer, by_title, split.evaluation, device
    )
    threshold, direction, train_accuracy = fit_threshold(train)
    return {
        "threshold": threshold,
        "greater_equal": direction,
        "train": score_threshold(train, threshold, direction),
        "evaluation": score_threshold(evaluation, threshold, direction),
        "fitted_train_accuracy": train_accuracy,
    }


def deterministic_token_code(token_id: int, dimensions: int = ADDRESS_DIMS) -> Tensor:
    digest = b""
    counter = 0
    while len(digest) * 8 < dimensions:
        digest += hashlib.sha256(
            f"t12-token-{token_id}-{counter}".encode()
        ).digest()
        counter += 1
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:dimensions]
    return torch.from_numpy(bits.astype(np.float32) * 2.0 - 1.0)


@torch.no_grad()
def compositional_address_probe(
    tokenizer: object,
    documents: list[dict[str, str]],
    split: data.Split,
    device: torch.device,
) -> dict[str, object]:
    titles = sorted({document["title"] for document in documents})
    title_index = {title: index for index, title in enumerate(titles)}
    cache: dict[int, Tensor] = {}

    def encode_phrase(text: str) -> Tensor:
        identifiers = tokenizer.encode(" " + text.strip(), add_special_tokens=False)
        codes = [
            cache.setdefault(identifier, deterministic_token_code(identifier).to(device))
            for identifier in identifiers
        ]
        return F.normalize(torch.stack(codes).mean(dim=0), dim=0)

    canonical = torch.stack([encode_phrase(title) for title in titles])
    similarities = canonical @ canonical.T
    similarities.fill_diagonal_(-2.0)
    canonical_margins = 1.0 - similarities.max(dim=1).values
    surface_margins: list[float] = []
    correct = 0
    surfaces = 0
    for row in split.evaluation:
        question = str(row["question"])
        for title in data.supporting_titles(row):
            match = re.search(re.escape(title), question, flags=re.I)
            if match is None:
                raise RuntimeError(f"title absent from frozen question: {title}")
            query = encode_phrase(match.group(0))
            scores = canonical @ query
            target = title_index[title]
            prediction = int(scores.argmax().item())
            correct += prediction == target
            surfaces += 1
            competitor = torch.cat((scores[:target], scores[target + 1 :])).max()
            surface_margins.append(float((scores[target] - competitor).item()))
    return {
        "dimensions": ADDRESS_DIMS,
        "candidate_titles": len(titles),
        "token_rows": len(cache),
        "canonical_collisions": int((canonical_margins <= 0.0).sum().item()),
        "canonical_minimum_margin": float(canonical_margins.min().item()),
        "sealed_surfaces": surfaces,
        "sealed_top1_accuracy": correct / surfaces,
        "sealed_minimum_margin": min(surface_margins),
    }


def save_checkpoint(
    model: t10.core.small.SharedInterpreterLM, path: Path
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "schema": "hotpot-semantic-address-t12-shared-base-v1",
            "model_seed": MODEL_SEED,
            "steps": STEPS,
            "state_dict": state,
        },
        path,
    )
    return sha256_file(path)


def verify_inputs() -> dict[str, object]:
    manifest = json.loads(DATA_MANIFEST.read_text())
    if not manifest["all_gates_pass"]:
        raise RuntimeError("T12 data manifest did not pass")
    expected_files = manifest["files"]
    checks = {
        "candidate_corpus": sha256_file(CANDIDATE_CORPUS)
        == expected_files["candidate_raw_prose"]["sha256"],
        "source_parquet": sha256_file(SOURCE_PARQUET) == data.SOURCE_SHA256,
        "natural_train": sha256_file(t10.TRAIN_FILE)
        == "1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a",
        "natural_validation": sha256_file(t10.VALIDATION_FILE)
        == "889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b",
    }
    if not all(checks.values()):
        raise RuntimeError(f"T12 input integrity failed: {checks}")
    return {"manifest": manifest, "checks": checks}


def run(device: torch.device, checkpoint_path: Path) -> dict[str, object]:
    verified = verify_inputs()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, revision=TOKENIZER_REVISION
    )
    documents = load_documents(CANDIDATE_CORPUS)
    document_stream = build_document_stream(documents, tokenizer)
    natural = t10.core.small.TokenStream(t10.TRAIN_FILE)
    validation = t10.core.small.TokenStream(t10.VALIDATION_FILE)
    split = load_probe_split()

    model, training = train_shared_base(
        natural, document_stream, validation, device
    )
    # The checkpoint is sealed before any label-bearing probe runs.
    checkpoint_sha256 = save_checkpoint(model, checkpoint_path)
    trained_semantic = semantic_probe(model, tokenizer, documents, split, device)
    address = compositional_address_probe(
        tokenizer, documents, split, device
    )

    random_model = t10.core.build_model(MODEL_SEED, device)
    random_semantic = semantic_probe(
        random_model, tokenizer, documents, split, device
    )
    del random_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    final_nll = training["checkpoints"][str(STEPS)]["nll"]
    trained_accuracy = trained_semantic["evaluation"]["accuracy"]
    random_accuracy = random_semantic["evaluation"]["accuracy"]
    gates = {
        "input_integrity": all(verified["checks"].values()),
        "finite_training": math.isfinite(training["maximum_loss"])
        and math.isfinite(training["maximum_gradient_norm"]),
        "no_failed_or_retried_steps": training["failed_or_retried_steps"] == 0,
        "token_ledger_exact": training["total_token_presentations"]
        == STEPS * BATCH_SIZE * CONTEXT,
        "natural_improved": final_nll < training["initial_natural"]["nll"],
        "natural_below_5p5": final_nll < 5.5,
        "canonical_addresses_distinct": address["canonical_collisions"] == 0
        and address["canonical_minimum_margin"] > 0.0,
        "sealed_entity_surface_top1_exact": address["sealed_top1_accuracy"] == 1.0
        and address["sealed_minimum_margin"] > 0.0,
        "trained_semantic_at_least_70pct": trained_accuracy >= 0.70,
        "trained_semantic_beats_lexical_by_10pp": trained_accuracy
        >= LEXICAL_REFERENCE_ACCURACY + 0.10,
        "trained_semantic_beats_random": trained_accuracy > random_accuracy,
        "checkpoint_exists_and_hashed": checkpoint_path.exists()
        and bool(checkpoint_sha256),
    }
    return {
        "schema": "hotpot-semantic-address-t12-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": t10.core.small.parameter_count(),
            "model_seed": MODEL_SEED,
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "context": CONTEXT,
            "hot_interval": HOT_INTERVAL,
            "muon_peak_lr": MUON_PEAK_LR,
            "adamw_peak_lr": ADAMW_PEAK_LR,
            "warmup_steps": WARMUP_STEPS,
            "candidate_documents": len(documents),
            "candidate_document_tokens": len(document_stream.tokens),
        },
        "training": training,
        "compositional_address": address,
        "semantic_probe": {
            "claim_boundary": (
                "Privileged post-checkpoint diagnostic using gold page identities "
                "and train-fitted threshold; not an admissible candidate."
            ),
            "random_initialization": random_semantic,
            "trained": trained_semantic,
            "privileged_lexical_reference_accuracy": LEXICAL_REFERENCE_ACCURACY,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "data_manifest_sha256": sha256_file(DATA_MANIFEST),
            "candidate_corpus_sha256": sha256_file(CANDIDATE_CORPUS),
            "source_parquet_sha256": sha256_file(SOURCE_PARQUET),
            "natural_train_sha256": sha256_file(t10.TRAIN_FILE),
            "natural_validation_sha256": sha256_file(t10.VALIDATION_FILE),
            "terminal_state_sha256": training["terminal_state_sha256"],
        },
        "claim_boundary": (
            "This is a from-scratch semantic/address feasibility gate. Passing "
            "admits compilation; it does not show autonomous extraction, natural "
            "question answering, or production superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
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
                "final_natural_nll": result["training"]["checkpoints"][str(STEPS)]["nll"],
                "semantic_evaluation_accuracy": result["semantic_probe"]["trained"]["evaluation"]["accuracy"],
                "entity_surface_top1": result["compositional_address"]["sealed_top1_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
