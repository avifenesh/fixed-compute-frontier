#!/usr/bin/env python3
"""T20a: raw first-order learning operators as 220-cell document records."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import self_latent_payload_t19c as t19c


t19b = t19c.t19b
PREREGISTRATION = ROOT / "results/functional-residual-operator-t20a-preregistration.md"
OUTPUT = ROOT / "results/functional-residual-operator-t20a.json"
SCHEMA = "functional-residual-operator-t20a-v1"
CHECKPOINT = t19c.CHECKPOINT
CHECKPOINT_SHA256 = t19c.CHECKPOINT_SHA256
CONTEXT = 128
BATCH_SIZE = 24
LEFT_RANK = 20
RIGHT_RANK = 11
PAYLOAD_DIMS = LEFT_RANK * RIGHT_RANK
RANDOM_SEED = 20_001
SHUFFLE_SEED = 20_005
T19B_ACCURACY = 0.6057692307692307
REQUIRED_ACCURACY = 0.75
REQUIRED_GAIN = 0.10


def sha256_file(path: Path) -> str:
    return t19b.sha256_file(path)


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def transition_batch(
    texts: list[str], tokenizer: object
) -> tuple[Tensor, Tensor, Tensor]:
    eos = int(tokenizer.eos_token_id)
    encoded: list[list[int]] = []
    lengths: list[int] = []
    for text in texts:
        tokens = [
            int(value)
            for value in tokenizer.encode(text, add_special_tokens=False)
        ]
        sequence = (tokens + [eos])[: CONTEXT + 1]
        if len(sequence) < 2:
            sequence = [eos, eos]
        length = len(sequence) - 1
        lengths.append(length)
        encoded.append(sequence + [eos] * (CONTEXT + 1 - len(sequence)))
    values = torch.tensor(encoded, dtype=torch.long)
    positions = torch.arange(CONTEXT)[None, :]
    mask = positions < torch.tensor(lengths, dtype=torch.long)[:, None]
    return values[:, :-1], values[:, 1:], mask


@torch.no_grad()
def learning_operator_batch(
    model: torch.nn.Module,
    texts: list[str],
    tokenizer: object,
    device: torch.device,
) -> Tensor:
    inputs, targets, mask = transition_batch(texts, tokenizer)
    inputs = inputs.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    embedding = model.token.weight
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        hidden = model.hidden(inputs)
        logits = hidden @ embedding.T
    probabilities = logits.float().softmax(dim=-1)
    del logits
    if device.type == "cuda":
        expected = probabilities.to(torch.bfloat16) @ embedding.to(torch.bfloat16)
    else:
        expected = probabilities @ embedding
    residual = F.embedding(targets, embedding).float() - expected.float()
    active = mask[..., None].float()
    operators = torch.einsum(
        "bth,btk->bhk", hidden.float() * active, residual * active
    )
    lengths = mask.sum(dim=1).clamp_min(1).float().sqrt()
    operators /= lengths[:, None, None]
    flat = operators.flatten(1)
    norms = flat.norm(dim=1)
    if not torch.isfinite(flat).all() or bool((norms == 0).any()):
        raise RuntimeError("nonfinite or zero functional residual operator")
    return (flat / norms[:, None]).view(
        len(texts), t19b.t19a.t10.core.HIDDEN, t19b.t19a.t10.core.HIDDEN
    )


def canonicalize_columns(matrix: Tensor) -> Tensor:
    result = matrix.clone()
    for column in range(result.shape[1]):
        values = result[:, column]
        pivot = int(values.abs().argmax().item())
        if float(values[pivot].item()) < 0.0:
            result[:, column].neg_()
    return result


def top_eigenspace(covariance: Tensor, rank: int) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    if not torch.isfinite(eigenvalues).all() or float(eigenvalues[-1]) <= 0.0:
        raise RuntimeError("invalid functional residual covariance spectrum")
    selected = eigenvectors[:, -rank:].flip(dims=(1,))
    return canonicalize_columns(selected)


def random_basis(dimensions: int, rank: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(dimensions, rank, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return canonicalize_columns(basis.float())


def project_operators(operators: Tensor, left: Tensor, right: Tensor) -> Tensor:
    projected = torch.einsum("hr,bhk,ks->brs", left, operators, right)
    flat = projected.flatten(1)
    norms = flat.norm(dim=1)
    if not torch.isfinite(flat).all() or bool((norms == 0).any()):
        raise RuntimeError("nonfinite or zero projected operator")
    return flat / norms[:, None]


@dataclass(frozen=True)
class OperatorPlane:
    payloads: Tensor
    quantized_payloads: Tensor
    random_payloads: Tensor
    quantized_random_payloads: Tensor
    left: Tensor
    right: Tensor
    random_left: Tensor
    random_right: Tensor
    minimum_bf16_cosine: float
    minimum_random_bf16_cosine: float
    hosvd_energy_fraction: float
    random_energy_fraction: float
    covariance_seconds: float
    projection_seconds: float


@torch.no_grad()
def build_raw_operator_plane(
    model: torch.nn.Module,
    documents: list[dict[str, str]],
    tokenizer: object,
    device: torch.device,
) -> OperatorPlane:
    """Compile operator payloads from the same raw-trained model and documents."""
    hidden = t19b.t19a.t10.core.HIDDEN
    texts = [document["text"] for document in documents]
    covariance_left = torch.zeros(hidden, hidden, device=device, dtype=torch.float32)
    covariance_right = torch.zeros_like(covariance_left)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(texts), BATCH_SIZE):
        operators = learning_operator_batch(
            model, texts[start : start + BATCH_SIZE], tokenizer, device
        )
        covariance_left += torch.einsum("bij,bkj->ik", operators, operators)
        covariance_right += torch.einsum("bij,bik->jk", operators, operators)
    left = top_eigenspace(covariance_left, LEFT_RANK)
    right = top_eigenspace(covariance_right, RIGHT_RANK)
    random_left = random_basis(hidden, LEFT_RANK, RANDOM_SEED).to(device)
    random_right = random_basis(hidden, RIGHT_RANK, RANDOM_SEED + 1).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    covariance_seconds = time.perf_counter() - started

    started = time.perf_counter()
    payloads: list[Tensor] = []
    random_payloads: list[Tensor] = []
    hosvd_energy = 0.0
    random_energy = 0.0
    for start in range(0, len(texts), BATCH_SIZE):
        operators = learning_operator_batch(
            model, texts[start : start + BATCH_SIZE], tokenizer, device
        )
        raw_hosvd = torch.einsum("hr,bhk,ks->brs", left, operators, right)
        raw_random = torch.einsum(
            "hr,bhk,ks->brs", random_left, operators, random_right
        )
        hosvd_energy += float(raw_hosvd.square().sum().item())
        random_energy += float(raw_random.square().sum().item())
        payloads.append(project_operators(operators, left, right).cpu())
        random_payloads.append(
            project_operators(operators, random_left, random_right).cpu()
        )
    values = torch.cat(payloads)
    random_values = torch.cat(random_payloads)
    quantized = values.to(torch.bfloat16).float()
    random_quantized = random_values.to(torch.bfloat16).float()
    cosine = F.cosine_similarity(values, quantized, dim=-1)
    random_cosine = F.cosine_similarity(random_values, random_quantized, dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    projection_seconds = time.perf_counter() - started
    return OperatorPlane(
        payloads=values,
        quantized_payloads=quantized,
        random_payloads=random_values,
        quantized_random_payloads=random_quantized,
        left=left.cpu(),
        right=right.cpu(),
        random_left=random_left.cpu(),
        random_right=random_right.cpu(),
        minimum_bf16_cosine=float(cosine.min().item()),
        minimum_random_bf16_cosine=float(random_cosine.min().item()),
        hosvd_energy_fraction=hosvd_energy / len(documents),
        random_energy_fraction=random_energy / len(documents),
        covariance_seconds=covariance_seconds,
        projection_seconds=projection_seconds,
    )


@dataclass(frozen=True)
class QuerySet:
    texts: tuple[str, ...]
    pairs: tuple[tuple[int, int], ...]
    targets: Tensor


def prepare_queries(
    rows: list[dict[str, object]], titles: tuple[str, ...]
) -> QuerySet:
    texts: list[str] = []
    pairs: list[tuple[int, int]] = []
    targets: list[float] = []
    for row in rows:
        question = str(row["question"])
        first_span, second_span, pair = t19b.nonoverlapping_title_pair(
            question, titles
        )
        texts.append(t19b.remove_spans(question, (first_span, second_span)))
        pairs.append(pair)
        targets.append(float(str(row["answer"]).lower() == "yes"))
    return QuerySet(
        texts=tuple(texts),
        pairs=tuple(pairs),
        targets=torch.tensor(targets, dtype=torch.float64),
    )


@torch.no_grad()
def encode_query_operators(
    model: torch.nn.Module,
    query_set: QuerySet,
    tokenizer: object,
    left: Tensor,
    right: Tensor,
    device: torch.device,
) -> Tensor:
    values: list[Tensor] = []
    left_device = left.to(device)
    right_device = right.to(device)
    for start in range(0, len(query_set.texts), BATCH_SIZE):
        operators = learning_operator_batch(
            model,
            list(query_set.texts[start : start + BATCH_SIZE]),
            tokenizer,
            device,
        )
        values.append(
            project_operators(operators, left_device, right_device).cpu()
        )
    return torch.cat(values).to(torch.bfloat16).float()


def build_reader_matrix(
    queries: Tensor, pairs: tuple[tuple[int, int], ...], payloads: Tensor
) -> Tensor:
    return torch.stack(
        [
            t19b.symmetric_reader_features(
                query, payloads[pair[0]], payloads[pair[1]]
            )
            for query, pair in zip(queries, pairs, strict=True)
        ]
    )


def build_shuffled_reader_matrix(
    queries: Tensor,
    pairs: tuple[tuple[int, int], ...],
    payloads: Tensor,
) -> tuple[Tensor, int]:
    generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
    flat = torch.tensor([value for pair in pairs for value in pair], dtype=torch.long)
    shuffled = flat[torch.randperm(len(flat), generator=generator)]
    shuffled_pairs = tuple(
        (int(shuffled[2 * index]), int(shuffled[2 * index + 1]))
        for index in range(len(pairs))
    )
    unchanged = sum(
        original == replacement
        for original, replacement in zip(pairs, shuffled_pairs, strict=True)
    )
    return build_reader_matrix(queries, shuffled_pairs, payloads), unchanged


@dataclass(frozen=True)
class FittedReader:
    metrics: dict[str, object]
    weights: Tensor
    bias: Tensor
    mean: Tensor
    scale: Tensor


def score_reader(
    inputs: Tensor, targets: Tensor, weights: Tensor, bias: Tensor
) -> dict[str, object]:
    probabilities = torch.sigmoid(inputs @ weights + bias)
    predictions = probabilities >= 0.5
    target_bool = targets.bool()
    return {
        "examples": len(targets),
        "accuracy": float((predictions == target_bool).double().mean().item()),
        "errors": int((predictions != target_bool).sum().item()),
        "predicted_positive_rate": float(predictions.double().mean().item()),
        "positive_rate": float(targets.mean().item()),
        "finite": bool(torch.isfinite(probabilities).all()),
    }


def fit_reader(
    train_inputs: Tensor,
    train_targets: Tensor,
    evaluation_inputs: Tensor,
    evaluation_targets: Tensor,
) -> FittedReader:
    mean = train_inputs.mean(dim=0)
    standard_deviation = train_inputs.std(dim=0, unbiased=False)
    scale = torch.where(
        standard_deviation > 1e-12,
        standard_deviation,
        torch.ones_like(standard_deviation),
    )
    train = (train_inputs - mean) / scale
    evaluation = (evaluation_inputs - mean) / scale
    weights = torch.zeros(train.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (weights, bias),
        lr=1.0,
        max_iter=t19b.LBFGS_ITERATIONS,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> Tensor:
        nonlocal evaluations
        optimizer.zero_grad()
        logits = train @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, train_targets)
        loss = loss + 0.5 * t19b.L2_COEFFICIENT * weights.square().sum()
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        terminal_logits = train @ weights + bias
        terminal_loss = float(
            (
                F.binary_cross_entropy_with_logits(
                    terminal_logits, train_targets
                )
                + 0.5 * t19b.L2_COEFFICIENT * weights.square().sum()
            ).item()
        )
        metrics = {
            "terminal_regularized_loss": terminal_loss,
            "closure_evaluations": evaluations,
            "weight_norm": float(weights.norm().item()),
            "finite": bool(
                torch.isfinite(weights).all() and torch.isfinite(bias)
            ),
            "train": score_reader(train, train_targets, weights, bias),
            "evaluation": score_reader(
                evaluation, evaluation_targets, weights, bias
            ),
        }
    return FittedReader(
        metrics=metrics,
        weights=weights.detach().clone(),
        bias=bias.detach().clone(),
        mean=mean,
        scale=scale,
    )


def orthonormal_error(matrix: Tensor) -> float:
    identity = torch.eye(matrix.shape[1], dtype=matrix.dtype)
    return float((matrix.T @ matrix - identity).abs().max().item())


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
    torch.set_float32_matmul_precision("high")
    input_checks = verify_inputs()
    documents = t19b.t19a.load_raw_documents(t19b.t19a.CANDIDATE_CORPUS)
    tokenizer = AutoTokenizer.from_pretrained(
        t19b.t19a.TOKENIZER, revision=t19b.t19a.TOKENIZER_REVISION
    )
    title_data = t19b.t19a.build_title_data(documents, tokenizer)
    writer, writer_metadata = t19c.load_writer(device)
    writer.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    plane = build_raw_operator_plane(writer, documents, tokenizer, device)

    train_rows = t19b.load_qa(t19b.TRAIN_QA)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    train_queries = prepare_queries(train_rows, title_data.titles)
    evaluation_queries = prepare_queries(evaluation_rows, title_data.titles)
    train_hosvd = encode_query_operators(
        writer, train_queries, tokenizer, plane.left, plane.right, device
    )
    evaluation_hosvd = encode_query_operators(
        writer, evaluation_queries, tokenizer, plane.left, plane.right, device
    )
    train_random = encode_query_operators(
        writer,
        train_queries,
        tokenizer,
        plane.random_left,
        plane.random_right,
        device,
    )
    evaluation_random = encode_query_operators(
        writer,
        evaluation_queries,
        tokenizer,
        plane.random_left,
        plane.random_right,
        device,
    )

    hosvd_train_matrix = build_reader_matrix(
        train_hosvd, train_queries.pairs, plane.quantized_payloads
    )
    hosvd_evaluation_matrix = build_reader_matrix(
        evaluation_hosvd, evaluation_queries.pairs, plane.quantized_payloads
    )
    random_train_matrix = build_reader_matrix(
        train_random, train_queries.pairs, plane.quantized_random_payloads
    )
    random_evaluation_matrix = build_reader_matrix(
        evaluation_random,
        evaluation_queries.pairs,
        plane.quantized_random_payloads,
    )
    shuffled_matrix, unchanged_pairs = build_shuffled_reader_matrix(
        evaluation_hosvd,
        evaluation_queries.pairs,
        plane.quantized_payloads,
    )
    hosvd_fit = fit_reader(
        hosvd_train_matrix,
        train_queries.targets,
        hosvd_evaluation_matrix,
        evaluation_queries.targets,
    )
    random_fit = fit_reader(
        random_train_matrix,
        train_queries.targets,
        random_evaluation_matrix,
        evaluation_queries.targets,
    )
    query_only_fit = fit_reader(
        train_hosvd.double(),
        train_queries.targets,
        evaluation_hosvd.double(),
        evaluation_queries.targets,
    )
    hosvd_reader = hosvd_fit.metrics
    random_reader = random_fit.metrics
    query_only_reader = query_only_fit.metrics

    with torch.no_grad():
        shuffled_standardized = (
            shuffled_matrix - hosvd_fit.mean
        ) / hosvd_fit.scale
        shuffled_logits = (
            shuffled_standardized @ hosvd_fit.weights + hosvd_fit.bias
        )
        shuffled_predictions = torch.sigmoid(shuffled_logits) >= 0.5
        shuffled_accuracy = float(
            (
                shuffled_predictions
                == evaluation_queries.targets.bool()
            ).double().mean().item()
        )

    control, _ = t19c.load_writer(device)
    installed, _ = t19c.load_writer(device)
    base_hashes_equal = (
        t19b.t19a.t10.core.small.state_sha256(control)
        == t19b.t19a.t10.core.small.state_sha256(installed)
    )
    control_schema = t19b.state_schema(control)
    physical = t19b.compile_physical_payload(
        installed, title_data, plane.quantized_payloads
    )
    installed_schema = t19b.state_schema(installed)
    parameter_counts = {
        sum(parameter.numel() for parameter in control.parameters()),
        sum(parameter.numel() for parameter in installed.parameters()),
    }

    hosvd_accuracy = float(hosvd_reader["evaluation"]["accuracy"])
    random_accuracy = float(random_reader["evaluation"]["accuracy"])
    query_accuracy = float(query_only_reader["evaluation"]["accuracy"])
    compiler_fields = set().union(*(document.keys() for document in documents))
    all_values = (
        plane.payloads,
        plane.quantized_payloads,
        plane.random_payloads,
        train_hosvd,
        evaluation_hosvd,
        plane.left,
        plane.right,
    )
    finite_values = all(torch.isfinite(value).all() for value in all_values)
    payload_norm_error = float(
        (plane.payloads.norm(dim=-1) - 1.0).abs().max().item()
    )
    gates = {
        "input_integrity": all(input_checks.values()),
        "same_model_raw_only_compiler": compiler_fields
        == {"document_id", "title", "text"}
        and tuple(inspect.signature(build_raw_operator_plane).parameters)
        == ("model", "documents", "tokenizer", "device"),
        "finite_nonzero_operators_and_readers": bool(
            finite_values
            and plane.payloads.norm(dim=-1).min() > 0
            and hosvd_reader["finite"]
            and random_reader["finite"]
            and query_only_reader["finite"]
            and torch.isfinite(shuffled_logits).all()
        ),
        "orthonormal_subspaces": orthonormal_error(plane.left) <= 1e-4
        and orthonormal_error(plane.right) <= 1e-4,
        "unit_payloads_and_bf16_fidelity": payload_norm_error <= 1e-4
        and plane.minimum_bf16_cosine >= 0.9999,
        "two_titles_and_both_classes": len(train_queries.pairs) == len(train_rows)
        and len(evaluation_queries.pairs) == len(evaluation_rows)
        and set(train_queries.targets.tolist()) == {0.0, 1.0}
        and set(evaluation_queries.targets.tolist()) == {0.0, 1.0},
        "identical_served_structure_and_write_budget": base_hashes_equal
        and control_schema == installed_schema
        and len(parameter_counts) == 1
        and physical["total_writes"] == t19b.EXPECTED_WRITES
        and physical["total_writes"] <= t19b.WRITE_BUDGET,
        "training_accuracy_at_least_95pct": float(
            hosvd_reader["train"]["accuracy"]
        )
        >= 0.95,
        "evaluation_accuracy_at_least_75pct": hosvd_accuracy
        >= REQUIRED_ACCURACY,
        "gain_over_t19b_at_least_10pp": hosvd_accuracy - T19B_ACCURACY
        >= REQUIRED_GAIN,
        "gain_over_random_at_least_10pp": hosvd_accuracy - random_accuracy
        >= REQUIRED_GAIN,
        "gain_over_query_only_at_least_10pp": hosvd_accuracy - query_accuracy
        >= REQUIRED_GAIN,
        "document_shuffle_drop_at_least_10pp": hosvd_accuracy
        - shuffled_accuracy
        >= REQUIRED_GAIN,
    }
    return {
        "schema": SCHEMA,
        "device": str(device),
        "input_checks": input_checks,
        "writer": writer_metadata,
        "compiler": {
            "document_fields": sorted(compiler_fields),
            "documents": len(documents),
            "context": CONTEXT,
            "left_rank": LEFT_RANK,
            "right_rank": RIGHT_RANK,
            "payload_dimensions": PAYLOAD_DIMS,
            "covariance_seconds": plane.covariance_seconds,
            "projection_seconds": plane.projection_seconds,
            "hosvd_energy_fraction": plane.hosvd_energy_fraction,
            "random_energy_fraction": plane.random_energy_fraction,
            "minimum_bf16_cosine": plane.minimum_bf16_cosine,
            "minimum_random_bf16_cosine": plane.minimum_random_bf16_cosine,
            "payload_norm_max_error": payload_norm_error,
            "left_orthonormal_max_error": orthonormal_error(plane.left),
            "right_orthonormal_max_error": orthonormal_error(plane.right),
            "left_sha256": tensor_sha256(plane.left),
            "right_sha256": tensor_sha256(plane.right),
            "payload_sha256": tensor_sha256(plane.quantized_payloads),
        },
        "physical": physical,
        "served": {
            "base_hashes_equal_before_write": base_hashes_equal,
            "identical_state_schema": control_schema == installed_schema,
            "parameter_count": next(iter(parameter_counts)),
            "parameter_count_variants": len(parameter_counts),
        },
        "reader": {
            "features": int(hosvd_train_matrix.shape[1]),
            "hosvd": hosvd_reader,
            "random_subspace": random_reader,
            "query_only": query_only_reader,
            "shuffled_evaluation": {
                "accuracy": shuffled_accuracy,
                "errors": int(
                    (
                        shuffled_predictions
                        != evaluation_queries.targets.bool()
                    ).sum().item()
                ),
                "unchanged_pairs": unchanged_pairs,
                "seed": SHUFFLE_SEED,
            },
        },
        "effects": {
            "gain_over_t19b_pp": 100.0 * (hosvd_accuracy - T19B_ACCURACY),
            "gain_over_random_pp": 100.0 * (hosvd_accuracy - random_accuracy),
            "gain_over_query_only_pp": 100.0 * (hosvd_accuracy - query_accuracy),
            "document_shuffle_drop_pp": 100.0
            * (hosvd_accuracy - shuffled_accuracy),
        },
        "runtime": {
            "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
        },
        "gates": gates,
        "admitted": all(gates.values()),
        "integrity": {
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "candidate_corpus_sha256": sha256_file(t19b.t19a.CANDIDATE_CORPUS),
            "train_qa_sha256": sha256_file(t19b.TRAIN_QA),
            "evaluation_qa_sha256": sha256_file(t19b.EVALUATION_QA),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
        },
        "limits": [
            "The query operator is a privileged gradient-side upper bound.",
            "The physical write is verified but the ordinary model is not trained to emit or consume the operator.",
            "Passing cannot establish a smarter production model without matched joint from-zero training.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device))
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
