#!/usr/bin/env python3
"""T23a: localize continuous, Q16, and one-pass forced-read boundaries."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import causal_write_read_plane_t22a as t22


core = t22.core
t7 = t22.t7
t19a = t22.t19a
t19b = t22.t19b
t21a = t22.t21a

PREREGISTRATION = ROOT / "results/heldout-record-oracle-t23a-preregistration.md"
PREREGISTRATION_SHA256 = (
    "1d7d69ac456b83402d979cb9e45789554e1c6285fb85fb87d8de8ffa36016f3f"
)
INTEGRITY_AMENDMENT = (
    ROOT / "results/heldout-record-oracle-t23a-integrity-amendment.md"
)
INTEGRITY_AMENDMENT_SHA256 = (
    "c5590e10b415d1ab619358f82d206ae881763a48c09877afa703d4cb36bdc3a4"
)
T22_SOURCE = ROOT / "experiments/causal_write_read_plane_t22a.py"
T22_SOURCE_SHA256 = (
    "c59457f73c4b7770afcbf6c82b5254ee0cda2cea233a8bdc7b859c41fa37f222"
)
T22_RESULT = ROOT / "results/causal-write-read-plane-t22a.json"
T22_RESULT_SHA256 = (
    "be7ce8c4d31250910e806770606fdb25381283584e20bd529d90008767bdd10b"
)
OUTPUT = ROOT / "results/heldout-record-oracle-t23a.json"
SCHEMA = "heldout-record-oracle-t23a-v1"

MODEL_SEED = t22.MODEL_SEED
META_CODE_SEED = 23_001
COMPILE_CODE_SEED = 23_003
COMPILE_MASK_SEEDS = (23_007, 23_009)
SHUFFLE_SEED = 23_011
RANDOM_CODE_SEED = 23_013
CODE_LR = 0.03
META_STEPS = 1_000
COMPILE_ROUNDS = 40
COMPILE_POSITIONS = 16
COMPILE_VIEWS_PER_PRESENTATION = 2
REQUIRED_NLL_GAIN = 0.20
REQUIRED_GAIN_RETENTION = 0.90
REQUIRED_ACCURACY_GAIN = 0.10
REQUIRED_NATURAL_TOLERANCE = 0.005


def sha256_file(path: Path) -> str:
    return t21a.sha256_file(path)


def continuous_records(parameters: Tensor) -> Tensor:
    return t22.CODE_AMPLITUDE * parameters.float().tanh()


def q16_records(parameters: Tensor) -> Tensor:
    return t22.quantize_16_ste(parameters)[0]


def nearest_q16(records: Tensor) -> Tensor:
    normalized = (records.float() / t22.CODE_AMPLITUDE).clamp(-1.0, 1.0)
    indices = torch.round((normalized + 1.0) * 7.5).clamp(0, 15)
    return t22.CODE_AMPLITUDE * (-1.0 + 2.0 * indices / 15.0)


def transform_records(parameters: Tensor, mode: str) -> Tensor:
    if mode == "continuous":
        return continuous_records(parameters)
    if mode == "q16":
        return q16_records(parameters)
    raise ValueError(f"unknown record mode: {mode}")


def initial_parameters(rows: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (0.05 * torch.randn(rows, t22.CODE_DIMS, generator=generator)).to(device)


@dataclasses.dataclass
class MetaReader:
    mode: str
    model: core.small.SharedInterpreterLM
    training_parameters: Tensor
    training: dict[str, object]


def train_meta_reader(
    mode: str,
    base_state: dict[str, Tensor],
    corpus: t22.RawCorpus,
    training_indices: Tensor,
    heldout_indices: Tensor,
    special: t22.SpecialTokens,
    device: torch.device,
) -> MetaReader:
    model = t22.clone_base(base_state, device)
    initial_model_hash = core.small.state_sha256(model)
    parameters = initial_parameters(
        len(corpus.titles), META_CODE_SEED, device
    ).requires_grad_(True)
    heldout_before = parameters.detach()[heldout_indices.to(device)].clone()
    model_optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=t22.SPECIALIZATION_MUON_LR
    )
    code_first_moment = torch.zeros_like(parameters)
    code_second_moment = torch.zeros_like(parameters)
    code_counts = torch.zeros(len(parameters), dtype=torch.long, device=device)
    maximum_loss = 0.0
    maximum_model_gradient = 0.0
    maximum_code_gradient = 0.0
    heldout_presentations = 0
    heldout_set = set(heldout_indices.tolist())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, META_STEPS + 1):
        model.train()
        model_optimizer.zero_grad()
        parameters.grad = None
        t22.set_learning_rate(
            model_optimizer,
            step,
            META_STEPS,
            t22.SPECIALIZATION_WARMUP,
            t22.SPECIALIZATION_MUON_LR,
            t22.SPECIALIZATION_ADAMW_LR,
        )
        indices = t22.sampled_indices(
            training_indices, t22.PROBE_SEED * 1_000_003 + step, t22.BATCH
        )
        heldout_presentations += sum(
            int(index) in heldout_set for index in indices.tolist()
        )
        if heldout_presentations:
            raise RuntimeError(f"{mode} meta-reader sampled a held-out document")
        probes = t22.build_probes(
            corpus,
            indices,
            t22.PROBE_SEED + step,
            special,
            t22.PROBES_PER_DOCUMENT,
        )
        selected = transform_records(
            parameters[probes.document_indices.to(device)], mode
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = t22.probe_logits(model, probes, device, selected)
        loss = F.cross_entropy(logits.float(), probes.targets.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {mode} meta-reader loss at {step}")
        loss.backward()
        model_gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        code_gradient = float(parameters.grad.norm().item())
        if not math.isfinite(model_gradient) or not math.isfinite(code_gradient):
            raise RuntimeError(f"nonfinite {mode} meta-reader gradient at {step}")
        model_optimizer.step()
        selected_rows = torch.unique(probes.document_indices.to(device))
        with torch.no_grad():
            row_adam_step(
                parameters,
                code_first_moment,
                code_second_moment,
                code_counts,
                selected_rows,
            )
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_model_gradient = max(maximum_model_gradient, model_gradient)
        maximum_code_gradient = max(maximum_code_gradient, code_gradient)
        if step % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "meta_reader",
                        "mode": mode,
                        "step": step,
                        "loss": float(loss.item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    heldout_after = parameters.detach()[heldout_indices.to(device)]
    heldout_unchanged = torch.equal(heldout_before, heldout_after)
    if not heldout_unchanged:
        raise RuntimeError(f"{mode} held-out parameter rows changed")
    result = {
        "initial_model_sha256": initial_model_hash,
        "updates": META_STEPS,
        "reader_sequence_token_positions": META_STEPS
        * t22.BATCH
        * t22.PROBES_PER_DOCUMENT
        * t22.CONTEXT,
        "heldout_model_gradient_presentations": heldout_presentations,
        "heldout_parameter_rows_unchanged": heldout_unchanged,
        "maximum_loss": maximum_loss,
        "maximum_model_gradient_norm": maximum_model_gradient,
        "maximum_code_gradient_norm": maximum_code_gradient,
        "minimum_selected_code_updates": int(
            code_counts[training_indices.to(device)].min().item()
        ),
        "maximum_selected_code_updates": int(
            code_counts[training_indices.to(device)].max().item()
        ),
        "code_optimizer_state_bytes": int(
            code_first_moment.numel() * code_first_moment.element_size()
            + code_second_moment.numel() * code_second_moment.element_size()
            + code_counts.numel() * code_counts.element_size()
        ),
        "failed_or_retried_steps": 0,
        "elapsed_seconds": elapsed,
        "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
    }
    del model_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return MetaReader(mode, model, parameters.detach().clone(), result)


def select_probe_rows(probes: t22.ProbeBatch, rows: list[int]) -> t22.ProbeBatch:
    indices = torch.tensor(rows, dtype=torch.long)
    return t22.ProbeBatch(
        inputs=probes.inputs[indices],
        lengths=probes.lengths[indices],
        title_ends=probes.title_ends[indices],
        targets=probes.targets[indices],
        document_indices=probes.document_indices[indices],
        target_positions=probes.target_positions[indices],
    )


def compile_position_rank(document_id: str, position: int) -> bytes:
    return hashlib.sha256(
        f"t23a-compile|{document_id}|{position}".encode()
    ).digest()


def make_compile_probe(
    corpus: t22.RawCorpus,
    document_index: int,
    target_position: int,
    mask_seed: int,
    special: t22.SpecialTokens,
) -> tuple[Tensor, int, int, int]:
    eos = int(corpus.writer_inputs[0, -1].item())
    length = int(corpus.writer_lengths[document_index].item())
    content = corpus.writer_inputs[document_index, : length - 1].clone()
    target = int(content[target_position].item())
    query = content.clone()
    generator = np.random.default_rng(
        mask_seed * 1_000_003 + document_index * 1_009 + target_position * 97
    )
    for position, value in zip(
        corpus.body_positions[document_index].tolist(),
        generator.random(len(corpus.body_positions[document_index])),
        strict=True,
    ):
        if int(position) == target_position:
            query[int(position)] = special.target_mask
        elif value < t22.OTHER_BODY_MASK_PROBABILITY:
            query[int(position)] = special.view_mask
    active = torch.cat((query, torch.tensor([special.answer_carrier])))
    padded = torch.full((t22.CONTEXT,), eos, dtype=torch.long)
    padded[: len(active)] = active
    return padded, len(active), target, int(corpus.title_ends[document_index])


@dataclasses.dataclass(frozen=True)
class CompileViews:
    probes: t22.ProbeBatch
    rows_by_document: dict[int, tuple[int, ...]]
    selected_positions: dict[int, tuple[int, ...]]


def build_compile_views(
    corpus: t22.RawCorpus,
    heldout_indices: Tensor,
    evaluation: t22.ProbeBatch,
    special: t22.SpecialTokens,
) -> CompileViews:
    evaluation_positions: dict[int, set[int]] = {
        int(document): set() for document in heldout_indices.tolist()
    }
    for document, position in zip(
        evaluation.document_indices.tolist(),
        evaluation.target_positions.tolist(),
        strict=True,
    ):
        evaluation_positions[int(document)].add(int(position))
    inputs: list[Tensor] = []
    lengths: list[int] = []
    title_ends: list[int] = []
    targets: list[int] = []
    documents: list[int] = []
    target_positions: list[int] = []
    rows_by_document: dict[int, tuple[int, ...]] = {}
    selected_positions: dict[int, tuple[int, ...]] = {}
    for document in heldout_indices.tolist():
        document = int(document)
        candidates = [
            int(position)
            for position in corpus.body_positions[document].tolist()
            if int(position) not in evaluation_positions[document]
        ]
        candidates.sort(
            key=lambda position: compile_position_rank(
                corpus.document_ids[document], position
            )
        )
        chosen = tuple(candidates[:COMPILE_POSITIONS])
        if not chosen:
            raise RuntimeError(f"held-out document {document} has no compile target")
        selected_positions[document] = chosen
        document_rows: list[int] = []
        for position in chosen:
            for mask_seed in COMPILE_MASK_SEEDS:
                padded, length, target, title_end = make_compile_probe(
                    corpus, document, position, mask_seed, special
                )
                document_rows.append(len(inputs))
                inputs.append(padded)
                lengths.append(length)
                title_ends.append(title_end)
                targets.append(target)
                documents.append(document)
                target_positions.append(position)
        rows_by_document[document] = tuple(document_rows)
    probes = t22.ProbeBatch(
        inputs=torch.stack(inputs),
        lengths=torch.tensor(lengths),
        title_ends=torch.tensor(title_ends),
        targets=torch.tensor(targets),
        document_indices=torch.tensor(documents),
        target_positions=torch.tensor(target_positions),
    )
    return CompileViews(probes, rows_by_document, selected_positions)


def row_adam_step(
    parameters: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    counts: Tensor,
    selected_rows: Tensor,
) -> None:
    if parameters.grad is None:
        raise RuntimeError("compile parameters lack a gradient")
    gradient = parameters.grad[selected_rows]
    first_moment[selected_rows] = 0.9 * first_moment[selected_rows] + 0.1 * gradient
    second_moment[selected_rows] = (
        0.999 * second_moment[selected_rows] + 0.001 * gradient.square()
    )
    counts[selected_rows] += 1
    count = counts[selected_rows].float()[:, None]
    corrected_first = first_moment[selected_rows] / (1.0 - 0.9**count)
    corrected_second = second_moment[selected_rows] / (1.0 - 0.999**count)
    parameters[selected_rows] -= CODE_LR * corrected_first / (
        corrected_second.sqrt() + 1e-8
    )
    parameters.grad = None


@dataclasses.dataclass
class CompiledOracle:
    mode: str
    records: Tensor
    training: dict[str, object]


def compile_oracle(
    mode: str,
    reader: core.small.SharedInterpreterLM,
    heldout_indices: Tensor,
    views: CompileViews,
    device: torch.device,
) -> CompiledOracle:
    reader.eval()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    parameters = initial_parameters(
        len(heldout_indices), COMPILE_CODE_SEED, device
    ).requires_grad_(True)
    first_moment = torch.zeros_like(parameters)
    second_moment = torch.zeros_like(parameters)
    counts = torch.zeros(len(heldout_indices), dtype=torch.long, device=device)
    global_to_local = {
        int(document): local
        for local, document in enumerate(heldout_indices.tolist())
    }
    maximum_loss = 0.0
    maximum_gradient = 0.0
    updates = 0
    target_presentations = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for round_index in range(COMPILE_ROUNDS):
        for start in range(0, len(heldout_indices), t22.BATCH):
            documents = [
                int(value)
                for value in heldout_indices[start : start + t22.BATCH].tolist()
            ]
            selected_probe_rows: list[int] = []
            selected_parameter_rows: list[int] = []
            for document in documents:
                available = views.rows_by_document[document]
                for offset in range(COMPILE_VIEWS_PER_PRESENTATION):
                    selected_probe_rows.append(
                        available[
                            (COMPILE_VIEWS_PER_PRESENTATION * round_index + offset)
                            % len(available)
                        ]
                    )
                    selected_parameter_rows.append(global_to_local[document])
            probes = select_probe_rows(views.probes, selected_probe_rows)
            local_rows = torch.tensor(
                selected_parameter_rows, dtype=torch.long, device=device
            )
            records = transform_records(parameters[local_rows], mode)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = t22.probe_logits(reader, probes, device, records)
            loss = F.cross_entropy(logits.float(), probes.targets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {mode} compile loss")
            loss.backward()
            gradient = float(parameters.grad.norm().item())
            if not math.isfinite(gradient):
                raise RuntimeError(f"nonfinite {mode} compile gradient")
            unique_rows = torch.tensor(
                [global_to_local[document] for document in documents],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                row_adam_step(
                    parameters,
                    first_moment,
                    second_moment,
                    counts,
                    unique_rows,
                )
            maximum_loss = max(maximum_loss, float(loss.item()))
            maximum_gradient = max(maximum_gradient, gradient)
            updates += 1
            target_presentations += len(selected_probe_rows)
        print(
            json.dumps(
                {
                    "phase": "heldout_compile",
                    "mode": mode,
                    "round": round_index + 1,
                    "loss": float(loss.item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    records = transform_records(parameters.detach(), mode).cpu()
    return CompiledOracle(
        mode,
        records,
        {
            "updates": updates,
            "rounds": COMPILE_ROUNDS,
            "target_token_presentations": target_presentations,
            "minimum_document_presentations": int(counts.min().item()),
            "maximum_document_presentations": int(counts.max().item()),
            "maximum_loss": maximum_loss,
            "maximum_gradient_norm": maximum_gradient,
            "failed_or_retried_steps": 0,
            "elapsed_seconds": elapsed,
            "optimizer_state_bytes": int(
                first_moment.numel() * first_moment.element_size()
                + second_moment.numel() * second_moment.element_size()
                + counts.numel() * counts.element_size()
            ),
            "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
        },
    )


def hash_bytes(document_id: str, counter: int) -> bytes:
    return hashlib.sha256(
        f"t23a-random|{RANDOM_CODE_SEED}|{document_id}|{counter}".encode()
    ).digest()


def fixed_random_records(
    corpus: t22.RawCorpus, heldout_indices: Tensor, mode: str
) -> Tensor:
    rows: list[Tensor] = []
    for document in heldout_indices.tolist():
        document_id = corpus.document_ids[int(document)]
        values: list[float] = []
        counter = 0
        while len(values) < t22.CODE_DIMS:
            digest = hash_bytes(document_id, counter)
            if mode == "continuous":
                values.extend(
                    -4.0 + 8.0 * int.from_bytes(digest[index : index + 2], "big") / 65535.0
                    for index in range(0, len(digest), 2)
                )
            elif mode == "q16":
                for byte in digest:
                    values.extend(
                        (
                            t22.CODE_AMPLITUDE * (-1.0 + 2.0 * (byte >> 4) / 15.0),
                            t22.CODE_AMPLITUDE * (-1.0 + 2.0 * (byte & 15) / 15.0),
                        )
                    )
            else:
                raise ValueError(mode)
            counter += 1
        rows.append(torch.tensor(values[: t22.CODE_DIMS], dtype=torch.float32))
    return torch.stack(rows)


def shuffle_records(records: Tensor) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
    return records[torch.randperm(len(records), generator=generator)]


@torch.no_grad()
def evaluate_record_table(
    reader: core.small.SharedInterpreterLM,
    probes: t22.ProbeBatch,
    heldout_indices: Tensor,
    records: Tensor,
    device: torch.device,
    name: str,
) -> dict[str, object]:
    reader.eval()
    global_to_local = torch.full(
        (int(heldout_indices.max().item()) + 1,), -1, dtype=torch.long
    )
    global_to_local[heldout_indices] = torch.arange(len(heldout_indices))
    losses = 0.0
    correct = 0
    total = 0
    for start in range(0, len(probes.inputs), t22.BATCH):
        stop = min(start + t22.BATCH, len(probes.inputs))
        batch = select_probe_rows(probes, list(range(start, stop)))
        local = global_to_local[batch.document_indices]
        if bool((local < 0).any()):
            raise RuntimeError("evaluation probe is outside the held-out table")
        selected = records[local].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = t22.probe_logits(reader, batch, device, selected)
        targets = batch.targets.to(device)
        losses += float(F.cross_entropy(logits.float(), targets, reduction="sum").item())
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total += len(targets)
    return {
        "name": name,
        "examples": total,
        "nll": losses / total,
        "accuracy": correct / total,
        "correct": correct,
        "finite": math.isfinite(losses),
    }


@dataclasses.dataclass
class WriterResult:
    records: Tensor
    training: dict[str, object]


def train_matched_writer(
    base_state: dict[str, Tensor],
    reader: core.small.SharedInterpreterLM,
    corpus: t22.RawCorpus,
    training_indices: Tensor,
    heldout_indices: Tensor,
    special: t22.SpecialTokens,
    device: torch.device,
) -> WriterResult:
    writer = t22.clone_base(base_state, device)
    reader.eval()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    optimizer = t7.build_optimizer(
        writer, "muon", muon_learning_rate=t22.SPECIALIZATION_MUON_LR
    )
    maximum_loss = 0.0
    maximum_gradient = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, META_STEPS + 1):
        writer.train()
        optimizer.zero_grad()
        t22.set_learning_rate(
            optimizer,
            step,
            META_STEPS,
            t22.SPECIALIZATION_WARMUP,
            t22.SPECIALIZATION_MUON_LR,
            t22.SPECIALIZATION_ADAMW_LR,
        )
        indices = t22.sampled_indices(
            training_indices, t22.PROBE_SEED * 1_000_003 + step, t22.BATCH
        )
        probes = t22.build_probes(
            corpus,
            indices,
            t22.PROBE_SEED + step,
            special,
            t22.PROBES_PER_DOCUMENT,
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            per_document = t22.writer_codes(writer, corpus, indices, device)
            records = per_document.repeat_interleave(
                t22.PROBES_PER_DOCUMENT, dim=0
            )
            logits = t22.probe_logits(reader, probes, device, records)
        loss = F.cross_entropy(logits.float(), probes.targets.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite matched-writer loss at {step}")
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient):
            raise RuntimeError(f"nonfinite matched-writer gradient at {step}")
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "matched_writer",
                        "step": step,
                        "loss": float(loss.item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    writer.eval()
    values: list[Tensor] = []
    compile_counts = torch.zeros(len(heldout_indices), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(heldout_indices), t22.BATCH):
            indices = heldout_indices[start : start + t22.BATCH]
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                values.append(t22.writer_codes(writer, corpus, indices, device).cpu())
            compile_counts[start : start + len(indices)] += 1
    result = WriterResult(
        torch.cat(values),
        {
            "updates": META_STEPS,
            "writer_sequence_token_positions": META_STEPS
            * t22.BATCH
            * t22.CONTEXT,
            "reader_sequence_token_positions": META_STEPS
            * t22.BATCH
            * t22.PROBES_PER_DOCUMENT
            * t22.CONTEXT,
            "heldout_compile_minimum": int(compile_counts.min().item()),
            "heldout_compile_maximum": int(compile_counts.max().item()),
            "maximum_loss": maximum_loss,
            "maximum_gradient_norm": maximum_gradient,
            "failed_or_retried_steps": 0,
            "elapsed_seconds": elapsed,
            "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
        },
    )
    del optimizer, writer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def gain_retention(reference: float, continuous: float, candidate: float) -> float:
    denominator = reference - continuous
    return (reference - candidate) / denominator if denominator > 0.0 else -math.inf


def cross_reader_gain_retention(
    continuous_reference: float,
    continuous_candidate: float,
    discrete_reference: float,
    discrete_candidate: float,
) -> float:
    continuous_gain = relative_gain(continuous_reference, continuous_candidate)
    discrete_gain = relative_gain(discrete_reference, discrete_candidate)
    return discrete_gain / continuous_gain if continuous_gain > 0.0 else -math.inf


def setup(device: torch.device) -> tuple[
    t22.RawCorpus,
    core.small.TokenStream,
    core.small.TokenStream,
    t22.SpecialTokens,
    Tensor,
    Tensor,
]:
    documents = t19a.load_raw_documents(t19a.CANDIDATE_CORPUS)
    tokenizer = AutoTokenizer.from_pretrained(
        t19a.TOKENIZER, revision=t19a.TOKENIZER_REVISION
    )
    corpus = t22.build_raw_corpus(documents, tokenizer)
    natural = core.small.TokenStream(t22.t12.TRAIN_FILE)
    validation = core.small.TokenStream(t22.t12.VALIDATION_FILE)
    special = t22.choose_special_tokens(corpus, tokenizer, natural, validation)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    title_data = t19a.build_title_data(documents, tokenizer)
    evaluation_qa = t21a.build_qa_data(
        evaluation_rows, title_data.titles, tokenizer
    )
    support_titles = t22.heldout_title_set(evaluation_rows)
    routed_titles = {
        corpus.titles[int(index)] for index in evaluation_qa.pairs.flatten().tolist()
    }
    heldout_titles = frozenset(set(support_titles) | routed_titles)
    training_indices, heldout_indices = t22.split_document_indices(
        corpus, heldout_titles
    )
    return corpus, natural, validation, special, training_indices, heldout_indices


def run(device: torch.device) -> dict[str, object]:
    torch.set_float32_matmul_precision("high")
    input_checks = {
        "preregistration": sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256,
        "integrity_amendment": sha256_file(INTEGRITY_AMENDMENT)
        == INTEGRITY_AMENDMENT_SHA256,
        "t22_source": sha256_file(T22_SOURCE) == T22_SOURCE_SHA256,
        "t22_result": sha256_file(T22_RESULT) == T22_RESULT_SHA256,
        "candidate_corpus": sha256_file(t19a.CANDIDATE_CORPUS)
        == t19a.CANDIDATE_SHA256,
        "natural_train": sha256_file(t22.t12.TRAIN_FILE)
        == t19a.NATURAL_TRAIN_SHA256,
        "natural_validation": sha256_file(t22.t12.VALIDATION_FILE)
        == t19a.NATURAL_VALIDATION_SHA256,
        "train_qa": sha256_file(t19b.TRAIN_QA) == t19b.TRAIN_QA_SHA256,
        "evaluation_qa": sha256_file(t19b.EVALUATION_QA)
        == t19a.EVALUATOR_SHA256,
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"input integrity failed: {input_checks}")
    corpus, natural, validation, special, training_indices, heldout_indices = setup(
        device
    )
    base_state, base_training = t22.train_common_base(natural, validation, device)
    base_model = t22.clone_base(base_state, device)
    base_natural = core.small.evaluate_natural(
        base_model,
        validation,
        device,
        core.EVALUATION_SEED + MODEL_SEED + 231_000,
        batches=32,
    )
    del base_model

    continuous_reader = train_meta_reader(
        "continuous",
        base_state,
        corpus,
        training_indices,
        heldout_indices,
        special,
        device,
    )
    q16_reader = train_meta_reader(
        "q16",
        base_state,
        corpus,
        training_indices,
        heldout_indices,
        special,
        device,
    )
    evaluation = t22.build_probes(
        corpus,
        heldout_indices,
        t22.EVALUATION_PROBE_SEED,
        special,
        t22.EVALUATION_PROBES_PER_DOCUMENT,
    )
    views = build_compile_views(corpus, heldout_indices, evaluation, special)
    continuous = compile_oracle(
        "continuous", continuous_reader.model, heldout_indices, views, device
    )
    q16 = compile_oracle(
        "q16", q16_reader.model, heldout_indices, views, device
    )
    projected = nearest_q16(continuous.records)
    random_continuous = fixed_random_records(corpus, heldout_indices, "continuous")
    random_q16 = fixed_random_records(corpus, heldout_indices, "q16")
    zero = torch.zeros_like(continuous.records)
    matched_writer = train_matched_writer(
        base_state,
        q16_reader.model,
        corpus,
        training_indices,
        heldout_indices,
        special,
        device,
    )

    conditions: dict[str, dict[str, object]] = {}
    for name, reader, records in (
        ("continuous_correct", continuous_reader.model, continuous.records),
        ("continuous_zero", continuous_reader.model, zero),
        ("continuous_shuffle", continuous_reader.model, shuffle_records(continuous.records)),
        ("continuous_random", continuous_reader.model, random_continuous),
        ("projected_q16_correct", continuous_reader.model, projected),
        ("projected_q16_zero", continuous_reader.model, zero),
        ("projected_q16_shuffle", continuous_reader.model, shuffle_records(projected)),
        ("projected_q16_random", continuous_reader.model, random_q16),
        ("ste_q16_correct", q16_reader.model, q16.records),
        ("ste_q16_zero", q16_reader.model, zero),
        ("ste_q16_shuffle", q16_reader.model, shuffle_records(q16.records)),
        ("ste_q16_random", q16_reader.model, random_q16),
        ("matched_writer", q16_reader.model, matched_writer.records),
        ("matched_writer_zero", q16_reader.model, zero),
        ("matched_writer_shuffle", q16_reader.model, shuffle_records(matched_writer.records)),
        ("matched_writer_random", q16_reader.model, random_q16),
    ):
        conditions[name] = evaluate_record_table(
            reader, evaluation, heldout_indices, records, device, name
        )

    final_natural = {
        "continuous": core.small.evaluate_natural(
            continuous_reader.model,
            validation,
            device,
            core.EVALUATION_SEED + MODEL_SEED + 231_000,
            batches=32,
        ),
        "q16": core.small.evaluate_natural(
            q16_reader.model,
            validation,
            device,
            core.EVALUATION_SEED + MODEL_SEED + 231_000,
            batches=32,
        ),
    }
    cont = conditions["continuous_correct"]
    cont_zero = conditions["continuous_zero"]
    proj = conditions["projected_q16_correct"]
    ste = conditions["ste_q16_correct"]
    ste_zero = conditions["ste_q16_zero"]
    continuous_controls = [
        conditions[name]
        for name in ("continuous_zero", "continuous_shuffle", "continuous_random")
    ]
    projected_controls = [
        conditions[name]
        for name in ("projected_q16_zero", "projected_q16_shuffle", "projected_q16_random")
    ]
    q16_controls = [
        conditions[name]
        for name in ("ste_q16_zero", "ste_q16_shuffle", "ste_q16_random")
    ]
    matched_controls = [
        conditions[name]
        for name in (
            "matched_writer_zero",
            "matched_writer_shuffle",
            "matched_writer_random",
        )
    ]
    continuous_pass = all(
        relative_gain(float(control["nll"]), float(cont["nll"]))
        >= REQUIRED_NLL_GAIN
        for control in continuous_controls
    ) and all(
        float(cont["accuracy"]) - float(control["accuracy"])
        >= REQUIRED_ACCURACY_GAIN
        for control in continuous_controls
    )
    projected_pass = (
        gain_retention(
            float(cont_zero["nll"]), float(cont["nll"]), float(proj["nll"])
        )
        >= REQUIRED_GAIN_RETENTION
        and all(
            float(proj["accuracy"]) - float(control["accuracy"])
            >= REQUIRED_ACCURACY_GAIN
            for control in projected_controls
        )
    )
    q16_pass = all(
        relative_gain(float(control["nll"]), float(ste["nll"]))
        >= REQUIRED_NLL_GAIN
        for control in q16_controls
    ) and all(
        float(ste["accuracy"]) - float(control["accuracy"])
        >= REQUIRED_ACCURACY_GAIN
        for control in q16_controls
    )
    q16_retention = cross_reader_gain_retention(
        float(cont_zero["nll"]),
        float(cont["nll"]),
        float(ste_zero["nll"]),
        float(ste["nll"]),
    )
    natural_pass = all(
        (float(result["nll"]) - float(base_natural["nll"]))
        / float(base_natural["nll"])
        <= REQUIRED_NATURAL_TOLERANCE
        for result in final_natural.values()
    )
    matched = conditions["matched_writer"]
    matched_pass = all(
        relative_gain(float(control["nll"]), float(matched["nll"]))
        >= REQUIRED_NLL_GAIN
        for control in matched_controls
    ) and all(
        float(matched["accuracy"]) - float(control["accuracy"])
        >= REQUIRED_ACCURACY_GAIN
        for control in matched_controls
    )
    q16_integrity: dict[str, dict[str, object]] = {}
    for name, records in (
        ("ste_q16", q16.records),
        ("projected_q16", projected),
        ("matched_writer", matched_writer.records),
    ):
        indices = t22.decoded_level_indices(records)
        bf16_indices = t22.decoded_level_indices(
            records.to(torch.bfloat16).float()
        )
        canonical = nearest_q16(records)
        q16_integrity[name] = {
            "minimum_level_index": int(indices.min().item()),
            "maximum_level_index": int(indices.max().item()),
            "unique_level_indices": int(torch.unique(indices).numel()),
            "bf16_level_indices_exact": bool(torch.equal(indices, bf16_indices)),
            "maximum_float_residual_from_canonical": float(
                (records.float() - canonical).abs().max().item()
            ),
        }
    all_q16_levels = all(
        int(check["minimum_level_index"]) >= 0
        and int(check["maximum_level_index"]) <= 15
        for check in q16_integrity.values()
    )
    bf16_levels_exact = all(
        bool(check["bf16_level_indices_exact"])
        for check in q16_integrity.values()
    )
    compile_targets = {
        (int(document), int(position))
        for document, positions in views.selected_positions.items()
        for position in positions
    }
    evaluation_targets = set(
        zip(
            evaluation.document_indices.tolist(),
            evaluation.target_positions.tolist(),
            strict=True,
        )
    )
    gates = {
        "input_integrity": all(input_checks.values()),
        "byte_identical_meta_reader_initial_models": continuous_reader.training[
            "initial_model_sha256"
        ]
        == q16_reader.training["initial_model_sha256"],
        "whole_document_holdout_and_disjoint_targets": len(training_indices) == 2_198
        and len(heldout_indices) == 207
        and not (compile_targets & evaluation_targets)
        and continuous_reader.training["heldout_model_gradient_presentations"] == 0
        and q16_reader.training["heldout_model_gradient_presentations"] == 0
        and continuous_reader.training["heldout_parameter_rows_unchanged"]
        and q16_reader.training["heldout_parameter_rows_unchanged"],
        "balanced_finite_compile": continuous.training["minimum_document_presentations"]
        == COMPILE_ROUNDS
        and continuous.training["maximum_document_presentations"] == COMPILE_ROUNDS
        and q16.training["minimum_document_presentations"] == COMPILE_ROUNDS
        and q16.training["maximum_document_presentations"] == COMPILE_ROUNDS,
        "q16_levels_and_bf16_exact": all_q16_levels and bf16_levels_exact,
        "continuous_oracle_pass": continuous_pass,
        "projected_q16_pass": projected_pass,
        "ste_q16_pass": q16_pass and q16_retention >= REQUIRED_GAIN_RETENTION,
        "matched_one_pass_writer_pass": matched_pass,
        "natural_nll_within_0_5pct": natural_pass,
    }
    integrity_gate_names = (
        "input_integrity",
        "byte_identical_meta_reader_initial_models",
        "whole_document_holdout_and_disjoint_targets",
        "balanced_finite_compile",
        "q16_levels_and_bf16_exact",
    )
    experiment_valid = all(bool(gates[name]) for name in integrity_gate_names)
    if not experiment_valid:
        localization = "invalid_experiment"
    elif not continuous_pass:
        localization = "continuous_reader_interface_wall"
    elif not projected_pass:
        localization = "local_q16_projection_wall"
    elif not (q16_pass and q16_retention >= REQUIRED_GAIN_RETENTION):
        localization = "ste_q16_optimization_wall"
    elif not matched_pass:
        localization = "one_pass_amortized_writer_wall"
    else:
        localization = "historical_shared_coadaptation_wall"
    record_plane_pass = (
        experiment_valid
        and continuous_pass
        and projected_pass
        and q16_pass
        and q16_retention >= REQUIRED_GAIN_RETENTION
    )
    if localization == "invalid_experiment":
        classification = localization
    elif record_plane_pass and not natural_pass:
        classification = "natural_capability_regression"
    else:
        classification = localization
    return {
        "schema": SCHEMA,
        "device": str(device),
        "input_checks": input_checks,
        "data": {
            "documents": len(corpus.titles),
            "training_documents": len(training_indices),
            "heldout_documents": len(heldout_indices),
            "evaluation_probes": len(evaluation.inputs),
            "compile_views": len(views.probes.inputs),
            "compile_target_pairs": len(compile_targets),
            "evaluation_target_pairs": len(evaluation_targets),
            "compile_evaluation_target_overlap": len(
                compile_targets & evaluation_targets
            ),
        },
        "base_training": base_training,
        "base_natural": base_natural,
        "meta_readers": {
            "continuous": continuous_reader.training,
            "q16": q16_reader.training,
        },
        "compile": {
            "continuous": continuous.training,
            "q16": q16.training,
            "matched_writer": matched_writer.training,
        },
        "q16_integrity": q16_integrity,
        "conditions": conditions,
        "final_natural": final_natural,
        "effects": {
            "continuous_gain_over_zero_pct": 100.0
            * relative_gain(float(cont_zero["nll"]), float(cont["nll"])),
            "projected_q16_continuous_gain_retention_pct": 100.0
            * gain_retention(
                float(cont_zero["nll"]), float(cont["nll"]), float(proj["nll"])
            ),
            "ste_q16_gain_over_zero_pct": 100.0
            * relative_gain(float(ste_zero["nll"]), float(ste["nll"])),
            "ste_q16_cross_reader_gain_retention_pct": 100.0 * q16_retention,
            "matched_writer_gain_over_zero_pct": 100.0
            * relative_gain(float(ste_zero["nll"]), float(matched["nll"])),
        },
        "gates": gates,
        "experiment_valid": experiment_valid,
        "record_boundary_localization": localization,
        "eligible_for_writer_rethink": record_plane_pass and natural_pass,
        "classification": classification,
        "claim_boundary": (
            "T23a localizes a virtual write/read boundary. It is not an autonomous "
            "compiler, physical export, or smarter-production-model result."
        ),
        "provenance": {
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "integrity_amendment_sha256": sha256_file(INTEGRITY_AMENDMENT),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "t22_source_sha256": sha256_file(T22_SOURCE),
            "t22_result_sha256": sha256_file(T22_RESULT),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(t22.json_native(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps(t22.json_native(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
