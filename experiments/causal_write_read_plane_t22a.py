#!/usr/bin/env python3
"""T22a: same-model one-pass raw-prose writer and causal read plane."""

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

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import joint_quantized_document_code_t21a as t21a


core = t21a.core
t7 = t21a.t7
t12 = t21a.t12
t19a = t21a.t19a
t19b = t21a.t19b

PREREGISTRATION = ROOT / "results/causal-write-read-plane-t22a-preregistration-v4.md"
PARENT_PREREGISTRATION = ROOT / "results/causal-write-read-plane-t22a-preregistration-v3.md"
PARENT_PREREGISTRATION_SHA256 = (
    "ad563abdc209e2ac465a21028d8975059cbe0bb351d3308015f42ce60663ddc5"
)
V2_PREREGISTRATION = ROOT / "results/causal-write-read-plane-t22a-preregistration-v2.md"
V2_PREREGISTRATION_SHA256 = (
    "66604efec55201e274d831bd74c97ae890082cc1e9c6bff818f73c1ec910ba17"
)
ROOT_PREREGISTRATION = ROOT / "results/causal-write-read-plane-t22a-preregistration.md"
ROOT_PREREGISTRATION_SHA256 = (
    "e6517016b796caef7d4ba8d4e628efa38d5aa16dd3b508d96f84d01205554445"
)
OUTPUT = ROOT / "results/causal-write-read-plane-t22a.json"
SCHEMA = "causal-write-read-plane-t22a-v1"

MODEL_SEED = 8_209
PROBE_SEED = 22_001
RANDOM_CODE_SEED = 22_003
SHUFFLE_SEED = 22_005
EVALUATION_PROBE_SEED = 22_007
t21a.SHUFFLE_SEED = SHUFFLE_SEED

CONTEXT = 128
BATCH = 16
PROBES_PER_DOCUMENT = 2
EVALUATION_PROBES_PER_DOCUMENT = 8
OTHER_BODY_MASK_PROBABILITY = 0.5
CODE_DIMS = 220
CODE_START = 64
CODE_AMPLITUDE = 4.0

BASE_STEPS = 19_000
BASE_WARMUP = 500
BASE_MUON_LR = 0.005
BASE_ADAMW_LR = 0.0003
READ_STEPS = 1_000
DENSE_STEPS = 3_000
SPECIALIZATION_WARMUP = 100
SPECIALIZATION_MUON_LR = 0.001
SPECIALIZATION_ADAMW_LR = 0.0001

REQUIRED_PROBE_GAIN = 0.20
REQUIRED_RANDOM_GAIN = 0.10
REQUIRED_QA_ACCURACY = 0.75
REQUIRED_QA_GAIN = 0.10


def sha256_file(path: Path) -> str:
    return t21a.sha256_file(path)


def tensor_sha256(value: Tensor) -> str:
    return t21a.tensor_sha256(value)


def json_native(value: object, path: str = "$") -> object:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TypeError(f"non-scalar tensor at {path}: {tuple(value.shape)}")
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: json_native(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_native(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def quantize_16_ste(values: Tensor) -> tuple[Tensor, Tensor]:
    bounded = values.float().tanh()
    indices = torch.round((bounded + 1.0) * 7.5).clamp(0, 15)
    quantized = -1.0 + 2.0 * indices / 15.0
    straight_through = bounded + (quantized - bounded).detach()
    return CODE_AMPLITUDE * straight_through, indices.to(torch.uint8)


def decoded_level_indices(values: Tensor) -> Tensor:
    normalized = values.float() / CODE_AMPLITUDE
    return torch.round((normalized + 1.0) * 7.5).clamp(0, 15).to(torch.uint8)


@dataclass(frozen=True)
class RawCorpus:
    document_ids: tuple[str, ...]
    titles: tuple[str, ...]
    writer_inputs: Tensor
    writer_lengths: Tensor
    title_ends: Tensor
    body_positions: tuple[Tensor, ...]
    causal_inputs: Tensor
    causal_targets: Tensor
    causal_mask: Tensor


def build_raw_corpus(
    documents: list[dict[str, str]], tokenizer: object
) -> RawCorpus:
    eos = int(tokenizer.eos_token_id)
    writer_inputs: list[list[int]] = []
    writer_lengths: list[int] = []
    title_ends: list[int] = []
    body_positions: list[Tensor] = []
    causal_inputs: list[list[int]] = []
    causal_targets: list[list[int]] = []
    causal_masks: list[list[bool]] = []
    for document in documents:
        prefix = [
            int(value)
            for value in tokenizer.encode(
                str(document["title"]) + "\n", add_special_tokens=False
            )
        ]
        body = [
            int(value)
            for value in tokenizer.encode(
                str(document["text"]), add_special_tokens=False
            )
        ]
        if not prefix or not body:
            raise RuntimeError("raw document produced an empty title/body tokenization")
        content = (prefix + body)[: CONTEXT - 1]
        active_prefix = min(len(prefix), len(content))
        positions = list(range(active_prefix, len(content)))
        if not positions:
            raise RuntimeError("raw document has no body target inside context")
        sequence = content + [eos]
        length = len(sequence)
        writer_inputs.append(sequence + [eos] * (CONTEXT - length))
        writer_lengths.append(length)
        title_ends.append(active_prefix - 1)
        body_positions.append(torch.tensor(positions, dtype=torch.long))

        source = sequence[:-1]
        target = sequence[1:]
        active = len(source)
        causal_inputs.append(source + [eos] * (CONTEXT - active))
        causal_targets.append(target + [eos] * (CONTEXT - active))
        causal_masks.append([True] * active + [False] * (CONTEXT - active))
    return RawCorpus(
        document_ids=tuple(str(document["document_id"]) for document in documents),
        titles=tuple(str(document["title"]) for document in documents),
        writer_inputs=torch.tensor(writer_inputs, dtype=torch.long),
        writer_lengths=torch.tensor(writer_lengths, dtype=torch.long),
        title_ends=torch.tensor(title_ends, dtype=torch.long),
        body_positions=tuple(body_positions),
        causal_inputs=torch.tensor(causal_inputs, dtype=torch.long),
        causal_targets=torch.tensor(causal_targets, dtype=torch.long),
        causal_mask=torch.tensor(causal_masks, dtype=torch.bool),
    )


@dataclass(frozen=True)
class SpecialTokens:
    target_mask: int
    view_mask: int
    answer_carrier: int
    unused_count: int


def choose_special_tokens(
    corpus: RawCorpus,
    tokenizer: object,
    natural: core.small.TokenStream,
    validation: core.small.TokenStream,
) -> SpecialTokens:
    used = np.zeros(core.small.VOCAB, dtype=np.bool_)
    used[np.asarray(natural.tokens, dtype=np.int64)] = True
    used[np.asarray(validation.tokens, dtype=np.int64)] = True
    used[corpus.writer_inputs.numpy().reshape(-1)] = True
    for path in (t19b.TRAIN_QA, t19b.EVALUATION_QA):
        for row in t19b.load_qa(path):
            identifiers = tokenizer.encode(
                str(row["question"]) + " " + str(row["answer"]),
                add_special_tokens=False,
            )
            used[np.asarray(identifiers, dtype=np.int64)] = True
    unused = np.flatnonzero(~used)
    if len(unused) < 3:
        raise RuntimeError("fewer than three unused carrier tokens")
    return SpecialTokens(
        target_mask=int(unused[0]),
        view_mask=int(unused[1]),
        answer_carrier=int(unused[2]),
        unused_count=len(unused),
    )


def heldout_title_set(rows: list[dict[str, object]]) -> frozenset[str]:
    values: set[str] = set()
    for row in rows:
        titles = row.get("supporting_titles")
        if not isinstance(titles, list):
            raise RuntimeError("development row lacks sealed supporting titles")
        values.update(str(title) for title in titles)
    return frozenset(values)


def split_document_indices(
    corpus: RawCorpus, heldout_titles: frozenset[str]
) -> tuple[Tensor, Tensor]:
    heldout = torch.tensor(
        [index for index, title in enumerate(corpus.titles) if title in heldout_titles],
        dtype=torch.long,
    )
    training = torch.tensor(
        [index for index, title in enumerate(corpus.titles) if title not in heldout_titles],
        dtype=torch.long,
    )
    if len(heldout) != len(heldout_titles):
        raise RuntimeError("held-out title is absent or duplicated in raw corpus")
    if len(training) + len(heldout) != len(corpus.titles):
        raise RuntimeError("writer split does not partition raw documents")
    return training, heldout


def sampled_indices(pool: Tensor, seed: int, count: int = BATCH) -> Tensor:
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(pool), size=count, dtype=np.int64)
    return pool[torch.from_numpy(choices)]


@dataclass(frozen=True)
class ProbeBatch:
    inputs: Tensor
    lengths: Tensor
    title_ends: Tensor
    targets: Tensor
    document_indices: Tensor
    target_positions: Tensor


def build_probes(
    corpus: RawCorpus,
    document_indices: Tensor,
    seed: int,
    special: SpecialTokens,
    probes_per_document: int,
) -> ProbeBatch:
    eos = int(corpus.writer_inputs[0, -1].item())
    inputs: list[Tensor] = []
    lengths: list[int] = []
    title_ends: list[int] = []
    targets: list[int] = []
    expanded_indices: list[int] = []
    target_positions: list[int] = []
    for row, document_index_value in enumerate(document_indices.tolist()):
        document_index = int(document_index_value)
        length = int(corpus.writer_lengths[document_index].item())
        content = corpus.writer_inputs[document_index, : length - 1].clone()
        positions = corpus.body_positions[document_index]
        for probe in range(probes_per_document):
            generator = np.random.default_rng(
                seed * 1_000_003 + row * 1_009 + probe * 97 + document_index
            )
            target_position = int(positions[generator.integers(0, len(positions))])
            target = int(content[target_position].item())
            query = content.clone()
            random_values = generator.random(len(positions))
            for position, value in zip(positions.tolist(), random_values, strict=True):
                if int(position) == target_position:
                    query[int(position)] = special.target_mask
                elif value < OTHER_BODY_MASK_PROBABILITY:
                    query[int(position)] = special.view_mask
            active = torch.cat(
                (query, torch.tensor([special.answer_carrier], dtype=torch.long))
            )
            padded = torch.full((CONTEXT,), eos, dtype=torch.long)
            padded[: len(active)] = active
            inputs.append(padded)
            lengths.append(len(active))
            title_ends.append(int(corpus.title_ends[document_index].item()))
            targets.append(target)
            expanded_indices.append(document_index)
            target_positions.append(target_position)
    return ProbeBatch(
        inputs=torch.stack(inputs),
        lengths=torch.tensor(lengths, dtype=torch.long),
        title_ends=torch.tensor(title_ends, dtype=torch.long),
        targets=torch.tensor(targets, dtype=torch.long),
        document_indices=torch.tensor(expanded_indices, dtype=torch.long),
        target_positions=torch.tensor(target_positions, dtype=torch.long),
    )


def fixed_random_code(document_id: str) -> Tensor:
    nibbles: list[int] = []
    counter = 0
    while len(nibbles) < CODE_DIMS:
        digest = hashlib.sha256(
            f"t22a-{RANDOM_CODE_SEED}-{document_id}-{counter}".encode()
        ).digest()
        for byte in digest:
            nibbles.extend((byte >> 4, byte & 15))
        counter += 1
    indices = torch.tensor(nibbles[:CODE_DIMS], dtype=torch.float32)
    return CODE_AMPLITUDE * (-1.0 + 2.0 * indices / 15.0)


def fixed_random_code_table(corpus: RawCorpus) -> Tensor:
    return torch.stack([fixed_random_code(value) for value in corpus.document_ids])


def writer_codes(
    model: core.small.SharedInterpreterLM,
    corpus: RawCorpus,
    indices: Tensor,
    device: torch.device,
) -> Tensor:
    inputs = corpus.writer_inputs[indices].to(device)
    lengths = corpus.writer_lengths[indices].to(device)
    hidden = model.hidden(inputs)
    rows = torch.arange(len(indices), device=device)
    state = hidden[rows, lengths - 1, CODE_START : CODE_START + CODE_DIMS]
    codes, _ = quantize_16_ste(state)
    return codes


def probe_logits(
    model: core.small.SharedInterpreterLM,
    probes: ProbeBatch,
    device: torch.device,
    records: Tensor | None,
) -> Tensor:
    inputs = probes.inputs.to(device)
    lengths = probes.lengths.to(device)
    positions = probes.title_ends.to(device)
    hidden = t21a.hidden_with_records(
        model,
        inputs,
        records[:, None, :] if records is not None else None,
        positions[:, None] if records is not None else None,
    )
    rows = torch.arange(len(inputs), device=device)
    final = hidden[rows, lengths - 1]
    return final @ model.token.weight.T


def set_learning_rate(
    optimizer: t7.OptimizerBundle,
    step: int,
    total_steps: int,
    warmup: int,
    muon_peak: float,
    adamw_peak: float,
) -> None:
    t21a.set_bundle_learning_rate(
        optimizer,
        step,
        total_steps,
        warmup,
        muon_peak,
        adamw_peak,
    )


def train_common_base(
    natural: core.small.TokenStream,
    validation: core.small.TokenStream,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    model = core.build_model(MODEL_SEED, device)
    initial_hash = core.small.state_sha256(model)
    optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=BASE_MUON_LR
    )
    maximum_loss = 0.0
    maximum_gradient = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, BASE_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        set_learning_rate(
            optimizer,
            step,
            BASE_STEPS,
            BASE_WARMUP,
            BASE_MUON_LR,
            BASE_ADAMW_LR,
        )
        inputs, targets = natural.batch(
            MODEL_SEED * 1_000_003 + step, BATCH, CONTEXT, device
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
        loss = F.cross_entropy(
            logits.float().reshape(-1, core.small.VOCAB), targets.reshape(-1)
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite common-base loss at step {step}")
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient):
            raise RuntimeError(f"nonfinite common-base gradient at step {step}")
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 1_000 == 0:
            print(
                json.dumps(
                    {"phase": "base", "step": step, "loss": float(loss.item())},
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    natural_result = core.small.evaluate_natural(
        model,
        validation,
        device,
        core.EVALUATION_SEED + MODEL_SEED + BASE_STEPS,
        batches=32,
    )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    terminal_hash = core.small.state_sha256(model)
    result = {
        "initial_state_sha256": initial_hash,
        "terminal_state_sha256": terminal_hash,
        "updates": BASE_STEPS,
        "token_presentations": BASE_STEPS * BATCH * CONTEXT,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
        "failed_or_retried_steps": 0,
        "elapsed_seconds": elapsed,
        "natural": natural_result,
        "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
    }
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, result


def clone_base(base_state: dict[str, Tensor], device: torch.device) -> core.small.SharedInterpreterLM:
    model = core.build_model(MODEL_SEED, device)
    model.load_state_dict(base_state)
    return model


@dataclass
class SpecializedArm:
    name: str
    model: core.small.SharedInterpreterLM
    codes: Tensor | None
    training: dict[str, object]
    probe_evaluation: dict[str, dict[str, object]]


def extract_writer_table(
    model: core.small.SharedInterpreterLM,
    corpus: RawCorpus,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    values: list[Tensor] = []
    compile_counts = torch.zeros(len(corpus.titles), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(corpus.titles), BATCH):
            indices = torch.arange(start, min(start + BATCH, len(corpus.titles)))
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                values.append(writer_codes(model, corpus, indices, device).float().cpu())
            compile_counts[indices] += 1
    return torch.cat(values), compile_counts


def train_read_arm(
    name: str,
    mode: str,
    base_state: dict[str, Tensor],
    corpus: RawCorpus,
    training_indices: Tensor,
    heldout_indices: Tensor,
    special: SpecialTokens,
    random_table: Tensor,
    device: torch.device,
) -> SpecializedArm:
    model = clone_base(base_state, device)
    initial_hash = core.small.state_sha256(model)
    optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=SPECIALIZATION_MUON_LR
    )
    maximum_loss = 0.0
    maximum_gradient = 0.0
    heldout_presentations = 0
    heldout_set = set(heldout_indices.tolist())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, READ_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        set_learning_rate(
            optimizer,
            step,
            READ_STEPS,
            SPECIALIZATION_WARMUP,
            SPECIALIZATION_MUON_LR,
            SPECIALIZATION_ADAMW_LR,
        )
        indices = sampled_indices(
            training_indices, PROBE_SEED * 1_000_003 + step, BATCH
        )
        heldout_presentations += sum(
            int(index) in heldout_set for index in indices.tolist()
        )
        if heldout_presentations:
            raise RuntimeError(f"{name} sampled a writer-held-out document")
        probes = build_probes(
            corpus,
            indices,
            PROBE_SEED + step,
            special,
            PROBES_PER_DOCUMENT,
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            if mode == "learned":
                per_document = writer_codes(model, corpus, indices, device)
                records = per_document.repeat_interleave(
                    PROBES_PER_DOCUMENT, dim=0
                )
            elif mode == "random":
                records = random_table[probes.document_indices].to(device)
            elif mode == "zero":
                records = None
            else:
                raise ValueError(f"unknown read mode: {mode}")
            logits = probe_logits(model, probes, device, records)
        targets = probes.targets.to(device)
        loss = F.cross_entropy(logits.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {name} loss at step {step}")
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient):
            raise RuntimeError(f"nonfinite {name} gradient at step {step}")
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "specialize",
                        "arm": name,
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
    if mode == "learned":
        codes, compile_counts = extract_writer_table(model, corpus, device)
    elif mode == "random":
        codes = random_table.detach().clone()
        compile_counts = torch.zeros(len(corpus.titles), dtype=torch.long)
    else:
        codes = None
        compile_counts = torch.zeros(len(corpus.titles), dtype=torch.long)
    heldout_compile_counts = compile_counts[heldout_indices]
    training = {
        "initial_state_sha256": initial_hash,
        "updates": READ_STEPS,
        "writer_sequence_token_positions": READ_STEPS * BATCH * CONTEXT
        if mode == "learned"
        else 0,
        "reader_sequence_token_positions": READ_STEPS
        * BATCH
        * PROBES_PER_DOCUMENT
        * CONTEXT,
        "heldout_gradient_presentations": heldout_presentations,
        "heldout_compile_passes": int(heldout_compile_counts.sum().item()),
        "heldout_compile_minimum": int(heldout_compile_counts.min().item())
        if len(heldout_compile_counts)
        else 0,
        "heldout_compile_maximum": int(heldout_compile_counts.max().item())
        if len(heldout_compile_counts)
        else 0,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
        "failed_or_retried_steps": 0,
        "elapsed_seconds": elapsed,
        "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
    }
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return SpecializedArm(name, model, codes, training, {})


def train_dense_arm(
    base_state: dict[str, Tensor],
    corpus: RawCorpus,
    device: torch.device,
) -> SpecializedArm:
    name = "dense_causal_3x"
    model = clone_base(base_state, device)
    initial_hash = core.small.state_sha256(model)
    optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=SPECIALIZATION_MUON_LR
    )
    maximum_loss = 0.0
    maximum_gradient = 0.0
    all_indices = torch.arange(len(corpus.titles))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, DENSE_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        set_learning_rate(
            optimizer,
            step,
            DENSE_STEPS,
            SPECIALIZATION_WARMUP,
            SPECIALIZATION_MUON_LR,
            SPECIALIZATION_ADAMW_LR,
        )
        indices = sampled_indices(
            all_indices, PROBE_SEED * 2_000_003 + step, BATCH
        )
        inputs = corpus.causal_inputs[indices].to(device)
        targets = corpus.causal_targets[indices].to(device)
        mask = corpus.causal_mask[indices].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
        loss = t21a.masked_lm_loss(logits, targets, mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite dense loss at step {step}")
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient):
            raise RuntimeError(f"nonfinite dense gradient at step {step}")
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 250 == 0:
            print(
                json.dumps(
                    {
                        "phase": "specialize",
                        "arm": name,
                        "step": step,
                        "loss": float(loss.item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training = {
        "initial_state_sha256": initial_hash,
        "updates": DENSE_STEPS,
        "sequence_token_positions": DENSE_STEPS * BATCH * CONTEXT,
        "heldout_documents_eligible_for_gradients": True,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
        "failed_or_retried_steps": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
    }
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return SpecializedArm(name, model, None, training, {})


@torch.no_grad()
def evaluate_probe_batch(
    model: core.small.SharedInterpreterLM,
    probes: ProbeBatch,
    device: torch.device,
    codes: Tensor | None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for start in range(0, len(probes.inputs), 64):
        stop = min(start + 64, len(probes.inputs))
        selected = ProbeBatch(
            inputs=probes.inputs[start:stop],
            lengths=probes.lengths[start:stop],
            title_ends=probes.title_ends[start:stop],
            targets=probes.targets[start:stop],
            document_indices=probes.document_indices[start:stop],
            target_positions=probes.target_positions[start:stop],
        )
        records = (
            codes[selected.document_indices].to(device) if codes is not None else None
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = probe_logits(model, selected, device, records)
        targets = selected.targets.to(device)
        losses = F.cross_entropy(logits.float(), targets, reduction="none")
        total_loss += float(losses.sum().item())
        correct += int((logits.float().argmax(dim=-1) == targets).sum().item())
        count += len(targets)
    return {
        "examples": count,
        "nll": total_loss / count,
        "accuracy": correct / count,
        "errors": count - correct,
        "finite": math.isfinite(total_loss),
    }


def shuffled_table(codes: Tensor) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
    return codes[torch.randperm(len(codes), generator=generator)]


def evaluate_probes(
    arms: dict[str, SpecializedArm],
    probes: ProbeBatch,
    device: torch.device,
) -> None:
    candidate = arms["learned_writer"]
    assert candidate.codes is not None
    candidate.probe_evaluation = {
        "correct": evaluate_probe_batch(
            candidate.model, probes, device, candidate.codes
        ),
        "zero": evaluate_probe_batch(
            candidate.model, probes, device, torch.zeros_like(candidate.codes)
        ),
        "shuffle": evaluate_probe_batch(
            candidate.model, probes, device, shuffled_table(candidate.codes)
        ),
    }
    for name in ("fixed_random_writer", "zero_record_read", "dense_causal_3x"):
        arm = arms[name]
        arm.probe_evaluation = {
            "correct": evaluate_probe_batch(arm.model, probes, device, arm.codes)
        }


def state_schema(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())


def verify_inputs() -> dict[str, bool]:
    checks = {
        "candidate_corpus": sha256_file(t19a.CANDIDATE_CORPUS)
        == t19a.CANDIDATE_SHA256,
        "train_qa": sha256_file(t19b.TRAIN_QA) == t19b.TRAIN_QA_SHA256,
        "evaluation_qa": sha256_file(t19b.EVALUATION_QA)
        == t19a.EVALUATOR_SHA256,
        "natural_train": sha256_file(t12.TRAIN_FILE)
        == t19a.NATURAL_TRAIN_SHA256,
        "natural_validation": sha256_file(t12.VALIDATION_FILE)
        == t19a.NATURAL_VALIDATION_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
        "parent_preregistration": sha256_file(PARENT_PREREGISTRATION)
        == PARENT_PREREGISTRATION_SHA256,
        "v2_preregistration": sha256_file(V2_PREREGISTRATION)
        == V2_PREREGISTRATION_SHA256,
        "root_preregistration": sha256_file(ROOT_PREREGISTRATION)
        == ROOT_PREREGISTRATION_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def run(device: torch.device) -> dict[str, object]:
    torch.set_float32_matmul_precision("high")
    input_checks = verify_inputs()
    documents = t19a.load_raw_documents(t19a.CANDIDATE_CORPUS)
    tokenizer = AutoTokenizer.from_pretrained(
        t19a.TOKENIZER, revision=t19a.TOKENIZER_REVISION
    )
    corpus = build_raw_corpus(documents, tokenizer)
    natural = core.small.TokenStream(t12.TRAIN_FILE)
    validation = core.small.TokenStream(t12.VALIDATION_FILE)
    special = choose_special_tokens(corpus, tokenizer, natural, validation)

    train_rows = t19b.load_qa(t19b.TRAIN_QA)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    title_data = t19a.build_title_data(documents, tokenizer)
    train_qa = t21a.build_qa_data(train_rows, title_data.titles, tokenizer)
    evaluation_qa = t21a.build_qa_data(
        evaluation_rows, title_data.titles, tokenizer
    )
    support_titles = heldout_title_set(evaluation_rows)
    routed_titles = {
        corpus.titles[int(index)] for index in evaluation_qa.pairs.flatten().tolist()
    }
    router_support_mismatches = sum(
        {corpus.titles[int(index)] for index in pair}
        != set(row["supporting_titles"])
        for row, pair in zip(
            evaluation_rows, evaluation_qa.pairs.tolist(), strict=True
        )
    )
    heldout_titles = frozenset(set(support_titles) | routed_titles)
    writer_training_indices, writer_heldout_indices = split_document_indices(
        corpus, heldout_titles
    )
    evaluation_pair_indices = set(evaluation_qa.pairs.flatten().tolist())
    heldout_index_set = set(writer_heldout_indices.tolist())
    train_pair_indices = set(train_qa.pairs.flatten().tolist())

    base_state, base_training = train_common_base(natural, validation, device)
    base_hash = hashlib.sha256()
    for name in sorted(base_state):
        base_hash.update(name.encode())
        base_hash.update(base_state[name].contiguous().numpy().tobytes())
    common_base_sha256 = base_hash.hexdigest()

    random_table = fixed_random_code_table(corpus)
    arms: dict[str, SpecializedArm] = {}
    arms["learned_writer"] = train_read_arm(
        "learned_writer",
        "learned",
        base_state,
        corpus,
        writer_training_indices,
        writer_heldout_indices,
        special,
        random_table,
        device,
    )
    arms["fixed_random_writer"] = train_read_arm(
        "fixed_random_writer",
        "random",
        base_state,
        corpus,
        writer_training_indices,
        writer_heldout_indices,
        special,
        random_table,
        device,
    )
    arms["zero_record_read"] = train_read_arm(
        "zero_record_read",
        "zero",
        base_state,
        corpus,
        writer_training_indices,
        writer_heldout_indices,
        special,
        random_table,
        device,
    )
    arms["dense_causal_3x"] = train_dense_arm(base_state, corpus, device)

    heldout_probes = build_probes(
        corpus,
        writer_heldout_indices,
        EVALUATION_PROBE_SEED,
        special,
        EVALUATION_PROBES_PER_DOCUMENT,
    )
    evaluate_probes(arms, heldout_probes, device)

    pretraining_code_hashes = {
        name: tensor_sha256(arm.codes) if arm.codes is not None else None
        for name, arm in arms.items()
    }
    qa_training: dict[str, dict[str, object]] = {}
    pretrain_wrappers: dict[str, t21a.PretrainedArm] = {}
    for name, arm in arms.items():
        wrapper = t21a.PretrainedArm(
            name=name,
            model=arm.model,
            code_parameters=None,
            quantized_codes=arm.codes,
            training=arm.training,
            pretrain_natural={},
            pretrain_documents={},
        )
        pretrain_wrappers[name] = wrapper
        qa_training[name] = t21a.train_qa_arm(wrapper, train_qa, device)

    qa_evaluation: dict[str, dict[str, dict[str, object]]] = {}
    for name, wrapper in pretrain_wrappers.items():
        qa_evaluation[name] = {
            "correct": t21a.evaluate_qa(wrapper, evaluation_qa, device, "correct")
        }
    candidate_wrapper = pretrain_wrappers["learned_writer"]
    qa_evaluation["learned_writer"]["zero"] = t21a.evaluate_qa(
        candidate_wrapper, evaluation_qa, device, "zero"
    )
    qa_evaluation["learned_writer"]["shuffle"] = t21a.evaluate_qa(
        candidate_wrapper, evaluation_qa, device, "shuffle"
    )

    final_natural = {
        name: core.small.evaluate_natural(
            arm.model,
            validation,
            device,
            core.EVALUATION_SEED + MODEL_SEED + 220_000,
            batches=32,
        )
        for name, arm in arms.items()
    }

    candidate = arms["learned_writer"]
    assert candidate.codes is not None
    code_values = candidate.codes.detach().cpu()
    code_indices = decoded_level_indices(code_values)
    bf16_indices = decoded_level_indices(code_values.to(torch.bfloat16).float())
    allowed_levels = torch.tensor(
        [CODE_AMPLITUDE * (-1.0 + 2.0 * index / 15.0) for index in range(16)]
    )
    all_on_levels = bool(
        ((code_values[..., None] - allowed_levels).abs().min(dim=-1).values < 1e-6)
        .all()
        .item()
    )
    post_qa_hashes = {
        name: tensor_sha256(arm.codes) if arm.codes is not None else None
        for name, arm in arms.items()
    }

    prospective = core.build_model(MODEL_SEED, device)
    physical = t19b.compile_physical_payload(prospective, title_data, code_values)
    schemas = {state_schema(arm.model) for arm in arms.values()}
    parameter_counts = {
        sum(parameter.numel() for parameter in arm.model.parameters())
        for arm in arms.values()
    }

    probe = candidate.probe_evaluation
    correct_probe_nll = float(probe["correct"]["nll"])
    zero_probe_nll = float(probe["zero"]["nll"])
    shuffle_probe_nll = float(probe["shuffle"]["nll"])
    random_probe_nll = float(
        arms["fixed_random_writer"].probe_evaluation["correct"]["nll"]
    )
    candidate_accuracy = float(
        qa_evaluation["learned_writer"]["correct"]["accuracy"]
    )
    best_control_accuracy = max(
        float(qa_evaluation[name]["correct"]["accuracy"])
        for name in (
            "fixed_random_writer",
            "zero_record_read",
            "dense_causal_3x",
        )
    )
    zero_accuracy = float(qa_evaluation["learned_writer"]["zero"]["accuracy"])
    shuffle_accuracy = float(
        qa_evaluation["learned_writer"]["shuffle"]["accuracy"]
    )
    candidate_natural = float(final_natural["learned_writer"]["nll"])
    dense_natural = float(final_natural["dense_causal_3x"]["nll"])

    initial_hashes = {arm.training["initial_state_sha256"] for arm in arms.values()}
    training_finite = all(
        arm.training["failed_or_retried_steps"] == 0
        and math.isfinite(float(arm.training["maximum_loss"]))
        and math.isfinite(float(arm.training["maximum_gradient_norm"]))
        and qa_training[name]["failed_or_retried_steps"] == 0
        and math.isfinite(float(qa_training[name]["maximum_loss"]))
        and math.isfinite(float(qa_training[name]["maximum_gradient_norm"]))
        for name, arm in arms.items()
    )
    raw_fields = set().union(*(document.keys() for document in documents))
    gates = {
        "input_integrity_and_raw_only_writer": all(input_checks.values())
        and raw_fields == {"document_id", "title", "text"}
        and tuple(inspect.signature(writer_codes).parameters)
        == ("model", "corpus", "indices", "device"),
        "identical_common_base_and_finite_training": len(initial_hashes) == 1
        and training_finite,
        "frozen_levels_bf16_and_code_hash": all_on_levels
        and bool(torch.isfinite(code_values).all())
        and torch.equal(code_indices, bf16_indices)
        and pretraining_code_hashes == post_qa_hashes,
        "writer_heldout_isolation_and_one_pass": evaluation_pair_indices
        <= heldout_index_set
        and not (train_pair_indices & heldout_index_set)
        and len(writer_training_indices) == 2_198
        and len(writer_heldout_indices) == 207
        and router_support_mismatches == 2
        and candidate.training["heldout_gradient_presentations"] == 0
        and candidate.training["heldout_compile_passes"]
        == len(writer_heldout_indices)
        and candidate.training["heldout_compile_minimum"] == 1
        and candidate.training["heldout_compile_maximum"] == 1,
        "probe_gain_over_zero_at_least_20pct": (
            zero_probe_nll - correct_probe_nll
        )
        / zero_probe_nll
        >= REQUIRED_PROBE_GAIN,
        "probe_gain_over_shuffle_at_least_20pct": (
            shuffle_probe_nll - correct_probe_nll
        )
        / shuffle_probe_nll
        >= REQUIRED_PROBE_GAIN,
        "probe_gain_over_random_at_least_10pct": (
            random_probe_nll - correct_probe_nll
        )
        / random_probe_nll
        >= REQUIRED_RANDOM_GAIN,
        "qa_accuracy_at_least_75pct": candidate_accuracy >= REQUIRED_QA_ACCURACY,
        "qa_gain_over_best_control_at_least_10pp": candidate_accuracy
        - best_control_accuracy
        >= REQUIRED_QA_GAIN,
        "zero_code_qa_drop_at_least_10pp": candidate_accuracy - zero_accuracy
        >= REQUIRED_QA_GAIN,
        "shuffle_code_qa_drop_at_least_10pp": candidate_accuracy
        - shuffle_accuracy
        >= REQUIRED_QA_GAIN,
        "natural_nll_within_0_5pct_of_dense": (
            candidate_natural - dense_natural
        )
        / dense_natural
        <= 0.005,
        "identical_served_structure_and_physical_ledger": len(schemas) == 1
        and len(parameter_counts) == 1
        and physical["total_writes"] == t19b.EXPECTED_WRITES
        and physical["total_writes"] <= t19b.WRITE_BUDGET,
    }

    return {
        "schema": SCHEMA,
        "device": str(device),
        "input_checks": input_checks,
        "data": {
            "documents": len(corpus.titles),
            "writer_training_documents": len(writer_training_indices),
            "writer_heldout_documents": len(writer_heldout_indices),
            "heldout_titles": len(heldout_titles),
            "sealed_support_titles": len(support_titles),
            "router_selected_titles": len(routed_titles),
            "router_support_mismatches": router_support_mismatches,
            "train_qa": len(train_rows),
            "evaluation_qa": len(evaluation_rows),
            "raw_fields": sorted(raw_fields),
            "special_tokens": {
                "target_mask": special.target_mask,
                "view_mask": special.view_mask,
                "answer_carrier": special.answer_carrier,
                "unused_count": special.unused_count,
            },
            "heldout_probe_examples": len(heldout_probes.inputs),
        },
        "common_base": {**base_training, "state_sha256": common_base_sha256},
        "arms": {
            name: {
                "training": arm.training,
                "probe_evaluation": arm.probe_evaluation,
                "qa_training": qa_training[name],
                "qa_evaluation": qa_evaluation[name],
                "final_natural": final_natural[name],
                "state_sha256": core.small.state_sha256(arm.model),
                "code_sha256_before_qa": pretraining_code_hashes[name],
                "code_sha256_after_qa": post_qa_hashes[name],
            }
            for name, arm in arms.items()
        },
        "digital_code": {
            "documents": len(code_values),
            "dimensions": CODE_DIMS,
            "logical_cells": code_values.numel(),
            "logical_bits": 4 * code_values.numel(),
            "unique_level_indices": sorted(set(code_indices.flatten().tolist())),
            "all_on_frozen_levels": all_on_levels,
            "bf16_level_indices_exact": torch.equal(code_indices, bf16_indices),
            "sha256": tensor_sha256(code_values),
            "per_document_optimized_parameters": 0,
        },
        "compute": {
            "candidate_specialization_sequence_token_positions": READ_STEPS
            * (BATCH + BATCH * PROBES_PER_DOCUMENT)
            * CONTEXT,
            "dense_specialization_sequence_token_positions": DENSE_STEPS
            * BATCH
            * CONTEXT,
            "one_pass_compile_sequence_token_positions": len(corpus.titles)
            * CONTEXT,
        },
        "physical": physical,
        "served": {
            "state_schema_variants": len(schemas),
            "parameter_count_variants": len(parameter_counts),
            "parameter_count": next(iter(parameter_counts)),
            "virtual_injection_exported": False,
        },
        "effects": {
            "probe_gain_over_zero_pct": 100.0
            * (zero_probe_nll - correct_probe_nll)
            / zero_probe_nll,
            "probe_gain_over_shuffle_pct": 100.0
            * (shuffle_probe_nll - correct_probe_nll)
            / shuffle_probe_nll,
            "probe_gain_over_random_pct": 100.0
            * (random_probe_nll - correct_probe_nll)
            / random_probe_nll,
            "qa_gain_over_best_control_pp": 100.0
            * (candidate_accuracy - best_control_accuracy),
            "qa_zero_code_drop_pp": 100.0 * (candidate_accuracy - zero_accuracy),
            "qa_shuffle_code_drop_pp": 100.0
            * (candidate_accuracy - shuffle_accuracy),
            "natural_nll_relative_to_dense_pct": 100.0
            * (candidate_natural - dense_natural)
            / dense_natural,
        },
        "gates": gates,
        "admitted": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "parent_preregistration_sha256": sha256_file(
                PARENT_PREREGISTRATION
            ),
            "v2_preregistration_sha256": sha256_file(V2_PREREGISTRATION),
            "root_preregistration_sha256": sha256_file(
                ROOT_PREREGISTRATION
            ),
            "candidate_corpus_sha256": sha256_file(t19a.CANDIDATE_CORPUS),
            "train_qa_sha256": sha256_file(t19b.TRAIN_QA),
            "evaluation_qa_sha256": sha256_file(t19b.EVALUATION_QA),
            "natural_train_sha256": sha256_file(t12.TRAIN_FILE),
            "natural_validation_sha256": sha256_file(t12.VALIDATION_FILE),
        },
        "limits": [
            "T22a uses exact title selection and virtual hidden-state addition.",
            "Passing admits physical export and new final evaluation; it is not itself a production claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = json_native(run(torch.device(arguments.device)))
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
