#!/usr/bin/env python3
"""T22a: raw-only quotient reads and equality over quantized document records."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import inspect
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import joint_quantized_document_code_t21a as t21a
from experiments import raw_self_query_functional_record_stage0 as stage0


PREREGISTRATION = ROOT / "results/raw-self-query-functional-record-t22a-preregistration.md"
STAGE0_RESULT = ROOT / "results/raw-self-query-functional-record-stage0.json"
OUTPUT = ROOT / "results/raw-self-query-functional-record-t22a.json"
SOURCE = Path(__file__).resolve()
TEST_SOURCE = ROOT / "tests/test_raw_self_query_functional_record_t22a.py"
SCHEMA = "raw-self-query-functional-record-t22a-v1"
STAGE0_SHA256 = "f46a36b5c617bb62e18e9fc81111dc3df2a8e8d351d5808e6173deaa79c1d660"
MODEL_SEED = 8_209
CODE_SEED = 22_001
SHUFFLE_SEED = 22_005
PROBE_SEED = 22_009
PAIR_SEED = 22_013
CODE_DIMS = t21a.CODE_DIMS
CODE_START = t21a.CODE_START
CODE_AMPLITUDE = t21a.CODE_AMPLITUDE
CONTEXT = 64
BATCH = 16
CYCLES = 1_000
ONE_X_NATURAL = 19
ONE_X_UNARY = 8
ONE_X_EQUALITY = 2
TWO_X_NATURAL = 19
TWO_X_UNARY = 16
TWO_X_EQUALITY = 4
ONE_X_STEPS = CYCLES * (ONE_X_NATURAL + ONE_X_UNARY + ONE_X_EQUALITY)
TWO_X_STEPS = CYCLES * (TWO_X_NATURAL + TWO_X_UNARY + TWO_X_EQUALITY)
WARMUP = 500
MUON_LR = 0.005
ADAMW_LR = 0.0003
CODE_LR = 0.03
REQUIRED_UNARY_GAIN = 0.20
REQUIRED_EQUALITY_ACCURACY = 0.80
REQUIRED_EQUALITY_DROP = 0.20
REQUIRED_QA_ACCURACY = 0.75
REQUIRED_QA_GAIN = 0.10
YES_TOKEN = t21a.YES_TOKEN
NO_TOKEN = t21a.NO_TOKEN


core = t21a.core
t7 = t21a.t7
t19a = t21a.t19a
t19b = t21a.t19b
t12 = t21a.t12


@dataclasses.dataclass(frozen=True)
class SpecialLayout:
    read: int
    mask: int
    answer: int
    compare: int
    second: int
    unused_available: int

    @property
    def all_special(self) -> tuple[int, ...]:
        return (self.read, self.mask, self.answer, self.compare, self.second)


def choose_special_layout(
    natural: core.small.TokenStream,
    validation: core.small.TokenStream,
    documents: list[dict[str, str]],
    tokenizer: object,
) -> SpecialLayout:
    used = np.zeros(core.small.VOCAB, dtype=np.bool_)
    for stream in (natural, validation):
        used[np.asarray(stream.tokens, dtype=np.int64)] = True
    for document in documents:
        identifiers = tokenizer.encode(
            document["title"] + "\n" + document["text"],
            add_special_tokens=False,
        )
        used[np.asarray(identifiers, dtype=np.int64)] = True
    unused = np.flatnonzero(~used).tolist()
    if len(unused) < 5:
        raise RuntimeError(f"need five unused token rows, found {len(unused)}")
    return SpecialLayout(*(int(value) for value in unused[:5]), len(unused))


@dataclasses.dataclass(frozen=True)
class EncodedProbe:
    document_index: int
    query_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    skeleton: tuple[tuple[int, str], ...]
    target_word: str
    structured: bool

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.query_ids + self.target_ids[:-1]


@dataclasses.dataclass(frozen=True)
class ProbeDataset:
    rows: tuple[EncodedProbe, ...]
    by_document: tuple[tuple[int, ...], ...]
    structured_indices: tuple[int, ...]
    maximum_input_tokens: int


@dataclasses.dataclass(frozen=True)
class EqualityExample:
    first_document: int
    second_document: int
    query_ids: tuple[int, ...]
    target: int


@dataclasses.dataclass(frozen=True)
class EqualityDataset:
    positive: tuple[EqualityExample, ...]
    negative: tuple[EqualityExample, ...]


@dataclasses.dataclass(frozen=True)
class FunctionalData:
    train_probes: ProbeDataset
    evaluation_probes: ProbeDataset
    train_equality: EqualityDataset
    evaluation_equality: EqualityDataset
    layout: SpecialLayout
    stage0_counts: dict[str, object]


def encode_skeleton(
    skeleton: tuple[tuple[int, str], ...],
    tokenizer: object,
    layout: SpecialLayout,
    prefix: int,
    suffix: int,
) -> tuple[int, ...]:
    anchors = dict(skeleton)
    identifiers = [prefix]
    for offset in range(-stage0.RADIUS, stage0.RADIUS + 1):
        if offset == 0:
            continue
        word = anchors.get(offset)
        if word is None:
            identifiers.append(layout.mask)
        else:
            values = tokenizer.encode(" " + word, add_special_tokens=False)
            if not values:
                raise RuntimeError(f"empty anchor encoding: {word!r}")
            identifiers.extend(int(value) for value in values)
    identifiers.append(suffix)
    return tuple(identifiers)


def representative_collision_probes(
    probes: list[stage0.Probe],
) -> tuple[
    dict[tuple[tuple[int, str], ...], set[tuple[int, str]]],
    dict[tuple[tuple[int, str], ...], dict[tuple[int, str], stage0.Probe]],
]:
    collisions, _, _ = stage0.single_valued_collision_groups(probes)
    representatives: dict[
        tuple[tuple[int, str], ...], dict[tuple[int, str], stage0.Probe]
    ] = {}
    grouped: dict[
        tuple[tuple[int, str], ...], dict[tuple[int, str], list[stage0.Probe]]
    ] = collections.defaultdict(lambda: collections.defaultdict(list))
    for probe in probes:
        grouped[probe.skeleton][(probe.document_index, probe.target)].append(probe)
    for skeleton, values in collisions.items():
        representatives[skeleton] = {
            value: min(grouped[skeleton][value], key=lambda probe: probe.target_index)
            for value in values
        }
    return collisions, representatives


def build_probe_dataset(
    probes: list[stage0.Probe],
    document_count: int,
    tokenizer: object,
    layout: SpecialLayout,
) -> tuple[ProbeDataset, dict[tuple[tuple[int, str], ...], set[tuple[int, str]]]]:
    collisions, representatives = representative_collision_probes(probes)
    selected: dict[tuple[int, tuple[tuple[int, str], ...], str], stage0.Probe] = {}
    for skeleton, values in representatives.items():
        for (document, target), probe in values.items():
            selected[(document, skeleton, target)] = probe
    structured_documents = {key[0] for key in selected}
    by_raw_document: dict[int, list[stage0.Probe]] = collections.defaultdict(list)
    for probe in probes:
        by_raw_document[probe.document_index].append(probe)
    for document in range(document_count):
        if document not in structured_documents:
            fallback = min(
                by_raw_document[document],
                key=lambda probe: (stage0.stable_u64(probe.identity), probe.target_index),
            )
            selected[(document, fallback.skeleton, fallback.target)] = fallback

    encoded: list[EncodedProbe] = []
    for key, probe in sorted(
        selected.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        query = encode_skeleton(
            probe.skeleton, tokenizer, layout, layout.read, layout.answer
        )
        target = tuple(
            int(value)
            for value in tokenizer.encode(" " + probe.target, add_special_tokens=False)
        )
        if not target:
            raise RuntimeError(f"empty target encoding: {probe.target!r}")
        row = EncodedProbe(
            document_index=probe.document_index,
            query_ids=query,
            target_ids=target,
            skeleton=probe.skeleton,
            target_word=probe.target,
            structured=probe.document_index in structured_documents
            and (probe.document_index, probe.target)
            in collisions.get(probe.skeleton, set()),
        )
        if len(row.input_ids) > CONTEXT:
            raise RuntimeError(
                f"probe needs {len(row.input_ids)} tokens, context is {CONTEXT}"
            )
        encoded.append(row)

    document_rows: list[list[int]] = [[] for _ in range(document_count)]
    for index, row in enumerate(encoded):
        document_rows[row.document_index].append(index)
    if any(not rows for rows in document_rows):
        raise RuntimeError("a document has no functional read")
    dataset = ProbeDataset(
        rows=tuple(encoded),
        by_document=tuple(tuple(rows) for rows in document_rows),
        structured_indices=tuple(
            index for index, row in enumerate(encoded) if row.structured
        ),
        maximum_input_tokens=max(len(row.input_ids) for row in encoded),
    )
    return dataset, collisions


def build_equality_dataset(
    collisions: dict[tuple[tuple[int, str], ...], set[tuple[int, str]]],
    tokenizer: object,
    layout: SpecialLayout,
) -> EqualityDataset:
    positives: list[EqualityExample] = []
    negatives: list[EqualityExample] = []
    for skeleton, values in sorted(collisions.items()):
        query = encode_skeleton(
            skeleton, tokenizer, layout, layout.compare, layout.second
        )
        by_target: dict[str, list[int]] = collections.defaultdict(list)
        by_document = dict(values)
        for document, target in sorted(values):
            by_target[target].append(document)
        group_positives: list[EqualityExample] = []
        for documents in by_target.values():
            for first, second in itertools.combinations(sorted(documents), 2):
                group_positives.append(
                    EqualityExample(first, second, query, 0)
                )
        if not group_positives:
            continue
        positives.extend(group_positives)
        for first, second in itertools.combinations(sorted(by_document), 2):
            if by_document[first] != by_document[second]:
                negatives.append(EqualityExample(first, second, query, 1))
    positives.sort(
        key=lambda row: stage0.stable_u64(
            f"positive:{row.first_document}:{row.second_document}:{row.query_ids}"
        )
    )
    negatives.sort(
        key=lambda row: stage0.stable_u64(
            f"negative:{row.first_document}:{row.second_document}:{row.query_ids}"
        )
    )
    if not positives or not negatives:
        raise RuntimeError("equality set lacks a class")
    return EqualityDataset(tuple(positives), tuple(negatives))


def build_functional_data(
    documents: list[dict[str, str]],
    tokenizer: object,
    natural: core.small.TokenStream,
    validation: core.small.TokenStream,
) -> FunctionalData:
    raw_probes, stage0_vocabulary = stage0.build_probes(documents)
    layout = choose_special_layout(natural, validation, documents, tokenizer)
    train_raw = [probe for probe in raw_probes if probe.split == "train"]
    evaluation_raw = [
        probe for probe in raw_probes if probe.split == "evaluation"
    ]
    train_probes, train_collisions = build_probe_dataset(
        train_raw, len(documents), tokenizer, layout
    )
    evaluation_probes, evaluation_collisions = build_probe_dataset(
        evaluation_raw, len(documents), tokenizer, layout
    )
    train_equality = build_equality_dataset(
        train_collisions, tokenizer, layout
    )
    evaluation_equality = build_equality_dataset(
        evaluation_collisions, tokenizer, layout
    )
    return FunctionalData(
        train_probes=train_probes,
        evaluation_probes=evaluation_probes,
        train_equality=train_equality,
        evaluation_equality=evaluation_equality,
        layout=layout,
        stage0_counts={
            **stage0_vocabulary,
            "raw_probes": len(raw_probes),
            "train_encoded_reads": len(train_probes.rows),
            "evaluation_encoded_reads": len(evaluation_probes.rows),
            "train_structured_reads": len(train_probes.structured_indices),
            "evaluation_structured_reads": len(
                evaluation_probes.structured_indices
            ),
            "train_positive_pairs": len(train_equality.positive),
            "evaluation_positive_pairs": len(evaluation_equality.positive),
        },
    )


def pack_unary_rows(
    rows: list[EncodedProbe], eos: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    inputs = torch.full((len(rows), CONTEXT), eos, dtype=torch.long)
    targets = torch.full_like(inputs, eos)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    documents = torch.tensor(
        [row.document_index for row in rows], dtype=torch.long
    )
    for index, row in enumerate(rows):
        values = row.input_ids
        inputs[index, : len(values)] = torch.tensor(values, dtype=torch.long)
        first_target_position = len(row.query_ids) - 1
        for offset, target in enumerate(row.target_ids):
            position = first_target_position + offset
            targets[index, position] = target
            mask[index, position] = True
    return (
        inputs.to(device),
        targets.to(device),
        mask.to(device),
        documents.to(device),
    )


def sample_unary_batch(
    dataset: ProbeDataset,
    seed: int,
    batch_size: int,
    eos: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = np.random.default_rng(seed)
    documents = generator.integers(
        0, len(dataset.by_document), size=batch_size, dtype=np.int64
    )
    indices = [
        choices[int(document)][
            int(generator.integers(0, len(choices[int(document)])))
        ]
        for document in documents
        for choices in (dataset.by_document,)
    ]
    return pack_unary_rows(
        [dataset.rows[index] for index in indices], eos, device
    )


def balanced_equality_rows(
    dataset: EqualityDataset,
    seed: int,
    batch_size: int,
) -> list[EqualityExample]:
    if batch_size % 2:
        raise ValueError("balanced equality batch must be even")
    generator = np.random.default_rng(seed)
    half = batch_size // 2
    positives = generator.integers(0, len(dataset.positive), size=half)
    negatives = generator.integers(0, len(dataset.negative), size=half)
    rows = [dataset.positive[int(index)] for index in positives]
    rows.extend(dataset.negative[int(index)] for index in negatives)
    generator.shuffle(rows)
    return rows


def pack_equality_rows(
    rows: list[EqualityExample], eos: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    inputs = torch.full((len(rows), CONTEXT), eos, dtype=torch.long)
    lengths = torch.tensor([len(row.query_ids) for row in rows], dtype=torch.long)
    pairs = torch.tensor(
        [(row.first_document, row.second_document) for row in rows],
        dtype=torch.long,
    )
    positions = torch.stack(
        (torch.zeros_like(lengths), lengths - 1), dim=1
    )
    targets = torch.tensor([row.target for row in rows], dtype=torch.long)
    for index, row in enumerate(rows):
        inputs[index, : len(row.query_ids)] = torch.tensor(
            row.query_ids, dtype=torch.long
        )
    return (
        inputs.to(device),
        lengths.to(device),
        pairs.to(device),
        positions.to(device),
        targets.to(device),
    )


def phase_for_step(arm: str, step: int) -> str:
    if arm in {"dense_probe_1x", "functional_record_1x"}:
        cycle = ONE_X_NATURAL + ONE_X_UNARY + ONE_X_EQUALITY
        position = (step - 1) % cycle
        if position < ONE_X_NATURAL:
            return "natural"
        if position < ONE_X_NATURAL + ONE_X_UNARY:
            return "unary"
        return "equality"
    if arm == "dense_probe_2x":
        cycle = TWO_X_NATURAL + TWO_X_UNARY + TWO_X_EQUALITY
        position = (step - 1) % cycle
        if position < TWO_X_NATURAL:
            return "natural"
        if position < TWO_X_NATURAL + TWO_X_UNARY:
            return "unary"
        return "equality"
    raise ValueError(f"unknown arm: {arm}")


def equality_logits(
    model: core.small.SharedInterpreterLM,
    inputs: Tensor,
    lengths: Tensor,
    positions: Tensor,
    records: Tensor | None,
) -> Tensor:
    hidden = t21a.hidden_with_records(
        model, inputs, records, positions if records is not None else None
    )
    final = hidden[torch.arange(len(inputs), device=inputs.device), lengths - 1]
    rows = model.token.weight[
        torch.tensor([YES_TOKEN, NO_TOKEN], dtype=torch.long, device=inputs.device)
    ]
    return final @ rows.T


def torch_optimizer_state_bytes(
    optimizer: torch.optim.Optimizer | None,
) -> int:
    if optimizer is None:
        return 0
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


@torch.no_grad()
def evaluate_unary(
    model: core.small.SharedInterpreterLM,
    dataset: ProbeDataset,
    eos: int,
    device: torch.device,
    codes: Tensor | None,
    mode: str = "correct",
) -> dict[str, object]:
    model.eval()
    selected_codes = codes
    if selected_codes is not None:
        if mode == "zero":
            selected_codes = torch.zeros_like(selected_codes)
        elif mode == "shuffle":
            generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
            permutation = torch.randperm(
                len(selected_codes), generator=generator
            ).to(selected_codes.device)
            selected_codes = selected_codes[permutation]
        elif mode != "correct":
            raise ValueError(f"unknown unary mode: {mode}")
    elif mode != "correct":
        raise ValueError("dense unary evaluation has no record ablation")
    total_loss = 0.0
    total_tokens = 0
    indices = dataset.structured_indices
    for start in range(0, len(indices), BATCH):
        chosen = indices[start : start + BATCH]
        rows = [dataset.rows[index] for index in chosen]
        inputs, targets, mask, documents = pack_unary_rows(rows, eos, device)
        records = (
            selected_codes[documents] if selected_codes is not None else None
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = t21a.document_forward(model, inputs, records)
        losses = F.cross_entropy(
            logits.float().reshape(-1, core.small.VOCAB),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        total_loss += float((losses * mask.float()).sum().item())
        total_tokens += int(mask.sum().item())
    return {
        "mode": mode,
        "structured_reads": len(indices),
        "target_tokens": total_tokens,
        "nll": total_loss / total_tokens,
        "finite": math.isfinite(total_loss),
    }


def balanced_evaluation_rows(dataset: EqualityDataset) -> list[EqualityExample]:
    count = min(len(dataset.positive), len(dataset.negative))
    rows = list(dataset.positive[:count]) + list(dataset.negative[:count])
    rows.sort(
        key=lambda row: stage0.stable_u64(
            f"eval:{row.target}:{row.first_document}:{row.second_document}:{row.query_ids}"
        )
    )
    return rows


@torch.no_grad()
def evaluate_equality(
    model: core.small.SharedInterpreterLM,
    dataset: EqualityDataset,
    eos: int,
    device: torch.device,
    codes: Tensor | None,
    mode: str = "correct",
) -> dict[str, object]:
    model.eval()
    selected_codes = codes
    if selected_codes is not None:
        if mode == "zero":
            selected_codes = torch.zeros_like(selected_codes)
        elif mode == "shuffle":
            generator = torch.Generator(device="cpu").manual_seed(
                SHUFFLE_SEED + 1
            )
            permutation = torch.randperm(
                len(selected_codes), generator=generator
            ).to(selected_codes.device)
            selected_codes = selected_codes[permutation]
        elif mode != "correct":
            raise ValueError(f"unknown equality mode: {mode}")
    elif mode != "correct":
        raise ValueError("dense equality evaluation has no record ablation")
    rows = balanced_evaluation_rows(dataset)
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        inputs, lengths, pairs, positions, batch_targets = pack_equality_rows(
            batch, eos, device
        )
        records = selected_codes[pairs] if selected_codes is not None else None
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = equality_logits(
                model, inputs, lengths, positions, records
            )
        predictions.append(logits.float().argmax(dim=-1).cpu())
        targets.append(batch_targets.cpu())
    predicted = torch.cat(predictions)
    expected = torch.cat(targets)
    return {
        "mode": mode,
        "examples": len(expected),
        "positive_examples": int((expected == 0).sum().item()),
        "accuracy": float((predicted == expected).float().mean().item()),
        "errors": int((predicted != expected).sum().item()),
        "predicted_positive_rate": float(
            (predicted == 0).float().mean().item()
        ),
    }


@dataclasses.dataclass
class PretrainedArm:
    name: str
    model: core.small.SharedInterpreterLM
    code_parameters: nn.Parameter | None
    quantized_codes: Tensor | None
    training: dict[str, object]
    pretrain_natural: dict[str, object]
    unary_evaluation: dict[str, dict[str, object]]
    equality_evaluation: dict[str, dict[str, object]]


def train_pretraining_arm(
    arm: str,
    natural: core.small.TokenStream,
    validation: core.small.TokenStream,
    data: FunctionalData,
    eos: int,
    device: torch.device,
) -> PretrainedArm:
    total_steps = TWO_X_STEPS if arm == "dense_probe_2x" else ONE_X_STEPS
    model = core.build_model(MODEL_SEED, device)
    initial_state_hash = core.small.state_sha256(model)
    optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=MUON_LR
    )
    code_parameters: nn.Parameter | None = None
    code_optimizer: torch.optim.Optimizer | None = None
    if arm == "functional_record_1x":
        generator = torch.Generator(device="cpu").manual_seed(CODE_SEED)
        initial = 0.05 * torch.randn(
            len(data.train_probes.by_document),
            CODE_DIMS,
            generator=generator,
        )
        code_parameters = nn.Parameter(initial.to(device))
        code_optimizer = torch.optim.Adam(
            (code_parameters,), lr=CODE_LR, weight_decay=0.0
        )
    updates = collections.Counter()
    token_presentations = collections.Counter()
    maximum_loss = 0.0
    maximum_model_gradient = 0.0
    maximum_code_gradient = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad()
        if code_optimizer is not None:
            code_optimizer.zero_grad(set_to_none=True)
        t21a.set_bundle_learning_rate(
            optimizer,
            step,
            total_steps,
            WARMUP,
            MUON_LR,
            ADAMW_LR,
        )
        phase = phase_for_step(arm, step)
        updates[phase] += 1
        if phase == "natural":
            inputs, targets = natural.batch(
                MODEL_SEED * 1_000_003 + updates[phase],
                BATCH,
                t21a.CONTEXT,
                device,
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, core.small.VOCAB),
                targets.reshape(-1),
            )
            token_presentations[phase] += BATCH * t21a.CONTEXT
        elif phase == "unary":
            inputs, targets, mask, documents = sample_unary_batch(
                data.train_probes,
                PROBE_SEED + updates[phase],
                BATCH,
                eos,
                device,
            )
            records = None
            if code_parameters is not None:
                codes, _ = t21a.quantize_16_ste(code_parameters)
                records = codes[documents]
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = t21a.document_forward(model, inputs, records)
            loss = t21a.masked_lm_loss(logits, targets, mask)
            token_presentations[phase] += int(mask.sum().item())
        else:
            rows = balanced_equality_rows(
                data.train_equality,
                PAIR_SEED + updates[phase],
                BATCH,
            )
            inputs, lengths, pairs, positions, targets = pack_equality_rows(
                rows, eos, device
            )
            records = None
            if code_parameters is not None:
                codes, _ = t21a.quantize_16_ste(code_parameters)
                records = codes[pairs]
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = equality_logits(
                    model, inputs, lengths, positions, records
                )
            loss = F.cross_entropy(logits.float(), targets)
            token_presentations[phase] += BATCH
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm}/{phase} loss at {step}")
        loss.backward()
        model_gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(model_gradient):
            raise RuntimeError(f"nonfinite {arm} model gradient at {step}")
        maximum_model_gradient = max(maximum_model_gradient, model_gradient)
        if code_parameters is not None and code_parameters.grad is not None:
            code_gradient = float(code_parameters.grad.norm().item())
            if not math.isfinite(code_gradient):
                raise RuntimeError(f"nonfinite code gradient at {step}")
            maximum_code_gradient = max(maximum_code_gradient, code_gradient)
        optimizer.step()
        if code_optimizer is not None and phase != "natural":
            code_optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step % 1_000 == 0:
            print(
                json.dumps(
                    {
                        "phase": "pretrain",
                        "arm": arm,
                        "step": step,
                        "last_update": phase,
                        "loss": float(loss.item()),
                        "updates": dict(updates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    quantized_codes = None
    if code_parameters is not None:
        with torch.no_grad():
            quantized_codes, _ = t21a.quantize_16_ste(code_parameters)
            quantized_codes = quantized_codes.detach().clone()
    natural_result = core.small.evaluate_natural(
        model,
        validation,
        device,
        core.EVALUATION_SEED + MODEL_SEED + total_steps,
        batches=32,
    )
    unary_modes = ("correct", "zero", "shuffle") if quantized_codes is not None else ("correct",)
    equality_modes = unary_modes
    unary_result = {
        mode: evaluate_unary(
            model,
            data.evaluation_probes,
            eos,
            device,
            quantized_codes,
            mode,
        )
        for mode in unary_modes
    }
    equality_result = {
        mode: evaluate_equality(
            model,
            data.evaluation_equality,
            eos,
            device,
            quantized_codes,
            mode,
        )
        for mode in equality_modes
    }
    model_optimizer_state_bytes = t7.optimizer_state_bytes(optimizer)
    code_optimizer_state_bytes = torch_optimizer_state_bytes(code_optimizer)
    del optimizer, code_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return PretrainedArm(
        name=arm,
        model=model,
        code_parameters=code_parameters,
        quantized_codes=quantized_codes,
        training={
            "initial_state_sha256": initial_state_hash,
            "total_updates": total_steps,
            "updates": dict(updates),
            "token_presentations": dict(token_presentations),
            "maximum_loss": maximum_loss,
            "maximum_model_gradient_norm": maximum_model_gradient,
            "maximum_code_gradient_norm": maximum_code_gradient,
            "failed_or_retried_steps": 0,
            "elapsed_seconds": elapsed,
            "peak_hbm_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            )
            if device.type == "cuda"
            else 0,
            "model_optimizer_state_bytes": model_optimizer_state_bytes,
            "code_optimizer_state_bytes": code_optimizer_state_bytes,
        },
        pretrain_natural=natural_result,
        unary_evaluation=unary_result,
        equality_evaluation=equality_result,
    )


def verify_inputs() -> dict[str, bool]:
    checks = {
        "candidate_corpus": t21a.sha256_file(t19a.CANDIDATE_CORPUS)
        == t19a.CANDIDATE_SHA256,
        "train_qa": t21a.sha256_file(t19b.TRAIN_QA)
        == t19b.TRAIN_QA_SHA256,
        "evaluation_qa": t21a.sha256_file(t19b.EVALUATION_QA)
        == t19a.EVALUATOR_SHA256,
        "natural_train": t21a.sha256_file(t12.TRAIN_FILE)
        == t19a.NATURAL_TRAIN_SHA256,
        "natural_validation": t21a.sha256_file(t12.VALIDATION_FILE)
        == t19a.NATURAL_VALIDATION_SHA256,
        "stage0_result": t21a.sha256_file(STAGE0_RESULT) == STAGE0_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference


def run(device: torch.device) -> dict[str, object]:
    torch.set_float32_matmul_precision("high")
    input_checks = verify_inputs()
    stage0_result = json.loads(STAGE0_RESULT.read_text())
    if not stage0_result["stage0_pass"]:
        raise RuntimeError("Stage 0 did not admit the GPU test")
    documents = t19a.load_raw_documents(t19a.CANDIDATE_CORPUS)
    compiler_fields = set().union(*(document.keys() for document in documents))
    tokenizer = AutoTokenizer.from_pretrained(
        t19a.TOKENIZER, revision=t19a.TOKENIZER_REVISION
    )
    if tokenizer.encode(" yes", add_special_tokens=False) != [YES_TOKEN]:
        raise RuntimeError("yes token drifted")
    if tokenizer.encode(" no", add_special_tokens=False) != [NO_TOKEN]:
        raise RuntimeError("no token drifted")
    natural = core.small.TokenStream(t12.TRAIN_FILE)
    validation = core.small.TokenStream(t12.VALIDATION_FILE)
    train_rows = t19b.load_qa(t19b.TRAIN_QA)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    all_qa_rows = [*train_rows, *evaluation_rows]
    data = build_functional_data(
        documents, tokenizer, natural, validation
    )
    train_edge_ids = {
        (row.document_index, row.skeleton, row.target_word)
        for row in data.train_probes.rows
    }
    evaluation_edge_ids = {
        (row.document_index, row.skeleton, row.target_word)
        for row in data.evaluation_probes.rows
    }
    functional_data_integrity = (
        data.stage0_counts["raw_probes"] == 108_570
        and len(data.train_probes.structured_indices) == 10_746
        and len(data.evaluation_probes.structured_indices) == 1_743
        and len(data.train_equality.positive) == 562
        and len(data.train_equality.negative) == 21_148
        and len(data.evaluation_equality.positive) == 36
        and len(data.evaluation_equality.negative) == 224
        and not (train_edge_ids & evaluation_edge_ids)
        and data.train_probes.maximum_input_tokens <= CONTEXT
        and data.evaluation_probes.maximum_input_tokens <= CONTEXT
    )
    qa_token_ids = {
        int(identifier)
        for row in all_qa_rows
        for identifier in tokenizer.encode(
            str(row["question"]), add_special_tokens=False
        )
    }
    if qa_token_ids & set(data.layout.all_special):
        raise RuntimeError("raw-selected control token occurs in QA text")
    title_data = t19a.build_title_data(documents, tokenizer)
    train_qa = t21a.build_qa_data(
        train_rows, title_data.titles, tokenizer
    )
    evaluation_qa = t21a.build_qa_data(
        evaluation_rows, title_data.titles, tokenizer
    )
    eos = int(tokenizer.eos_token_id)

    arms: dict[str, PretrainedArm] = {}
    for name in (
        "dense_probe_1x",
        "functional_record_1x",
        "dense_probe_2x",
    ):
        arms[name] = train_pretraining_arm(
            name, natural, validation, data, eos, device
        )

    qa_training: dict[str, dict[str, object]] = {}
    for name, arm in arms.items():
        qa_training[name] = t21a.train_qa_arm(arm, train_qa, device)
    qa_results: dict[str, dict[str, dict[str, object]]] = {}
    for name, arm in arms.items():
        qa_results[name] = {
            "correct": t21a.evaluate_qa(
                arm, evaluation_qa, device, "correct"
            )
        }
    candidate = arms["functional_record_1x"]
    qa_results[candidate.name]["zero"] = t21a.evaluate_qa(
        candidate, evaluation_qa, device, "zero"
    )
    qa_results[candidate.name]["shuffle"] = t21a.evaluate_qa(
        candidate, evaluation_qa, device, "shuffle"
    )
    final_natural = {
        name: core.small.evaluate_natural(
            arm.model,
            validation,
            device,
            core.EVALUATION_SEED + MODEL_SEED + 120_000,
            batches=32,
        )
        for name, arm in arms.items()
    }

    assert candidate.quantized_codes is not None
    code_values = candidate.quantized_codes.detach().cpu()
    code_indices = t21a.decoded_level_indices(code_values)
    bf16_indices = t21a.decoded_level_indices(
        code_values.to(torch.bfloat16).float()
    )
    allowed_levels = torch.tensor(
        [
            CODE_AMPLITUDE * (-1.0 + 2.0 * index / 15.0)
            for index in range(16)
        ]
    )
    all_on_levels = bool(
        (
            (code_values[..., None] - allowed_levels)
            .abs()
            .min(dim=-1)
            .values
            < 1e-6
        )
        .all()
    )
    prospective = core.build_model(MODEL_SEED, device)
    physical = t19b.compile_physical_payload(
        prospective, title_data, code_values
    )
    schemas = {t21a.state_schema(arm.model) for arm in arms.values()}
    parameter_counts = {
        sum(parameter.numel() for parameter in arm.model.parameters())
        for arm in arms.values()
    }

    unary = candidate.unary_evaluation
    unary_correct = float(unary["correct"]["nll"])
    unary_zero = float(unary["zero"]["nll"])
    unary_shuffle = float(unary["shuffle"]["nll"])
    equality = candidate.equality_evaluation
    equality_correct = float(equality["correct"]["accuracy"])
    equality_zero = float(equality["zero"]["accuracy"])
    equality_shuffle = float(equality["shuffle"]["accuracy"])
    best_dense_equality = max(
        float(arms[name].equality_evaluation["correct"]["accuracy"])
        for name in ("dense_probe_1x", "dense_probe_2x")
    )
    candidate_qa = float(qa_results[candidate.name]["correct"]["accuracy"])
    zero_qa = float(qa_results[candidate.name]["zero"]["accuracy"])
    shuffle_qa = float(qa_results[candidate.name]["shuffle"]["accuracy"])
    best_dense_qa = max(
        float(qa_results[name]["correct"]["accuracy"])
        for name in ("dense_probe_1x", "dense_probe_2x")
    )
    candidate_natural = float(final_natural[candidate.name]["nll"])
    matched_natural = float(final_natural["dense_probe_1x"]["nll"])
    training_finite = all(
        arm.training["failed_or_retried_steps"] == 0
        and math.isfinite(float(arm.training["maximum_loss"]))
        and math.isfinite(
            float(arm.training["maximum_model_gradient_norm"])
        )
        and qa_training[name]["failed_or_retried_steps"] == 0
        and math.isfinite(float(qa_training[name]["maximum_loss"]))
        and math.isfinite(
            float(qa_training[name]["maximum_gradient_norm"])
        )
        for name, arm in arms.items()
    )
    gates = {
        "input_integrity_and_raw_only_record_acquisition": all(
            input_checks.values()
        )
        and compiler_fields == {"document_id", "title", "text"}
        and tuple(inspect.signature(stage0.build_probes).parameters)
        == ("documents",)
        and not (qa_token_ids & set(data.layout.all_special))
        and functional_data_integrity,
        "identical_initial_models_and_finite_training": len(
            {
                arm.training["initial_state_sha256"]
                for arm in arms.values()
            }
        )
        == 1
        and training_finite,
        "frozen_16_levels_and_bf16_indices": all_on_levels
        and torch.equal(code_indices, bf16_indices)
        and bool(torch.isfinite(code_values).all()),
        "record_frozen_during_qa": bool(
            qa_training[candidate.name]["code_unchanged"]
        ),
        "identical_served_structure_and_physical_ledger": len(schemas) == 1
        and len(parameter_counts) == 1
        and physical["total_writes"] == t19b.EXPECTED_WRITES
        and physical["total_writes"] <= t19b.WRITE_BUDGET,
        "unary_nll_gain_over_zero_at_least_20pct": relative_gain(
            unary_zero, unary_correct
        )
        >= REQUIRED_UNARY_GAIN,
        "unary_nll_gain_over_shuffle_at_least_20pct": relative_gain(
            unary_shuffle, unary_correct
        )
        >= REQUIRED_UNARY_GAIN,
        "equality_accuracy_at_least_80pct": equality_correct
        >= REQUIRED_EQUALITY_ACCURACY,
        "equality_gain_over_best_dense_at_least_20pp": equality_correct
        - best_dense_equality
        >= REQUIRED_EQUALITY_DROP,
        "zero_record_equality_drop_at_least_20pp": equality_correct
        - equality_zero
        >= REQUIRED_EQUALITY_DROP,
        "shuffled_record_equality_drop_at_least_20pp": equality_correct
        - equality_shuffle
        >= REQUIRED_EQUALITY_DROP,
        "qa_accuracy_at_least_75pct": candidate_qa
        >= REQUIRED_QA_ACCURACY,
        "qa_gain_over_best_dense_at_least_10pp": candidate_qa
        - best_dense_qa
        >= REQUIRED_QA_GAIN,
        "zero_record_qa_drop_at_least_10pp": candidate_qa - zero_qa
        >= REQUIRED_QA_GAIN,
        "shuffled_record_qa_drop_at_least_10pp": candidate_qa - shuffle_qa
        >= REQUIRED_QA_GAIN,
        "natural_nll_within_0_5pct_of_matched_dense": (
            candidate_natural - matched_natural
        )
        / matched_natural
        <= 0.005,
    }
    effects = {
        "unary_nll_gain_over_zero_pct": 100.0
        * relative_gain(unary_zero, unary_correct),
        "unary_nll_gain_over_shuffle_pct": 100.0
        * relative_gain(unary_shuffle, unary_correct),
        "equality_gain_over_best_dense_pp": 100.0
        * (equality_correct - best_dense_equality),
        "equality_zero_record_drop_pp": 100.0
        * (equality_correct - equality_zero),
        "equality_shuffle_record_drop_pp": 100.0
        * (equality_correct - equality_shuffle),
        "qa_gain_over_best_dense_pp": 100.0
        * (candidate_qa - best_dense_qa),
        "qa_zero_record_drop_pp": 100.0 * (candidate_qa - zero_qa),
        "qa_shuffle_record_drop_pp": 100.0
        * (candidate_qa - shuffle_qa),
        "natural_nll_relative_to_matched_dense_pct": 100.0
        * (candidate_natural - matched_natural)
        / matched_natural,
    }
    return {
        "schema": SCHEMA,
        "device": str(device),
        "input_checks": input_checks,
        "data": {
            "documents": len(documents),
            "train_qa": len(train_rows),
            "evaluation_qa": len(evaluation_rows),
            "compiler_fields": sorted(compiler_fields),
            "stage0": data.stage0_counts,
            "unused_special_tokens_available": data.layout.unused_available,
            "special_token_ids": list(data.layout.all_special),
            "maximum_train_probe_tokens": data.train_probes.maximum_input_tokens,
            "maximum_evaluation_probe_tokens": data.evaluation_probes.maximum_input_tokens,
            "train_positive_equality_pairs": len(
                data.train_equality.positive
            ),
            "train_negative_equality_pairs": len(
                data.train_equality.negative
            ),
            "evaluation_positive_equality_pairs": len(
                data.evaluation_equality.positive
            ),
            "evaluation_negative_equality_pairs": len(
                data.evaluation_equality.negative
            ),
        },
        "arms": {
            name: {
                "training": arm.training,
                "pretrain_natural": arm.pretrain_natural,
                "unary_evaluation": arm.unary_evaluation,
                "equality_evaluation": arm.equality_evaluation,
                "qa_training": qa_training[name],
                "qa_evaluation": qa_results[name],
                "final_natural": final_natural[name],
                "state_sha256": core.small.state_sha256(arm.model),
            }
            for name, arm in arms.items()
        },
        "digital_record": {
            "documents": len(code_values),
            "dimensions": CODE_DIMS,
            "logical_cells": code_values.numel(),
            "logical_bits": 4 * code_values.numel(),
            "amplitude": CODE_AMPLITUDE,
            "unique_level_indices": sorted(
                set(code_indices.flatten().tolist())
            ),
            "all_on_frozen_levels": all_on_levels,
            "bf16_level_indices_exact": torch.equal(
                code_indices, bf16_indices
            ),
            "sha256": t21a.tensor_sha256(code_values),
            "training_parameter_bytes": candidate.code_parameters.numel()
            * candidate.code_parameters.element_size()
            if candidate.code_parameters is not None
            else 0,
        },
        "physical": physical,
        "served": {
            "state_schema_variants": len(schemas),
            "parameter_count_variants": len(parameter_counts),
            "parameter_count": next(iter(parameter_counts)),
            "virtual_injection_exported": False,
        },
        "effects": effects,
        "gates": gates,
        "admitted": all(gates.values()),
        "integrity": {
            "candidate_corpus_sha256": t21a.sha256_file(
                t19a.CANDIDATE_CORPUS
            ),
            "train_qa_sha256": t21a.sha256_file(t19b.TRAIN_QA),
            "evaluation_qa_sha256": t21a.sha256_file(
                t19b.EVALUATION_QA
            ),
            "natural_train_sha256": t21a.sha256_file(t12.TRAIN_FILE),
            "natural_validation_sha256": t21a.sha256_file(
                t12.VALIDATION_FILE
            ),
            "stage0_result_sha256": t21a.sha256_file(STAGE0_RESULT),
            "preregistration_sha256": t21a.sha256_file(PREREGISTRATION),
            "executed_source_sha256": t21a.sha256_file(SOURCE),
            "test_source_sha256": t21a.sha256_file(TEST_SOURCE),
        },
        "limits": [
            "The reused Hotpot split is development evidence, not untouched final evidence.",
            "Exact title lookup and virtual hidden-state record addition remain privileged.",
            "A pass admits physical ordinary-weight export, untouched evaluation, replication, and scale; it does not complete the production claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = t21a.json_native(run(torch.device(arguments.device)))
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
