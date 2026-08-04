#!/usr/bin/env python3
"""T10: write compiler-recovered prose facts and equality into a standard LM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import consensus_permutation_raw_prose as raw
from experiments import digital_relation_plane_t8 as t8
from experiments import digital_relation_plane_t9_small_core as core
from experiments import prepare_raw_prose_equality_plane_t10_data as natural_data


OUTPUT = ROOT / "results/raw-prose-equality-plane-t10-quick.json"
PREREGISTRATION = ROOT / "results/raw-prose-equality-plane-t10-preregistration.md"
TRAIN_FILE = t8.TRAIN_FILE
VALIDATION_FILE = t8.VALIDATION_FILE

ENTITIES = 384
RELATIONS = 8
VALUES = 16
ALIAS_VIEWS = 2
PARAPHRASES = 2
VALUE_BITS = 4
RELATION_BITS = 3
MEMORY_COORDS = RELATIONS * VALUE_BITS
PROGRAM_HEADS = 2
PROGRAM_CHANNELS = 64

MEMORY_START = 0
RELATION_A_START = 32
RELATION_B_START = 35
ENTITY_TYPE = 38
RELATION_A_TYPE = 39
RELATION_B_TYPE = 40
QUERY_ROUTE = 41
CONST = 42
VALUE_A_START = 43
VALUE_B_START = 47
PRODUCT_START = 51
PROGRAM_DIMS = 55

HIDDEN = core.HIDDEN
LAYERS = core.LAYERS
HEAD_DIM = core.HEAD_DIM
FFN_WIDTH = core.FFN_WIDTH
RMS_EPSILON = core.RMS_EPSILON
BETA = 16.0


@dataclass(frozen=True)
class ProseLayout:
    alias_token: dict[str, int]
    frame_anchor_word: dict[raw.Frame, str]
    frame_anchor_token: dict[raw.Frame, int]
    frame_role: dict[raw.Frame, str]
    value_token: dict[str, int]
    direct_query: int
    equality_query: int
    result_same: int
    result_different: int
    filler_token: dict[str, int]

    @property
    def all_special(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(self.alias_token.values())
                | set(self.frame_anchor_token.values())
                | set(self.value_token.values())
                | {
                    self.direct_query,
                    self.equality_query,
                    self.result_same,
                    self.result_different,
                }
            )
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + RMS_EPSILON * HIDDEN))


def bipolar_code(index: int, bits: int) -> Tensor:
    return torch.tensor(
        [1.0 if (index >> bit) & 1 else -1.0 for bit in range(bits)]
    )


def discover_frame_anchors(compiled: raw.CompileResult) -> dict[raw.Frame, str]:
    counts = Counter(
        token
        for frame in compiled.induced_frames
        for token in set(frame.tokens)
        if token != raw.SLOT
    )
    anchors: dict[raw.Frame, str] = {}
    for frame in compiled.induced_frames:
        candidates = sorted(
            token
            for token in frame.tokens
            if token != raw.SLOT
            and counts[token] == 1
            and re.fullmatch(r"[a-z]+", token)
        )
        if len(candidates) != 1:
            raise raw.NonIdentifiable(
                f"frame has {len(candidates)} unique opaque anchors"
            )
        anchors[frame] = candidates[0]
    if len(set(anchors.values())) != len(anchors):
        raise raw.NonIdentifiable("surface anchors are not unique")
    return anchors


def choose_layout(
    natural: core.small.TokenStream,
    validation: core.small.TokenStream,
    corpus: raw.GeneratedCorpus,
    compiled: raw.CompileResult,
) -> tuple[ProseLayout, int]:
    counts = np.zeros(core.small.VOCAB, dtype=np.int64)
    for stream in (natural, validation):
        counts += np.bincount(
            np.asarray(stream.tokens, dtype=np.int64), minlength=core.small.VOCAB
        )
    unused = [int(index) for index in np.flatnonzero(counts == 0)]
    frame_anchor_word = discover_frame_anchors(compiled)
    aliases = sorted(compiled.alias_to_canonical)
    values = sorted({value for row in compiled.canonical_values for value in row})
    required = len(aliases) + len(frame_anchor_word) + len(values) + 4
    if len(unused) < required:
        raise RuntimeError(f"need {required} unused tokens, found {len(unused)}")
    cursor = 0

    def allocate(words: list[str]) -> dict[str, int]:
        nonlocal cursor
        result = {word: unused[cursor + index] for index, word in enumerate(words)}
        cursor += len(words)
        return result

    alias_token = allocate(aliases)
    anchor_word_to_token = allocate(sorted(frame_anchor_word.values()))
    value_token = allocate(values)
    direct_query, equality_query, result_same, result_different = unused[
        cursor : cursor + 4
    ]

    frames_by_relation: dict[int, list[raw.Frame]] = defaultdict(list)
    for frame, relation in compiled.frame_to_relation.items():
        frames_by_relation[relation].append(frame)
    frame_role: dict[raw.Frame, str] = {}
    for relation in range(compiled.canonical_relation_count):
        frames = sorted(frames_by_relation[relation])
        if len(frames) < 2 or len(frames) % 2:
            raise raw.NonIdentifiable("each relation needs an even number of surfaces")
        midpoint = len(frames) // 2
        for frame in frames[:midpoint]:
            frame_role[frame] = "a"
        for frame in frames[midpoint:]:
            frame_role[frame] = "b"

    special_words = set(alias_token) | set(value_token) | set(anchor_word_to_token)
    filler_words = sorted(
        {
            token
            for sentence in corpus.sentences
            for token in raw.tokenize(sentence)
            if token not in special_words
        }
    )
    used_ids = [int(index) for index in np.flatnonzero(counts > 0)]
    if len(used_ids) < len(filler_words):
        raise RuntimeError("not enough used natural tokens for prose fillers")
    filler_token = dict(
        zip(filler_words, used_ids[: len(filler_words)], strict=True)
    )
    return (
        ProseLayout(
            alias_token=alias_token,
            frame_anchor_word=frame_anchor_word,
            frame_anchor_token={
                frame: anchor_word_to_token[word]
                for frame, word in frame_anchor_word.items()
            },
            frame_role=frame_role,
            value_token=value_token,
            direct_query=direct_query,
            equality_query=equality_query,
            result_same=result_same,
            result_different=result_different,
            filler_token=filler_token,
        ),
        len(unused),
    )


def value_codes(compiled: raw.CompileResult) -> dict[str, Tensor]:
    values = sorted({value for row in compiled.canonical_values for value in row})
    if len(values) != VALUES:
        raise raw.NonIdentifiable(f"expected {VALUES} values, recovered {len(values)}")
    return {value: bipolar_code(index, VALUE_BITS) for index, value in enumerate(values)}


@torch.no_grad()
def compile_equality_plane(
    model: core.small.SharedInterpreterLM,
    compiled: raw.CompileResult,
    layout: ProseLayout,
) -> tuple[t8.FrozenEntries, dict[str, object]]:
    """Write only recovered strings and equivalence maps; no latent world accepted."""
    if compiled.canonical_relation_count != RELATIONS:
        raise ValueError("relation count mismatch")
    codes = value_codes(compiled)
    named = dict(model.named_parameters())
    masks = {
        name: torch.zeros_like(parameter, dtype=torch.bool)
        for name, parameter in named.items()
    }

    def fix(name: str, index: object, value: float = 0.0) -> None:
        named[name][index] = value
        masks[name][index] = True

    fix("token.weight", (slice(None), slice(0, PROGRAM_DIMS)))
    for token_id in layout.all_special:
        fix("token.weight", (token_id, slice(None)))

    canonical_index = {
        alias: index for index, alias in enumerate(compiled.canonical_entities)
    }
    for surface_alias, canonical_alias in compiled.alias_to_canonical.items():
        token_id = layout.alias_token[surface_alias]
        row = compiled.canonical_values[canonical_index[canonical_alias]]
        named["token.weight"][token_id, ENTITY_TYPE] = 1.0
        for relation, value in enumerate(row):
            start = MEMORY_START + relation * VALUE_BITS
            named["token.weight"][token_id, start : start + VALUE_BITS] = codes[
                value
            ].to(model.token.weight.device)

    for frame, relation in compiled.frame_to_relation.items():
        token_id = layout.frame_anchor_token[frame]
        relation_code = bipolar_code(relation, RELATION_BITS).to(
            model.token.weight.device
        )
        if layout.frame_role[frame] == "a":
            named["token.weight"][
                token_id, RELATION_A_START : RELATION_A_START + RELATION_BITS
            ] = relation_code
            named["token.weight"][token_id, RELATION_A_TYPE] = 1.0
        else:
            named["token.weight"][
                token_id, RELATION_B_START : RELATION_B_START + RELATION_BITS
            ] = relation_code
            named["token.weight"][token_id, RELATION_B_TYPE] = 1.0

    for query in (layout.direct_query, layout.equality_query):
        named["token.weight"][query, QUERY_ROUTE] = 1.0
        named["token.weight"][query, CONST] = 1.0
    for value, token_id in layout.value_token.items():
        named["token.weight"][
            token_id, VALUE_A_START : VALUE_A_START + VALUE_BITS
        ] = 30.0 * codes[value].to(model.token.weight.device)
    named["token.weight"][
        layout.result_same, PRODUCT_START : PRODUCT_START + VALUE_BITS
    ] = 30.0
    named["token.weight"][layout.result_same, CONST] = -90.0
    named["token.weight"][
        layout.result_different, PRODUCT_START : PRODUCT_START + VALUE_BITS
    ] = -30.0
    named["token.weight"][layout.result_different, CONST] = 90.0

    program_rows = slice(0, PROGRAM_HEADS * HEAD_DIM)
    language_rows = slice(PROGRAM_HEADS * HEAD_DIM, HIDDEN)
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}."
        for norm in ("attention_norm.weight", "ffn_norm.weight"):
            fix(prefix + norm, slice(0, PROGRAM_DIMS), 1.0)
        for projection in ("attention.q.weight", "attention.k.weight", "attention.v.weight"):
            fix(prefix + projection, (program_rows, slice(None)))
            fix(prefix + projection, (language_rows, slice(0, PROGRAM_DIMS)))
        fix(prefix + "attention.o.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "attention.o.weight", (slice(PROGRAM_DIMS, None), program_rows))
        for projection in ("gate.weight", "up.weight"):
            fix(prefix + projection, (slice(0, PROGRAM_CHANNELS), slice(None)))
            fix(prefix + projection, (slice(PROGRAM_CHANNELS, None), slice(0, PROGRAM_DIMS)))
        fix(prefix + "down.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "down.weight", (slice(PROGRAM_DIMS, None), slice(0, PROGRAM_CHANNELS)))
    fix("final_norm.weight", slice(0, PROGRAM_DIMS), 1.0)

    block0 = model.blocks[0]
    alpha_entity = rms_scale(MEMORY_COORDS + 1.0)
    alpha_relation = rms_scale(RELATION_BITS + 1.0)
    alpha_query = rms_scale(2.0)
    entity_route = math.sqrt(
        math.sqrt(HEAD_DIM) * 60.0 / (alpha_entity * alpha_query)
    )
    block0.attention.q.weight[63, QUERY_ROUTE] = entity_route
    block0.attention.k.weight[63, ENTITY_TYPE] = entity_route
    for coordinate in range(MEMORY_COORDS):
        block0.attention.v.weight[coordinate, MEMORY_START + coordinate] = 1.0
        block0.attention.o.weight[MEMORY_START + coordinate, coordinate] = (
            1.0 / alpha_entity
        )

    second_head = HEAD_DIM
    relation_route = math.sqrt(
        math.sqrt(HEAD_DIM) * 60.0 / (alpha_relation * alpha_query)
    )
    block0.attention.q.weight[second_head + 62, QUERY_ROUTE] = relation_route
    block0.attention.q.weight[second_head + 63, QUERY_ROUTE] = relation_route
    block0.attention.k.weight[second_head + 62, RELATION_A_TYPE] = relation_route
    block0.attention.k.weight[second_head + 63, RELATION_B_TYPE] = relation_route
    for bit in range(RELATION_BITS):
        block0.attention.v.weight[
            second_head + bit, RELATION_A_START + bit
        ] = 1.0
        block0.attention.v.weight[
            second_head + RELATION_BITS + bit, RELATION_B_START + bit
        ] = 1.0
        block0.attention.o.weight[
            RELATION_A_START + bit, second_head + bit
        ] = 2.0 / alpha_relation
        block0.attention.o.weight[
            RELATION_B_START + bit, second_head + RELATION_BITS + bit
        ] = 2.0 / alpha_relation

    alpha_select = rms_scale(MEMORY_COORDS + 2 * RELATION_BITS + 2.0)
    block1 = model.blocks[1]
    channel = 0
    for role, relation_start, output_start in (
        ("a", RELATION_A_START, VALUE_A_START),
        ("b", RELATION_B_START, VALUE_B_START),
    ):
        del role
        for relation in range(RELATIONS):
            code = bipolar_code(relation, RELATION_BITS).to(model.token.weight.device)
            for bit in range(VALUE_BITS):
                block1.gate.weight[
                    channel, relation_start : relation_start + RELATION_BITS
                ] = BETA * code / alpha_select
                block1.gate.weight[channel, CONST] = (
                    -BETA * (RELATION_BITS - 1) / alpha_select
                )
                memory_coordinate = MEMORY_START + relation * VALUE_BITS + bit
                block1.up.weight[channel, memory_coordinate] = 1.0 / (
                    BETA * alpha_select
                )
                block1.down.weight[output_start + bit, channel] = 1.0
                channel += 1
    if channel != PROGRAM_CHANNELS:
        raise AssertionError("selection channel accounting mismatch")

    alpha_product = rms_scale(
        MEMORY_COORDS + 2 * RELATION_BITS + 2.0 + 2 * VALUE_BITS
    )
    block2 = model.blocks[2]
    for bit in range(VALUE_BITS):
        positive = 2 * bit
        negative = positive + 1
        block2.gate.weight[positive, VALUE_A_START + bit] = BETA / alpha_product
        block2.gate.weight[negative, VALUE_A_START + bit] = -BETA / alpha_product
        for product_channel in (positive, negative):
            block2.up.weight[product_channel, VALUE_B_START + bit] = 1.0 / (
                BETA * alpha_product
            )
        block2.down.weight[PRODUCT_START + bit, positive] = 1.0
        block2.down.weight[PRODUCT_START + bit, negative] = -1.0

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = t8.FrozenEntries(
        masks,
        values,
        sum(int(mask.sum().item()) for mask in masks.values()),
    )
    frozen.enforce(model)
    fixed_nonzero = sum(
        int(((values[name] != 0) & mask).sum().item())
        for name, mask in masks.items()
    )
    return frozen, {
        "recovered_entities": len(compiled.canonical_entities),
        "surface_aliases": len(compiled.alias_to_canonical),
        "relations": compiled.canonical_relation_count,
        "surface_frames": len(compiled.frame_to_relation),
        "direct_facts": len(compiled.canonical_entities)
        * compiled.canonical_relation_count,
        "payload_scalars": len(compiled.alias_to_canonical)
        * compiled.canonical_relation_count
        * VALUE_BITS,
        "program_hidden_coordinates": PROGRAM_DIMS,
        "program_attention_heads": PROGRAM_HEADS,
        "selection_channels": PROGRAM_CHANNELS,
        "product_channels": 2 * VALUE_BITS,
        "frozen_parameter_entries": frozen.count,
        "fixed_nonzero_entries": fixed_nonzero,
    }


def frames_by_relation_and_role(
    compiled: raw.CompileResult, layout: ProseLayout
) -> dict[tuple[int, str], tuple[raw.Frame, ...]]:
    grouped: dict[tuple[int, str], list[raw.Frame]] = defaultdict(list)
    for frame, relation in compiled.frame_to_relation.items():
        grouped[(relation, layout.frame_role[frame])].append(frame)
    return {key: tuple(sorted(frames)) for key, frames in grouped.items()}


def direct_dataset(
    compiled: raw.CompileResult, layout: ProseLayout
) -> tuple[Tensor, Tensor]:
    """All alias/paraphrase direct queries, derived only from compiler output."""
    grouped = frames_by_relation_and_role(compiled, layout)
    canonical_index = {
        alias: index for index, alias in enumerate(compiled.canonical_entities)
    }
    values = sorted(layout.value_token)
    value_index = {value: index for index, value in enumerate(values)}
    rows: list[list[int]] = []
    targets: list[int] = []
    for alias, canonical in sorted(compiled.alias_to_canonical.items()):
        entity = canonical_index[canonical]
        for relation in range(RELATIONS):
            fallback_b = grouped[(relation, "b")][0]
            for frame_a in grouped[(relation, "a")]:
                rows.append(
                    [
                        layout.alias_token[alias],
                        layout.frame_anchor_token[frame_a],
                        layout.frame_anchor_token[fallback_b],
                        layout.direct_query,
                    ]
                )
                targets.append(
                    value_index[compiled.canonical_values[entity][relation]]
                )
    return torch.tensor(rows, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def equality_dataset(
    compiled: raw.CompileResult, layout: ProseLayout
) -> tuple[Tensor, Tensor]:
    """All ordered relation-pair equality queries over every recovered surface."""
    grouped = frames_by_relation_and_role(compiled, layout)
    canonical_index = {
        alias: index for index, alias in enumerate(compiled.canonical_entities)
    }
    rows: list[list[int]] = []
    targets: list[int] = []
    for alias, canonical in sorted(compiled.alias_to_canonical.items()):
        entity = canonical_index[canonical]
        for first in range(RELATIONS):
            for second in range(RELATIONS):
                for frame_a in grouped[(first, "a")]:
                    for frame_b in grouped[(second, "b")]:
                        rows.append(
                            [
                                layout.alias_token[alias],
                                layout.frame_anchor_token[frame_a],
                                layout.frame_anchor_token[frame_b],
                                layout.equality_query,
                            ]
                        )
                        targets.append(
                            int(
                                compiled.canonical_values[entity][first]
                                == compiled.canonical_values[entity][second]
                            )
                        )
    return torch.tensor(rows, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def encode_raw_corpus(
    corpus: raw.GeneratedCorpus, layout: ProseLayout
) -> dict[int, Tensor]:
    """Encode strings without template or latent metadata; group only by length."""
    lexical = {
        **layout.alias_token,
        **layout.value_token,
        **{
            word: layout.frame_anchor_token[frame]
            for frame, word in layout.frame_anchor_word.items()
        },
        **layout.filler_token,
    }
    grouped: dict[int, list[list[int]]] = defaultdict(list)
    for sentence in corpus.sentences:
        tokens = raw.tokenize(sentence)
        try:
            encoded = [lexical[token] for token in tokens]
        except KeyError as error:
            raise raw.NonIdentifiable(f"unmapped raw token {error.args[0]}") from error
        grouped[len(encoded)].append(encoded)
    return {
        length: torch.tensor(rows, dtype=torch.long)
        for length, rows in sorted(grouped.items())
    }


@torch.no_grad()
def evaluate(
    model: core.small.SharedInterpreterLM,
    compiled: raw.CompileResult,
    layout: ProseLayout,
    device: torch.device,
    *,
    batch_size: int = 4_096,
) -> dict[str, object]:
    model.eval()
    values = sorted(layout.value_token)
    value_rows = torch.tensor(
        [layout.value_token[value] for value in values],
        dtype=torch.long,
        device=device,
    )
    direct_tensor, direct_target_tensor = direct_dataset(compiled, layout)

    direct_predictions: list[Tensor] = []
    direct_margins: list[Tensor] = []
    for start in range(0, len(direct_tensor), batch_size):
        tokens = direct_tensor[start : start + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.hidden(tokens)[:, -1]
            logits = hidden @ model.token.weight[value_rows].T
        ordered = logits.float().topk(2, dim=-1).values
        direct_predictions.append(logits.argmax(-1).cpu())
        direct_margins.append((ordered[:, 0] - ordered[:, 1]).cpu())
    predicted_direct = torch.cat(direct_predictions)
    direct_correct = predicted_direct == direct_target_tensor

    equality_tensor, equality_target_tensor = equality_dataset(compiled, layout)
    equality_result_rows = torch.tensor(
        [layout.result_different, layout.result_same],
        dtype=torch.long,
        device=device,
    )
    equality_predictions: list[Tensor] = []
    equality_margins: list[Tensor] = []
    for start in range(0, len(equality_tensor), batch_size):
        tokens = equality_tensor[start : start + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.hidden(tokens)[:, -1]
            logits = hidden @ model.token.weight[equality_result_rows].T
        ordered = logits.float().topk(2, dim=-1).values
        equality_predictions.append(logits.argmax(-1).cpu())
        equality_margins.append((ordered[:, 0] - ordered[:, 1]).cpu())
    predicted_equality = torch.cat(equality_predictions)
    equality_correct = predicted_equality == equality_target_tensor
    return {
        "direct_accuracy": float(direct_correct.float().mean().item()),
        "direct_errors": int((~direct_correct).sum().item()),
        "direct_queries": len(direct_tensor),
        "direct_minimum_margin": float(torch.cat(direct_margins).min().item()),
        "equality_accuracy": float(equality_correct.float().mean().item()),
        "equality_errors": int((~equality_correct).sum().item()),
        "equality_queries": len(equality_tensor),
        "equality_positive_rate": float(equality_target_tensor.float().mean().item()),
        "equality_minimum_margin": float(torch.cat(equality_margins).min().item()),
    }


@torch.no_grad()
def perturb_and_enforce(
    model: core.small.SharedInterpreterLM, frozen: t8.FrozenEntries
) -> bool:
    named = dict(model.named_parameters())
    for name, mask in frozen.masks.items():
        if mask.any():
            named[name][mask] += 1.0
    frozen.enforce(model)
    return all(
        torch.equal(named[name][mask], frozen.values[name][mask])
        for name, mask in frozen.masks.items()
        if mask.any()
    )


def run(device: torch.device) -> dict[str, object]:
    if TRAIN_FILE.parent.resolve() != natural_data.OUTPUT_DIR.resolve():
        raise RuntimeError(
            f"T10 data directory drifted: {TRAIN_FILE.parent} != {natural_data.OUTPUT_DIR}"
        )
    verified_natural_data = natural_data.verify_data(TRAIN_FILE.parent)
    generation_started = time.perf_counter()
    corpus = raw.make_corpus(
        seed=6_000_059,
        entities=ENTITIES,
        relations=RELATIONS,
        values=VALUES,
        views=ALIAS_VIEWS,
        paraphrases=PARAPHRASES,
        repeats=1,
        corruption_rate=0.0,
    )
    generation_seconds = time.perf_counter() - generation_started
    extraction_started = time.perf_counter()
    compiled = raw.compile_corpus(corpus.sentences)
    extraction_seconds = time.perf_counter() - extraction_started
    extraction = raw.evaluate_recovery(corpus, compiled)

    natural = core.small.TokenStream(TRAIN_FILE)
    validation = core.small.TokenStream(VALIDATION_FILE)
    layout, unused = choose_layout(natural, validation, corpus, compiled)
    model = core.build_model(6_101, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    writer_started = time.perf_counter()
    frozen, writer = compile_equality_plane(model, compiled, layout)
    if device.type == "cuda":
        torch.cuda.synchronize()
    writer_seconds = time.perf_counter() - writer_started
    evaluation = evaluate(model, compiled, layout, device)
    enforcement_exact = perturb_and_enforce(model, frozen)
    evaluation_after_enforce = evaluate(model, compiled, layout, device)

    gates = {
        "raw_compiler_direct_exact": extraction["direct_fact_accuracy"] == 1.0,
        "raw_compiler_alias_exact": extraction["entity_alias_accuracy"] == 1.0,
        "raw_compiler_relation_exact": extraction["relation_paraphrase_accuracy"]
        == 1.0,
        "same_parameter_count": parameter_count == core.small.parameter_count(),
        "direct_exact": evaluation["direct_accuracy"] == 1.0
        and evaluation["direct_errors"] == 0,
        "direct_positive_margin": evaluation["direct_minimum_margin"] > 0.0,
        "equality_exact": evaluation["equality_accuracy"] == 1.0
        and evaluation["equality_errors"] == 0,
        "equality_positive_margin": evaluation["equality_minimum_margin"] > 0.0,
        "all_surfaces_present": len(layout.alias_token) == ENTITIES * ALIAS_VIEWS
        and len(layout.frame_anchor_token)
        == RELATIONS * ALIAS_VIEWS * PARAPHRASES,
        "writer_accepts_no_latent_world": tuple(
            __import__("inspect").signature(compile_equality_plane).parameters
        )
        == ("model", "compiled", "layout"),
        "frozen_perturb_enforce_exact": enforcement_exact,
        "post_enforce_outputs_exact": evaluation_after_enforce == evaluation,
    }
    return {
        "schema": "raw-prose-equality-plane-t10-quick-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": parameter_count,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": core.HEADS,
            "ffn_width": FFN_WIDTH,
            "entities": ENTITIES,
            "relations": RELATIONS,
            "values": VALUES,
            "alias_views": ALIAS_VIEWS,
            "surface_frames": len(compiled.induced_frames),
            "raw_sentences": len(corpus.sentences),
            "unused_tokens_available": unused,
            "special_tokens_consumed": len(layout.all_special),
        },
        "natural_data": verified_natural_data,
        "extraction": extraction,
        "writer": writer,
        "evaluation": evaluation,
        "resource_ledger": {
            "generation_cpu_seconds": generation_seconds,
            "extraction_cpu_seconds": extraction_seconds,
            "writer_gpu_seconds": writer_seconds if device.type == "cuda" else 0.0,
            "compiler_gpu_seconds": 0.0,
        },
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "extractor_source_sha256": sha256_file(Path(raw.__file__)),
            "data_preparer_source_sha256": sha256_file(Path(natural_data.__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "stage0b_sha256": sha256_file(
                ROOT / "results/consensus-permutation-raw-prose-stage0b.json"
            ),
            "raw_corpus_sha256": raw.corpus_sha256(corpus.sentences),
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VALIDATION_FILE),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "claim_boundary": (
            "This is exact controlled-surface representation integration. "
            "The query protocol is four lexical tokens, not unrestricted natural QA; "
            "matched gradient training has not yet been run."
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
                "evaluation": result["evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
