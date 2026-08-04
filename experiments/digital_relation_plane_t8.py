#!/usr/bin/env python3
"""T8: compile dense arbitrary facts into an in-graph digital relation plane."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import heterogeneous_family_discovery_t7_scale as t7


OUTPUT = ROOT / "results/digital-relation-plane-t8-quick.json"
TRAIN_FILE = t7.t6.t3.TRAIN_FILE
VALIDATION_FILE = t7.t6.t3.VALIDATION_FILE

VOCAB = t7.VOCAB
HIDDEN = t7.HIDDEN
LAYERS = t7.LAYERS
HEAD_DIM = t7.HEAD_DIM
FFN_WIDTH = t7.FFN_WIDTH
RMS_EPSILON = t7.RMS_EPSILON

STRUCTURED_ENTITIES = 512
UNSEEN_ENTITIES = 64
ENTITIES = STRUCTURED_ENTITIES + UNSEEN_ENTITIES
RELATIONS = 64
RELATION_BITS = 6
VALUES = 16
FACTS = STRUCTURED_ENTITIES * RELATIONS
PROGRAM_HEADS = 2
PROGRAM_CHANNELS = 64

MEMORY_START = 0
COMPENSATION = 64
RELATION_START = 65
ENTITY_TYPE = 71
QUERY_ROUTE = 72
CONST = 73
CODE = 74
OUTPUT_START = 75
PROGRAM_DIMS = 79

MAX_LEVEL = VALUES - 1
ENTITY_NORM_SQUARED = RELATIONS * MAX_LEVEL**2 + 1.0
WORLD_SEED = 4_000_037
EVALUATION_SEED = 4_100_041

ScaleLM = t7.ScaleLM
TokenStream = t7.TokenStream
FrozenEntries = t7.FrozenEntries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RelationLayout:
    entity: tuple[int, ...]
    relation_query: tuple[int, ...]
    result: tuple[int, ...]

    @property
    def all_special(self) -> tuple[int, ...]:
        return self.entity + self.relation_query + self.result


@dataclass(frozen=True)
class RelationWorld:
    values: Tensor


def make_world(seed: int = WORLD_SEED) -> RelationWorld:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(
        0,
        VALUES,
        (STRUCTURED_ENTITIES, RELATIONS),
        generator=generator,
        dtype=torch.long,
    )
    return RelationWorld(values)


def relation_code(relation: int) -> Tensor:
    return torch.tensor(
        [1.0 if (relation >> bit) & 1 else -1.0 for bit in range(RELATION_BITS)]
    )


def value_level(value: Tensor) -> Tensor:
    return value.float().mul(2.0).sub(MAX_LEVEL)


def value_code(value: int) -> Tensor:
    return torch.tensor(
        [1.0 if (value >> bit) & 1 else -1.0 for bit in range(4)]
    )


def choose_layout(
    train: TokenStream, validation: TokenStream
) -> tuple[RelationLayout, int]:
    counts = np.zeros(VOCAB, dtype=np.int64)
    for stream in (train, validation):
        counts += np.bincount(np.asarray(stream.tokens, dtype=np.int64), minlength=VOCAB)
    required = ENTITIES + RELATIONS + VALUES
    unused = np.flatnonzero(counts == 0)
    if len(unused) < required:
        raise RuntimeError(f"need {required} unused tokens, found {len(unused)}")
    selected = tuple(int(value) for value in unused[:required])
    cursor = 0
    entities = selected[cursor : cursor + ENTITIES]
    cursor += ENTITIES
    queries = selected[cursor : cursor + RELATIONS]
    cursor += RELATIONS
    results = selected[cursor : cursor + VALUES]
    return RelationLayout(entities, queries, results), len(unused)


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + RMS_EPSILON * HIDDEN))


def encode_queries(
    entities: Tensor,
    relations: Tensor,
    layout: RelationLayout,
    device: torch.device,
) -> Tensor:
    entity_tokens = torch.tensor(layout.entity, dtype=torch.long)[entities]
    relation_tokens = torch.tensor(layout.relation_query, dtype=torch.long)[relations]
    return torch.stack((entity_tokens, relation_tokens), dim=1).to(device)


@torch.no_grad()
def compile_relation_plane(
    model: ScaleLM,
    world: RelationWorld,
    layout: RelationLayout,
) -> tuple[FrozenEntries, dict[str, object]]:
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

    # One amplitude-coded scalar stores one arbitrary 16-way fact.  A single
    # compensation scalar makes every entity row have exactly the same norm,
    # so RMSNorm cannot distort the 16 code levels differently per entity.
    levels = value_level(world.values)
    for entity, token_id in enumerate(layout.entity):
        named["token.weight"][token_id, ENTITY_TYPE] = 1.0
        if entity < STRUCTURED_ENTITIES:
            row = levels[entity].to(model.token.weight.device)
            named["token.weight"][token_id, MEMORY_START : MEMORY_START + RELATIONS] = row
            used_squared = float(row.float().pow(2).sum().item()) + 1.0
        else:
            used_squared = 1.0
        compensation = math.sqrt(max(ENTITY_NORM_SQUARED - used_squared, 0.0))
        named["token.weight"][token_id, COMPENSATION] = compensation

    for relation, token_id in enumerate(layout.relation_query):
        named["token.weight"][token_id, RELATION_START : RELATION_START + RELATION_BITS] = (
            relation_code(relation).to(model.token.weight.device)
        )
        named["token.weight"][token_id, QUERY_ROUTE] = 1.0
        named["token.weight"][token_id, CONST] = 1.0

    for value, token_id in enumerate(layout.result):
        named["token.weight"][token_id, OUTPUT_START : OUTPUT_START + 4] = (
            30.0 * value_code(value).to(model.token.weight.device)
        )

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

    # Block 0: two existing heads copy the entity's 64 fact amplitudes and its
    # norm compensation into the relation-specific query token.
    block0 = model.blocks[0]
    alpha_entity = rms_scale(ENTITY_NORM_SQUARED)
    alpha_query = rms_scale(RELATION_BITS + 2.0)
    route = math.sqrt(
        math.sqrt(HEAD_DIM) * 60.0 / (alpha_entity * alpha_query)
    )
    block0.attention.q.weight[63, QUERY_ROUTE] = route
    block0.attention.k.weight[63, ENTITY_TYPE] = route
    for coordinate in range(RELATIONS):
        block0.attention.v.weight[coordinate, MEMORY_START + coordinate] = 1.0
        block0.attention.o.weight[MEMORY_START + coordinate, coordinate] = 1.0 / alpha_entity

    second_head = HEAD_DIM
    block0.attention.q.weight[second_head + 63, QUERY_ROUTE] = route
    block0.attention.k.weight[second_head + 63, ENTITY_TYPE] = route
    block0.attention.v.weight[second_head, COMPENSATION] = 1.0
    block0.attention.o.weight[COMPENSATION, second_head] = 1.0 / alpha_entity

    # The copied entity payload has norm ENTITY_NORM_SQUARED - 1.  Six relation
    # bits plus query/constant markers add eight, hence this is constant.
    query_norm_squared = ENTITY_NORM_SQUARED + 7.0
    alpha_select = rms_scale(query_norm_squared)

    # Block 1: each of 64 shared channels recognizes one 6-bit relation code,
    # selects its amplitude, and moves it into CODE without changing total norm.
    block1 = model.blocks[1]
    beta = 16.0
    for relation in range(RELATIONS):
        code = relation_code(relation).to(model.token.weight.device)
        block1.gate.weight[
            relation, RELATION_START : RELATION_START + RELATION_BITS
        ] = beta * code / alpha_select
        block1.gate.weight[relation, CONST] = (
            -beta * (RELATION_BITS - 1) / alpha_select
        )
        block1.up.weight[relation, MEMORY_START + relation] = 1.0 / (
            beta * alpha_select
        )
        block1.down.weight[CODE, relation] = 1.0
        block1.down.weight[MEMORY_START + relation, relation] = -1.0

    # Block 2: a 48-channel shared triangular decoder turns the selected odd
    # amplitude into four output bits.  Result-token rows decode by Hamming dot.
    block2 = model.blocks[2]
    alpha_decode = rms_scale(query_norm_squared)
    channel = 0
    for value in range(VALUES):
        level = 2 * value - MAX_LEVEL
        bits = value_code(value).to(model.token.weight.device)
        for threshold, coefficient in (
            (level - 1, 1.0),
            (level, -2.0),
            (level + 1, 1.0),
        ):
            block2.gate.weight[channel, CODE] = beta / alpha_decode
            block2.gate.weight[channel, CONST] = -beta * threshold / alpha_decode
            block2.up.weight[channel, QUERY_ROUTE] = 1.0 / (beta * alpha_decode)
            for bit in range(4):
                block2.down.weight[OUTPUT_START + bit, channel] = (
                    bits[bit] * coefficient
                )
            channel += 1

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = FrozenEntries(
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
        "structured_entities": STRUCTURED_ENTITIES,
        "unseen_entities": UNSEEN_ENTITIES,
        "relations": RELATIONS,
        "values": VALUES,
        "compiled_facts": FACTS,
        "payload_scalars": FACTS,
        "payload_bits": FACTS * math.log2(VALUES),
        "bits_per_payload_scalar": math.log2(VALUES),
        "fact_payload_scalars_per_fact": 1.0,
        "program_hidden_coordinates": PROGRAM_DIMS,
        "program_attention_heads": PROGRAM_HEADS,
        "selection_channels": RELATIONS,
        "decoder_channels": channel,
        "frozen_parameter_entries": frozen.count,
        "fixed_nonzero_entries": fixed_nonzero,
        "constant_entity_norm_squared": ENTITY_NORM_SQUARED,
    }


@torch.no_grad()
def evaluate(
    model: ScaleLM,
    world: RelationWorld,
    layout: RelationLayout,
    device: torch.device,
    *,
    batch_size: int = 4_096,
) -> dict[str, object]:
    model.eval()
    entities = torch.arange(STRUCTURED_ENTITIES).repeat_interleave(RELATIONS)
    relations = torch.arange(RELATIONS).repeat(STRUCTURED_ENTITIES)
    targets = world.values.reshape(-1).to(device)
    predictions: list[Tensor] = []
    margins: list[Tensor] = []
    result_rows = torch.tensor(layout.result, dtype=torch.long, device=device)
    for start in range(0, len(entities), batch_size):
        tokens = encode_queries(
            entities[start : start + batch_size],
            relations[start : start + batch_size],
            layout,
            device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.hidden(tokens)[:, -1]
            logits = hidden @ model.token.weight[result_rows].T
        ordered = logits.float().topk(2, dim=-1).values
        predictions.append(logits.argmax(-1))
        margins.append(ordered[:, 0] - ordered[:, 1])
    predicted = torch.cat(predictions)
    margin = torch.cat(margins)
    correct = predicted == targets

    unseen_generator = torch.Generator(device="cpu").manual_seed(EVALUATION_SEED)
    unseen_entities = torch.arange(
        STRUCTURED_ENTITIES, ENTITIES
    ).repeat_interleave(RELATIONS)
    unseen_relations = torch.arange(RELATIONS).repeat(UNSEEN_ENTITIES)
    unseen_targets = torch.randint(
        0,
        VALUES,
        (len(unseen_entities),),
        generator=unseen_generator,
        dtype=torch.long,
    ).to(device)
    unseen_predictions: list[Tensor] = []
    for start in range(0, len(unseen_entities), batch_size):
        tokens = encode_queries(
            unseen_entities[start : start + batch_size],
            unseen_relations[start : start + batch_size],
            layout,
            device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.hidden(tokens)[:, -1]
            logits = hidden @ model.token.weight[result_rows].T
        unseen_predictions.append(logits.argmax(-1))
    unseen_predicted = torch.cat(unseen_predictions)
    return {
        "structured_accuracy": float(correct.float().mean().item()),
        "structured_errors": int((~correct).sum().item()),
        "minimum_logit_margin": float(margin.min().item()),
        "mean_logit_margin": float(margin.mean().item()),
        "unseen_random_accuracy": float(
            (unseen_predicted == unseen_targets).float().mean().item()
        ),
        "per_relation_accuracy": [
            float(correct.view(STRUCTURED_ENTITIES, RELATIONS)[:, relation].float().mean().item())
            for relation in range(RELATIONS)
        ],
    }


def run(device: torch.device) -> dict[str, object]:
    train = TokenStream(TRAIN_FILE)
    validation = TokenStream(VALIDATION_FILE)
    layout, unused = choose_layout(train, validation)
    world = make_world()
    model = t7.build_model(5_003, device)
    before = sum(parameter.numel() for parameter in model.parameters())
    frozen, compiler = compile_relation_plane(model, world, layout)
    after = sum(parameter.numel() for parameter in model.parameters())
    result = evaluate(model, world, layout, device)
    gates = {
        "same_parameter_count": before == after == t7.parameter_count(),
        "all_structured_facts_exact": result["structured_accuracy"] == 1.0,
        "zero_structured_errors": result["structured_errors"] == 0,
        "positive_minimum_margin": result["minimum_logit_margin"] > 0.0,
        "every_relation_exact": min(result["per_relation_accuracy"]) == 1.0,
        "unseen_near_chance": 0.04 <= result["unseen_random_accuracy"] <= 0.09,
        "one_payload_scalar_per_fact": compiler["fact_payload_scalars_per_fact"] == 1.0,
        "decoder_fits_reserved_channels": compiler["decoder_channels"] <= PROGRAM_CHANNELS,
    }
    return {
        "schema": "digital-relation-plane-t8-quick-v1",
        "status": "complete",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": before,
            "vocab": VOCAB,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": t7.HEADS,
            "ffn_width": FFN_WIDTH,
            "structured_entities": STRUCTURED_ENTITIES,
            "unseen_entities": UNSEEN_ENTITIES,
            "relations": RELATIONS,
            "values": VALUES,
            "compiled_facts": FACTS,
            "unused_tokens_available": unused,
        },
        "compiler": compiler,
        "evaluation": result,
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "train_sha256": sha256_file(TRAIN_FILE),
            "validation_sha256": sha256_file(VALIDATION_FILE),
            "predecessor_sha256": sha256_file(
                ROOT / "results/heterogeneous-family-discovery-t7.json"
            ),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "all_gates_pass": result["all_gates_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
