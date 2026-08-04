#!/usr/bin/env python3
"""T11: capacity-preserving triangular isolation for the T10 equality plane."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor

LOCAL_ROOT = Path(__file__).resolve().parents[1]
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from experiments import raw_prose_equality_plane_t10 as base


ROOT = base.ROOT
OUTPUT = ROOT / "results/raw-prose-equality-plane-t11-quick.json"
PREREGISTRATION = ROOT / "results/raw-prose-equality-plane-t11-preregistration.md"
TRAIN_FILE = base.TRAIN_FILE
VALIDATION_FILE = base.VALIDATION_FILE

# Re-export the frozen corpus/model contract so the matched-training harness can
# use T11 as a drop-in writer without duplicating the extractor or query split.
raw = base.raw
core = base.core
natural_data = base.natural_data
t8 = base.t8
ProseLayout = base.ProseLayout
ENTITIES = base.ENTITIES
RELATIONS = base.RELATIONS
VALUES = base.VALUES
ALIAS_VIEWS = base.ALIAS_VIEWS
PARAPHRASES = base.PARAPHRASES
VALUE_BITS = base.VALUE_BITS
RELATION_BITS = base.RELATION_BITS
MEMORY_COORDS = base.MEMORY_COORDS
PROGRAM_HEADS = base.PROGRAM_HEADS
MEMORY_START = base.MEMORY_START
RELATION_A_START = base.RELATION_A_START
RELATION_B_START = base.RELATION_B_START
ENTITY_TYPE = base.ENTITY_TYPE
RELATION_A_TYPE = base.RELATION_A_TYPE
RELATION_B_TYPE = base.RELATION_B_TYPE
QUERY_ROUTE = base.QUERY_ROUTE
CONST = base.CONST
VALUE_A_START = base.VALUE_A_START
VALUE_B_START = base.VALUE_B_START
PRODUCT_START = base.PRODUCT_START
PROGRAM_DIMS = base.PROGRAM_DIMS
HIDDEN = base.HIDDEN
LAYERS = base.LAYERS
HEAD_DIM = base.HEAD_DIM
FFN_WIDTH = base.FFN_WIDTH
RMS_EPSILON = base.RMS_EPSILON
BETA = base.BETA

sha256_file = base.sha256_file
rms_scale = base.rms_scale
bipolar_code = base.bipolar_code
discover_frame_anchors = base.discover_frame_anchors
choose_layout = base.choose_layout
value_codes = base.value_codes
frames_by_relation_and_role = base.frames_by_relation_and_role
direct_dataset = base.direct_dataset
equality_dataset = base.equality_dataset
encode_raw_corpus = base.encode_raw_corpus
evaluate = base.evaluate
perturb_and_enforce = base.perturb_and_enforce

BLOCK1_PROGRAM_CHANNELS = 64
BLOCK2_PROGRAM_CHANNELS = 2 * VALUE_BITS
OUTPUT_SCALE = 2.0
PROTECTED_OUTPUT_DIMS = (
    CONST,
    *range(VALUE_A_START, VALUE_A_START + VALUE_BITS),
    *range(PRODUCT_START, PRODUCT_START + VALUE_BITS),
)
BLOCK0_VALUE_COORDS = (
    *range(MEMORY_COORDS),
    *range(HEAD_DIM, HEAD_DIM + 2 * RELATION_BITS),
)


@torch.no_grad()
def compile_equality_plane(
    model: core.small.SharedInterpreterLM,
    compiled: raw.CompileResult,
    layout: ProseLayout,
) -> tuple[t8.FrozenEntries, dict[str, object]]:
    """Write T10's exact program while isolating only causally required entries."""
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

    # Special rows are program-only.  Natural token rows retain every hidden
    # coordinate, unlike T10's global 55-coordinate embedding reservation.
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
            named["token.weight"][token_id, RELATION_A_START : RELATION_A_START + RELATION_BITS] = relation_code
            named["token.weight"][token_id, RELATION_A_TYPE] = 1.0
        else:
            named["token.weight"][token_id, RELATION_B_START : RELATION_B_START + RELATION_BITS] = relation_code
            named["token.weight"][token_id, RELATION_B_TYPE] = 1.0

    for query in (layout.direct_query, layout.equality_query):
        named["token.weight"][query, QUERY_ROUTE] = 1.0
        named["token.weight"][query, CONST] = 1.0
    for value, token_id in layout.value_token.items():
        named["token.weight"][token_id, VALUE_A_START : VALUE_A_START + VALUE_BITS] = OUTPUT_SCALE * codes[value].to(model.token.weight.device)
    named["token.weight"][layout.result_same, PRODUCT_START : PRODUCT_START + VALUE_BITS] = OUTPUT_SCALE
    named["token.weight"][layout.result_same, CONST] = -3.0 * OUTPUT_SCALE
    named["token.weight"][layout.result_different, PRODUCT_START : PRODUCT_START + VALUE_BITS] = -OUTPUT_SCALE
    named["token.weight"][layout.result_different, CONST] = 3.0 * OUTPUT_SCALE

    program_columns = slice(0, PROGRAM_DIMS)
    program_head_rows = slice(0, PROGRAM_HEADS * HEAD_DIM)

    # Block 0 attention is active.  Special inputs have no language component,
    # so only weights that read program columns or emit used V coordinates need
    # isolation.  The unused 90 coordinates of the two nominal program heads
    # remain available to ordinary language features.
    prefix = "blocks.0."
    fix(prefix + "attention_norm.weight", program_columns, 1.0)
    for projection in ("attention.q.weight", "attention.k.weight"):
        fix(prefix + projection, (program_head_rows, program_columns))
    fix(prefix + "attention.v.weight", (slice(None), program_columns))
    for coordinate in BLOCK0_VALUE_COORDS:
        fix(prefix + "attention.o.weight", (slice(None), coordinate))
    # The block-0 FFN must be zero on program-only inputs; one triangular zero
    # in the multiplicative up branch is sufficient.
    fix(prefix + "up.weight", (slice(None), program_columns))

    # Blocks 1 and 2 do not use attention.  Zeroing only V's program columns
    # makes those branches exactly zero for special inputs while leaving Q/K
    # and every language-input column trainable.
    for layer in (1, 2):
        prefix = f"blocks.{layer}."
        fix(prefix + "attention.v.weight", (slice(None), program_columns))
        fix(prefix + "ffn_norm.weight", program_columns, 1.0)

    def isolate_active_ffn(layer: int, channels: int) -> None:
        prefix = f"blocks.{layer}."
        active = slice(0, channels)
        inactive = slice(channels, FFN_WIDTH)
        for projection in ("gate.weight", "up.weight"):
            fix(prefix + projection, (active, slice(None)))
        # For program-only inputs the inactive up branch is zero, so its
        # product is zero even though the corresponding gate stays trainable.
        fix(prefix + "up.weight", (inactive, program_columns))
        # Active channels may write only the explicitly compiled registers.
        fix(prefix + "down.weight", (slice(None), active))

    isolate_active_ffn(1, BLOCK1_PROGRAM_CHANNELS)
    isolate_active_ffn(2, BLOCK2_PROGRAM_CHANNELS)

    # Once block 2 has produced the answers, later layers may freely read the
    # program.  They may not write the nine answer coordinates.  Residual
    # identity therefore preserves the answer while the remaining 375 hidden
    # coordinates and all branch inputs remain trainable.
    protected_rows = torch.tensor(
        PROTECTED_OUTPUT_DIMS, dtype=torch.long, device=model.token.weight.device
    )
    for layer in range(3, LAYERS):
        prefix = f"blocks.{layer}."
        fix(prefix + "attention.o.weight", (protected_rows, slice(None)))
        fix(prefix + "down.weight", (protected_rows, slice(None)))
    fix("final_norm.weight", protected_rows, 1.0)

    # Exact T10 algebra, now written into the smaller triangular mask.
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
        block0.attention.o.weight[MEMORY_START + coordinate, coordinate] = 1.0 / alpha_entity

    second_head = HEAD_DIM
    relation_route = math.sqrt(
        math.sqrt(HEAD_DIM) * 60.0 / (alpha_relation * alpha_query)
    )
    block0.attention.q.weight[second_head + 62, QUERY_ROUTE] = relation_route
    block0.attention.q.weight[second_head + 63, QUERY_ROUTE] = relation_route
    block0.attention.k.weight[second_head + 62, RELATION_A_TYPE] = relation_route
    block0.attention.k.weight[second_head + 63, RELATION_B_TYPE] = relation_route
    for bit in range(RELATION_BITS):
        block0.attention.v.weight[second_head + bit, RELATION_A_START + bit] = 1.0
        block0.attention.v.weight[second_head + RELATION_BITS + bit, RELATION_B_START + bit] = 1.0
        block0.attention.o.weight[RELATION_A_START + bit, second_head + bit] = 2.0 / alpha_relation
        block0.attention.o.weight[RELATION_B_START + bit, second_head + RELATION_BITS + bit] = 2.0 / alpha_relation

    alpha_select = rms_scale(MEMORY_COORDS + 2 * RELATION_BITS + 2.0)
    block1 = model.blocks[1]
    channel = 0
    for relation_start, output_start in (
        (RELATION_A_START, VALUE_A_START),
        (RELATION_B_START, VALUE_B_START),
    ):
        for relation in range(RELATIONS):
            code = bipolar_code(relation, RELATION_BITS).to(model.token.weight.device)
            for bit in range(VALUE_BITS):
                block1.gate.weight[channel, relation_start : relation_start + RELATION_BITS] = BETA * code / alpha_select
                block1.gate.weight[channel, CONST] = -BETA * (RELATION_BITS - 1) / alpha_select
                memory_coordinate = MEMORY_START + relation * VALUE_BITS + bit
                block1.up.weight[channel, memory_coordinate] = 1.0 / (BETA * alpha_select)
                block1.down.weight[output_start + bit, channel] = 1.0
                channel += 1
    if channel != BLOCK1_PROGRAM_CHANNELS:
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
            block2.up.weight[product_channel, VALUE_B_START + bit] = 1.0 / (BETA * alpha_product)
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
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return frozen, {
        "recovered_entities": len(compiled.canonical_entities),
        "surface_aliases": len(compiled.alias_to_canonical),
        "relations": compiled.canonical_relation_count,
        "surface_frames": len(compiled.frame_to_relation),
        "direct_facts": len(compiled.canonical_entities) * compiled.canonical_relation_count,
        "payload_scalars": len(compiled.alias_to_canonical) * compiled.canonical_relation_count * VALUE_BITS,
        "program_hidden_coordinates_during_blocks_0_to_2": PROGRAM_DIMS,
        "protected_hidden_coordinates_after_block_2": len(PROTECTED_OUTPUT_DIMS),
        "block0_program_value_coordinates": len(BLOCK0_VALUE_COORDS),
        "block1_selection_channels": BLOCK1_PROGRAM_CHANNELS,
        "block2_product_channels": BLOCK2_PROGRAM_CHANNELS,
        "output_scale": OUTPUT_SCALE,
        "frozen_parameter_entries": frozen.count,
        "frozen_parameter_fraction": frozen.count / total_parameters,
        "fixed_nonzero_entries": fixed_nonzero,
    }


def run(device: torch.device) -> dict[str, object]:
    """Reuse the sealed T10 corpus/evaluator with only the T11 writer swapped."""
    previous_writer = base.compile_equality_plane
    previous_preregistration = base.PREREGISTRATION
    base.compile_equality_plane = compile_equality_plane
    base.PREREGISTRATION = PREREGISTRATION
    try:
        result = base.run(device)
    finally:
        base.compile_equality_plane = previous_writer
        base.PREREGISTRATION = previous_preregistration
    result["schema"] = "raw-prose-equality-plane-t11-quick-v1"
    result["integrity"]["base_t10_source_sha256"] = sha256_file(Path(base.__file__))
    result["integrity"]["source_sha256"] = sha256_file(Path(__file__))
    result["claim_boundary"] = (
        "This tests whether minimal triangular isolation preserves T10's exact "
        "controlled capability. It is not a production claim; matched training "
        "must also pass without natural-language degradation."
    )
    return result


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
                "writer": result["writer"],
                "evaluation": result["evaluation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
