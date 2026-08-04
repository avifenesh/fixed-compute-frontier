#!/usr/bin/env python3
"""T5: nibble-packed rule memory with a constant-amplitude regeneration boundary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import packed_rule_memory_t4 as t4


PREREGISTRATION = ROOT / "results/regenerated-packed-memory-t5-preregistration.md"
TEST_SOURCE = ROOT / "tests/test_regenerated_packed_memory_t5.py"
PREDECESSOR = ROOT / "results/packed-rule-memory-t4-quick.json"
OUTPUT = ROOT / "results/regenerated-packed-memory-t5.json"

VOCAB = t4.VOCAB
HIDDEN = t4.HIDDEN
LAYERS = t4.LAYERS
HEAD_DIM = t4.HEAD_DIM
INPUT_DIM = t4.INPUT_DIM
RULES = t4.RULES
NIBBLES = t4.NIBBLES
SUPPORT_SIZE = t4.SUPPORT_SIZE
PROGRAM_DIMS = 81
PROGRAM_HEADS = 2
DECODE_CHANNELS = 160
REGENERATE_CHANNELS = 96
EXECUTE_CHANNELS = 64
CLIP_LEVEL = 1.0 / 16.0
REGEN_BETA = 32.0

NIBBLE_START = 0
REFERENCE = 8
RAW_START = 9
CANONICAL_START = 41
SUM = 73
COPY_SUM = 74
CONST = 75
TASK = 76
BIT_TYPE = 77
PARITY = 78
COPY = 79
OUT = 80


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + t4.RMS_EPSILON * HIDDEN))


def _autocast_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def compile_regenerated_interpreter(
    model: t4.SharedInterpreterLM,
    recovered: torch.Tensor,
    layout: t4.TokenLayout,
) -> tuple[t4.FrozenEntries, dict[str, object]]:
    named = dict(model.named_parameters())
    masks = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in named.items()}

    def fix(name: str, index: object, value: float = 0.0) -> None:
        named[name][index] = value
        masks[name][index] = True

    fix("token.weight", (slice(None), slice(0, PROGRAM_DIMS)))
    for token_id in layout.all_special:
        fix("token.weight", (token_id, slice(None)))

    packed = t4.pack_supports(recovered).to(model.token.weight.device)
    for rule, token_id in enumerate(layout.rule):
        named["token.weight"][token_id, NIBBLE_START : NIBBLE_START + NIBBLES] = packed[
            rule
        ].float()
        named["token.weight"][token_id, REFERENCE] = 1.0
        named["token.weight"][token_id, TASK] = 1.0
    for position, token_id in enumerate(layout.zero):
        named["token.weight"][token_id, CANONICAL_START + position] = 1.0
        named["token.weight"][token_id, BIT_TYPE] = 1.0
        named["token.weight"][token_id, CONST] = 1.0
    for position, token_id in enumerate(layout.one):
        named["token.weight"][token_id, CANONICAL_START + position] = 1.0
        named["token.weight"][token_id, BIT_TYPE] = 1.0
        named["token.weight"][token_id, CONST] = -1.0
    named["token.weight"][layout.parity_query, CONST] = 1.0
    named["token.weight"][layout.parity_query, PARITY] = 1.0
    named["token.weight"][layout.copy_query, CONST] = 1.0
    named["token.weight"][layout.copy_query, COPY] = 1.0
    named["token.weight"][layout.result_zero, OUT] = 30.0
    named["token.weight"][layout.result_one, OUT] = -30.0

    program_rows = slice(0, PROGRAM_HEADS * HEAD_DIM)
    language_rows = slice(PROGRAM_HEADS * HEAD_DIM, HIDDEN)
    layer_channels = (
        DECODE_CHANNELS,
        REGENERATE_CHANNELS,
        EXECUTE_CHANNELS,
        EXECUTE_CHANNELS,
        EXECUTE_CHANNELS,
    )
    for layer, channels in enumerate(layer_channels):
        prefix = f"blocks.{layer}."
        for norm in ("attention_norm.weight", "ffn_norm.weight"):
            fix(prefix + norm, slice(0, PROGRAM_DIMS), 1.0)
        for projection in ("attention.q.weight", "attention.k.weight", "attention.v.weight"):
            fix(prefix + projection, (program_rows, slice(None)))
            fix(prefix + projection, (language_rows, slice(0, PROGRAM_DIMS)))
        fix(prefix + "attention.o.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "attention.o.weight", (slice(PROGRAM_DIMS, None), program_rows))
        for projection in ("gate.weight", "up.weight"):
            fix(prefix + projection, (slice(0, channels), slice(None)))
            fix(prefix + projection, (slice(channels, None), slice(0, PROGRAM_DIMS)))
        fix(prefix + "down.weight", (slice(0, PROGRAM_DIMS), slice(None)))
        fix(prefix + "down.weight", (slice(PROGRAM_DIMS, None), slice(0, channels)))
    for layer in range(5, LAYERS):
        fix(f"blocks.{layer}.attention.o.weight", (OUT, slice(None)))
        fix(f"blocks.{layer}.down.weight", (OUT, slice(None)))
    fix("final_norm.weight", OUT, 1.0)

    # Block 0: T4's sealed shared nibble decoder, now writing transient RAW.
    block0 = model.blocks[0]
    beta = 16.0
    channel = 0
    block0.gate.weight[channel, REFERENCE] = 1.0
    block0.gate.weight[channel + 1, REFERENCE] = -1.0
    block0.up.weight[channel, TASK] = 1.0
    block0.up.weight[channel + 1, TASK] = 1.0
    for output in range(INPUT_DIM):
        block0.down.weight[RAW_START + output, channel] = -1.0
        block0.down.weight[RAW_START + output, channel + 1] = 1.0
    channel += 2
    for nibble in range(NIBBLES):
        block0.gate.weight[channel, NIBBLE_START + nibble] = 1.0
        block0.gate.weight[channel + 1, NIBBLE_START + nibble] = -1.0
        block0.up.weight[channel, TASK] = 1.0
        block0.up.weight[channel + 1, TASK] = 1.0
        for bit in range(4):
            slope0 = t4.bit_sign(1, bit) - t4.bit_sign(0, bit)
            output = RAW_START + 4 * nibble + bit
            block0.down.weight[output, channel] = slope0
            block0.down.weight[output, channel + 1] = -slope0
        channel += 2
        for knot in range(1, 15):
            block0.gate.weight[channel, NIBBLE_START + nibble] = beta
            block0.gate.weight[channel, REFERENCE] = -beta * knot
            block0.up.weight[channel, TASK] = 1.0 / beta
            for bit in range(4):
                previous_slope = t4.bit_sign(knot, bit) - t4.bit_sign(knot - 1, bit)
                next_slope = t4.bit_sign(knot + 1, bit) - t4.bit_sign(knot, bit)
                output = RAW_START + 4 * nibble + bit
                block0.down.weight[output, channel] = next_slope - previous_slope
            channel += 1
    decoder_channels = channel

    # Block 1: regenerate a constant-amplitude alphabet in separate coordinates.
    block1 = model.blocks[1]
    channel = 0
    block1.gate.weight[channel, REFERENCE] = 1.0
    block1.gate.weight[channel + 1, REFERENCE] = -1.0
    block1.up.weight[channel, TASK] = 1.0
    block1.up.weight[channel + 1, TASK] = 1.0
    for output in range(INPUT_DIM):
        block1.down.weight[CANONICAL_START + output, channel] = -CLIP_LEVEL
        block1.down.weight[CANONICAL_START + output, channel + 1] = CLIP_LEVEL
    channel += 2
    for output in range(INPUT_DIM):
        raw = RAW_START + output
        canonical = CANONICAL_START + output
        # +ReLU(x+a)
        block1.gate.weight[channel, raw] = REGEN_BETA
        block1.gate.weight[channel, REFERENCE] = REGEN_BETA * CLIP_LEVEL
        block1.up.weight[channel, TASK] = 1.0 / REGEN_BETA
        block1.down.weight[canonical, channel] = 1.0
        channel += 1
        # -ReLU(x-a)
        block1.gate.weight[channel, raw] = REGEN_BETA
        block1.gate.weight[channel, REFERENCE] = -REGEN_BETA * CLIP_LEVEL
        block1.up.weight[channel, TASK] = 1.0 / REGEN_BETA
        block1.down.weight[canonical, channel] = -1.0
        channel += 1
    regeneration_channels = channel

    # Execute the first two stages on every rule to seal the actual BF16 symbol
    # margin used to calibrate attention.  This is compiler analysis, not an
    # inference-time operation.
    rule_ids = torch.tensor(layout.rule, device=model.token.weight.device)
    with _autocast_context(model.token.weight.device):
        task_hidden = model.token(rule_ids[:, None])
        task_hidden = block0(task_hidden)
        raw_values = task_hidden[:, 0, RAW_START : RAW_START + INPUT_DIM].float()
        task_hidden = block1(task_hidden)
        canonical_values = task_hidden[
            :, 0, CANONICAL_START : CANONICAL_START + INPUT_DIM
        ].float()
    expected = recovered.to(canonical_values.device).float().mul(2.0).sub(1.0)
    raw_sign_accuracy = float((raw_values.sign() == expected).float().mean().item())
    canonical_sign_accuracy = float(
        (canonical_values.sign() == expected).float().mean().item()
    )
    magnitudes = canonical_values.abs()
    per_rule_ratio = magnitudes.max(1).values / magnitudes.min(1).values
    maximum_canonical_ratio = float(per_rule_ratio.max().item())

    # Block 2: bit tokens self-attend while parity queries copy canonical signs.
    block2 = model.blocks[2]
    alpha_bit_2 = rms_scale(3.0)
    alpha_query_2 = rms_scale(2.0)
    task_scale_2 = task_hidden.float().pow(2).mean(-1).add(t4.RMS_EPSILON).rsqrt()[:, 0]
    gap = 30.0
    base = 30.0
    q_a = math.sqrt(math.sqrt(HEAD_DIM) * gap) / alpha_bit_2
    q_type = math.sqrt(math.sqrt(HEAD_DIM) * base) / alpha_bit_2
    minimum_task_scale = float(task_scale_2.min().item())
    q_query = math.sqrt(
        math.sqrt(HEAD_DIM) * 60.0 / (alpha_query_2 * minimum_task_scale)
    )
    for component in range(INPUT_DIM):
        block2.attention.q.weight[component, CANONICAL_START + component] = q_a
        block2.attention.k.weight[component, CANONICAL_START + component] = q_a
        block2.attention.v.weight[component, CANONICAL_START + component] = 1.0
        block2.attention.o.weight[CANONICAL_START + component, component] = 1.0 / alpha_bit_2
    block2.attention.q.weight[32, BIT_TYPE] = q_type
    block2.attention.k.weight[32, BIT_TYPE] = q_type
    block2.attention.q.weight[33, PARITY] = q_query
    block2.attention.k.weight[33, TASK] = q_query

    # Simulate query-copy states for all rules; selector temperature is chosen
    # from the smallest normalized canonical coordinate.
    zero_bits = torch.zeros(RULES, INPUT_DIM, dtype=torch.uint8)
    rules = torch.arange(RULES)
    route = torch.ones(RULES, dtype=torch.bool)
    tokens = t4.encode_algorithm(zero_bits, rules, route, layout, model.token.weight.device)
    with _autocast_context(model.token.weight.device):
        hidden = model.token(tokens)
        hidden = block0(hidden)
        hidden = block1(hidden)
        hidden = block2(hidden)
    query_state = hidden[:, -1].float()
    query_scale = query_state.pow(2).mean(-1).add(t4.RMS_EPSILON).rsqrt()
    query_canonical = query_state[:, CANONICAL_START : CANONICAL_START + INPUT_DIM]
    normalized_query = query_canonical.abs() * query_scale[:, None]
    minimum_normalized_query = float(normalized_query.min().item())

    # Block 3 support selection.  Large negative type penalties remove task and
    # self keys without depending on rule-specific amplitude.
    block3 = model.blocks[3]
    bit_amplitude = 2.0
    alpha_bit_3 = rms_scale(bit_amplitude**2 + 2.0)
    q_select = math.sqrt(
        math.sqrt(HEAD_DIM)
        * 30.0
        / (minimum_normalized_query * alpha_bit_3 * bit_amplitude)
    )
    for component in range(INPUT_DIM):
        block3.attention.q.weight[component, CANONICAL_START + component] = q_select
        block3.attention.k.weight[component, CANONICAL_START + component] = q_select
    block3.attention.q.weight[63, PARITY] = 1.0
    block3.attention.k.weight[63, TASK] = -1_000_000.0
    block3.attention.k.weight[63, PARITY] = -1_000_000.0
    block3.attention.v.weight[0, CONST] = 1.0
    block3.attention.o.weight[SUM, 0] = SUPPORT_SIZE / alpha_bit_3

    copy_head = HEAD_DIM
    strong = 20.0
    block3.attention.q.weight[copy_head, COPY] = strong
    block3.attention.k.weight[copy_head, CANONICAL_START] = strong
    block3.attention.q.weight[copy_head + 1, COPY] = strong
    block3.attention.k.weight[copy_head + 1, BIT_TYPE] = strong
    block3.attention.v.weight[copy_head, CONST] = 1.0
    block3.attention.o.weight[COPY_SUM, copy_head] = 1.0 / alpha_bit_3

    # Block 4 count-to-parity decoder and protected-copy product.
    block4 = model.blocks[4]
    beta = 16.0
    channel = 0
    for integer_sum in range(-SUPPORT_SIZE, SUPPORT_SIZE + 1, 2):
        negatives = (SUPPORT_SIZE - integer_sum) // 2
        parity_sign = -1.0 if negatives % 2 else 1.0
        for threshold, coefficient in (
            (integer_sum - 1, 1.0),
            (integer_sum, -2.0),
            (integer_sum + 1, 1.0),
        ):
            block4.gate.weight[channel, SUM] = beta
            block4.gate.weight[channel, CONST] = -beta * threshold
            block4.up.weight[channel, PARITY] = 1.0 / beta
            block4.down.weight[OUT, channel] = parity_sign * coefficient
            channel += 1
    block4.gate.weight[channel, COPY_SUM] = 1.0
    block4.gate.weight[channel + 1, COPY_SUM] = -1.0
    block4.up.weight[channel, COPY] = 1.0
    block4.up.weight[channel + 1, COPY] = 1.0
    block4.down.weight[OUT, channel] = 1.0
    block4.down.weight[OUT, channel + 1] = -1.0
    execute_decoder_channels = channel + 2

    values = {name: parameter.detach().clone() for name, parameter in named.items()}
    frozen = t4.FrozenEntries(
        masks, values, sum(int(mask.sum().item()) for mask in masks.values())
    )
    frozen.enforce(model)
    fixed_nonzero = sum(
        int(((values[name] != 0) & mask).sum().item()) for name, mask in masks.items()
    )
    return frozen, {
        "rules": RULES,
        "description_entries": RULES * NIBBLES,
        "bits_per_description_entry": 4,
        "decoded_support_bits": RULES * INPUT_DIM,
        "density_vs_t3_rule_table": 12.0,
        "program_hidden_coordinates": PROGRAM_DIMS,
        "decoder_channels": decoder_channels,
        "regeneration_channels": regeneration_channels,
        "execute_decoder_channels": execute_decoder_channels,
        "shared_width_independent_of_rules": True,
        "frozen_parameter_entries": frozen.count,
        "fixed_nonzero_entries": fixed_nonzero,
        "raw_sign_accuracy": raw_sign_accuracy,
        "canonical_sign_accuracy": canonical_sign_accuracy,
        "maximum_canonical_magnitude_ratio": maximum_canonical_ratio,
        "canonical_abs_min": float(magnitudes.min().item()),
        "canonical_abs_max": float(magnitudes.max().item()),
        "minimum_normalized_query_support": minimum_normalized_query,
        "clip_level": CLIP_LEVEL,
        "regeneration_beta": REGEN_BETA,
    }


def run(device: torch.device, quick: bool = False) -> dict[str, object]:
    original_compiler = t4.compile_packed_interpreter
    original_preregistration = t4.PREREGISTRATION
    original_test_source = t4.TEST_SOURCE
    original_predecessor = t4.PREDECESSOR
    try:
        t4.compile_packed_interpreter = compile_regenerated_interpreter
        t4.PREREGISTRATION = PREREGISTRATION
        t4.TEST_SOURCE = TEST_SOURCE
        t4.PREDECESSOR = PREDECESSOR
        result = t4.run(device, quick=quick)
    finally:
        t4.compile_packed_interpreter = original_compiler
        t4.PREREGISTRATION = original_preregistration
        t4.TEST_SOURCE = original_test_source
        t4.PREDECESSOR = original_predecessor

    result["schema"] = "regenerated-packed-memory-t5-v1"
    result["integrity"]["source_sha256"] = t4.sha256_file(Path(__file__))
    result["integrity"]["test_sha256"] = t4.sha256_file(TEST_SOURCE)
    result["integrity"]["preregistration_sha256"] = t4.sha256_file(PREREGISTRATION)
    result["integrity"]["predecessor_sha256"] = t4.sha256_file(PREDECESSOR)
    if quick:
        world = next(iter(result["worlds"].values()))
        compiler = world["compiler"]
        algorithm = world["algorithm"]
        result["gates"] = {
            "raw_signs_100": compiler["raw_sign_accuracy"] == 1.0,
            "canonical_signs_100": compiler["canonical_sign_accuracy"] == 1.0,
            "canonical_ratio_le_1p05": compiler[
                "maximum_canonical_magnitude_ratio"
            ]
            <= 1.05,
            "every_rule_ge_99": algorithm["parity_min"] >= 0.99,
            "copy_ge_99": algorithm["protected_copy"] >= 0.99,
        }
        result["all_gates_pass"] = all(result["gates"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device), quick=arguments.quick)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "all_gates_pass": result["all_gates_pass"]}, indent=2))


if __name__ == "__main__":
    main()
