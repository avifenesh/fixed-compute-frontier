#!/usr/bin/env python3
"""T9 core: 36.6M replication compiler and Muon calibration."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments import digital_relation_plane_t8 as t8
from experiments import heterogeneous_family_discovery_t7_scale as t7


ROOT = Path(__file__).resolve().parents[1]
PILOT_PROTOCOL = ROOT / "results/digital-relation-plane-t9-small-muon-pilot-protocol.md"
PILOT_OUTPUT = ROOT / "results/digital-relation-plane-t9-small-muon-pilot.json"
QUICK_OUTPUT = ROOT / "results/digital-relation-plane-t9-small-quick.json"

small = t7.t6.t3
HIDDEN = small.HIDDEN
LAYERS = small.LAYERS
HEADS = small.HEADS
HEAD_DIM = small.HEAD_DIM
FFN_WIDTH = small.FFN_WIDTH
RMS_EPSILON = small.RMS_EPSILON
PROGRAM_DIMS = t8.PROGRAM_DIMS
PROGRAM_HEADS = t8.PROGRAM_HEADS
PROGRAM_CHANNELS = t8.PROGRAM_CHANNELS
WORLD_SEED = 5_000_051
EVALUATION_SEED = 5_100_053

PILOT_SEED = 5_003
PILOT_STEPS = 250
PILOT_CHECKPOINTS = (50, 100, 250)
MUON_RATES = (0.0025, 0.005, 0.01, 0.02)
NATURAL_BATCH = 16


def rms_scale(norm_squared: float) -> float:
    return math.sqrt(HIDDEN / (norm_squared + RMS_EPSILON * HIDDEN))


def build_model(seed: int, device: torch.device) -> small.SharedInterpreterLM:
    return small.build_model(seed, device)


@torch.no_grad()
def compile_relation_plane(
    model: small.SharedInterpreterLM,
    world: t8.RelationWorld,
    layout: t8.RelationLayout,
) -> tuple[t8.FrozenEntries, dict[str, object]]:
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

    levels = t8.value_level(world.values)
    for entity, token_id in enumerate(layout.entity):
        named["token.weight"][token_id, t8.ENTITY_TYPE] = 1.0
        if entity < t8.STRUCTURED_ENTITIES:
            row = levels[entity].to(model.token.weight.device)
            named["token.weight"][token_id, t8.MEMORY_START : t8.MEMORY_START + t8.RELATIONS] = row
            used_squared = float(row.float().pow(2).sum().item()) + 1.0
        else:
            used_squared = 1.0
        named["token.weight"][token_id, t8.COMPENSATION] = math.sqrt(
            max(t8.ENTITY_NORM_SQUARED - used_squared, 0.0)
        )
    for relation, token_id in enumerate(layout.relation_query):
        named["token.weight"][token_id, t8.RELATION_START : t8.RELATION_START + t8.RELATION_BITS] = (
            t8.relation_code(relation).to(model.token.weight.device)
        )
        named["token.weight"][token_id, t8.QUERY_ROUTE] = 1.0
        named["token.weight"][token_id, t8.CONST] = 1.0
    for value, token_id in enumerate(layout.result):
        named["token.weight"][token_id, t8.OUTPUT_START : t8.OUTPUT_START + 4] = (
            30.0 * t8.value_code(value).to(model.token.weight.device)
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

    block0 = model.blocks[0]
    alpha_entity = rms_scale(t8.ENTITY_NORM_SQUARED)
    alpha_query = rms_scale(t8.RELATION_BITS + 2.0)
    route = math.sqrt(
        math.sqrt(HEAD_DIM) * 60.0 / (alpha_entity * alpha_query)
    )
    block0.attention.q.weight[63, t8.QUERY_ROUTE] = route
    block0.attention.k.weight[63, t8.ENTITY_TYPE] = route
    for coordinate in range(t8.RELATIONS):
        block0.attention.v.weight[coordinate, t8.MEMORY_START + coordinate] = 1.0
        block0.attention.o.weight[t8.MEMORY_START + coordinate, coordinate] = 1.0 / alpha_entity
    second_head = HEAD_DIM
    block0.attention.q.weight[second_head + 63, t8.QUERY_ROUTE] = route
    block0.attention.k.weight[second_head + 63, t8.ENTITY_TYPE] = route
    block0.attention.v.weight[second_head, t8.COMPENSATION] = 1.0
    block0.attention.o.weight[t8.COMPENSATION, second_head] = 1.0 / alpha_entity

    query_norm_squared = t8.ENTITY_NORM_SQUARED + 7.0
    alpha_select = rms_scale(query_norm_squared)
    block1 = model.blocks[1]
    beta = 16.0
    for relation in range(t8.RELATIONS):
        code = t8.relation_code(relation).to(model.token.weight.device)
        block1.gate.weight[
            relation, t8.RELATION_START : t8.RELATION_START + t8.RELATION_BITS
        ] = beta * code / alpha_select
        block1.gate.weight[relation, t8.CONST] = (
            -beta * (t8.RELATION_BITS - 1) / alpha_select
        )
        block1.up.weight[relation, t8.MEMORY_START + relation] = 1.0 / (
            beta * alpha_select
        )
        block1.down.weight[t8.CODE, relation] = 1.0
        block1.down.weight[t8.MEMORY_START + relation, relation] = -1.0

    block2 = model.blocks[2]
    alpha_decode = rms_scale(query_norm_squared)
    channel = 0
    for value in range(t8.VALUES):
        level = 2 * value - t8.MAX_LEVEL
        bits = t8.value_code(value).to(model.token.weight.device)
        for threshold, coefficient in (
            (level - 1, 1.0),
            (level, -2.0),
            (level + 1, 1.0),
        ):
            block2.gate.weight[channel, t8.CODE] = beta / alpha_decode
            block2.gate.weight[channel, t8.CONST] = -beta * threshold / alpha_decode
            block2.up.weight[channel, t8.QUERY_ROUTE] = 1.0 / (beta * alpha_decode)
            for bit in range(4):
                block2.down.weight[t8.OUTPUT_START + bit, channel] = bits[bit] * coefficient
            channel += 1

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
        "compiled_facts": t8.FACTS,
        "payload_scalars": t8.FACTS,
        "logical_payload_bits": t8.FACTS * math.log2(t8.VALUES),
        "program_hidden_coordinates": PROGRAM_DIMS,
        "program_attention_heads": PROGRAM_HEADS,
        "selection_channels": t8.RELATIONS,
        "decoder_channels": channel,
        "frozen_parameter_entries": frozen.count,
        "fixed_nonzero_entries": fixed_nonzero,
    }


def run_pilot(device: torch.device) -> dict[str, object]:
    natural = t8.TokenStream(t8.TRAIN_FILE)
    validation = t8.TokenStream(t8.VALIDATION_FILE)
    specifications = (("adamw", t7.ADAMW_LEARNING_RATE),) + tuple(
        ("muon", rate) for rate in MUON_RATES
    )
    arms: dict[str, object] = {}
    for kind, rate in specifications:
        label = kind if kind == "adamw" else f"muon_{rate:g}"
        model = build_model(PILOT_SEED, device)
        initial_hash = small.state_sha256(model)
        optimizer = t7.build_optimizer(model, kind, muon_learning_rate=rate)
        checkpoints: dict[str, object] = {}
        maximum_loss = 0.0
        maximum_gradient_norm = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(1, PILOT_STEPS + 1):
            model.train()
            optimizer.set_step(step)
            optimizer.zero_grad()
            inputs, targets = natural.batch(
                PILOT_SEED * 1_000_003 + step,
                NATURAL_BATCH,
                small.CONTEXT,
                device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
            loss = F.cross_entropy(logits.float().reshape(-1, small.VOCAB), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {label} pilot")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optimizer.step()
            maximum_loss = max(maximum_loss, float(loss.item()))
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            if step in PILOT_CHECKPOINTS:
                checkpoints[str(step)] = small.evaluate_natural(
                    model, validation, device, EVALUATION_SEED, batches=16
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        arms[label] = {
            "kind": optimizer.kind,
            "learning_rate": rate,
            "initial_hash": initial_hash,
            "checkpoints": checkpoints,
            "maximum_loss": maximum_loss,
            "maximum_gradient_norm": maximum_gradient_norm,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_state_bytes": t7.optimizer_state_bytes(optimizer),
            "peak_hbm_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        }
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    terminal = {
        label: result["checkpoints"][str(PILOT_STEPS)]["nll"]
        for label, result in arms.items()
    }
    eligible = {label: value for label, value in terminal.items() if label.startswith("muon_")}
    winner = min(eligible, key=eligible.get)
    return {
        "schema": "digital-relation-plane-t9-small-muon-pilot-v1",
        "device": str(device),
        "configuration": {
            "parameter_count": small.parameter_count(),
            "pilot_seed": PILOT_SEED,
            "steps": PILOT_STEPS,
            "muon_rates": MUON_RATES,
        },
        "integrity": {
            "source_sha256": t8.sha256_file(Path(__file__)),
            "pilot_protocol_sha256": t8.sha256_file(PILOT_PROTOCOL),
            "train_sha256": t8.sha256_file(t8.TRAIN_FILE),
            "validation_sha256": t8.sha256_file(t8.VALIDATION_FILE),
        },
        "identical_initial_hashes": len({v["initial_hash"] for v in arms.values()}) == 1,
        "arms": arms,
        "terminal_nll": terminal,
        "selected_arm": winner,
        "selected_muon_learning_rate": float(winner.removeprefix("muon_")),
    }


def run_quick(device: torch.device) -> dict[str, object]:
    natural = t8.TokenStream(t8.TRAIN_FILE)
    validation = t8.TokenStream(t8.VALIDATION_FILE)
    layout, unused = t8.choose_layout(natural, validation)
    world = t8.make_world(WORLD_SEED)
    model = build_model(5_123, device)
    before = sum(parameter.numel() for parameter in model.parameters())
    frozen, compiler = compile_relation_plane(model, world, layout)
    frozen.enforce(model)
    evaluation = t8.evaluate(model, world, layout, device)
    gates = {
        "same_parameter_count": before == small.parameter_count(),
        "exact": evaluation["structured_accuracy"] == 1.0,
        "zero_errors": evaluation["structured_errors"] == 0,
        "positive_margin": evaluation["minimum_logit_margin"] > 0.0,
        "unseen_near_chance": 0.04 <= evaluation["unseen_random_accuracy"] <= 0.09,
    }
    return {
        "schema": "digital-relation-plane-t9-small-quick-v1",
        "device": str(device),
        "configuration": {
            "parameter_count": before,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": HEADS,
            "ffn_width": FFN_WIDTH,
            "facts": t8.FACTS,
            "unused_tokens_available": unused,
        },
        "compiler": compiler,
        "evaluation": evaluation,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.pilot == arguments.quick:
        raise SystemExit("select exactly one of --pilot or --quick")
    result = run_pilot(torch.device(arguments.device)) if arguments.pilot else run_quick(torch.device(arguments.device))
    output = arguments.output or (PILOT_OUTPUT if arguments.pilot else QUICK_OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "selected": result.get("selected_arm"), "all_gates_pass": result.get("all_gates_pass")}, indent=2))


if __name__ == "__main__":
    main()
