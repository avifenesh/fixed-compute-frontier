#!/usr/bin/env python3
"""T9: sealed 36.6M second-scale replication of the digital relation plane."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments import digital_relation_plane_t8 as t8
from experiments import digital_relation_plane_t8_train as t8train
from experiments import digital_relation_plane_t9_small_core as core
from experiments import heterogeneous_family_discovery_t7_scale as t7


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "results/digital-relation-plane-t9-small-preregistration.md"
PILOT_ARTIFACT = ROOT / "results/digital-relation-plane-t9-small-muon-pilot.json"
PREDECESSOR = ROOT / "results/digital-relation-plane-t8-train.json"
OUTPUT = ROOT / "results/digital-relation-plane-t9-small-train.json"

PILOT_SHA256 = "1dd04991690a7c1867d84a1ec5d88a068b56be5cd1506186ef07c08ed7df6026"
MUON_LEARNING_RATE = 0.005
MODEL_SEEDS = (5_123, 5_461, 5_819)
NATURAL_BATCH = 16
KNOWLEDGE_BATCH = 64
MIXED_KNOWLEDGE_BATCH = 256
KNOWLEDGE_INTERVAL = 20
CHECKPOINTS_1X = (250, 500, 1_000)
STEPS_2X = 2_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pilot() -> None:
    if sha256_file(PILOT_ARTIFACT) != PILOT_SHA256:
        raise RuntimeError("small-scale pilot hash mismatch")
    artifact = json.loads(PILOT_ARTIFACT.read_text())
    if artifact["selected_muon_learning_rate"] != MUON_LEARNING_RATE:
        raise RuntimeError("small-scale Muon selection mismatch")


def knowledge_logits(
    model: core.small.SharedInterpreterLM,
    tokens: Tensor,
    layout: t8.RelationLayout,
) -> Tensor:
    final = model.hidden(tokens)[:, -1]
    rows = model.token.weight[
        torch.tensor(layout.result, dtype=torch.long, device=final.device)
    ]
    return final @ rows.T


@torch.no_grad()
def evaluate(
    model: core.small.SharedInterpreterLM,
    world: t8.RelationWorld,
    layout: t8.RelationLayout,
    device: torch.device,
    *,
    batch_size: int = 4_096,
) -> dict[str, object]:
    model.eval()
    entities = torch.arange(t8.STRUCTURED_ENTITIES).repeat_interleave(t8.RELATIONS)
    relations = torch.arange(t8.RELATIONS).repeat(t8.STRUCTURED_ENTITIES)
    targets = world.values.reshape(-1).to(device)
    predictions: list[Tensor] = []
    margins: list[Tensor] = []
    for start in range(0, len(entities), batch_size):
        tokens = t8.encode_queries(
            entities[start : start + batch_size],
            relations[start : start + batch_size],
            layout,
            device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = knowledge_logits(model, tokens, layout)
        ordered = logits.float().topk(2, dim=-1).values
        predictions.append(logits.argmax(-1))
        margins.append(ordered[:, 0] - ordered[:, 1])
    predicted = torch.cat(predictions)
    margin = torch.cat(margins)
    correct = predicted == targets

    generator = torch.Generator(device="cpu").manual_seed(core.EVALUATION_SEED)
    unseen_entities = torch.arange(
        t8.STRUCTURED_ENTITIES, t8.ENTITIES
    ).repeat_interleave(t8.RELATIONS)
    unseen_relations = torch.arange(t8.RELATIONS).repeat(t8.UNSEEN_ENTITIES)
    unseen_targets = torch.randint(
        0, t8.VALUES, (len(unseen_entities),), generator=generator
    ).to(device)
    unseen_predictions: list[Tensor] = []
    for start in range(0, len(unseen_entities), batch_size):
        tokens = t8.encode_queries(
            unseen_entities[start : start + batch_size],
            unseen_relations[start : start + batch_size],
            layout,
            device,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            unseen_predictions.append(knowledge_logits(model, tokens, layout).argmax(-1))
    return {
        "structured_accuracy": float(correct.float().mean().item()),
        "structured_errors": int((~correct).sum().item()),
        "minimum_logit_margin": float(margin.min().item()),
        "mean_logit_margin": float(margin.mean().item()),
        "unseen_random_accuracy": float(
            (torch.cat(unseen_predictions) == unseen_targets).float().mean().item()
        ),
    }


def train_arm(
    arm: str,
    seed: int,
    world: t8.RelationWorld,
    layout: t8.RelationLayout,
    natural: t8.TokenStream,
    validation: t8.TokenStream,
    device: torch.device,
) -> dict[str, object]:
    compiler_arm = arm == "compiler_muon_1x"
    optimizer_kind = "adamw" if arm == "adamw_2x" else "muon"
    steps = CHECKPOINTS_1X[-1] if arm.endswith("1x") else STEPS_2X
    prefix_passes = 0 if compiler_arm else (2 if arm.endswith("2x") else 1)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    model = core.build_model(seed, device)
    initial_hash = core.small.state_sha256(model)
    frozen: t8.FrozenEntries | None = None
    compiler: dict[str, object] | None = None
    if compiler_arm:
        frozen, compiler = core.compile_relation_plane(model, world, layout)
    optimizer = t7.build_optimizer(
        model, optimizer_kind, muon_learning_rate=MUON_LEARNING_RATE
    )

    entities, relations, targets = t8train.prefix_examples(world)
    prefix_losses: list[float] = []
    maximum_gradient_norm = 0.0
    optimizer_updates = 0
    knowledge_model_input_tokens = 0
    for pass_index in range(prefix_passes):
        permutation = torch.randperm(
            t8.FACTS,
            generator=torch.Generator().manual_seed(seed + 50_000 + pass_index),
        )
        for start in range(0, t8.FACTS, KNOWLEDGE_BATCH):
            indices = permutation[start : start + KNOWLEDGE_BATCH]
            tokens = t8.encode_queries(
                entities[indices], relations[indices], layout, device
            )
            batch_targets = targets[indices].to(device)
            optimizer_updates += 1
            optimizer.set_step(min(optimizer_updates, STEPS_2X))
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = knowledge_logits(model, tokens, layout)
            loss = F.cross_entropy(logits.float(), batch_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {arm} prefix loss")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"nonfinite {arm} prefix gradient")
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            optimizer.step()
            prefix_losses.append(float(loss.item()))
            knowledge_model_input_tokens += len(indices) * 2

    checkpoints: dict[str, object] = {}
    maximum_loss = max(prefix_losses, default=0.0)
    natural_tokens = 0
    power_samples: list[float] = []
    checkpoint_set = set(CHECKPOINTS_1X) | ({STEPS_2X} if steps == STEPS_2X else set())
    for step in range(1, steps + 1):
        model.train()
        optimizer_updates += 1
        optimizer.set_step(step)
        optimizer.zero_grad()
        if step % KNOWLEDGE_INTERVAL == 0:
            tokens, batch_targets = t8train.knowledge_batch(
                world,
                layout,
                MIXED_KNOWLEDGE_BATCH,
                seed * 1_000_003 + step,
                device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = knowledge_logits(model, tokens, layout)
            loss = F.cross_entropy(logits.float(), batch_targets)
            knowledge_model_input_tokens += MIXED_KNOWLEDGE_BATCH * 2
        else:
            inputs, batch_targets = natural.batch(
                seed * 1_000_003 + step,
                NATURAL_BATCH,
                core.small.CONTEXT,
                device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, core.small.VOCAB),
                batch_targets.reshape(-1),
            )
            natural_tokens += NATURAL_BATCH * core.small.CONTEXT
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm} mixed loss at {step}")
        loss.backward()
        if frozen is not None:
            frozen.mask_gradients(model)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"nonfinite {arm} mixed gradient")
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        if frozen is not None:
            frozen.enforce(model)
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step % 20 == 0 and device.type == "cuda":
            sample = t7.read_gpu_power_watts()
            if sample is not None:
                power_samples.append(sample)
        if step in checkpoint_set:
            checkpoints[str(step)] = {
                "natural": core.small.evaluate_natural(
                    model,
                    validation,
                    device,
                    core.EVALUATION_SEED + seed,
                    batches=32,
                ),
                "knowledge": evaluate(model, world, layout, device),
                "natural_input_tokens": natural_tokens,
                "knowledge_model_input_tokens": knowledge_model_input_tokens,
            }
    if device.type == "cuda":
        torch.cuda.synchronize()
    terminal_hash = core.small.state_sha256(model)
    elapsed = time.perf_counter() - started
    mean_power = float(np.mean(power_samples)) if power_samples else None
    return {
        "arm": arm,
        "optimizer": optimizer.kind,
        "steps": steps,
        "prefix_passes": prefix_passes,
        "prefix_updates": prefix_passes * math.ceil(t8.FACTS / KNOWLEDGE_BATCH),
        "prefix_loss_first": prefix_losses[0] if prefix_losses else None,
        "prefix_loss_last": prefix_losses[-1] if prefix_losses else None,
        "initial_hash": initial_hash,
        "terminal_hash": terminal_hash,
        "compiler": compiler,
        "checkpoints": checkpoints,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient_norm,
        "resource_ledger": {
            "elapsed_seconds": elapsed,
            "gpu_seconds": elapsed if device.type == "cuda" else 0.0,
            "mean_sampled_power_watts": mean_power,
            "estimated_joules": mean_power * elapsed if mean_power is not None else None,
            "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "optimizer_state_bytes": t7.optimizer_state_bytes(optimizer),
            "natural_input_token_presentations": natural_tokens,
            "knowledge_raw_facts_consumed": t8.FACTS * max(prefix_passes, 1),
            "knowledge_model_input_token_presentations": knowledge_model_input_tokens,
            "optimizer_updates": optimizer_updates,
            "failed_or_retried_steps": 0,
        },
    }


def run(device: torch.device) -> dict[str, object]:
    verify_pilot()
    natural = t8.TokenStream(t8.TRAIN_FILE)
    validation = t8.TokenStream(t8.VALIDATION_FILE)
    layout, unused = t8.choose_layout(natural, validation)
    world = t8.make_world(core.WORLD_SEED)
    arm_orders = (
        ("muon_1x", "compiler_muon_1x", "muon_2x", "adamw_2x"),
        ("compiler_muon_1x", "adamw_2x", "muon_1x", "muon_2x"),
        ("muon_2x", "muon_1x", "adamw_2x", "compiler_muon_1x"),
    )
    worlds: dict[str, object] = {}
    for seed_index, seed in enumerate(MODEL_SEEDS):
        arms = {
            arm: train_arm(arm, seed, world, layout, natural, validation, device)
            for arm in arm_orders[seed_index]
        }
        worlds[str(seed)] = {
            "execution_order": arm_orders[seed_index],
            "arms": arms,
            "identical_initial_hashes": len({a["initial_hash"] for a in arms.values()}) == 1,
        }

    gates: dict[str, bool] = {}
    for seed, result in worlds.items():
        arms = result["arms"]
        candidate = arms["compiler_muon_1x"]
        baseline = arms["muon_1x"]
        gates[f"{seed}_same_initialization"] = result["identical_initial_hashes"]
        for checkpoint in CHECKPOINTS_1X:
            key = str(checkpoint)
            knowledge = candidate["checkpoints"][key]["knowledge"]
            gates[f"{seed}_{key}_candidate_exact"] = (
                knowledge["structured_accuracy"] == 1.0
                and knowledge["structured_errors"] == 0
                and knowledge["minimum_logit_margin"] > 0.0
            )
            gates[f"{seed}_{key}_unseen_near_chance"] = 0.04 <= knowledge["unseen_random_accuracy"] <= 0.09
            candidate_nll = candidate["checkpoints"][key]["natural"]["nll"]
            baseline_nll = baseline["checkpoints"][key]["natural"]["nll"]
            gates[f"{seed}_{key}_natural_within_0p5pct"] = candidate_nll <= 1.005 * baseline_nll
        for control in ("muon_2x", "adamw_2x"):
            accuracy = arms[control]["checkpoints"][str(STEPS_2X)]["knowledge"]["structured_accuracy"]
            gates[f"{seed}_{control}_below_50pct"] = accuracy < 0.50
        gates[f"{seed}_candidate_terminal_improves"] = (
            candidate["checkpoints"]["1000"]["natural"]["nll"]
            <= candidate["checkpoints"]["500"]["natural"]["nll"]
        )
        gates[f"{seed}_finite"] = all(
            math.isfinite(arm["maximum_loss"])
            and math.isfinite(arm["maximum_gradient_norm"])
            and arm["maximum_loss"] < 100.0
            for arm in arms.values()
        )
    return {
        "schema": "digital-relation-plane-t9-small-training-v1",
        "status": "complete",
        "device": str(device),
        "configuration": {
            "parameter_count": core.small.parameter_count(),
            "hidden": core.HIDDEN,
            "layers": core.LAYERS,
            "heads": core.HEADS,
            "ffn_width": core.FFN_WIDTH,
            "model_seeds": MODEL_SEEDS,
            "world_seed": core.WORLD_SEED,
            "evaluation_seed": core.EVALUATION_SEED,
            "compiled_facts": t8.FACTS,
            "logical_payload_bits": t8.FACTS * math.log2(t8.VALUES),
            "unused_tokens_available": unused,
            "muon_learning_rate": MUON_LEARNING_RATE,
        },
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "core_source_sha256": sha256_file(Path(core.__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "pilot_artifact_sha256": sha256_file(PILOT_ARTIFACT),
            "predecessor_sha256": sha256_file(PREDECESSOR),
            "train_sha256": sha256_file(t8.TRAIN_FILE),
            "validation_sha256": sha256_file(t8.VALIDATION_FILE),
        },
        "worlds": worlds,
        "gates": gates,
        "all_gates_pass": bool(gates) and all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    result = run(torch.device(arguments.device))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "all_gates_pass": result["all_gates_pass"]}, indent=2))


if __name__ == "__main__":
    main()
