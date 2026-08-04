#!/usr/bin/env python3
"""Matched T10 training pilot: direct writer versus compiler-label controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import raw_prose_equality_plane_t10 as t10
from experiments import heterogeneous_family_discovery_t7_scale as t7


OUTPUT = ROOT / "results/raw-prose-equality-plane-t10-training-pilot.json"
PROTOCOL = ROOT / "results/raw-prose-equality-plane-t10-training-pilot-protocol.md"
REPRESENTATION = ROOT / "results/raw-prose-equality-plane-t10-quick.json"

MODEL_SEED = 6_401
MUON_LEARNING_RATE = 0.005
RAW_BATCH = 64
QUERY_BATCH = 256
NATURAL_BATCH = 16
KNOWLEDGE_INTERVAL = 20
CHECKPOINTS_1X = (250, 500, 1_000)
STEPS_2X = 2_000


@dataclass(frozen=True)
class QuerySplit:
    direct_train_tokens: Tensor
    direct_train_targets: Tensor
    equality_train_tokens: Tensor
    equality_train_targets: Tensor
    direct_heldout_tokens: Tensor
    direct_heldout_targets: Tensor
    equality_heldout_tokens: Tensor
    equality_heldout_targets: Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_representation() -> dict[str, object]:
    if not REPRESENTATION.exists():
        raise RuntimeError("run the T10 representation gate first")
    artifact = json.loads(REPRESENTATION.read_text())
    if not artifact.get("all_gates_pass"):
        raise RuntimeError("T10 representation artifact did not pass")
    expected_source = sha256_file(Path(t10.__file__))
    if artifact["integrity"]["source_sha256"] != expected_source:
        raise RuntimeError("T10 source changed after its representation run")
    return artifact


def build_query_split(
    compiled: t10.raw.CompileResult, layout: t10.ProseLayout
) -> QuerySplit:
    grouped = t10.frames_by_relation_and_role(compiled, layout)
    aliases_by_canonical: dict[str, list[str]] = {
        canonical: [] for canonical in compiled.canonical_entities
    }
    for alias, canonical in compiled.alias_to_canonical.items():
        aliases_by_canonical[canonical].append(alias)
    for aliases in aliases_by_canonical.values():
        aliases.sort()
    canonical_index = {
        alias: index for index, alias in enumerate(compiled.canonical_entities)
    }
    values = sorted(layout.value_token)
    value_index = {value: index for index, value in enumerate(values)}

    direct_train_rows: list[list[int]] = []
    direct_train_targets: list[int] = []
    for canonical in compiled.canonical_entities:
        entity = canonical_index[canonical]
        alias = aliases_by_canonical[canonical][0]
        for relation in range(t10.RELATIONS):
            direct_train_rows.append(
                [
                    layout.alias_token[alias],
                    layout.frame_anchor_token[grouped[(relation, "a")][0]],
                    layout.frame_anchor_token[grouped[(relation, "b")][0]],
                    layout.direct_query,
                ]
            )
            direct_train_targets.append(
                value_index[compiled.canonical_values[entity][relation]]
            )

    equality_train_rows: list[list[int]] = []
    equality_train_targets: list[int] = []
    for entity, canonical in enumerate(compiled.canonical_entities):
        if entity % 4:
            continue
        alias = aliases_by_canonical[canonical][0]
        for first in range(t10.RELATIONS):
            for second in range(t10.RELATIONS):
                if (first + second) % 2:
                    continue
                equality_train_rows.append(
                    [
                        layout.alias_token[alias],
                        layout.frame_anchor_token[grouped[(first, "a")][0]],
                        layout.frame_anchor_token[grouped[(second, "b")][0]],
                        layout.equality_query,
                    ]
                )
                equality_train_targets.append(
                    int(
                        compiled.canonical_values[entity][first]
                        == compiled.canonical_values[entity][second]
                    )
                )

    direct_all_tokens, direct_all_targets = t10.direct_dataset(compiled, layout)
    equality_all_tokens, equality_all_targets = t10.equality_dataset(compiled, layout)
    direct_train_set = {tuple(row) for row in direct_train_rows}
    equality_train_set = {tuple(row) for row in equality_train_rows}
    direct_heldout = [
        index
        for index, row in enumerate(direct_all_tokens.tolist())
        if tuple(row) not in direct_train_set
    ]
    equality_heldout = [
        index
        for index, row in enumerate(equality_all_tokens.tolist())
        if tuple(row) not in equality_train_set
    ]
    return QuerySplit(
        direct_train_tokens=torch.tensor(direct_train_rows, dtype=torch.long),
        direct_train_targets=torch.tensor(direct_train_targets, dtype=torch.long),
        equality_train_tokens=torch.tensor(equality_train_rows, dtype=torch.long),
        equality_train_targets=torch.tensor(equality_train_targets, dtype=torch.long),
        direct_heldout_tokens=direct_all_tokens[direct_heldout],
        direct_heldout_targets=direct_all_targets[direct_heldout],
        equality_heldout_tokens=equality_all_tokens[equality_heldout],
        equality_heldout_targets=equality_all_targets[equality_heldout],
    )


def query_logits(
    model: t10.core.small.SharedInterpreterLM,
    tokens: Tensor,
    layout: t10.ProseLayout,
    kind: str,
) -> Tensor:
    final = model.hidden(tokens)[:, -1]
    if kind == "direct":
        values = sorted(layout.value_token)
        rows = torch.tensor(
            [layout.value_token[value] for value in values],
            dtype=torch.long,
            device=final.device,
        )
    elif kind == "equality":
        rows = torch.tensor(
            [layout.result_different, layout.result_same],
            dtype=torch.long,
            device=final.device,
        )
    else:
        raise ValueError(kind)
    return final @ model.token.weight[rows].T


@torch.no_grad()
def evaluate_query_slice(
    model: t10.core.small.SharedInterpreterLM,
    tokens: Tensor,
    targets: Tensor,
    layout: t10.ProseLayout,
    kind: str,
    device: torch.device,
    *,
    batch_size: int = 4_096,
) -> dict[str, object]:
    model.eval()
    predictions: list[Tensor] = []
    margins: list[Tensor] = []
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start : start + batch_size].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = query_logits(model, batch, layout, kind)
        ordered = logits.float().topk(2, dim=-1).values
        predictions.append(logits.argmax(-1).cpu())
        margins.append((ordered[:, 0] - ordered[:, 1]).cpu())
    predicted = torch.cat(predictions)
    correct = predicted == targets
    return {
        "accuracy": float(correct.float().mean().item()),
        "errors": int((~correct).sum().item()),
        "queries": len(tokens),
        "minimum_margin": float(torch.cat(margins).min().item()),
    }


def evaluate_model(
    model: t10.core.small.SharedInterpreterLM,
    split: QuerySplit,
    layout: t10.ProseLayout,
    validation: t10.core.small.TokenStream,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    return {
        "natural": t10.core.small.evaluate_natural(
            model, validation, device, seed, batches=32
        ),
        "direct_heldout": evaluate_query_slice(
            model,
            split.direct_heldout_tokens,
            split.direct_heldout_targets,
            layout,
            "direct",
            device,
        ),
        "equality_heldout": evaluate_query_slice(
            model,
            split.equality_heldout_tokens,
            split.equality_heldout_targets,
            layout,
            "equality",
            device,
        ),
    }


def shuffled_batches(
    count: int, batch_size: int, seed: int
) -> list[Tensor]:
    permutation = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed)
    )
    return [
        permutation[start : start + batch_size]
        for start in range(0, count, batch_size)
    ]


def apply_update(
    model: t10.core.small.SharedInterpreterLM,
    optimizer: t7.OptimizerBundle,
    loss: Tensor,
    frozen: t10.t8.FrozenEntries | None,
) -> float:
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite training loss")
    loss.backward()
    if frozen is not None:
        frozen.mask_gradients(model)
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    )
    if not math.isfinite(gradient_norm):
        raise RuntimeError("nonfinite gradient")
    optimizer.step()
    if frozen is not None:
        frozen.enforce(model)
    return gradient_norm


def train_query_pass(
    model: t10.core.small.SharedInterpreterLM,
    optimizer: t7.OptimizerBundle,
    tokens: Tensor,
    targets: Tensor,
    layout: t10.ProseLayout,
    kind: str,
    device: torch.device,
    seed: int,
    update_offset: int,
) -> tuple[int, int, float, float]:
    maximum_loss = 0.0
    maximum_gradient = 0.0
    updates = 0
    presentations = 0
    for indices in shuffled_batches(len(tokens), QUERY_BATCH, seed):
        updates += 1
        optimizer.set_step(min(update_offset + updates, STEPS_2X))
        optimizer.zero_grad()
        batch = tokens[indices].to(device)
        target = targets[indices].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = query_logits(model, batch, layout, kind)
        loss = F.cross_entropy(logits.float(), target)
        gradient = apply_update(model, optimizer, loss, None)
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        presentations += len(indices)
    return updates, presentations, maximum_loss, maximum_gradient


def train_raw_pass(
    model: t10.core.small.SharedInterpreterLM,
    optimizer: t7.OptimizerBundle,
    encoded: dict[int, Tensor],
    device: torch.device,
    seed: int,
    update_offset: int,
) -> tuple[int, int, float, float]:
    maximum_loss = 0.0
    maximum_gradient = 0.0
    updates = 0
    token_presentations = 0
    for length, rows in encoded.items():
        for indices in shuffled_batches(len(rows), RAW_BATCH, seed + length):
            updates += 1
            optimizer.set_step(min(update_offset + updates, STEPS_2X))
            optimizer.zero_grad()
            batch = rows[indices].to(device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.float().reshape(-1, t10.core.small.VOCAB),
                batch[:, 1:].reshape(-1),
            )
            gradient = apply_update(model, optimizer, loss, None)
            maximum_loss = max(maximum_loss, float(loss.item()))
            maximum_gradient = max(maximum_gradient, gradient)
            token_presentations += len(indices) * (length - 1)
    return updates, token_presentations, maximum_loss, maximum_gradient


def train_arm(
    arm: str,
    corpus: t10.raw.GeneratedCorpus,
    compiled: t10.raw.CompileResult,
    layout: t10.ProseLayout,
    encoded_raw: dict[int, Tensor],
    split: QuerySplit,
    natural: t10.core.small.TokenStream,
    validation: t10.core.small.TokenStream,
    device: torch.device,
) -> dict[str, object]:
    if arm not in {
        "compiler_muon_1x",
        "muon_labels_1x",
        "muon_labels_2x",
        "adamw_labels_2x",
    }:
        raise ValueError(arm)
    compiler_arm = arm == "compiler_muon_1x"
    optimizer_kind = "adamw" if arm.startswith("adamw") else "muon"
    prefix_passes = 0 if compiler_arm else (2 if arm.endswith("2x") else 1)
    steps = CHECKPOINTS_1X[-1] if arm.endswith("1x") else STEPS_2X
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    model = t10.core.build_model(MODEL_SEED, device)
    initial_hash = t10.core.small.state_sha256(model)
    frozen: t10.t8.FrozenEntries | None = None
    writer: dict[str, object] | None = None
    writer_seconds = 0.0
    if compiler_arm:
        writer_started = time.perf_counter()
        frozen, writer = t10.compile_equality_plane(model, compiled, layout)
        if device.type == "cuda":
            torch.cuda.synchronize()
        writer_seconds = time.perf_counter() - writer_started
    optimizer = t7.build_optimizer(
        model, optimizer_kind, muon_learning_rate=MUON_LEARNING_RATE
    )

    prefix_updates = 0
    raw_token_presentations = 0
    direct_target_presentations = 0
    equality_target_presentations = 0
    maximum_loss = 0.0
    maximum_gradient = 0.0
    for pass_index in range(prefix_passes):
        result = train_raw_pass(
            model,
            optimizer,
            encoded_raw,
            device,
            MODEL_SEED + 10_000 * pass_index,
            prefix_updates,
        )
        updates, presentations, loss, gradient = result
        prefix_updates += updates
        raw_token_presentations += presentations
        maximum_loss = max(maximum_loss, loss)
        maximum_gradient = max(maximum_gradient, gradient)
        result = train_query_pass(
            model,
            optimizer,
            split.direct_train_tokens,
            split.direct_train_targets,
            layout,
            "direct",
            device,
            MODEL_SEED + 20_000 * pass_index,
            prefix_updates,
        )
        updates, presentations, loss, gradient = result
        prefix_updates += updates
        direct_target_presentations += presentations
        maximum_loss = max(maximum_loss, loss)
        maximum_gradient = max(maximum_gradient, gradient)
        result = train_query_pass(
            model,
            optimizer,
            split.equality_train_tokens,
            split.equality_train_targets,
            layout,
            "equality",
            device,
            MODEL_SEED + 30_000 * pass_index,
            prefix_updates,
        )
        updates, presentations, loss, gradient = result
        prefix_updates += updates
        equality_target_presentations += presentations
        maximum_loss = max(maximum_loss, loss)
        maximum_gradient = max(maximum_gradient, gradient)

    checkpoints: dict[str, object] = {}
    natural_tokens = 0
    mixed_direct_presentations = 0
    mixed_equality_presentations = 0
    power_samples: list[float] = []
    checkpoint_set = set(CHECKPOINTS_1X) | ({STEPS_2X} if steps == STEPS_2X else set())
    for step in range(1, steps + 1):
        model.train()
        optimizer.set_step(step)
        optimizer.zero_grad()
        if step % KNOWLEDGE_INTERVAL:
            inputs, targets = natural.batch(
                MODEL_SEED * 1_000_003 + step,
                NATURAL_BATCH,
                t10.core.small.CONTEXT,
                device,
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, t10.core.small.VOCAB),
                targets.reshape(-1),
            )
            natural_tokens += NATURAL_BATCH * t10.core.small.CONTEXT
        else:
            kind = "direct" if step % (2 * KNOWLEDGE_INTERVAL) else "equality"
            tokens = (
                split.direct_train_tokens
                if kind == "direct"
                else split.equality_train_tokens
            )
            targets = (
                split.direct_train_targets
                if kind == "direct"
                else split.equality_train_targets
            )
            generator = torch.Generator().manual_seed(MODEL_SEED * 1_000_003 + step)
            indices = torch.randint(0, len(tokens), (QUERY_BATCH,), generator=generator)
            batch = tokens[indices].to(device)
            target = targets[indices].to(device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = query_logits(model, batch, layout, kind)
            loss = F.cross_entropy(logits.float(), target)
            if kind == "direct":
                mixed_direct_presentations += QUERY_BATCH
            else:
                mixed_equality_presentations += QUERY_BATCH
        gradient = apply_update(model, optimizer, loss, frozen)
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 20 == 0 and device.type == "cuda":
            sample = t7.read_gpu_power_watts()
            if sample is not None:
                power_samples.append(sample)
        if step in checkpoint_set:
            checkpoints[str(step)] = evaluate_model(
                model,
                split,
                layout,
                validation,
                device,
                t10.core.EVALUATION_SEED + MODEL_SEED + step,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    mean_power = float(np.mean(power_samples)) if power_samples else None
    terminal_hash = t10.core.small.state_sha256(model)
    result = {
        "arm": arm,
        "optimizer": optimizer.kind,
        "steps": steps,
        "prefix_passes": prefix_passes,
        "initial_hash": initial_hash,
        "terminal_hash": terminal_hash,
        "writer": writer,
        "checkpoints": checkpoints,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
        "resource_ledger": {
            "elapsed_seconds": elapsed,
            "gpu_seconds": elapsed if device.type == "cuda" else 0.0,
            "writer_seconds": writer_seconds,
            "mean_sampled_power_watts": mean_power,
            "estimated_joules": mean_power * elapsed if mean_power is not None else None,
            "peak_hbm_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "optimizer_state_bytes": t7.optimizer_state_bytes(optimizer),
            "prefix_optimizer_updates": prefix_updates,
            "mixed_optimizer_updates": steps,
            "raw_prose_token_presentations": raw_token_presentations,
            "direct_target_presentations": direct_target_presentations
            + mixed_direct_presentations,
            "equality_target_presentations": equality_target_presentations
            + mixed_equality_presentations,
            "natural_input_token_presentations": natural_tokens,
            "failed_or_retried_steps": 0,
        },
    }
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(device: torch.device) -> dict[str, object]:
    representation = verify_representation()
    generated_started = time.perf_counter()
    corpus = t10.raw.make_corpus(
        seed=6_000_059,
        entities=t10.ENTITIES,
        relations=t10.RELATIONS,
        values=t10.VALUES,
        views=t10.ALIAS_VIEWS,
        paraphrases=t10.PARAPHRASES,
        repeats=1,
        corruption_rate=0.0,
    )
    compiled = t10.raw.compile_corpus(corpus.sentences)
    compiler_seconds = time.perf_counter() - generated_started
    natural = t10.core.small.TokenStream(t10.TRAIN_FILE)
    validation = t10.core.small.TokenStream(t10.VALIDATION_FILE)
    layout, unused = t10.choose_layout(natural, validation, corpus, compiled)
    encoded_raw = t10.encode_raw_corpus(corpus, layout)
    split = build_query_split(compiled, layout)

    arm_order = (
        "muon_labels_1x",
        "compiler_muon_1x",
        "adamw_labels_2x",
        "muon_labels_2x",
    )
    arms = {
        arm: train_arm(
            arm,
            corpus,
            compiled,
            layout,
            encoded_raw,
            split,
            natural,
            validation,
            device,
        )
        for arm in arm_order
    }
    candidate = arms["compiler_muon_1x"]
    baseline = arms["muon_labels_1x"]
    gates: dict[str, bool] = {
        "representation_precondition_passed": bool(representation["all_gates_pass"]),
        "identical_initial_hashes": len({arm["initial_hash"] for arm in arms.values()})
        == 1,
        "finite": all(
            math.isfinite(arm["maximum_loss"])
            and math.isfinite(arm["maximum_gradient_norm"])
            and arm["maximum_loss"] < 100.0
            for arm in arms.values()
        ),
    }
    for checkpoint in CHECKPOINTS_1X:
        key = str(checkpoint)
        result = candidate["checkpoints"][key]
        gates[f"candidate_{key}_direct_exact"] = (
            result["direct_heldout"]["accuracy"] == 1.0
            and result["direct_heldout"]["errors"] == 0
            and result["direct_heldout"]["minimum_margin"] > 0.0
        )
        gates[f"candidate_{key}_equality_exact"] = (
            result["equality_heldout"]["accuracy"] == 1.0
            and result["equality_heldout"]["errors"] == 0
            and result["equality_heldout"]["minimum_margin"] > 0.0
        )
        gates[f"candidate_{key}_natural_within_0p5pct"] = (
            result["natural"]["nll"]
            <= 1.005 * baseline["checkpoints"][key]["natural"]["nll"]
        )
    controls = (arms["muon_labels_2x"], arms["adamw_labels_2x"])
    best_direct = max(
        arm["checkpoints"][str(STEPS_2X)]["direct_heldout"]["accuracy"]
        for arm in controls
    )
    best_equality = max(
        arm["checkpoints"][str(STEPS_2X)]["equality_heldout"]["accuracy"]
        for arm in controls
    )
    gates["qualitative_control_gap"] = best_direct < 0.90 or best_equality < 0.90
    return {
        "schema": "raw-prose-equality-plane-t10-training-pilot-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": t10.core.small.parameter_count(),
            "model_seed": MODEL_SEED,
            "muon_learning_rate": MUON_LEARNING_RATE,
            "raw_sentences": len(corpus.sentences),
            "direct_train_queries": len(split.direct_train_tokens),
            "equality_train_queries": len(split.equality_train_tokens),
            "direct_heldout_queries": len(split.direct_heldout_tokens),
            "equality_heldout_queries": len(split.equality_heldout_tokens),
            "unused_tokens_available": unused,
        },
        "compiler_seconds_charged_to_every_arm": compiler_seconds,
        "execution_order": arm_order,
        "arms": arms,
        "best_2x_control": {
            "direct_heldout_accuracy": best_direct,
            "equality_heldout_accuracy": best_equality,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "representation_sha256": sha256_file(REPRESENTATION),
            "representation_source_sha256": sha256_file(Path(t10.__file__)),
            "extractor_source_sha256": sha256_file(Path(t10.raw.__file__)),
            "protocol_sha256": sha256_file(PROTOCOL),
            "raw_corpus_sha256": t10.raw.corpus_sha256(corpus.sentences),
            "train_sha256": sha256_file(t10.TRAIN_FILE),
            "validation_sha256": sha256_file(t10.VALIDATION_FILE),
        },
        "claim_boundary": (
            "Single-seed matched-training pilot. Passing admits a three-seed run; "
            "it does not establish real-world prose extraction or production capability."
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
                "best_2x_control": result["best_2x_control"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
