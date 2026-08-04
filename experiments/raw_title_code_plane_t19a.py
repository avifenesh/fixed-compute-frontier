#!/usr/bin/env python3
"""T19a: co-train a raw-title digital address inside an ordinary Transformer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import re
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

from experiments import heterogeneous_family_discovery_t7_scale as t7
from experiments import hotpot_real_prose_t12_data as data
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/raw-title-code-plane-t19a-preregistration.md"
OUTPUT = ROOT / "results/raw-title-code-plane-t19a.json"
CANDIDATE_CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
DEVELOPMENT_EVALUATOR = ROOT / "data/hotpot-real-prose-t12/sealed-evaluator.jsonl"

TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
CANDIDATE_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
EVALUATOR_SHA256 = "5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b"
NATURAL_TRAIN_SHA256 = "1871a8a790e2b2ae5273f1a46bc9fa2e39cbe95d5cb7537bde5a2b3e35cb464a"
NATURAL_VALIDATION_SHA256 = "889188f0e4c43cd49b2933ac462e8bd367520f15fe08d324f169f7eca962ce7b"

MODEL_SEED = 9_019
CODE_DIMS = 32
CODE_AMPLITUDE = 0.02
WRITE_BUDGET = 743_734
STEPS_1X = 3_000
STEPS_2X = 6_000
WARMUP_STEPS = 500
NATURAL_BATCH = 16
ROUTER_BATCH = 64
ROUTER_CONTEXT = 64
MUON_PEAK_LR = 0.005
ADAMW_PEAK_LR = 0.0003
NATURAL_EVALUATION_BATCHES = 32
TRAIN_WRAPPERS = ("{title}", "Article about {title}", "Document title: {title}")
HELD_WRAPPERS = ("Information concerning {title}", "Read the entry for {title}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_token_code(token_id: int, dimensions: int = CODE_DIMS) -> Tensor:
    digest = b""
    counter = 0
    while len(digest) * 8 < dimensions:
        digest += hashlib.sha256(
            f"t19a-token-{token_id}-{counter}".encode()
        ).digest()
        counter += 1
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:dimensions]
    return torch.from_numpy(bits.astype(np.float32) * 2.0 - 1.0)


@dataclass(frozen=True)
class TitleData:
    titles: tuple[str, ...]
    document_ids: tuple[str, ...]
    canonical_tokens: tuple[tuple[int, ...], ...]
    prototypes: Tensor
    active_tokens: Tensor
    active_codes: Tensor


def load_raw_documents(path: Path) -> list[dict[str, str]]:
    documents = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {"document_id", "title", "text"}
    if len(documents) != 2_405:
        raise RuntimeError(f"expected 2,405 documents, found {len(documents)}")
    if any(set(document) != expected for document in documents):
        raise RuntimeError("compiler corpus contains a non-raw field")
    if len({document["title"] for document in documents}) != len(documents):
        raise RuntimeError("document titles are not unique")
    return sorted(documents, key=lambda document: document["document_id"])


def build_title_data(documents: list[dict[str, str]], tokenizer: object) -> TitleData:
    titles = tuple(document["title"] for document in documents)
    document_ids = tuple(document["document_id"] for document in documents)
    canonical_tokens: list[tuple[int, ...]] = []
    active: set[int] = set()
    cache: dict[int, Tensor] = {}
    prototypes: list[Tensor] = []
    for title in titles:
        identifiers = tuple(
            int(value)
            for value in tokenizer.encode(" " + title, add_special_tokens=False)
        )
        if not identifiers:
            raise RuntimeError(f"empty title tokenization: {title!r}")
        canonical_tokens.append(identifiers)
        active.update(identifiers)
        codes = [
            cache.setdefault(identifier, deterministic_token_code(identifier))
            for identifier in identifiers
        ]
        prototypes.append(F.normalize(torch.stack(codes).mean(dim=0), dim=0))
    active_tokens = torch.tensor(sorted(active), dtype=torch.long)
    active_codes = torch.stack(
        [cache.setdefault(int(token), deterministic_token_code(int(token))) for token in active_tokens]
    )
    return TitleData(
        titles=titles,
        document_ids=document_ids,
        canonical_tokens=tuple(canonical_tokens),
        prototypes=torch.stack(prototypes),
        active_tokens=active_tokens,
        active_codes=active_codes,
    )


def encode_tail(text: str, tokenizer: object) -> Tensor:
    identifiers = [
        int(value) for value in tokenizer.encode(text, add_special_tokens=False)
    ]
    if not identifiers:
        raise RuntimeError(f"empty routing input: {text!r}")
    eos = int(tokenizer.eos_token_id)
    identifiers = identifiers[-ROUTER_CONTEXT:]
    padded = [eos] * (ROUTER_CONTEXT - len(identifiers)) + identifiers
    return torch.tensor(padded, dtype=torch.long)


def build_wrapper_examples(
    title_data: TitleData, tokenizer: object, wrappers: tuple[str, ...]
) -> tuple[Tensor, Tensor]:
    rows: list[Tensor] = []
    targets: list[int] = []
    for target, title in enumerate(title_data.titles):
        for wrapper in wrappers:
            rows.append(encode_tail(wrapper.format(title=title), tokenizer))
            targets.append(target)
    return torch.stack(rows), torch.tensor(targets, dtype=torch.long)


def build_natural_surface_examples(
    title_data: TitleData, tokenizer: object, path: Path
) -> tuple[Tensor, Tensor]:
    title_index = {title: index for index, title in enumerate(title_data.titles)}
    rows: list[Tensor] = []
    targets: list[int] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        question = str(record["question"])
        for title in record["supporting_titles"]:
            match = re.search(re.escape(title), question, flags=re.I)
            if match is None:
                raise RuntimeError(f"development title absent from question: {title}")
            rows.append(encode_tail(question[: match.end()], tokenizer))
            targets.append(title_index[title])
    if len(rows) != 208:
        raise RuntimeError(f"expected 208 natural surfaces, found {len(rows)}")
    return torch.stack(rows), torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def compile_token_codes(
    model: t10.core.small.SharedInterpreterLM, title_data: TitleData
) -> dict[str, object]:
    token_ids = title_data.active_tokens.to(model.token.weight.device)
    values = (CODE_AMPLITUDE * title_data.active_codes).to(
        model.token.weight.device, dtype=model.token.weight.dtype
    )
    model.token.weight[token_ids, :CODE_DIMS] = values
    return {
        "active_token_rows": len(token_ids),
        "code_dimensions": CODE_DIMS,
        "written_entries": len(token_ids) * CODE_DIMS,
        "values": values.detach().clone(),
    }


@torch.no_grad()
def enforce_token_codes(
    model: t10.core.small.SharedInterpreterLM,
    title_data: TitleData,
    compiled: dict[str, object],
) -> None:
    token_ids = title_data.active_tokens.to(model.token.weight.device)
    values = compiled["values"]
    assert isinstance(values, Tensor)
    model.token.weight[token_ids, :CODE_DIMS] = values


@torch.no_grad()
def compiled_cells_exact(
    model: t10.core.small.SharedInterpreterLM,
    title_data: TitleData,
    compiled: dict[str, object],
) -> bool:
    token_ids = title_data.active_tokens.to(model.token.weight.device)
    values = compiled["values"]
    assert isinstance(values, Tensor)
    return torch.equal(model.token.weight[token_ids, :CODE_DIMS], values)


def schedule_fraction(step: int, total_steps: int) -> float:
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return 0.1 + 0.9 * cosine


def set_learning_rate(
    optimizer: t7.OptimizerBundle, step: int, total_steps: int
) -> None:
    fraction = schedule_fraction(step, total_steps)
    if optimizer.matrix is not None:
        for group in optimizer.matrix.param_groups:
            group["lr"] = MUON_PEAK_LR * fraction
    for group in optimizer.auxiliary.param_groups:
        group["lr"] = ADAMW_PEAK_LR * fraction


@torch.no_grad()
def evaluate_router(
    model: t10.core.small.SharedInterpreterLM,
    examples: tuple[Tensor, Tensor],
    prototypes: Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, object]:
    model.eval()
    rows, targets = examples
    prototype_values = F.normalize(prototypes.float(), dim=-1).to(device)
    correct = 0
    margins: list[Tensor] = []
    for start in range(0, len(rows), batch_size):
        tokens = rows[start : start + batch_size].to(device)
        target = targets[start : start + batch_size].to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden = model.hidden(tokens)[:, -1, :CODE_DIMS]
        query = F.normalize(hidden.float(), dim=-1)
        scores = query @ prototype_values.T
        ordered = scores.topk(2, dim=-1)
        correct += int((ordered.indices[:, 0] == target).sum().item())
        target_scores = scores.gather(1, target[:, None]).squeeze(1)
        masked = scores.clone()
        masked.scatter_(1, target[:, None], float("-inf"))
        margins.append((target_scores - masked.max(dim=-1).values).cpu())
    all_margins = torch.cat(margins)
    return {
        "examples": len(rows),
        "accuracy": correct / len(rows),
        "errors": len(rows) - correct,
        "minimum_target_margin": float(all_margins.min().item()),
        "mean_target_margin": float(all_margins.mean().item()),
    }


@torch.no_grad()
def natural_nll(
    model: t10.core.small.SharedInterpreterLM,
    validation: t10.core.small.TokenStream,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    return t10.core.small.evaluate_natural(
        model,
        validation,
        device,
        seed,
        batches=NATURAL_EVALUATION_BATCHES,
    )


def state_schema(model: t10.core.small.SharedInterpreterLM) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())


def train_arm(
    label: str,
    total_steps: int,
    compiled_arm: bool,
    title_data: TitleData,
    train_examples: tuple[Tensor, Tensor],
    evaluations: dict[str, tuple[Tensor, Tensor]],
    natural: t10.core.small.TokenStream,
    validation: t10.core.small.TokenStream,
    device: torch.device,
) -> tuple[t10.core.small.SharedInterpreterLM, dict[str, object]]:
    model = t10.core.build_model(MODEL_SEED, device)
    base_state_hash = t10.core.small.state_sha256(model)
    schema = state_schema(model)
    compiled: dict[str, object] | None = None
    compiler_started = time.perf_counter()
    if compiled_arm:
        compiled = compile_token_codes(model, title_data)
    compiler_seconds = time.perf_counter() - compiler_started
    post_compile_hash = t10.core.small.state_sha256(model)
    optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=MUON_PEAK_LR
    )
    rows, targets = train_examples
    prototypes_device = title_data.prototypes.to(device)
    maximum_loss = 0.0
    maximum_natural_loss = 0.0
    maximum_router_loss = 0.0
    maximum_gradient = 0.0
    failed_or_retried = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad()
        set_learning_rate(optimizer, step, total_steps)

        natural_inputs, natural_targets = natural.batch(
            MODEL_SEED * 1_000_003 + step,
            NATURAL_BATCH,
            t10.core.small.CONTEXT,
            device,
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(natural_inputs)
        natural_loss = F.cross_entropy(
            logits.float().reshape(-1, t10.core.small.VOCAB),
            natural_targets.reshape(-1),
        )
        if not torch.isfinite(natural_loss):
            raise RuntimeError(f"nonfinite natural loss in {label} step {step}")
        natural_loss.backward()
        natural_loss_value = float(natural_loss.item())
        del logits, natural_loss

        generator = np.random.default_rng(MODEL_SEED * 2_000_033 + step)
        selected = torch.from_numpy(
            generator.integers(0, len(rows), size=ROUTER_BATCH, dtype=np.int64)
        )
        route_inputs = rows[selected].to(device)
        route_targets = targets[selected].to(device)
        target_codes = prototypes_device[route_targets]
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            predicted_codes = model.hidden(route_inputs)[:, -1, :CODE_DIMS]
        normalized = F.normalize(predicted_codes.float(), dim=-1)
        router_loss = (1.0 - (normalized * target_codes).sum(dim=-1)).mean()
        if not torch.isfinite(router_loss):
            raise RuntimeError(f"nonfinite router loss in {label} step {step}")
        router_loss.backward()
        router_loss_value = float(router_loss.item())

        if compiled is not None and model.token.weight.grad is not None:
            token_ids = title_data.active_tokens.to(device)
            model.token.weight.grad[token_ids, :CODE_DIMS] = 0.0
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient):
            raise RuntimeError(f"nonfinite gradient in {label} step {step}")
        optimizer.step()
        if compiled is not None:
            enforce_token_codes(model, title_data, compiled)

        combined = natural_loss_value + router_loss_value
        maximum_loss = max(maximum_loss, combined)
        maximum_natural_loss = max(maximum_natural_loss, natural_loss_value)
        maximum_router_loss = max(maximum_router_loss, router_loss_value)
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 500 == 0 or step == total_steps:
            print(
                json.dumps(
                    {
                        "arm": label,
                        "step": step,
                        "natural_loss": natural_loss_value,
                        "router_loss": router_loss_value,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = {
        name: evaluate_router(
            model, examples, title_data.prototypes, device
        )
        for name, examples in evaluations.items()
    }
    terminal_nll = natural_nll(
        model,
        validation,
        device,
        t10.core.EVALUATION_SEED + MODEL_SEED + total_steps,
    )
    cells_exact = (
        compiled_cells_exact(model, title_data, compiled)
        if compiled is not None
        else None
    )
    result = {
        "label": label,
        "steps": total_steps,
        "compiled": compiled_arm,
        "base_state_sha256": base_state_hash,
        "post_compile_state_sha256": post_compile_hash,
        "terminal_state_sha256": t10.core.small.state_sha256(model),
        "state_schema": schema,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "compiler_seconds": compiler_seconds,
        "training_seconds": elapsed,
        "maximum_combined_loss": maximum_loss,
        "maximum_natural_loss": maximum_natural_loss,
        "maximum_router_loss": maximum_router_loss,
        "maximum_gradient_norm": maximum_gradient,
        "failed_or_retried_steps": failed_or_retried,
        "route": metrics,
        "natural_nll": terminal_nll,
        "compiled_cells_exact": cells_exact,
        "peak_hbm_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "optimizer_state_bytes": t7.optimizer_state_bytes(optimizer),
    }
    if compiled is not None:
        result["compiler"] = {
            key: value for key, value in compiled.items() if key != "values"
        }
    del optimizer
    return model, result


@torch.no_grad()
def shuffled_code_ablation(
    model: t10.core.small.SharedInterpreterLM,
    title_data: TitleData,
    evaluations: dict[str, tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, object]:
    token_ids = title_data.active_tokens.to(device)
    original = model.token.weight[token_ids, :CODE_DIMS].detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(19_019)
    permutation = torch.randperm(len(token_ids), generator=generator).to(device)
    model.token.weight[token_ids, :CODE_DIMS] = original[permutation]
    shuffled = {
        name: evaluate_router(model, examples, title_data.prototypes, device)
        for name, examples in evaluations.items()
    }
    model.token.weight[token_ids, :CODE_DIMS] = original
    restored = torch.equal(model.token.weight[token_ids, :CODE_DIMS], original)
    return {"route": shuffled, "state_restored_exact": restored}


def verify_inputs() -> dict[str, bool]:
    checks = {
        "candidate_corpus": sha256_file(CANDIDATE_CORPUS) == CANDIDATE_SHA256,
        "development_evaluator": sha256_file(DEVELOPMENT_EVALUATOR)
        == EVALUATOR_SHA256,
        "natural_train": sha256_file(t10.TRAIN_FILE) == NATURAL_TRAIN_SHA256,
        "natural_validation": sha256_file(t10.VALIDATION_FILE)
        == NATURAL_VALIDATION_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def run(device: torch.device) -> dict[str, object]:
    input_checks = verify_inputs()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER, revision=TOKENIZER_REVISION
    )
    documents = load_raw_documents(CANDIDATE_CORPUS)
    title_data = build_title_data(documents, tokenizer)
    train_examples = build_wrapper_examples(title_data, tokenizer, TRAIN_WRAPPERS)
    evaluations = {
        "canonical": build_wrapper_examples(title_data, tokenizer, ("{title}",)),
        "held_wrappers": build_wrapper_examples(title_data, tokenizer, HELD_WRAPPERS),
        "natural_question_surfaces": build_natural_surface_examples(
            title_data, tokenizer, DEVELOPMENT_EVALUATOR
        ),
    }
    natural = t10.core.small.TokenStream(t10.TRAIN_FILE)
    validation = t10.core.small.TokenStream(t10.VALIDATION_FILE)

    arms: dict[str, object] = {}
    compiled_model, compiled_result = train_arm(
        "compiled_1x",
        STEPS_1X,
        True,
        title_data,
        train_examples,
        evaluations,
        natural,
        validation,
        device,
    )
    arms["compiled_1x"] = compiled_result
    ablation = shuffled_code_ablation(
        compiled_model, title_data, evaluations, device
    )
    del compiled_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for label, steps in (("gradient_1x", STEPS_1X), ("gradient_2x", STEPS_2X)):
        model, result = train_arm(
            label,
            steps,
            False,
            title_data,
            train_examples,
            evaluations,
            natural,
            validation,
            device,
        )
        arms[label] = result
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate = arms["compiled_1x"]
    control_1x = arms["gradient_1x"]
    control_2x = arms["gradient_2x"]
    assert isinstance(candidate, dict)
    assert isinstance(control_1x, dict)
    assert isinstance(control_2x, dict)
    write_entries = len(title_data.active_tokens) * CODE_DIMS
    surface_names = ("canonical", "held_wrappers", "natural_question_surfaces")
    exact_candidate = all(
        candidate["route"][name]["accuracy"] == 1.0
        and candidate["route"][name]["minimum_target_margin"] > 0.0
        for name in surface_names
    )
    separation = all(
        candidate["route"][name]["accuracy"]
        - control_2x["route"][name]["accuracy"]
        >= 0.10
        for name in surface_names
    )
    natural_drop = (
        candidate["route"]["natural_question_surfaces"]["accuracy"]
        - ablation["route"]["natural_question_surfaces"]["accuracy"]
    )
    candidate_nll = float(candidate["natural_nll"]["nll"])
    control_nll = float(control_1x["natural_nll"]["nll"])
    natural_nll_delta = candidate_nll / control_nll - 1.0
    schemas = {tuple(arm["state_schema"]) for arm in arms.values()}
    parameter_counts = {int(arm["parameter_count"]) for arm in arms.values()}
    finite_training = all(
        math.isfinite(float(arm[key]))
        for arm in arms.values()
        for key in (
            "maximum_combined_loss",
            "maximum_natural_loss",
            "maximum_router_loss",
            "maximum_gradient_norm",
        )
    )
    gates = {
        "input_integrity": all(input_checks.values()),
        "raw_only_compiler_interface": tuple(
            inspect.signature(compile_token_codes).parameters
        )
        == ("model", "title_data"),
        "write_budget": write_entries <= WRITE_BUDGET,
        "compiled_cells_exact": candidate["compiled_cells_exact"] is True,
        "identical_served_structure": len(schemas) == 1
        and len(parameter_counts) == 1
        and next(iter(parameter_counts)) == t10.core.small.parameter_count(),
        "candidate_exact_all_surfaces": exact_candidate,
        "candidate_beats_gradient_2x_by_10pp_all_surfaces": separation,
        "shuffle_causal_drop": natural_drop >= 0.30
        and ablation["route"]["natural_question_surfaces"]["accuracy"] < 0.70
        and ablation["state_restored_exact"],
        "natural_nll_within_0p5pct": natural_nll_delta <= 0.005,
        "finite_training": finite_training,
        "no_failed_or_retried_steps": all(
            arm["failed_or_retried_steps"] == 0 for arm in arms.values()
        ),
        "identical_base_initialization": len(
            {str(arm["base_state_sha256"]) for arm in arms.values()}
        )
        == 1,
    }
    return {
        "schema": "raw-title-code-plane-t19a-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "parameter_count": t10.core.small.parameter_count(),
            "documents": len(documents),
            "titles": len(title_data.titles),
            "active_token_rows": len(title_data.active_tokens),
            "code_dimensions": CODE_DIMS,
            "code_amplitude": CODE_AMPLITUDE,
            "written_entries": write_entries,
            "write_budget": WRITE_BUDGET,
            "steps_1x": STEPS_1X,
            "steps_2x": STEPS_2X,
            "natural_batch": NATURAL_BATCH,
            "router_batch": ROUTER_BATCH,
            "router_context": ROUTER_CONTEXT,
            "train_wrappers": TRAIN_WRAPPERS,
            "held_wrappers": HELD_WRAPPERS,
        },
        "arms": arms,
        "shuffle_ablation": ablation,
        "comparisons": {
            "candidate_minus_gradient_2x_accuracy": {
                name: candidate["route"][name]["accuracy"]
                - control_2x["route"][name]["accuracy"]
                for name in surface_names
            },
            "natural_question_shuffle_drop": natural_drop,
            "candidate_vs_gradient_1x_natural_nll_relative": natural_nll_delta,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "candidate_corpus_sha256": sha256_file(CANDIDATE_CORPUS),
            "development_evaluator_sha256": sha256_file(DEVELOPMENT_EVALUATOR),
            "natural_train_sha256": sha256_file(t10.TRAIN_FILE),
            "natural_validation_sha256": sha256_file(t10.VALIDATION_FILE),
        },
        "claim_boundary": (
            "This tests co-trained real-title addressing only. Passing admits a "
            "raw-prose payload writer; it does not establish knowledge, reasoning, "
            "or a smarter production model."
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
                "failed_gates": sorted(
                    name for name, passed in result["gates"].items() if not passed
                ),
                "route": {
                    arm: values["route"] for arm, values in result["arms"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
