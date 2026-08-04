#!/usr/bin/env python3
"""T21a: learn 16-level per-document codes directly from raw LM loss."""

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
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import functional_residual_operator_t20a as t20a


t19c = t20a.t19c
t19b = t20a.t19b
t19a = t19b.t19a
t12 = t19a.t10
t7 = t19a.t7
core = t12.core

PREREGISTRATION = ROOT / "results/joint-quantized-document-code-t21a-preregistration.md"
OUTPUT = ROOT / "results/joint-quantized-document-code-t21a.json"
SCHEMA = "joint-quantized-document-code-t21a-v1"
MODEL_SEED = 8_209
CODE_SEED = 21_001
SHUFFLE_SEED = 21_005
CODE_DIMS = 220
CODE_START = 64
CODE_AMPLITUDE = 4.0
CONTEXT = 128
PRETRAIN_BATCH = 16
PRETRAIN_STEPS_1X = 20_000
PRETRAIN_STEPS_2X = 21_000
PRETRAIN_WARMUP = 500
MUON_PRETRAIN_LR = 0.005
ADAMW_PRETRAIN_LR = 0.0003
CODE_LR = 0.03
QA_STEPS = 3_000
QA_BATCH = 32
QA_WARMUP = 300
MUON_QA_LR = 0.001
ADAMW_QA_LR = 0.0001
REQUIRED_QA_ACCURACY = 0.75
REQUIRED_GAIN = 0.10
YES_TOKEN = 9_805
NO_TOKEN = 787


def sha256_file(path: Path) -> str:
    return t19b.sha256_file(path)


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_native(value: object, path: str = "$") -> object:
    """Convert scalar library values while rejecting accidental tensor payloads."""
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


def quantize_16_ste(parameters: Tensor) -> tuple[Tensor, Tensor]:
    bounded = parameters.tanh()
    indices = torch.round((bounded + 1.0) * 7.5).clamp(0, 15)
    quantized = -1.0 + 2.0 * indices / 15.0
    straight_through = bounded + (quantized - bounded).detach()
    return CODE_AMPLITUDE * straight_through, indices.to(torch.uint8)


def decoded_level_indices(values: Tensor) -> Tensor:
    normalized = values.float() / CODE_AMPLITUDE
    return torch.round((normalized + 1.0) * 7.5).clamp(0, 15).to(torch.uint8)


@dataclass(frozen=True)
class DocumentCorpus:
    inputs: Tensor
    targets: Tensor
    mask: Tensor

    def batch(
        self, seed: int, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        generator = np.random.default_rng(seed)
        indices = torch.from_numpy(
            generator.integers(0, len(self.inputs), size=batch_size, dtype=np.int64)
        )
        return (
            self.inputs[indices].to(device),
            self.targets[indices].to(device),
            self.mask[indices].to(device),
            indices.to(device),
        )


def build_document_corpus(
    documents: list[dict[str, str]], tokenizer: object
) -> DocumentCorpus:
    eos = int(tokenizer.eos_token_id)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    masks: list[list[bool]] = []
    for document in documents:
        tokens = [
            int(value)
            for value in tokenizer.encode(
                document["title"] + "\n" + document["text"],
                add_special_tokens=False,
            )
        ]
        sequence = (tokens + [eos])[: CONTEXT + 1]
        if len(sequence) < 2:
            sequence = [eos, eos]
        length = len(sequence) - 1
        padded = sequence + [eos] * (CONTEXT + 1 - len(sequence))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
        masks.append([True] * length + [False] * (CONTEXT - length))
    return DocumentCorpus(
        inputs=torch.tensor(inputs, dtype=torch.long),
        targets=torch.tensor(targets, dtype=torch.long),
        mask=torch.tensor(masks, dtype=torch.bool),
    )


def hidden_with_records(
    model: core.small.SharedInterpreterLM,
    tokens: Tensor,
    records: Tensor | None,
    positions: Tensor | None,
) -> Tensor:
    hidden = model.token(tokens)
    if records is not None:
        if positions is None or records.ndim != 3 or positions.ndim != 2:
            raise ValueError("record injection requires [batch,slots,*] tensors")
        if records.shape[:2] != positions.shape or records.shape[2] != CODE_DIMS:
            raise ValueError("record injection shape mismatch")
        hidden = hidden.clone()
        rows = torch.arange(len(tokens), device=tokens.device)
        for slot in range(records.shape[1]):
            selected = positions[:, slot]
            hidden[rows, selected, CODE_START : CODE_START + CODE_DIMS] = (
                hidden[rows, selected, CODE_START : CODE_START + CODE_DIMS]
                + records[:, slot]
            )
    for block in model.blocks:
        hidden = block(hidden)
    return model.final_norm(hidden)


def masked_lm_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    losses = F.cross_entropy(
        logits.float().reshape(-1, core.small.VOCAB),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    return (losses * mask.float()).sum() / mask.sum().clamp_min(1)


def schedule_fraction(step: int, total_steps: int, warmup: int) -> float:
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return 0.1 + 0.9 * cosine


def set_bundle_learning_rate(
    optimizer: t7.OptimizerBundle,
    step: int,
    total_steps: int,
    warmup: int,
    muon_peak: float,
    adamw_peak: float,
) -> None:
    fraction = schedule_fraction(step, total_steps, warmup)
    if optimizer.matrix is not None:
        for group in optimizer.matrix.param_groups:
            group["lr"] = muon_peak * fraction
    for group in optimizer.auxiliary.param_groups:
        group["lr"] = adamw_peak * fraction


def arm_is_document_step(arm: str, step: int) -> bool:
    if arm in {"dense_1x", "joint_code_1x"}:
        return step % 20 == 0
    if arm == "dense_2x_doc":
        return step <= 20_000 and step % 10 == 0
    raise ValueError(f"unknown arm: {arm}")


@dataclass
class PretrainedArm:
    name: str
    model: core.small.SharedInterpreterLM
    code_parameters: nn.Parameter | None
    quantized_codes: Tensor | None
    training: dict[str, object]
    pretrain_natural: dict[str, object]
    pretrain_documents: dict[str, object]


def document_forward(
    model: core.small.SharedInterpreterLM,
    inputs: Tensor,
    records: Tensor | None,
) -> Tensor:
    if records is None:
        return model(inputs)
    positions = torch.zeros(
        len(inputs), 1, dtype=torch.long, device=inputs.device
    )
    hidden = hidden_with_records(model, inputs, records[:, None, :], positions)
    return hidden @ model.token.weight.T


@torch.no_grad()
def evaluate_documents(
    model: core.small.SharedInterpreterLM,
    corpus: DocumentCorpus,
    device: torch.device,
    codes: Tensor | None,
    mode: str = "correct",
) -> dict[str, object]:
    model.eval()
    if codes is not None:
        codes = codes.to(device)
        if mode == "zero":
            codes = torch.zeros_like(codes)
        elif mode == "shuffle":
            generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
            permutation = torch.randperm(len(codes), generator=generator).to(device)
            codes = codes[permutation]
        elif mode != "correct":
            raise ValueError(f"unknown document evaluation mode: {mode}")
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(corpus.inputs), PRETRAIN_BATCH):
        stop = min(start + PRETRAIN_BATCH, len(corpus.inputs))
        inputs = corpus.inputs[start:stop].to(device)
        targets = corpus.targets[start:stop].to(device)
        mask = corpus.mask[start:stop].to(device)
        records = codes[start:stop] if codes is not None else None
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = document_forward(model, inputs, records)
        losses = F.cross_entropy(
            logits.float().reshape(-1, core.small.VOCAB),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        total_loss += float((losses * mask.float()).sum().item())
        total_tokens += int(mask.sum().item())
    return {
        "mode": mode,
        "tokens": total_tokens,
        "nll": total_loss / total_tokens,
        "finite": math.isfinite(total_loss),
    }


def train_pretraining_arm(
    arm: str,
    natural: core.small.TokenStream,
    validation: core.small.TokenStream,
    documents: DocumentCorpus,
    device: torch.device,
) -> PretrainedArm:
    total_steps = (
        PRETRAIN_STEPS_2X if arm == "dense_2x_doc" else PRETRAIN_STEPS_1X
    )
    model = core.build_model(MODEL_SEED, device)
    model_optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=MUON_PRETRAIN_LR
    )
    code_parameters: nn.Parameter | None = None
    code_optimizer: torch.optim.Optimizer | None = None
    if arm == "joint_code_1x":
        generator = torch.Generator(device="cpu").manual_seed(CODE_SEED)
        initial = 0.05 * torch.randn(
            len(documents.inputs), CODE_DIMS, generator=generator
        )
        code_parameters = nn.Parameter(initial.to(device))
        code_optimizer = torch.optim.Adam(
            (code_parameters,), lr=CODE_LR, weight_decay=0.0
        )
    natural_updates = 0
    document_updates = 0
    natural_tokens = 0
    document_tokens = 0
    maximum_loss = 0.0
    maximum_model_gradient = 0.0
    maximum_code_gradient = 0.0
    failed_or_retried = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        model.train()
        model_optimizer.zero_grad()
        if code_optimizer is not None:
            code_optimizer.zero_grad(set_to_none=True)
        set_bundle_learning_rate(
            model_optimizer,
            step,
            total_steps,
            PRETRAIN_WARMUP,
            MUON_PRETRAIN_LR,
            ADAMW_PRETRAIN_LR,
        )
        if arm_is_document_step(arm, step):
            document_updates += 1
            inputs, targets, mask, indices = documents.batch(
                MODEL_SEED * 1_000_003 + document_updates,
                PRETRAIN_BATCH,
                device,
            )
            records = None
            if code_parameters is not None:
                all_codes, _ = quantize_16_ste(code_parameters)
                records = all_codes[indices]
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = document_forward(model, inputs, records)
            loss = masked_lm_loss(logits, targets, mask)
            document_tokens += int(mask.sum().item())
        else:
            natural_updates += 1
            inputs, targets = natural.batch(
                MODEL_SEED * 2_000_003 + natural_updates,
                PRETRAIN_BATCH,
                CONTEXT,
                device,
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(inputs)
            loss = F.cross_entropy(
                logits.float().reshape(-1, core.small.VOCAB), targets.reshape(-1)
            )
            natural_tokens += PRETRAIN_BATCH * CONTEXT
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm} loss at step {step}")
        loss.backward()
        model_gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(model_gradient):
            raise RuntimeError(f"nonfinite {arm} model gradient at step {step}")
        maximum_model_gradient = max(maximum_model_gradient, model_gradient)
        if code_parameters is not None and code_parameters.grad is not None:
            code_gradient = float(code_parameters.grad.norm().item())
            if not math.isfinite(code_gradient):
                raise RuntimeError(f"nonfinite code gradient at step {step}")
            maximum_code_gradient = max(maximum_code_gradient, code_gradient)
        model_optimizer.step()
        if code_optimizer is not None and arm_is_document_step(arm, step):
            code_optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        if step % 1_000 == 0:
            print(
                json.dumps(
                    {
                        "phase": "pretrain",
                        "arm": arm,
                        "step": step,
                        "loss": float(loss.item()),
                        "natural_updates": natural_updates,
                        "document_updates": document_updates,
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
            quantized_codes, _ = quantize_16_ste(code_parameters)
            quantized_codes = quantized_codes.detach().clone()
    natural_result = core.small.evaluate_natural(
        model,
        validation,
        device,
        core.EVALUATION_SEED + MODEL_SEED + total_steps,
        batches=32,
    )
    document_result: dict[str, object]
    if quantized_codes is None:
        document_result = {
            "correct": evaluate_documents(model, documents, device, None)
        }
    else:
        document_result = {
            mode: evaluate_documents(
                model, documents, device, quantized_codes, mode
            )
            for mode in ("correct", "zero", "shuffle")
        }
    del model_optimizer, code_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return PretrainedArm(
        name=arm,
        model=model,
        code_parameters=code_parameters,
        quantized_codes=quantized_codes,
        training={
            "total_updates": total_steps,
            "natural_updates": natural_updates,
            "document_updates": document_updates,
            "natural_token_presentations": natural_tokens,
            "document_token_presentations": document_tokens,
            "maximum_loss": maximum_loss,
            "maximum_model_gradient_norm": maximum_model_gradient,
            "maximum_code_gradient_norm": maximum_code_gradient,
            "failed_or_retried_steps": failed_or_retried,
            "elapsed_seconds": elapsed,
            "peak_hbm_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            )
            if device.type == "cuda"
            else 0,
        },
        pretrain_natural=natural_result,
        pretrain_documents=document_result,
    )


@dataclass(frozen=True)
class QAData:
    inputs: Tensor
    lengths: Tensor
    targets: Tensor
    pairs: Tensor
    positions: Tensor


def build_qa_data(
    rows: list[dict[str, object]],
    titles: tuple[str, ...],
    tokenizer: object,
) -> QAData:
    eos = int(tokenizer.eos_token_id)
    inputs: list[list[int]] = []
    lengths: list[int] = []
    targets: list[int] = []
    pairs: list[tuple[int, int]] = []
    positions: list[tuple[int, int]] = []
    for row in rows:
        question = str(row["question"])
        first_span, second_span, pair = t19b.nonoverlapping_title_pair(
            question, titles
        )
        identifiers = [
            int(value)
            for value in tokenizer.encode(question, add_special_tokens=False)
        ]
        if not identifiers or len(identifiers) > CONTEXT:
            raise RuntimeError(f"invalid QA token length: {len(identifiers)}")
        ends = []
        for span in (first_span, second_span):
            prefix = tokenizer.encode(
                question[: span[1]], add_special_tokens=False
            )
            ends.append(len(prefix) - 1)
        if min(ends) < 0 or max(ends) >= len(identifiers):
            raise RuntimeError("title token position outside question")
        lengths.append(len(identifiers))
        inputs.append(identifiers + [eos] * (CONTEXT - len(identifiers)))
        targets.append(0 if str(row["answer"]).lower() == "yes" else 1)
        pairs.append(pair)
        positions.append((ends[0], ends[1]))
    return QAData(
        inputs=torch.tensor(inputs, dtype=torch.long),
        lengths=torch.tensor(lengths, dtype=torch.long),
        targets=torch.tensor(targets, dtype=torch.long),
        pairs=torch.tensor(pairs, dtype=torch.long),
        positions=torch.tensor(positions, dtype=torch.long),
    )


def qa_batch_indices(step: int, count: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        MODEL_SEED * 3_000_017 + step
    )
    return torch.randint(0, count, (QA_BATCH,), generator=generator)


def qa_logits(
    model: core.small.SharedInterpreterLM,
    data: QAData,
    indices: Tensor,
    device: torch.device,
    codes: Tensor | None,
    pair_override: Tensor | None = None,
    zero_codes: bool = False,
) -> Tensor:
    inputs = data.inputs[indices].to(device)
    lengths = data.lengths[indices].to(device)
    positions = data.positions[indices].to(device)
    records = None
    if codes is not None:
        pairs = (
            pair_override[indices] if pair_override is not None else data.pairs[indices]
        ).to(device)
        records = codes.to(device)[pairs]
        if zero_codes:
            records = torch.zeros_like(records)
    hidden = hidden_with_records(model, inputs, records, positions if records is not None else None)
    final = hidden[torch.arange(len(inputs), device=device), lengths - 1]
    rows = model.token.weight[
        torch.tensor([YES_TOKEN, NO_TOKEN], dtype=torch.long, device=device)
    ]
    return final @ rows.T


def train_qa_arm(
    arm: PretrainedArm,
    train: QAData,
    device: torch.device,
) -> dict[str, object]:
    model = arm.model
    optimizer = t7.build_optimizer(
        model, "muon", muon_learning_rate=MUON_QA_LR
    )
    maximum_loss = 0.0
    maximum_gradient = 0.0
    failed_or_retried = 0
    before_code_hash = (
        tensor_sha256(arm.quantized_codes)
        if arm.quantized_codes is not None
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, QA_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        set_bundle_learning_rate(
            optimizer,
            step,
            QA_STEPS,
            QA_WARMUP,
            MUON_QA_LR,
            ADAMW_QA_LR,
        )
        indices = qa_batch_indices(step, len(train.inputs))
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = qa_logits(
                model,
                train,
                indices,
                device,
                arm.quantized_codes,
            )
        targets = train.targets[indices].to(device)
        loss = F.cross_entropy(logits.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm.name} QA loss at step {step}")
        loss.backward()
        gradient = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        if not math.isfinite(gradient):
            raise RuntimeError(f"nonfinite {arm.name} QA gradient at step {step}")
        optimizer.step()
        maximum_loss = max(maximum_loss, float(loss.item()))
        maximum_gradient = max(maximum_gradient, gradient)
        if step % 500 == 0:
            print(
                json.dumps(
                    {
                        "phase": "qa",
                        "arm": arm.name,
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
    after_code_hash = (
        tensor_sha256(arm.quantized_codes)
        if arm.quantized_codes is not None
        else None
    )
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "updates": QA_STEPS,
        "maximum_loss": maximum_loss,
        "maximum_gradient_norm": maximum_gradient,
        "failed_or_retried_steps": failed_or_retried,
        "elapsed_seconds": elapsed,
        "peak_hbm_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "code_sha256_before": before_code_hash,
        "code_sha256_after": after_code_hash,
        "code_unchanged": before_code_hash == after_code_hash,
    }


@torch.no_grad()
def evaluate_qa(
    arm: PretrainedArm,
    data: QAData,
    device: torch.device,
    mode: str = "correct",
) -> dict[str, object]:
    arm.model.eval()
    pair_override = None
    zero_codes = False
    if arm.quantized_codes is not None:
        if mode == "zero":
            zero_codes = True
        elif mode == "shuffle":
            generator = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
            flat = data.pairs.flatten()
            shuffled = flat[torch.randperm(len(flat), generator=generator)]
            pair_override = shuffled.view_as(data.pairs)
        elif mode != "correct":
            raise ValueError(f"unknown QA evaluation mode: {mode}")
    elif mode != "correct":
        raise ValueError("dense arm has no record ablation")
    predictions: list[Tensor] = []
    for start in range(0, len(data.inputs), QA_BATCH):
        indices = torch.arange(start, min(start + QA_BATCH, len(data.inputs)))
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = qa_logits(
                arm.model,
                data,
                indices,
                device,
                arm.quantized_codes,
                pair_override=pair_override,
                zero_codes=zero_codes,
            )
        predictions.append(logits.float().argmax(dim=-1).cpu())
    values = torch.cat(predictions)
    return {
        "mode": mode,
        "examples": len(values),
        "accuracy": float((values == data.targets).float().mean().item()),
        "errors": int((values != data.targets).sum().item()),
        "positive_rate": float((data.targets == 0).float().mean().item()),
        "predicted_positive_rate": float((values == 0).float().mean().item()),
    }


def state_schema(model: nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((name, tuple(value.shape)) for name, value in model.state_dict().items())


def verify_inputs() -> dict[str, bool]:
    checks = {
        "candidate_corpus": sha256_file(t19a.CANDIDATE_CORPUS)
        == t19a.CANDIDATE_SHA256,
        "train_qa": sha256_file(t19b.TRAIN_QA) == t19b.TRAIN_QA_SHA256,
        "evaluation_qa": sha256_file(t19b.EVALUATION_QA)
        == t19a.EVALUATOR_SHA256,
        "natural_train": sha256_file(t12.TRAIN_FILE) == t19a.NATURAL_TRAIN_SHA256,
        "natural_validation": sha256_file(t12.VALIDATION_FILE)
        == t19a.NATURAL_VALIDATION_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


def run(device: torch.device) -> dict[str, object]:
    torch.set_float32_matmul_precision("high")
    input_checks = verify_inputs()
    documents = t19a.load_raw_documents(t19a.CANDIDATE_CORPUS)
    compiler_fields = set().union(*(document.keys() for document in documents))
    tokenizer = AutoTokenizer.from_pretrained(
        t19a.TOKENIZER, revision=t19a.TOKENIZER_REVISION
    )
    if tokenizer.encode(" yes", add_special_tokens=False) != [YES_TOKEN]:
        raise RuntimeError("yes token drifted")
    if tokenizer.encode(" no", add_special_tokens=False) != [NO_TOKEN]:
        raise RuntimeError("no token drifted")
    title_data = t19a.build_title_data(documents, tokenizer)
    document_corpus = build_document_corpus(documents, tokenizer)
    natural = core.small.TokenStream(t12.TRAIN_FILE)
    validation = core.small.TokenStream(t12.VALIDATION_FILE)

    initial_hashes = []
    for _ in range(3):
        probe = core.build_model(MODEL_SEED, device)
        initial_hashes.append(core.small.state_sha256(probe))
        del probe
    arms: dict[str, PretrainedArm] = {}
    for name in ("dense_1x", "joint_code_1x", "dense_2x_doc"):
        arms[name] = train_pretraining_arm(
            name, natural, validation, document_corpus, device
        )

    train_rows = t19b.load_qa(t19b.TRAIN_QA)
    evaluation_rows = t19b.load_qa(t19b.EVALUATION_QA)
    train_qa = build_qa_data(train_rows, title_data.titles, tokenizer)
    evaluation_qa = build_qa_data(
        evaluation_rows, title_data.titles, tokenizer
    )
    qa_training: dict[str, object] = {}
    for name, arm in arms.items():
        qa_training[name] = train_qa_arm(arm, train_qa, device)

    qa_results: dict[str, dict[str, object]] = {}
    for name, arm in arms.items():
        qa_results[name] = {
            "correct": evaluate_qa(arm, evaluation_qa, device, "correct")
        }
    candidate = arms["joint_code_1x"]
    qa_results["joint_code_1x"]["zero"] = evaluate_qa(
        candidate, evaluation_qa, device, "zero"
    )
    qa_results["joint_code_1x"]["shuffle"] = evaluate_qa(
        candidate, evaluation_qa, device, "shuffle"
    )

    final_natural = {
        name: core.small.evaluate_natural(
            arm.model,
            validation,
            device,
            core.EVALUATION_SEED + MODEL_SEED + 90_000,
            batches=32,
        )
        for name, arm in arms.items()
    }

    assert candidate.quantized_codes is not None
    code_values = candidate.quantized_codes.detach().cpu()
    code_indices = decoded_level_indices(code_values)
    bf16_indices = decoded_level_indices(code_values.to(torch.bfloat16).float())
    allowed_levels = torch.tensor(
        [CODE_AMPLITUDE * (-1.0 + 2.0 * index / 15.0) for index in range(16)]
    )
    distances = (code_values[..., None] - allowed_levels).abs()
    all_on_levels = bool((distances.min(dim=-1).values < 1e-6).all())

    schemas = {state_schema(arm.model) for arm in arms.values()}
    parameter_counts = {
        sum(parameter.numel() for parameter in arm.model.parameters())
        for arm in arms.values()
    }
    prospective = core.build_model(MODEL_SEED, device)
    physical = t19b.compile_physical_payload(
        prospective, title_data, code_values
    )
    candidate_document = candidate.pretrain_documents
    correct_document_nll = float(candidate_document["correct"]["nll"])
    zero_document_nll = float(candidate_document["zero"]["nll"])
    shuffled_document_nll = float(candidate_document["shuffle"]["nll"])
    candidate_accuracy = float(
        qa_results["joint_code_1x"]["correct"]["accuracy"]
    )
    dense_accuracy = max(
        float(qa_results["dense_1x"]["correct"]["accuracy"]),
        float(qa_results["dense_2x_doc"]["correct"]["accuracy"]),
    )
    zero_accuracy = float(qa_results["joint_code_1x"]["zero"]["accuracy"])
    shuffle_accuracy = float(
        qa_results["joint_code_1x"]["shuffle"]["accuracy"]
    )
    candidate_natural = float(final_natural["joint_code_1x"]["nll"])
    dense1_natural = float(final_natural["dense_1x"]["nll"])
    training_finite = all(
        arm.training["failed_or_retried_steps"] == 0
        and math.isfinite(float(arm.training["maximum_loss"]))
        and math.isfinite(float(arm.training["maximum_model_gradient_norm"]))
        and qa_training[name]["failed_or_retried_steps"] == 0
        and math.isfinite(float(qa_training[name]["maximum_loss"]))
        and math.isfinite(float(qa_training[name]["maximum_gradient_norm"]))
        for name, arm in arms.items()
    )
    gates = {
        "input_integrity_and_raw_only_code": all(input_checks.values())
        and compiler_fields == {"document_id", "title", "text"}
        and tuple(inspect.signature(build_document_corpus).parameters)
        == ("documents", "tokenizer"),
        "identical_initial_models_and_finite_training": len(set(initial_hashes)) == 1
        and training_finite,
        "frozen_16_levels_and_bf16_indices": all_on_levels
        and torch.equal(code_indices, bf16_indices)
        and bool(torch.isfinite(code_values).all()),
        "code_frozen_during_qa": bool(
            qa_training["joint_code_1x"]["code_unchanged"]
        ),
        "two_titles_and_both_classes": len(train_qa.inputs) == len(train_rows)
        and len(evaluation_qa.inputs) == len(evaluation_rows)
        and set(train_qa.targets.tolist()) == {0, 1}
        and set(evaluation_qa.targets.tolist()) == {0, 1},
        "identical_served_structure_and_physical_ledger": len(schemas) == 1
        and len(parameter_counts) == 1
        and physical["total_writes"] == t19b.EXPECTED_WRITES
        and physical["total_writes"] <= t19b.WRITE_BUDGET,
        "document_nll_gain_over_zero_at_least_10pct": (
            zero_document_nll - correct_document_nll
        )
        / zero_document_nll
        >= 0.10,
        "document_nll_gain_over_shuffle_at_least_10pct": (
            shuffled_document_nll - correct_document_nll
        )
        / shuffled_document_nll
        >= 0.10,
        "qa_accuracy_at_least_75pct": candidate_accuracy
        >= REQUIRED_QA_ACCURACY,
        "qa_gain_over_best_dense_at_least_10pp": candidate_accuracy
        - dense_accuracy
        >= REQUIRED_GAIN,
        "zero_code_qa_drop_at_least_10pp": candidate_accuracy - zero_accuracy
        >= REQUIRED_GAIN,
        "shuffled_code_qa_drop_at_least_10pp": candidate_accuracy
        - shuffle_accuracy
        >= REQUIRED_GAIN,
        "natural_nll_within_0_5pct_of_dense1": (
            candidate_natural - dense1_natural
        )
        / dense1_natural
        <= 0.005,
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
            "document_active_tokens": int(document_corpus.mask.sum().item()),
        },
        "initial_model_sha256": initial_hashes,
        "arms": {
            name: {
                "training": arm.training,
                "pretrain_natural": arm.pretrain_natural,
                "pretrain_documents": arm.pretrain_documents,
                "qa_training": qa_training[name],
                "qa_evaluation": qa_results[name],
                "final_natural": final_natural[name],
                "state_sha256": core.small.state_sha256(arm.model),
            }
            for name, arm in arms.items()
        },
        "digital_code": {
            "documents": len(code_values),
            "dimensions": CODE_DIMS,
            "logical_cells": code_values.numel(),
            "logical_bits": 4 * code_values.numel(),
            "amplitude": CODE_AMPLITUDE,
            "unique_level_indices": sorted(set(code_indices.flatten().tolist())),
            "all_on_frozen_levels": all_on_levels,
            "bf16_level_indices_exact": torch.equal(code_indices, bf16_indices),
            "sha256": tensor_sha256(code_values),
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
        "effects": {
            "document_nll_gain_over_zero_pct": 100.0
            * (zero_document_nll - correct_document_nll)
            / zero_document_nll,
            "document_nll_gain_over_shuffle_pct": 100.0
            * (shuffled_document_nll - correct_document_nll)
            / shuffled_document_nll,
            "qa_gain_over_best_dense_pp": 100.0
            * (candidate_accuracy - dense_accuracy),
            "qa_zero_code_drop_pp": 100.0
            * (candidate_accuracy - zero_accuracy),
            "qa_shuffle_code_drop_pp": 100.0
            * (candidate_accuracy - shuffle_accuracy),
            "natural_nll_relative_to_dense1_pct": 100.0
            * (candidate_natural - dense1_natural)
            / dense1_natural,
        },
        "gates": gates,
        "admitted": all(gates.values()),
        "integrity": {
            "candidate_corpus_sha256": sha256_file(t19a.CANDIDATE_CORPUS),
            "train_qa_sha256": sha256_file(t19b.TRAIN_QA),
            "evaluation_qa_sha256": sha256_file(t19b.EVALUATION_QA),
            "natural_train_sha256": sha256_file(t12.TRAIN_FILE),
            "natural_validation_sha256": sha256_file(t12.VALIDATION_FILE),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
        },
        "limits": [
            "Exact title lookup and hidden-state addition are privileged in T21a.",
            "The quantized table fits the prospective write ledger but has not been exported into the ordinary graph.",
            "Passing requires a physical export, final untouched evaluation, and replication before any production claim.",
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
