#!/usr/bin/env python3
"""T18 Stage 0: exact-budget soft record operator and raw-only compiler boundary."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import hotpot_semantic_address_t12 as t12
from experiments import raw_prose_equality_plane_t10 as t10


PREREGISTRATION = ROOT / "results/budget-neutral-soft-record-t18-stage0-preregistration.md"
OUTPUT = ROOT / "results/budget-neutral-soft-record-t18-stage0.json"

HIDDEN = t10.core.small.HIDDEN
BASELINE_FFN = t10.core.small.FFN_WIDTH
ENTITIES = 2_405
RECORD_SLOTS = ENTITIES + 1
SMALL_FFN = 444
SEED = 10_103
REFERENCE_BATCH = 2
REFERENCE_CONTEXT = 128
VALUE_TOKENS = 512
BASELINE_REPLACED_SCALARS = 6 * HIDDEN * BASELINE_FFN
CANDIDATE_REPLACED_SCALARS = 2 * RECORD_SLOTS * HIDDEN + 3 * HIDDEN * SMALL_FFN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DenseRecordMemory(nn.Module):
    def __init__(self, slots: int = RECORD_SLOTS, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.slots = slots
        self.hidden = hidden
        self.keys = nn.Parameter(torch.empty(slots, hidden))
        self.values = nn.Parameter(torch.empty(slots, hidden))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.keys, mean=0.0, std=0.02)
        nn.init.normal_(self.values, mean=0.0, std=0.02 / math.sqrt(2))
        self.values.data[0].zero_()

    def forward(self, hidden: Tensor) -> Tensor:
        scores = hidden @ self.keys.T / math.sqrt(self.hidden)
        return scores.softmax(dim=-1) @ self.values


class RecordBlock(nn.Module):
    def __init__(self, depth_scale: float) -> None:
        super().__init__()
        small = t10.core.small
        self.attention_norm = small.RMSNorm()
        self.attention = small.CausalAttention(depth_scale)
        self.ffn_norm = small.RMSNorm()
        self.memory = DenseRecordMemory()

    def reset_parameters(self) -> None:
        self.attention_norm.weight.data.fill_(1.0)
        self.ffn_norm.weight.data.fill_(1.0)
        self.attention.reset_parameters()
        self.memory.reset_parameters()

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.memory(self.ffn_norm(hidden))


class SmallSwiGLUBlock(nn.Module):
    def __init__(self, depth_scale: float) -> None:
        super().__init__()
        small = t10.core.small
        self.attention_norm = small.RMSNorm()
        self.attention = small.CausalAttention(depth_scale)
        self.ffn_norm = small.RMSNorm()
        self.gate = nn.Linear(HIDDEN, SMALL_FFN, bias=False)
        self.up = nn.Linear(HIDDEN, SMALL_FFN, bias=False)
        self.down = nn.Linear(SMALL_FFN, HIDDEN, bias=False)
        self.depth_scale = depth_scale

    def reset_parameters(self) -> None:
        self.attention_norm.weight.data.fill_(1.0)
        self.ffn_norm.weight.data.fill_(1.0)
        self.attention.reset_parameters()
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02 * self.depth_scale)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        normalized = self.ffn_norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class SoftRecordInterpreterLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        small = t10.core.small
        self.token = nn.Embedding(small.VOCAB, HIDDEN)
        depth_scale = 1.0 / math.sqrt(2 * small.LAYERS)
        self.blocks = nn.ModuleList(
            [small.Block(depth_scale) for _ in range(small.LAYERS - 2)]
            + [RecordBlock(depth_scale), SmallSwiGLUBlock(depth_scale)]
        )
        self.final_norm = small.RMSNorm()
        self.reset_parameters()

    @property
    def record_block(self) -> RecordBlock:
        block = self.blocks[-2]
        assert isinstance(block, RecordBlock)
        return block

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            block.reset_parameters()
        self.final_norm.weight.data.fill_(1.0)

    def hidden(self, tokens: Tensor) -> Tensor:
        hidden = self.token(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.hidden(tokens) @ self.token.weight.T


def build_candidate(seed: int, device: torch.device) -> SoftRecordInterpreterLM:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return SoftRecordInterpreterLM().to(device)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matrix_ledger() -> dict[str, int]:
    return {
        "baseline_two_swiglu_parameters": BASELINE_REPLACED_SCALARS,
        "baseline_two_swiglu_macs_per_token": BASELINE_REPLACED_SCALARS,
        "candidate_record_parameters": 2 * RECORD_SLOTS * HIDDEN,
        "candidate_record_macs_per_token": 2 * RECORD_SLOTS * HIDDEN,
        "candidate_small_swiglu_parameters": 3 * HIDDEN * SMALL_FFN,
        "candidate_small_swiglu_macs_per_token": 3 * HIDDEN * SMALL_FFN,
        "candidate_total_parameters": CANDIDATE_REPLACED_SCALARS,
        "candidate_total_macs_per_token": CANDIDATE_REPLACED_SCALARS,
    }


def activation_ledger(batch: int, context: int, bytes_per_scalar: int = 2) -> dict[str, int]:
    tokens = batch * context
    return {
        "tokens": tokens,
        "baseline_total_gate_up_product_scalars": tokens * 6 * BASELINE_FFN,
        "baseline_peak_gate_up_product_scalars": tokens * 3 * BASELINE_FFN,
        "candidate_materialized_score_probability_scalars": tokens * 2 * RECORD_SLOTS,
        "candidate_small_gate_up_product_scalars": tokens * 3 * SMALL_FFN,
        "baseline_total_gate_up_product_bytes": tokens
        * 6
        * BASELINE_FFN
        * bytes_per_scalar,
        "baseline_peak_gate_up_product_bytes": tokens
        * 3
        * BASELINE_FFN
        * bytes_per_scalar,
        "candidate_materialized_score_probability_bytes": tokens
        * 2
        * RECORD_SLOTS
        * bytes_per_scalar,
        "candidate_small_gate_up_product_bytes": tokens
        * 3
        * SMALL_FFN
        * bytes_per_scalar,
        "baseline_silu_evaluations_per_token": 2 * BASELINE_FFN,
        "baseline_elementwise_products_per_token": 2 * BASELINE_FFN,
        "candidate_softmax_exponentials_per_token": RECORD_SLOTS,
        "candidate_softmax_reduction_terms_per_token": RECORD_SLOTS,
        "candidate_silu_evaluations_per_token": SMALL_FFN,
        "candidate_elementwise_products_per_token": SMALL_FFN,
    }


@torch.no_grad()
def compile_raw_records(
    model: SoftRecordInterpreterLM,
    tokenizer: object,
    documents: list[dict[str, str]],
) -> dict[str, object]:
    expected = {"document_id", "title", "text"}
    if not documents or any(set(document) != expected for document in documents):
        raise RuntimeError("raw compiler accepts only document_id/title/text rows")
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for document in documents:
        grouped[document["title"]].append(document["text"])
    titles = sorted(grouped)
    if len(titles) != ENTITIES:
        raise RuntimeError(f"expected {ENTITIES} canonical titles, found {len(titles)}")
    device = model.token.weight.device
    keys: list[Tensor] = []
    values: list[Tensor] = []
    for title in titles:
        title_ids = tokenizer.encode(" " + title, add_special_tokens=False)
        text_ids = tokenizer.encode(
            "\n".join(sorted(set(grouped[title]))), add_special_tokens=False
        )[:VALUE_TOKENS]
        if not title_ids or not text_ids:
            raise RuntimeError(f"empty title or raw prose for {title}")
        title_tensor = torch.tensor(title_ids, dtype=torch.long, device=device)
        text_tensor = torch.tensor(text_ids, dtype=torch.long, device=device)
        keys.append(F.normalize(model.token(title_tensor).float().mean(dim=0), dim=0))
        values.append(model.token(text_tensor).float().mean(dim=0))
    memory = model.record_block.memory
    memory.keys[0].zero_()
    memory.values[0].zero_()
    memory.keys[1:].copy_(torch.stack(keys).to(memory.keys.dtype))
    memory.values[1:].copy_(torch.stack(values).to(memory.values.dtype))
    return {
        "documents": len(documents),
        "canonical_titles": len(titles),
        "record_slots_written": len(titles),
        "null_key_zero": bool((memory.keys[0] == 0).all().item()),
        "null_value_zero": bool((memory.values[0] == 0).all().item()),
        "accepted_fields": sorted(expected),
    }


def bounded_reference(device: torch.device) -> dict[str, object]:
    torch.manual_seed(SEED + 1)
    hidden = torch.randn(2, 5, 7, device=device, requires_grad=True)
    memory = DenseRecordMemory(slots=11, hidden=7).to(device)
    output = memory(hidden)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    actual = {
        "output": output.detach().clone(),
        "hidden_gradient": hidden.grad.detach().clone(),
        "key_gradient": memory.keys.grad.detach().clone(),
        "value_gradient": memory.values.grad.detach().clone(),
    }

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    keys_ref = memory.keys.detach().clone().requires_grad_(True)
    values_ref = memory.values.detach().clone().requires_grad_(True)
    scores = torch.einsum("bth,sh->bts", hidden_ref, keys_ref) / math.sqrt(7)
    probability = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
    probability = probability / probability.sum(dim=-1, keepdim=True)
    output_ref = torch.einsum("bts,sh->bth", probability, values_ref)
    output_ref.backward(gradient)
    errors = {
        "forward": float((actual["output"] - output_ref.detach()).abs().max().item()),
        "input_gradient": float(
            (actual["hidden_gradient"] - hidden_ref.grad).abs().max().item()
        ),
        "key_gradient": float(
            (actual["key_gradient"] - keys_ref.grad).abs().max().item()
        ),
        "value_gradient": float(
            (actual["value_gradient"] - values_ref.grad).abs().max().item()
        ),
    }

    null_memory = DenseRecordMemory(slots=5, hidden=7).to(device)
    query = F.normalize(torch.randn(1, 1, 7, device=device), dim=-1)
    with torch.no_grad():
        null_memory.keys[0].copy_(query[0, 0] * 100.0)
        null_memory.values[0].zero_()
        null_memory.keys[1:].copy_(-query[0, 0] * 100.0)
        null_memory.values[1:].normal_()
        null_output = null_memory(query)
    return {
        "maximum_errors": errors,
        "null_output_max_abs": float(null_output.abs().max().item()),
    }


def verify_inputs() -> dict[str, bool]:
    checks = {
        "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS)
        == "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a",
        "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST)
        == "90e539279f540a7516e3c90c3ee59539d3f8654cc84e88a82dc950ce6fb5036e",
        "checkpoint_sha256": sha256_file(t12.CHECKPOINT)
        == "a2900585a6e9afe4a9fba4f55afca30df5bd79c335d646cda5b9e940b72d693b",
    }
    if not all(checks.values()):
        raise RuntimeError(f"T18 input integrity failed: {checks}")
    return checks


def run(device: torch.device) -> dict[str, object]:
    checks = verify_inputs()
    tokenizer = t12.AutoTokenizer.from_pretrained(
        t12.TOKENIZER, revision=t12.TOKENIZER_REVISION
    )
    documents = t12.load_documents(t12.CANDIDATE_CORPUS)
    baseline = t10.core.build_model(SEED, device)
    candidate = build_candidate(SEED, device)
    baseline_parameters = parameter_count(baseline)
    candidate_parameters_before = parameter_count(candidate)
    ledger = matrix_ledger()
    activations = activation_ledger(REFERENCE_BATCH, REFERENCE_CONTEXT)
    reference = bounded_reference(device)
    compilation = compile_raw_records(candidate, tokenizer, documents)
    candidate_parameters_after = parameter_count(candidate)

    malformed = dict(documents[0])
    malformed["answer"] = "forbidden"
    rejected_extra_field = False
    try:
        compile_raw_records(candidate, tokenizer, [malformed])
    except RuntimeError:
        rejected_extra_field = True

    generator = torch.Generator(device=device).manual_seed(SEED + 2)
    tokens = torch.randint(
        0,
        t10.core.small.VOCAB,
        (REFERENCE_BATCH, REFERENCE_CONTEXT + 1),
        generator=generator,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    candidate.zero_grad(set_to_none=True)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        logits = candidate(tokens[:, :-1])
    loss = F.cross_entropy(
        logits.float().reshape(-1, t10.core.small.VOCAB),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    gradients = [parameter.grad for parameter in candidate.parameters()]
    gradients_finite = all(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in gradients
    )
    peak_hbm = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    maximum_error = max(reference["maximum_errors"].values())
    gates = {
        "input_integrity": all(checks.values()),
        "total_parameter_count_exact": baseline_parameters
        == candidate_parameters_before
        == candidate_parameters_after,
        "replaced_parameter_ledger_exact": BASELINE_REPLACED_SCALARS
        == CANDIDATE_REPLACED_SCALARS
        == ledger["candidate_total_parameters"],
        "matrix_mac_ledger_exact": ledger["baseline_two_swiglu_macs_per_token"]
        == ledger["candidate_total_macs_per_token"],
        "logit_shape_exact": tuple(logits.shape)
        == (REFERENCE_BATCH, REFERENCE_CONTEXT, t10.core.small.VOCAB),
        "finite_logits_and_loss": torch.isfinite(logits).all().item()
        and torch.isfinite(loss).item(),
        "finite_complete_backward": gradients_finite,
        "dense_reference_within_tolerance": maximum_error <= 2e-6,
        "dominant_null_suppresses_output": reference["null_output_max_abs"]
        < 1e-5,
        "raw_compiler_writes_every_entity": compilation["canonical_titles"]
        == ENTITIES
        and compilation["record_slots_written"] == ENTITIES,
        "raw_compiler_null_exact_zero": compilation["null_key_zero"]
        and compilation["null_value_zero"],
        "raw_compiler_rejects_extra_field": rejected_extra_field,
        "activation_and_nonmatrix_ledger_emitted": bool(activations),
    }
    return {
        "schema": "budget-neutral-soft-record-t18-stage0-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "torch_version": torch.__version__,
        "configuration": {
            "hidden": HIDDEN,
            "baseline_ffn_width": BASELINE_FFN,
            "entities": ENTITIES,
            "record_slots_including_null": RECORD_SLOTS,
            "candidate_small_ffn_width": SMALL_FFN,
            "reference_batch": REFERENCE_BATCH,
            "reference_context": REFERENCE_CONTEXT,
        },
        "parameter_counts": {
            "baseline": baseline_parameters,
            "candidate_before_compile": candidate_parameters_before,
            "candidate_after_compile": candidate_parameters_after,
        },
        "matrix_ledger": ledger,
        "activation_and_nonmatrix_ledger": activations,
        "execution": {
            "logit_shape": list(logits.shape),
            "loss": float(loss.item()),
            "all_gradients_present_and_finite": gradients_finite,
            "peak_hbm_allocated_bytes": peak_hbm,
        },
        "bounded_reference": reference,
        "raw_compilation": compilation,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "candidate_corpus_sha256": sha256_file(t12.CANDIDATE_CORPUS),
            "data_manifest_sha256": sha256_file(t12.DATA_MANIFEST),
            "checkpoint_sha256": sha256_file(t12.CHECKPOINT),
        },
        "claim_boundary": (
            "Stage 0 proves only operator/reference/resource/compiler contracts. "
            "It does not prove language quality, record usefulness, QA gain, "
            "runtime noninferiority, or a strict Pareto result."
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
                "baseline_parameters": result["parameter_counts"]["baseline"],
                "candidate_parameters": result["parameter_counts"]["candidate_after_compile"],
                "matrix_macs_per_token": result["matrix_ledger"]["candidate_total_macs_per_token"],
                "peak_hbm_allocated_bytes": result["execution"]["peak_hbm_allocated_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
