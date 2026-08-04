#!/usr/bin/env python3
"""Fresh-seed weight-RMS successor screen for address-factored transport."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "experiments" / "address_factored_transport_learning.py"
BASE_SOURCE_SHA = "3c969819e558007090bc5dfb031bf59d9c268ac75ee57425215bb3fae12cb8a0"
PRIOR_RESULT = ROOT / "results" / "address-factored-transport-learning.json"
PRIOR_RESULT_SHA = "77a3db085a0187133204543a7682807cb86e04322f2badd3c096f30a75874885"
PREREGISTRATION = (
    ROOT / "results" / "address-factored-transport-weight-rms-preregistration.md"
)
PREREGISTRATION_SHA = "1b8f578ca493bf0f980c0e285fb0169b0e5c8d7357470e4e188e30d20803a5d9"
DEFAULT_OUTPUT = ROOT / "results" / "address-factored-transport-weight-rms.json"
SEEDS = (4144, 2555, 1506, 1060, 2778)
RMS_EPSILON = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(BASE_SOURCE) != BASE_SOURCE_SHA:
    raise RuntimeError("frozen learned-screen dependency mismatch")
if sha256(PRIOR_RESULT) != PRIOR_RESULT_SHA:
    raise RuntimeError("frozen failed-result prerequisite mismatch")
if json.loads(PRIOR_RESULT.read_text())["pass"]:
    raise RuntimeError("successor requires the frozen failed result")
if sha256(PREREGISTRATION) != PREREGISTRATION_SHA:
    raise RuntimeError("weight-RMS preregistration mismatch")

spec = importlib.util.spec_from_file_location("afta_learning_base", BASE_SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen learned-screen source")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

OrdinaryAttentionClassifier = base.AttentionClassifier
EXPORT_AUDITS: dict[str, dict[str, Any]] = {}


class WeightRMSAttentionClassifier(OrdinaryAttentionClassifier):
    """Training reparameterization whose exported state is an ordinary model."""

    def __init__(self, config: Any, arm: str):
        super().__init__(config, arm)
        self.register_buffer(
            "q_weight_rms_reference", torch.zeros(12), persistent=False
        )
        self.register_buffer(
            "k_weight_rms_reference", torch.zeros(12), persistent=False
        )
        self._refresh_weight_rms_references()

    def map_slices(self) -> tuple[tuple[int, int], ...]:
        head = self.config.head_dim
        if self.arm.startswith("split12"):
            return tuple((index * head, (index + 1) * head) for index in range(12))
        wide = 4 * head
        return ((0, wide),) + tuple(
            (wide + index * head, wide + (index + 1) * head)
            for index in range(8)
        )

    @torch.no_grad()
    def _refresh_weight_rms_references(self) -> None:
        self.q_weight_rms_reference.zero_()
        self.k_weight_rms_reference.zero_()
        for index, (start, end) in enumerate(self.map_slices()):
            q_block = self.q_proj.weight[start:end]
            k_block = self.k_proj.weight[start:end]
            self.q_weight_rms_reference[index] = torch.sqrt(
                q_block.square().mean() + RMS_EPSILON
            )
            self.k_weight_rms_reference[index] = torch.sqrt(
                k_block.square().mean() + RMS_EPSILON
            )

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._refresh_weight_rms_references()
        return result

    def _effective_weight(
        self, weight: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        blocks = []
        for index, (start, end) in enumerate(self.map_slices()):
            block = weight[start:end]
            denominator = torch.sqrt(block.square().mean() + RMS_EPSILON)
            blocks.append(block * (references[index] / denominator))
        return torch.cat(blocks, dim=0)

    def effective_qk_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._effective_weight(self.q_proj.weight, self.q_weight_rms_reference),
            self._effective_weight(self.k_proj.weight, self.k_weight_rms_reference),
        )

    def forward(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        *,
        ablate_wide: bool = False,
        return_scores: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        q_weight, k_weight = self.effective_qk_weights()
        q = F.linear(query, q_weight)
        k = F.linear(context, k_weight)
        v = self.v_proj(context)
        outputs: list[torch.Tensor] = []
        raw_scores: list[torch.Tensor] = []
        width = self.config.total_width
        head = self.config.head_dim

        if self.arm.startswith("split12"):
            temperature = 2.0 if self.arm == "split12_temp2" else 1.0
            for index in range(width // head):
                start = index * head
                score = torch.einsum(
                    "bg,blg->bl",
                    q[:, start : start + head],
                    k[:, :, start : start + head],
                ) / math.sqrt(head)
                raw_scores.append(score)
                attention = torch.softmax(score * temperature, dim=-1)
                outputs.append(
                    base.transported_read(
                        attention, v[:, :, start : start + head], index % 4
                    )
                )
        else:
            wide = 4 * head
            score = torch.einsum(
                "bg,blg->bl", q[:, :wide], k[:, :, :wide]
            ) / math.sqrt(wide)
            raw_scores.append(score)
            attention = torch.softmax(score, dim=-1)
            if self.arm == "hybrid_afta":
                offsets = (0, 1, 2, 3)
            elif self.arm == "hybrid_common1":
                offsets = (1, 1, 1, 1)
            elif self.arm == "hybrid_ordinary0":
                offsets = (0, 0, 0, 0)
            else:
                raise AssertionError(self.arm)
            for relation, offset in enumerate(offsets):
                start = relation * head
                value = base.transported_read(
                    attention, v[:, :, start : start + head], offset
                )
                outputs.append(torch.zeros_like(value) if ablate_wide else value)
            for index in range(8):
                start = wide + index * head
                score = torch.einsum(
                    "bg,blg->bl",
                    q[:, start : start + head],
                    k[:, :, start : start + head],
                ) / math.sqrt(head)
                raw_scores.append(score)
                attention = torch.softmax(score, dim=-1)
                outputs.append(
                    base.transported_read(
                        attention, v[:, :, start : start + head], index % 4
                    )
                )
        joined = torch.cat(outputs, dim=-1)
        logits = self.readout(self.norm(self.o_proj(joined)))
        return logits, raw_scores if return_scores else []

    @torch.no_grad()
    def materialize_qk_weights(self) -> None:
        q_weight, k_weight = self.effective_qk_weights()
        self.q_proj.weight.copy_(q_weight)
        self.k_proj.weight.copy_(k_weight)


def train_arm(
    arm: str,
    config: Any,
    seed: int,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[OrdinaryAttentionClassifier, str]:
    model = WeightRMSAttentionClassifier(config, arm).to(device)
    model.load_state_dict(base_state)
    qk_parameters = [model.q_proj.weight, model.k_proj.weight]
    qk_ids = {id(parameter) for parameter in qk_parameters}
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in qk_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": qk_parameters, "weight_decay": 0.0},
            {"params": other_parameters, "weight_decay": config.weight_decay},
        ],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    model.train()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 100_000 + 17)
    data_trace = hashlib.sha256()
    for step in range(config.train_steps):
        batch = base.make_batch(
            config,
            config.batch_size,
            generator,
            device,
            task_id=None,
            noise=config.train_noise,
        )
        data_trace.update(step.to_bytes(8, byteorder="little"))
        data_trace.update(generator.get_state().cpu().contiguous().numpy().tobytes())
        if step in (0, config.train_steps - 1):
            data_trace.update(base.batch_hash(batch).encode())
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.context, batch.query)
        loss = F.cross_entropy(logits, batch.labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"nonfinite loss for {arm} seed {seed} step {step}"
            )
        loss.backward()
        optimizer.step()

    model.eval()
    diagnostic_generator = torch.Generator(device=device)
    diagnostic_generator.manual_seed(seed * 1_000_000 + 911)
    diagnostic = base.make_batch(
        config,
        64,
        diagnostic_generator,
        device,
        task_id=None,
        noise=config.train_noise,
    )
    with torch.inference_mode():
        before, _ = model(diagnostic.context, diagnostic.query)
        training_parameter_count = sum(p.numel() for p in model.parameters())
        model.materialize_qk_weights()
        exported = OrdinaryAttentionClassifier(config, arm).to(device)
        exported.load_state_dict(model.state_dict(), strict=True)
        exported.eval()
        after, _ = exported(diagnostic.context, diagnostic.query)
        export_error = float((before - after).abs().max().item())
        exported_parameter_count = sum(p.numel() for p in exported.parameters())
    EXPORT_AUDITS[f"{seed}:{arm}"] = {
        "seed": seed,
        "arm": arm,
        "max_logit_error": export_error,
        "strict_ordinary_load": True,
        "training_parameter_count": training_parameter_count,
        "exported_parameter_count": exported_parameter_count,
        "qk_weight_decay": 0.0,
        "other_weight_decay": config.weight_decay,
        "temporary_reference_scalars": 24,
    }
    return exported, data_trace.hexdigest()


def selected_output() -> Path:
    for index, argument in enumerate(sys.argv[1:]):
        if argument == "--output":
            return Path(sys.argv[index + 2])
        if argument.startswith("--output="):
            return Path(argument.split("=", 1)[1])
    return DEFAULT_OUTPUT


def training_only_ledger(arm: str) -> dict[str, int]:
    maps = 12 if arm.startswith("split12") else 9
    processed_scalars = 2 * 192 * 264
    return {
        "processed_qk_weight_scalars_per_forward": processed_scalars,
        "scaled_qk_weight_scalars_per_forward": processed_scalars,
        "block_rms_reductions_per_forward": 2 * maps,
        "reference_slots": 24,
        "used_reference_slots": 2 * maps,
    }


def main() -> None:
    output = selected_output()
    base.AttentionClassifier = WeightRMSAttentionClassifier
    base.train_arm = train_arm
    base.SEEDS = SEEDS
    base.PREREGISTRATION = PREREGISTRATION
    base.PREREGISTRATION_SHA = PREREGISTRATION_SHA
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.__file__ = str(Path(__file__).resolve())
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        base.main()

    result = json.loads(output.read_text())
    result["schema"] = "address-factored-transport-weight-rms-v1"
    result["prior_failed_result_sha256"] = PRIOR_RESULT_SHA
    result["base_source_sha256"] = BASE_SOURCE_SHA
    result["weight_rms_reparameterization"] = {
        "epsilon": RMS_EPSILON,
        "fresh_seed_derivation": "1000 + first_five_result_sha_u32_chunks_mod_8000",
        "qk_weight_decay": 0.0,
        "other_weight_decay": result["config"]["weight_decay"],
        "temporary_reference_scalars_per_arm": 24,
        "evaluated_model": "strictly_loaded_ordinary_export",
        "served_extra_parameters": 0,
        "served_extra_operations": 0,
        "claim": "per-map effective QK Frobenius-RMS radial constraint",
        "not_claimed": "raw-parameter gauge quotient or sole causal diagnosis",
    }
    result["training_only_ledgers"] = {
        arm: training_only_ledger(arm) for arm in base.ARMS
    }
    result["export_audits"] = EXPORT_AUDITS
    expected_exports = len(SEEDS) * len(base.ARMS)
    prior_parameter_count = 192_208
    export_integrity = (
        len(EXPORT_AUDITS) == expected_exports
        and all(
            audit["max_logit_error"] <= 2e-6
            and audit["strict_ordinary_load"]
            and audit["training_parameter_count"] == prior_parameter_count
            and audit["exported_parameter_count"] == prior_parameter_count
            and audit["qk_weight_decay"] == 0.0
            and audit["other_weight_decay"] == result["config"]["weight_decay"]
            for audit in EXPORT_AUDITS.values()
        )
        and all(
            ledger
            == {
                "processed_qk_weight_scalars_per_forward": 101_376,
                "scaled_qk_weight_scalars_per_forward": 101_376,
                "block_rms_reductions_per_forward": (
                    24 if arm.startswith("split12") else 18
                ),
                "reference_slots": 24,
                "used_reference_slots": (
                    24 if arm.startswith("split12") else 18
                ),
            }
            for arm, ledger in result["training_only_ledgers"].items()
        )
    )
    result["gates"]["export_integrity"] = export_integrity
    result["pass"] = all(result["gates"].values())
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "gates": result["gates"],
                "pass": result["pass"],
                "summary": result["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
