#!/usr/bin/env python3
"""Frozen T84 effective-v4 implementation.

IMPORTANT: source exists for audit only.  The accepted protocol does not permit
executing this module until an independent source receipt authorizes a named
stage.  The CLI enforces append-only receipts and never has a generic "run all"
mode.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN_ROOT = RESULTS / "t84-effective-v4-run"
V2 = RESULTS / "shared-score-heterogeneous-attention-t84-cpu-preregistration-v2.md"
V3 = RESULTS / "shared-score-heterogeneous-attention-t84-cpu-preregistration-v3.md"
V4 = RESULTS / "shared-score-heterogeneous-attention-t84-cpu-preregistration-v4.md"
ACCEPTANCE = RESULTS / "shared-score-heterogeneous-attention-t84-effective-v4-acceptance.md"
MIXER_FILE = RESULTS / "shared-score-heterogeneous-attention-t84-mixer-v2.txt"
SOURCE_MANIFEST = RESULTS / "shared-score-heterogeneous-attention-t84-source-manifest-v2.json"
ENVIRONMENT_EXPECTATION = RESULTS / "shared-score-heterogeneous-attention-t84-environment.json"

AUDITED_SOURCE_PATHS = (
    Path("experiments/shared_score_heterogeneous_attention_t84.py"),
    Path("experiments/shared_score_heterogeneous_attention_t84_controller.py"),
    Path("experiments/shared_score_heterogeneous_attention_t84_entrypoint.py"),
    Path("experiments/shared_score_heterogeneous_attention_t84_decoder.py"),
    Path("experiments/shared_score_heterogeneous_attention_t84_power.py"),
    Path("tests/test_shared_score_heterogeneous_attention_t84_static.py"),
    Path("results/shared-score-heterogeneous-attention-t84-power-canonical.json"),
    Path("results/shared-score-heterogeneous-attention-t84-environment.json"),
    Path("results/shared-score-heterogeneous-attention-t84-static-architecture-manifest.json"),
)

EXPECTED_HASHES = {
    V2: "a40b7928078d3aa9bf8514b7a6349ab09a76477ebdf071426e5d4e92f4b049d5",
    V3: "b10c376e386c916bfb25bc11a0af2ef3f2433439b99d98c5e5515af2047b6bbf",
    V4: "085abc5e703ee4a064a837f66cf3ffb82e282c1fe78fec12cc53870a0db774b8",
    MIXER_FILE: "f84aa219820c84314da2eede2e0e8b45bc83c881a1b01d2b1d6973d1cd8537c8",
}
MIXER_BINARY_HASH = "2c1ac147b8f896e889695097fa732e2811ec8aeabe45fddc211482f08a45c973"
TCRIT = 4.1014945453569363208
FAMILY_SIZE = 106
EVAL_SEEDS = tuple(range(842100, 842116))
OOD_NAMES = ("l16", "l32", "v2", "l32v2")

ARCHITECTURES = (
    "candidate",
    "soft_standard",
    "soft_grouped",
    "soft_narrow",
    "soft_narrow_grouped",
    "hybrid_independent",
    "hybrid_whole_head",
    "hard_shared",
    "finite_beta_8",
    "all_max_shared",
    "tropical_official",
    "deepsets_conditioned",
    "soft_time_matched",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen hash mismatch {path}: {actual} != {expected}")


def assert_audited_sources() -> Mapping[str, str]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    bound_inputs = manifest.get("bound_inputs")
    if not isinstance(bound_inputs, list) or not bound_inputs:
        raise RuntimeError("audited source manifest has no frozen input bindings")
    for binding in bound_inputs:
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or not isinstance(binding["sha256"], str)
        ):
            raise RuntimeError("invalid frozen input binding in source manifest")
        relative = Path(binding["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("source manifest input path escapes project")
        if sha256_file(ROOT / relative) != binding["sha256"]:
            raise RuntimeError(f"source manifest frozen input changed: {relative}")
    declared = manifest.get("source_hashes")
    expected_paths = {str(path) for path in AUDITED_SOURCE_PATHS}
    if not isinstance(declared, dict) or set(declared) != expected_paths:
        raise RuntimeError("audited source manifest path set mismatch")
    for relative in AUDITED_SOURCE_PATHS:
        expected = declared[str(relative)]
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(
                f"audited source changed {relative}: {actual} != {expected}"
            )
    return declared


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one new artifact; overwriting is forbidden by v4."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    if temporary.exists():
        raise FileExistsError(temporary)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temporary, path)


def load_mixer() -> np.ndarray:
    rows: list[list[float]] = []
    for line in MIXER_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(("0x", "-0x")):
            rows.append([float.fromhex(item) for item in line.split()])
    mixer = np.asarray(rows, dtype="<f8")
    if mixer.shape != (14, 14):
        raise RuntimeError(f"bad mixer shape {mixer.shape}")
    actual = hashlib.sha256(mixer.tobytes(order="C")).hexdigest()
    if actual != MIXER_BINARY_HASH:
        raise RuntimeError(f"bad mixer payload {actual}")
    return mixer


MIXER = load_mixer()


def tensor_seed(training_seed: int, canonical_path: str) -> int:
    payload = f"T84v3|{training_seed}|{canonical_path}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def generator_for(training_seed: int, canonical_path: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(tensor_seed(training_seed, canonical_path))
    return generator


@dataclass(frozen=True)
class Example:
    tokens: np.ndarray
    ys: int
    ym: int
    y: int
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    u: np.ndarray
    scores: np.ndarray


def draw_example(rng: np.random.Generator, n: int, a: float) -> Example:
    q = rng.uniform(-a, a, size=(4,)).astype(np.float64)
    k = rng.uniform(-a, a, size=(n, 4)).astype(np.float64)
    v = rng.uniform(-a, a, size=(n, 2)).astype(np.float64)
    u = rng.uniform(-a, a, size=(n, 2)).astype(np.float64)
    scores = (k @ q) / 2.0
    shifted = scores - scores.max()
    probability = np.exp(shifted) / np.exp(shifted).sum()
    soft = probability @ v
    maxplus = np.max(scores[:, None] + u, axis=0)
    ys = int(soft[0] > soft[1])
    ym = int(maxplus[0] > maxplus[1])
    order = rng.permutation(n)
    raw = np.zeros((n + 1, 14), dtype=np.float64)
    raw[:n, 0] = 1.0
    raw[:n, 2:6] = k[order]
    raw[:n, 6:8] = v[order]
    raw[:n, 8:10] = u[order]
    raw[n, 1] = 1.0
    raw[n, 10:14] = q
    mixed = np.asarray(raw @ MIXER, dtype="<f4", order="C")
    return Example(mixed, ys, ym, ys ^ ym, q, k[order], v[order], u[order], scores[order])


def balanced_examples(
    rng: np.random.Generator, n: int, a: float, per_quadrant: int
) -> list[Example]:
    examples, _ = balanced_examples_with_permutation(rng, n, a, per_quadrant)
    return examples


def balanced_examples_with_permutation(
    rng: np.random.Generator, n: int, a: float, per_quadrant: int
) -> tuple[list[Example], np.ndarray]:
    bins: dict[tuple[int, int], list[Example]] = {
        (0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []
    }
    while any(len(values) < per_quadrant for values in bins.values()):
        example = draw_example(rng, n, a)
        cell = bins[(example.ys, example.ym)]
        if len(cell) < per_quadrant:
            cell.append(example)
    ordered = [example for key in ((0, 0), (0, 1), (1, 0), (1, 1)) for example in bins[key]]
    permutation = rng.permutation(len(ordered))
    return [ordered[int(index)] for index in permutation], permutation.astype("<i8", copy=False)


def training_batch(training_seed: int, step: int) -> tuple[Tensor, dict[str, Tensor], int]:
    rng = np.random.Generator(np.random.PCG64(10_000_000 + 10_000 * training_seed + step))
    n = int(rng.integers(4, 9))
    examples = balanced_examples(rng, n=n, a=1.0, per_quadrant=32)
    tokens = torch.from_numpy(np.stack([example.tokens for example in examples]))
    labels = {
        "ys": torch.tensor([example.ys for example in examples], dtype=torch.long),
        "ym": torch.tensor([example.ym for example in examples], dtype=torch.long),
        "y": torch.tensor([example.y for example in examples], dtype=torch.long),
    }
    return tokens, labels, n


def causal_mask(n: int) -> Tensor:
    length = n + 1
    mask = torch.zeros((length, length), dtype=torch.bool)
    for row in range(n):
        mask[row, : row + 1] = True
    mask[n, :n] = True
    if not bool(mask.any(dim=-1).all()):
        raise RuntimeError("all-masked row")
    return mask


def group_rms(x: Tensor, group: int = 4) -> Tensor:
    if x.shape[-1] % group:
        raise ValueError("group mismatch")
    shaped = x.reshape(*x.shape[:-1], x.shape[-1] // group, group)
    shaped = shaped / torch.sqrt(shaped.square().mean(dim=-1, keepdim=True) + 1e-6)
    return shaped.reshape_as(x)


class FrozenRMSNorm(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class FeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.up = nn.Linear(32, width, bias=False)
        self.down = nn.Linear(width, 32, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.up(x), approximate="none"))


@dataclass
class AttentionTrace:
    scores: Tensor
    raw_max: Tensor | None
    winners: Tensor | None
    pre_o: Tensor
    soft_pre_o: Tensor | None
    max_pre_o: Tensor | None


ScoreOverride = Mapping[Literal["soft", "max"], Tensor]


def scaled_masked_scores(q: Tensor, k: Tensor, mask: Tensor, head_dim: int) -> Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    return scores.masked_fill(~mask, -torch.inf)


def soft_reduce(scores: Tensor, value: Tensor) -> Tensor:
    return torch.matmul(torch.softmax(scores, dim=-1), value)


def maxplus_reduce(scores: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
    return (scores.unsqueeze(-1) + value.unsqueeze(-3)).max(dim=-2)


def finite_beta_reduce(scores: Tensor, value: Tensor, beta: float = 8.0) -> Tensor:
    return torch.logsumexp(
        beta * (scores.unsqueeze(-1) + value.unsqueeze(-3)), dim=-2
    ) / beta


class MixedAttention(nn.Module):
    def __init__(self, architecture: str) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture in {"soft_narrow", "soft_narrow_grouped"}:
            self.heads, self.head_dim = 32, 1
        elif architecture == "hybrid_independent":
            self.heads, self.head_dim = 8, 4
        else:
            self.heads, self.head_dim = 4, 8
        self.q = nn.Linear(32, 32, bias=False)
        self.k = nn.Linear(32, 32, bias=False)
        self.v = nn.Linear(32, 32, bias=False)
        self.o = nn.Linear(32, 32, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _replace_query(scores: Tensor, replacement: Tensor | None) -> Tensor:
        if replacement is None:
            return scores
        result = scores.clone()
        result[:, :, -1, :] = replacement
        return result

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        overrides: ScoreOverride | None = None,
        lesion: Literal["soft", "max"] | None = None,
        collect_trace: bool = False,
    ) -> tuple[Tensor, AttentionTrace | None]:
        q, k, value = self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x))
        arch = self.architecture
        scores = scaled_masked_scores(
            q, k, mask[None, None, :, :], self.head_dim
        )
        soft_scores = self._replace_query(
            scores, None if overrides is None else overrides.get("soft")
        )
        max_scores = self._replace_query(
            scores, None if overrides is None else overrides.get("max")
        )
        soft_part: Tensor | None = None
        max_part: Tensor | None = None
        raw_max: Tensor | None = None
        winners: Tensor | None = None
        if arch in {"soft_standard", "soft_narrow"}:
            context = soft_reduce(soft_scores, value)
        elif arch in {"soft_grouped", "soft_time_matched"}:
            context = group_rms(soft_reduce(soft_scores, value))
        elif arch == "soft_narrow_grouped":
            soft = soft_reduce(soft_scores, value)
            merged = soft.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], 32)
            context = group_rms(merged).reshape(x.shape[0], x.shape[1], self.heads, 1).transpose(1, 2)
        elif arch == "candidate":
            soft_part = soft_reduce(soft_scores, value[..., :4])
            max_part, winners = maxplus_reduce(max_scores, value[..., 4:])
            context = torch.cat((group_rms(soft_part), group_rms(max_part)), dim=-1)
            raw_max = max_part
        elif arch == "finite_beta_8":
            soft_part = soft_reduce(soft_scores, value[..., :4])
            max_part = finite_beta_reduce(max_scores, value[..., 4:], 8.0)
            context = torch.cat((group_rms(soft_part), group_rms(max_part)), dim=-1)
            raw_max = max_part
        elif arch == "all_max_shared":
            max_part, winners = maxplus_reduce(max_scores, value)
            raw_max = max_part
            context = group_rms(max_part)
        elif arch == "hard_shared":
            score_winner = max_scores.argmax(dim=-1)
            gather_index = score_winner[..., None, None].expand(*score_winner.shape, 1, self.head_dim)
            copied = torch.gather(value.unsqueeze(-3).expand(-1, -1, value.shape[-2], -1, -1), -2, gather_index).squeeze(-2)
            max_part = copied
            context = group_rms(copied)
            winners = score_winner[..., None].expand(*score_winner.shape, self.head_dim)
        elif arch == "hybrid_independent":
            soft_part = soft_reduce(soft_scores[:, :4], value[:, :4])
            max_part, winners = maxplus_reduce(max_scores[:, 4:], value[:, 4:])
            raw_max = max_part
            context = torch.cat((group_rms(soft_part), group_rms(max_part)), dim=1)
        elif arch == "hybrid_whole_head":
            soft_part = soft_reduce(soft_scores[:, :2], value[:, :2])
            max_part, winners = maxplus_reduce(max_scores[:, 2:], value[:, 2:])
            raw_max = max_part
            context = torch.cat((group_rms(soft_part), group_rms(max_part)), dim=1)
        else:
            raise ValueError(f"unsupported mixed attention {arch}")

        if lesion is not None:
            if arch != "candidate":
                raise ValueError("branch lesions are candidate-only")
            context = context.clone()
            if lesion == "soft":
                context[..., :4] = 0.0
            else:
                context[..., 4:] = 0.0

        pre_o = context.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], 32)
        soft_pre_o = max_pre_o = None
        if collect_trace and arch == "candidate":
            if soft_part is None or max_part is None:
                raise RuntimeError("candidate branch tensors missing")
            soft_context = torch.zeros_like(context)
            max_context = torch.zeros_like(context)
            soft_context[..., :4] = group_rms(soft_part)
            max_context[..., 4:] = group_rms(max_part)
            soft_pre_o = soft_context.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], 32)
            max_pre_o = max_context.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], 32)
        trace = (
            AttentionTrace(scores, raw_max, winners, pre_o, soft_pre_o, max_pre_o)
            if collect_trace
            else None
        )
        return self.o(pre_o), trace


class TropicalLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(8, 8))

    def forward(self, x: Tensor) -> Tensor:
        return (x.unsqueeze(-2) + self.W).max(dim=-1).values


class TropicalOfficial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_trop = TropicalLinear()
        self.key_trop = TropicalLinear()
        self.value_trop = TropicalLinear()
        self.out = nn.Linear(32, 32, bias=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        overrides: ScoreOverride | None = None,
        collect_trace: bool = False,
    ) -> tuple[Tensor, AttentionTrace | None]:
        if overrides:
            raise ValueError("tropical control has no branch override")
        z = torch.log1p(F.relu(x)).reshape(x.shape[0], x.shape[1], 4, 8).transpose(1, 2)
        q, k, value = self.query_trop(z), self.key_trop(z), self.value_trop(z)
        difference = q.unsqueeze(-2) - k.unsqueeze(-3)
        scores = -(difference.max(dim=-1).values - difference.min(dim=-1).values)
        scores = scores.masked_fill(~mask[None, None, :, :], -torch.inf)
        context, winners = maxplus_reduce(scores, value)
        pre_o = torch.expm1(context).transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], 32)
        trace = (
            AttentionTrace(scores, context, winners, pre_o, None, pre_o)
            if collect_trace
            else None
        )
        return self.out(pre_o), trace


class TransformerBlock(nn.Module):
    def __init__(self, architecture: str, ffn_width: int = 64) -> None:
        super().__init__()
        self.norm_attn = FrozenRMSNorm()
        self.attn: nn.Module = TropicalOfficial() if architecture == "tropical_official" else MixedAttention(architecture)
        self.norm_ffn = FrozenRMSNorm()
        self.ffn = FeedForward(ffn_width)

    def forward(
        self,
        h: Tensor,
        mask: Tensor,
        *,
        r_attention: int = 1,
        r_ffn: int = 1,
        overrides: Sequence[ScoreOverride | None] | None = None,
        lesion: Literal["soft", "max"] | None = None,
        collect_traces: bool = False,
    ) -> tuple[Tensor, list[AttentionTrace]]:
        traces: list[AttentionTrace] = []
        for repeat in range(r_attention):
            override = None if overrides is None else overrides[repeat]
            if isinstance(self.attn, MixedAttention):
                update, trace = self.attn(
                    self.norm_attn(h), mask, override, lesion, collect_traces
                )
            else:
                update, trace = self.attn(
                    self.norm_attn(h), mask, override, collect_traces
                )
            h = h + update
            if trace is not None:
                traces.append(trace)
        for _ in range(r_ffn):
            h = h + self.ffn(self.norm_ffn(h))
        return h, traces


class TinyTransformer(nn.Module):
    def __init__(self, architecture: str, *, r_attention: int = 1, r_ffn: int = 1) -> None:
        super().__init__()
        self.architecture = architecture
        self.r_attention = r_attention
        self.r_ffn = r_ffn
        internal_arch = "soft_time_matched" if architecture.startswith("soft_time_matched") else architecture
        ffn_width = 109 if architecture == "tropical_official" else 64
        self.input = nn.Linear(14, 32, bias=False)
        self.block = nn.ModuleList([TransformerBlock(internal_arch, ffn_width) for _ in range(2)])
        self.final_norm = FrozenRMSNorm()
        self.readout = nn.ModuleDict({name: nn.Linear(32, 2, bias=True) for name in ("ys", "ym", "y")})

    def _forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        *,
        overrides: Sequence[ScoreOverride | None] | None = None,
        lesion: Literal["soft", "max"] | None = None,
        collect_traces: bool = False,
    ) -> tuple[dict[str, Tensor], list[AttentionTrace]]:
        h = self.input(tokens)
        traces: list[AttentionTrace] = []
        cursor = 0
        for block in self.block:
            block_overrides = None if overrides is None else overrides[cursor : cursor + self.r_attention]
            h, block_traces = block(
                h,
                mask,
                r_attention=self.r_attention,
                r_ffn=self.r_ffn,
                overrides=block_overrides,
                lesion=lesion,
                collect_traces=collect_traces,
            )
            traces.extend(block_traces)
            cursor += self.r_attention
        query = self.final_norm(h[:, -1])
        return {name: head(query) for name, head in self.readout.items()}, traces

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        *,
        lesion: Literal["soft", "max"] | None = None,
    ) -> dict[str, Tensor]:
        logits, traces = self._forward(tokens, mask, lesion=lesion, collect_traces=False)
        if traces:
            raise RuntimeError("trace-free forward unexpectedly retained diagnostics")
        return logits

    def trace_forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        *,
        overrides: Sequence[ScoreOverride | None] | None = None,
        lesion: Literal["soft", "max"] | None = None,
    ) -> tuple[dict[str, Tensor], list[AttentionTrace]]:
        return self._forward(
            tokens,
            mask,
            overrides=overrides,
            lesion=lesion,
            collect_traces=True,
        )


class ConditionedDeepSets(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phi1 = nn.Linear(28, 74, bias=False)
        self.phi2 = nn.Linear(74, 74, bias=False)
        self.pool_proj = nn.Linear(162, 32, bias=False)
        self.norm = FrozenRMSNorm()
        self.ffn = FeedForward(64)
        self.final_norm = FrozenRMSNorm()
        self.readout = nn.ModuleDict({name: nn.Linear(32, 2, bias=True) for name in ("ys", "ym", "y")})

    def forward(self, tokens: Tensor, mask: Tensor) -> dict[str, Tensor]:
        del mask
        records, query = tokens[:, :-1], tokens[:, -1]
        paired = torch.cat((records, query[:, None].expand(-1, records.shape[1], -1)), dim=-1)
        features = self.phi2(F.gelu(self.phi1(paired), approximate="none"))
        pooled = torch.cat((features.mean(dim=1), features.max(dim=1).values, query), dim=-1)
        h = self.pool_proj(pooled)
        h = h + self.ffn(self.norm(h))
        h = self.final_norm(h)
        return {name: head(h) for name, head in self.readout.items()}


def canonical_architecture_id(architecture: str, r_attention: int, r_ffn: int) -> str:
    if architecture == "soft_time_matched":
        return f"soft_time_matched.ra{r_attention}.rf{r_ffn}"
    return architecture


def build_model(
    architecture: str, training_seed: int, *, r_attention: int = 1, r_ffn: int = 1
) -> nn.Module:
    if architecture not in ARCHITECTURES:
        raise ValueError(architecture)
    model: nn.Module
    if architecture == "deepsets_conditioned":
        model = ConditionedDeepSets()
    else:
        model = TinyTransformer(architecture, r_attention=r_attention, r_ffn=r_ffn)
    architecture_id = canonical_architecture_id(architecture, r_attention, r_ffn)
    initialize_model(model, architecture_id, training_seed)
    return model


def initialize_model(model: nn.Module, architecture_id: str, training_seed: int) -> None:
    for local_path, parameter in sorted(model.named_parameters()):
        canonical = f"arch.{architecture_id}.{local_path}"
        generator = generator_for(training_seed, canonical)
        if local_path.endswith("bias"):
            nn.init.zeros_(parameter)
        elif local_path.endswith("norm.weight") or "norm_" in local_path and local_path.endswith("weight"):
            nn.init.ones_(parameter)
        elif architecture_id == "tropical_official" and local_path.endswith(("query_trop.W", "key_trop.W", "value_trop.W")):
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
        elif architecture_id == "tropical_official" and local_path.endswith("attn.out.weight"):
            nn.init.kaiming_uniform_(parameter, a=math.sqrt(5), generator=generator)
        else:
            nn.init.xavier_uniform_(parameter, gain=1.0, generator=generator)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_loss(logits: Mapping[str, Tensor], labels: Mapping[str, Tensor]) -> Tensor:
    return (
        0.5 * F.cross_entropy(logits["ys"], labels["ys"])
        + 0.5 * F.cross_entropy(logits["ym"], labels["ym"])
        + F.cross_entropy(logits["y"], labels["y"])
    )


def optimizer_for(model: nn.Module, learning_rate: float, *, weight_decay: float = 0.01) -> torch.optim.AdamW:
    ordered = [parameter for _, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    return torch.optim.AdamW(
        ordered,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    tokens: Tensor,
    labels: Mapping[str, Tensor],
    mask: Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens, mask)
    loss = total_loss(logits, labels)
    loss.backward()
    ordered = [parameter for _, parameter in sorted(model.named_parameters()) if parameter.requires_grad]
    torch.nn.utils.clip_grad_norm_(
        ordered, max_norm=1.0, norm_type=2.0, error_if_nonfinite=True, foreach=False
    )
    optimizer.step()
    return float(loss.detach())


def exact_sattolo(seed: int, size: int = 10_000) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    permutation = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        permutation[index], permutation[other] = permutation[other], permutation[index]
    if not np.array_equal(np.sort(permutation), np.arange(size)):
        raise RuntimeError("not a permutation")
    if np.any(permutation == np.arange(size)):
        raise RuntimeError("fixed point")
    return permutation


def confidence_bounds(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 16:
        raise ValueError("frozen inference requires 16 seeds")
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    radius = TCRIT * deviation / 4.0
    return mean - radius, mean + radius


def current_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def linux_vmhwm_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise RuntimeError("unexpected VmHWM format")
            return int(fields[1]) * 1024
    raise RuntimeError("VmHWM missing")


def verify_runtime() -> None:
    expected = json.loads(ENVIRONMENT_EXPECTATION.read_text(encoding="utf-8"))
    actual = {
        "runtime": str(Path(sys.executable).absolute().relative_to(ROOT)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "mpmath": metadata.version("mpmath"),
        "machine": platform.machine(),
        "kernel_release": platform.release(),
        "governor_cpu16": Path(
            "/sys/devices/system/cpu/cpu16/cpufreq/scaling_governor"
        ).read_text(encoding="utf-8").strip(),
    }
    cpu_fields: dict[str, str] = {}
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if not line:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            cpu_fields[key.strip()] = value.strip()
    actual["cpu_model"] = cpu_fields.get("model name", "")
    actual["microcode"] = cpu_fields.get("microcode", "")
    blas = np.__config__.CONFIG["Build Dependencies"]["blas"]
    actual["blas"] = blas["openblas configuration"]
    if actual != expected:
        raise RuntimeError(f"frozen environment mismatch: {actual!r} != {expected!r}")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise RuntimeError("OMP_NUM_THREADS and MKL_NUM_THREADS must both equal one")
    if not sys.dont_write_bytecode:
        raise RuntimeError("authorized Python must run with -B")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA-visible runtime forbidden")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("thread counts must both equal one")
    if sorted(os.sched_getaffinity(0)) != [16]:
        raise RuntimeError("process must be pinned to logical CPU 16")
    torch.use_deterministic_algorithms(True)


def prior_ledger_binding() -> Mapping[str, Any] | None:
    """Return the latest valid close while allowing exactly one current open."""
    ledger = RUN_ROOT / "ledger"
    if not ledger.exists():
        return None
    paths = sorted(ledger.glob("*.json"))
    recognized: set[Path] = set()
    previous_hash = "0" * 64
    latest: tuple[Path, Mapping[str, Any]] | None = None
    index = 0
    while True:
        opening_path = ledger / f"{index:04d}-open.json"
        prepare_path = ledger / f"{index:04d}-prepare.json"
        closing_path = ledger / f"{index:04d}-close.json"
        if not opening_path.exists():
            if closing_path.exists():
                raise RuntimeError("ledger close lacks its opening")
            break
        recognized.add(opening_path)
        opening = json.loads(opening_path.read_text(encoding="utf-8"))
        early_value = opening.get("early_marker_path")
        receipt_value = opening.get("authorization_receipt_path")
        if not isinstance(early_value, str) or not isinstance(receipt_value, str):
            raise RuntimeError("ledger opening marker or receipt binding missing")
        early_path = RUN_ROOT / early_value
        receipt_path = ROOT / receipt_value
        if (
            opening.get("index") != index
            or opening.get("previous_close_hash") != previous_hash
            or not early_path.is_file()
            or opening.get("early_marker_sha256") != sha256_file(early_path)
            or receipt_path.is_relative_to(RUN_ROOT)
            or not receipt_path.is_file()
            or opening.get("authorization_receipt_sha256")
            != sha256_file(receipt_path)
        ):
            raise RuntimeError("receipt encountered an invalid ledger opening")
        recognized.add(early_path)
        if prepare_path.exists():
            recognized.add(prepare_path)
        if not closing_path.exists():
            index += 1
            break
        if not prepare_path.exists():
            raise RuntimeError("ledger close lacks its fsynced prepare")
        recognized.add(closing_path)
        closing = json.loads(closing_path.read_text(encoding="utf-8"))
        if (
            closing.get("index") != index
            or closing.get("stage") != opening.get("stage")
            or closing.get("open_sha256") != sha256_file(opening_path)
            or closing.get("prepare_sha256") != sha256_file(prepare_path)
            or closing.get("previous_close_hash") != previous_hash
        ):
            raise RuntimeError("receipt encountered an invalid ledger close")
        if latest is not None and latest[1].get("status") != "CONTINUE":
            raise RuntimeError("ledger continued after a terminal close")
        latest = closing_path, closing
        previous_hash = sha256_file(closing_path)
        index += 1
    if set(paths) != recognized:
        raise RuntimeError("receipt encountered unexpected ledger files")
    if latest is None:
        return None
    close_path, closing = latest
    prepare_path = close_path.with_name(close_path.name.replace("-close.json", "-prepare.json"))
    return {
        "close": {
            "path": str(close_path.relative_to(ROOT)),
            "sha256": sha256_file(close_path),
        },
        "prepare": {
            "path": str(prepare_path.relative_to(ROOT)),
            "sha256": sha256_file(prepare_path),
        },
        "complete_artifact_manifest": {
            "path": closing.get("complete_artifact_manifest_path"),
            "sha256": closing.get("complete_artifact_manifest_sha256"),
        },
        "terminal_artifacts": closing.get("terminal_artifacts"),
        "decision_word": {
            "path": closing.get("decision_word_path"),
            "sha256": closing.get("decision_word_sha256"),
        },
        "status": closing.get("status"),
    }


def verify_bound_artifact_manifest(binding: Mapping[str, Any]) -> set[str]:
    path_value, hash_value = binding.get("path"), binding.get("sha256")
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise RuntimeError("artifact manifest binding fields are invalid")
    path = ROOT / path_value
    try:
        path.relative_to(RUN_ROOT)
    except ValueError as error:
        raise RuntimeError("artifact manifest is outside the run root") from error
    if sha256_file(path) != hash_value:
        raise RuntimeError("bound artifact manifest changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict) or payload.get("artifact_count") != len(files):
        raise RuntimeError("bound artifact manifest shape changed")
    verified: set[str] = set()
    for relative_value, expected in files.items():
        if not isinstance(relative_value, str) or not isinstance(expected, dict):
            raise RuntimeError("invalid artifact manifest entry")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("artifact manifest entry escapes the run root")
        if set(expected) != {"sha256", "logical_bytes", "allocated_bytes"}:
            raise RuntimeError("artifact manifest entry fields changed")
        artifact = RUN_ROOT / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError(f"bound artifact missing: {relative}")
        stat = artifact.stat()
        if stat.st_nlink != 1 or stat.st_blocks * 512 < stat.st_size:
            raise RuntimeError(f"bound artifact is linked or sparse: {relative}")
        if (
            sha256_file(artifact) != expected["sha256"]
            or stat.st_size != expected["logical_bytes"]
            or stat.st_blocks * 512 != expected["allocated_bytes"]
        ):
            raise RuntimeError(f"bound artifact changed: {relative}")
        verified.add(str(relative))
    return verified


def verify_terminal_artifacts(bindings: Mapping[str, Any]) -> set[str]:
    expected_names = {
        "decoder_envelope",
        "decoder_input",
        "decoder_vmhwm",
        "decoder_worker",
        "decision_word",
    }
    if set(bindings) != expected_names:
        raise RuntimeError("terminal artifact binding set changed")
    verified: set[str] = set()
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise RuntimeError(f"terminal artifact binding invalid: {name}")
        path_value, hash_value = binding["path"], binding["sha256"]
        if not isinstance(path_value, str) or not isinstance(hash_value, str):
            raise RuntimeError(f"terminal artifact binding fields invalid: {name}")
        path = ROOT / path_value
        try:
            relative = path.relative_to(RUN_ROOT)
        except ValueError as error:
            raise RuntimeError("terminal artifact escaped the run root") from error
        if path.is_symlink() or not path.is_file() or sha256_file(path) != hash_value:
            raise RuntimeError(f"terminal artifact changed: {name}")
        verified.add(str(relative))
    return verified


def unmatched_ledger_open() -> Path:
    ledger = RUN_ROOT / "ledger"
    pending = [
        path
        for path in sorted(ledger.glob("*-open.json"))
        if not path.with_name(path.name.replace("-open.json", "-close.json")).exists()
    ]
    if len(pending) != 1:
        raise RuntimeError("authorized action requires exactly one current ledger open")
    return pending[0]


def exact_current_prefix(expected_prior: set[str]) -> None:
    expected = set(expected_prior)
    expected.add(str(unmatched_ledger_open().relative_to(RUN_ROOT)))
    for path in RUN_ROOT.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("run artifact prefix contains a symlink")
    actual = {
        str(path.relative_to(RUN_ROOT))
        for path in RUN_ROOT.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError("run artifact prefix differs from the exact prior close")


def require_receipt(
    stage: str,
    receipt_path: Path,
    *,
    validate_prior_artifacts: bool = True,
) -> Mapping[str, Any]:
    audited_sources = assert_audited_sources()
    receipt_path = receipt_path.resolve()
    try:
        receipt_path.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError("authorization receipt must live inside the project") from error
    if receipt_path.is_relative_to(RUN_ROOT):
        raise RuntimeError("authorization receipt cannot be a mutable run artifact")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("authorization") != stage:
        raise RuntimeError(f"receipt does not authorize {stage}")
    for path, expected in EXPECTED_HASHES.items():
        if receipt.get("frozen_hashes", {}).get(str(path.relative_to(ROOT))) != expected:
            raise RuntimeError(f"receipt missing frozen hash for {path}")
    for relative in AUDITED_SOURCE_PATHS:
        actual = sha256_file(ROOT / relative)
        if (
            receipt.get("source_hashes", {}).get(str(relative)) != actual
            or audited_sources[str(relative)] != actual
        ):
            raise RuntimeError(f"receipt missing current audited source hash for {relative}")
    if receipt.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST):
        raise RuntimeError("receipt does not bind the current source manifest")
    prior = prior_ledger_binding()
    if stage == "FIXTURES AND TIMING":
        if prior is not None:
            raise RuntimeError("first-stage receipt cannot follow an existing close")
        if any(
            key not in receipt or receipt[key] is not None
            for key in (
                "prior_close",
                "prior_prepare",
                "prior_complete_artifact_manifest",
                "prior_terminal_artifacts",
                "prior_decision_word",
            )
        ):
            raise RuntimeError("first-stage receipt must explicitly bind no prior artifacts")
        if validate_prior_artifacts:
            exact_current_prefix(set())
    else:
        if prior is None or prior.get("status") != "CONTINUE":
            raise RuntimeError("later-stage receipt requires a prior CONTINUE close")
        expected_bindings = {
            "prior_close": prior["close"],
            "prior_prepare": prior["prepare"],
            "prior_complete_artifact_manifest": prior[
                "complete_artifact_manifest"
            ],
            "prior_terminal_artifacts": prior["terminal_artifacts"],
            "prior_decision_word": prior["decision_word"],
        }
        for key, expected in expected_bindings.items():
            if receipt.get(key) != expected:
                raise RuntimeError(f"receipt does not bind exact {key}")
            if key == "prior_terminal_artifacts":
                continue
            path_value = expected.get("path")
            hash_value = expected.get("sha256")
            if not isinstance(path_value, str) or not isinstance(hash_value, str):
                raise RuntimeError(f"prior close lacks valid {key} fields")
            if sha256_file(ROOT / path_value) != hash_value:
                raise RuntimeError(f"bound {key} changed after receipt")
        if validate_prior_artifacts:
            closing_files = verify_bound_artifact_manifest(
                prior["complete_artifact_manifest"]
            )
            complete_manifest_path = Path(
                prior["complete_artifact_manifest"]["path"]
            ).relative_to(RUN_ROOT.relative_to(ROOT))
            terminal_files = verify_terminal_artifacts(prior["terminal_artifacts"])
            close_path = Path(prior["close"]["path"]).relative_to(
                RUN_ROOT.relative_to(ROOT)
            )
            prepare_path = Path(prior["prepare"]["path"]).relative_to(
                RUN_ROOT.relative_to(ROOT)
            )
            closing_files.update(
                (
                    str(complete_manifest_path),
                    *terminal_files,
                    str(prepare_path),
                    str(close_path),
                )
            )
            exact_current_prefix(closing_files)
    return receipt


def static_manifest() -> dict[str, Any]:
    """Pure source-defined manifest; execution requires source-audit permission."""
    return {
        "architectures": ARCHITECTURES,
        "family_size": FAMILY_SIZE,
        "tcrit": TCRIT,
        "eval_seeds": EVAL_SEEDS,
        "expected_parameters": {"standard": 17190, "tropical_official": 17190, "deepsets": 17090},
        "frozen_hashes": {str(path.relative_to(ROOT)): value for path, value in EXPECTED_HASHES.items()},
        "audited_source_paths": tuple(str(path) for path in AUDITED_SOURCE_PATHS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("manifest", "fixtures", "timing", "stage0", "probes", "train"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_frozen_inputs()
    if args.stage == "manifest":
        raise RuntimeError("manifest execution requires the source auditor's dedicated wrapper")
    if args.receipt is None:
        raise RuntimeError("an audited authorization receipt is required")
    require_receipt(args.stage.upper(), args.receipt)
    raise RuntimeError(f"stage {args.stage} is implemented in the audited controller, not direct CLI")


if __name__ == "__main__":
    main()
