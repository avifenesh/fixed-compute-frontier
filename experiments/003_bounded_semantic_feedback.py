#!/usr/bin/env python3
"""Small causal screen for bounded semantic feedback memory.

This is deliberately not an LM-quality experiment.  It asks whether writing a
single two-tier memory from a token's top representation helps more than writing
the identical memory from its input representation, and whether both memory
tiers are causally useful.  All global variants execute the same reads, writes,
and learned modules; endpoint ablations mask a read contribution rather than
removing its work.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "top_both",
    "input_both",
    "top_exact_masked",
    "top_summary_masked",
    "layer_local_both",
)
CONTROL_MODES = ("gru_control",)
ALL_MODES = MODES + CONTROL_MODES


def token_layout(keys: int, values: int) -> dict[str, int]:
    set_base = 0
    get_base = set_base + keys * values
    add_base = get_base + keys
    xor_base = add_base + values
    ask = xor_base + values
    return {
        "set": set_base,
        "get": get_base,
        "add": add_base,
        "xor": xor_base,
        "ask": ask,
        "vocab": ask + 1,
    }


def make_batch(
    batch: int,
    steps: int,
    keys: int,
    values: int,
    task: str,
    state_modulus: int,
    ring_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate binding, finite-state, or mixed episodes.

    Categories: 0=no target, 1=recent GET, 2=old GET, 3=state ASK.

    The two state mutations are deliberately small but noncommuting:
    ADVANCE maps s -> s+1 mod m, while FLIP maps s -> s xor 1.
    """

    if state_modulus > values or state_modulus < 4:
        raise ValueError("state_modulus must be in [4, values]")
    if state_modulus & (state_modulus - 1):
        raise ValueError("state_modulus must be a power of two")
    if task not in {"mixed", "binding", "state"}:
        raise ValueError(f"unknown task: {task}")

    layout = token_layout(keys, values)
    tokens = torch.empty(batch, steps, dtype=torch.long, device=device)
    targets = torch.full((batch, steps), -100, dtype=torch.long, device=device)
    categories = torch.zeros(batch, steps, dtype=torch.long, device=device)
    table = torch.full((batch, keys), -1, dtype=torch.long, device=device)
    last_set = torch.full((batch, keys), -10_000, dtype=torch.long, device=device)
    last_answer_record = torch.full(
        (batch, keys), -10_000, dtype=torch.long, device=device
    )
    accumulator = torch.zeros(batch, dtype=torch.long, device=device)
    state_touched = torch.zeros(batch, dtype=torch.bool, device=device)
    rows = torch.arange(batch, device=device)
    event_thresholds = torch.tensor((0.30, 0.60, 0.75, 0.85), device=device)

    for t in range(steps):
        p = torch.rand(batch, device=device, generator=generator)
        if task == "mixed":
            event = torch.bucketize(p, event_thresholds)
        elif task == "binding":
            event = torch.where(
                p < 0.55,
                torch.zeros_like(p, dtype=torch.long),
                torch.ones_like(p, dtype=torch.long),
            )
        else:
            event = torch.where(
                p < 0.35,
                torch.full_like(p, 2, dtype=torch.long),
                torch.where(
                    p < 0.70,
                    torch.full_like(p, 3, dtype=torch.long),
                    torch.full_like(p, 4, dtype=torch.long),
                ),
            )
        has_any = table.ge(0).any(dim=1)
        event = torch.where((event == 1) & ~has_any, torch.zeros_like(event), event)
        # Do not score the untouched, constant-zero accumulator.  An episode's
        # first state event must mutate it before ASK becomes eligible.
        event = torch.where(
            (event == 4) & ~state_touched,
            torch.full_like(event, 2),
            event,
        )
        key = torch.randint(keys, (batch,), device=device, generator=generator)
        value = torch.randint(values, (batch,), device=device, generator=generator)

        is_set = event == 0
        tokens[:, t] = layout["set"] + key * values + value
        if is_set.any():
            set_rows = rows[is_set]
            set_keys = key[is_set]
            table[set_rows, set_keys] = value[is_set]
            last_set[set_rows, set_keys] = t
            last_answer_record[set_rows, set_keys] = t

        is_get = event == 1
        if is_get.any():
            random_rank = torch.rand(
                batch, keys, device=device, generator=generator
            ).masked_fill(table.lt(0), -1.0)
            get_key = random_rank.argmax(dim=1)
            answer = table.gather(1, get_key[:, None]).squeeze(1)
            # A top-level GET representation may itself carry the retrieved
            # answer.  Classify it as recent until that explicit answer record,
            # not merely the original SET, has left the exact ring.
            age = t - last_answer_record.gather(1, get_key[:, None]).squeeze(1)
            tokens[is_get, t] = layout["get"] + get_key[is_get]
            targets[is_get, t] = answer[is_get]
            categories[is_get, t] = torch.where(
                age[is_get] > ring_size,
                torch.full_like(age[is_get], 2),
                torch.ones_like(age[is_get]),
            )
            get_rows = rows[is_get]
            selected_keys = get_key[is_get]
            last_answer_record[get_rows, selected_keys] = t

        is_add = event == 2
        if is_add.any():
            tokens[is_add, t] = layout["add"]
            accumulator[is_add] = (accumulator[is_add] + 1) % state_modulus
            state_touched[is_add] = True

        is_xor = event == 3
        if is_xor.any():
            tokens[is_xor, t] = layout["xor"]
            accumulator[is_xor] = torch.bitwise_xor(
                accumulator[is_xor], 1
            )
            state_touched[is_xor] = True

        is_ask = event == 4
        if is_ask.any():
            tokens[is_ask, t] = layout["ask"]
            targets[is_ask, t] = accumulator[is_ask]
            categories[is_ask, t] = 3

    return tokens, targets, categories


class MemoryModel(nn.Module):
    def __init__(
        self,
        mode: str,
        vocab: int,
        values: int,
        d_model: int,
        layers: int,
        ring_size: int,
        memory_width: int,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.layers = layers
        self.ring_size = ring_size
        self.memory_width = memory_width
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model)
        self.query = nn.ModuleList(
            nn.Linear(d_model, memory_width, bias=False) for _ in range(layers)
        )
        self.read_out = nn.ModuleList(
            nn.Linear(2 * d_model, d_model, bias=False) for _ in range(layers)
        )
        self.norm1 = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(layers))
        self.norm2 = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(layers))
        self.mlp = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, 2 * d_model, bias=False),
                nn.SiLU(),
                nn.Linear(2 * d_model, d_model, bias=False),
            )
            for _ in range(layers)
        )
        # One writer is intentionally shared across depths in every variant.
        self.write_key = nn.Linear(d_model, memory_width, bias=False)
        self.write_value = nn.Linear(d_model, d_model, bias=False)
        self.decay_logit = nn.Parameter(torch.tensor(math.log(0.995 / 0.005)))
        self.final_norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, values, bias=False)

    def state_elements_per_sequence(self) -> int:
        one = self.ring_size * (self.memory_width + self.d_model)
        one += self.memory_width * self.d_model + self.memory_width
        return one * (self.layers if self.mode == "layer_local_both" else 1)

    def _read(
        self,
        q: torch.Tensor,
        ring_k: torch.Tensor | None,
        ring_v: torch.Tensor | None,
        summary: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ring_k is None:
            exact = torch.zeros_like(summary[:, 0, :])
        else:
            scores = torch.einsum("bm,bcm->bc", q, ring_k)
            # q and k are unit vectors.  Multiplying by sqrt(m) gives random
            # cosine logits O(1) while retaining enough range for a sharp hit.
            weights = scores.mul(math.sqrt(self.memory_width)).softmax(dim=-1)
            exact = torch.einsum("bc,bcd->bd", weights, ring_v)
        feature_q = F.elu(q) + 1.0
        numerator = torch.einsum("bm,bmd->bd", feature_q, summary)
        denominator = torch.einsum("bm,bm->b", feature_q, normalizer)
        compressed = numerator / denominator.clamp_min(1e-4)[:, None]
        return exact, compressed

    def _write(
        self,
        source: torch.Tensor,
        ring_k: torch.Tensor | None,
        ring_v: torch.Tensor | None,
        summary: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = F.normalize(self.write_key(source), dim=-1)
        value = torch.tanh(self.write_value(source))
        if ring_k is None:
            return key[:, None, :], value[:, None, :], summary, normalizer
        if ring_k.shape[1] < self.ring_size:
            return (
                torch.cat((ring_k, key[:, None, :]), dim=1),
                torch.cat((ring_v, value[:, None, :]), dim=1),
                summary,
                normalizer,
            )
        evicted_key = ring_k[:, 0, :]
        evicted_value = ring_v[:, 0, :]
        feature_key = F.elu(evicted_key) + 1.0
        decay = torch.sigmoid(self.decay_logit)
        summary = decay * summary + torch.einsum(
            "bm,bd->bmd", feature_key, evicted_value
        )
        normalizer = decay * normalizer + feature_key
        return (
            torch.cat((ring_k[:, 1:, :], key[:, None, :]), dim=1),
            torch.cat((ring_v[:, 1:, :], value[:, None, :]), dim=1),
            summary,
            normalizer,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, steps = tokens.shape
        zero_summary = tokens.new_zeros(
            batch, self.memory_width, self.d_model, dtype=self.embedding.weight.dtype
        )
        zero_normalizer = tokens.new_zeros(
            batch, self.memory_width, dtype=self.embedding.weight.dtype
        )
        if self.mode == "layer_local_both":
            ring_k: list[torch.Tensor | None] = [None] * self.layers
            ring_v: list[torch.Tensor | None] = [None] * self.layers
            summaries = [zero_summary.clone() for _ in range(self.layers)]
            normalizers = [zero_normalizer.clone() for _ in range(self.layers)]
        else:
            global_k: torch.Tensor | None = None
            global_v: torch.Tensor | None = None
            global_summary = zero_summary
            global_normalizer = zero_normalizer

        logits = []
        for t in range(steps):
            token_input = self.embedding(tokens[:, t])
            x = token_input
            layer_outputs = []
            for layer in range(self.layers):
                q = F.normalize(self.query[layer](self.norm1[layer](x)), dim=-1)
                if self.mode == "layer_local_both":
                    exact, compressed = self._read(
                        q,
                        ring_k[layer],
                        ring_v[layer],
                        summaries[layer],
                        normalizers[layer],
                    )
                else:
                    exact, compressed = self._read(
                        q, global_k, global_v, global_summary, global_normalizer
                    )
                # Masked endpoint controls retain both physical read paths.
                if self.mode == "top_exact_masked":
                    exact = exact * 0.0
                elif self.mode == "top_summary_masked":
                    compressed = compressed * 0.0
                x = x + self.read_out[layer](torch.cat((exact, compressed), dim=-1))
                x = x + self.mlp[layer](self.norm2[layer](x))
                layer_outputs.append(x)

            logits.append(self.output(self.final_norm(x)))
            # Every read above used the t-1 snapshot.  Commits happen only now.
            if self.mode == "layer_local_both":
                for layer, source in enumerate(layer_outputs):
                    (
                        ring_k[layer],
                        ring_v[layer],
                        summaries[layer],
                        normalizers[layer],
                    ) = self._write(
                        source,
                        ring_k[layer],
                        ring_v[layer],
                        summaries[layer],
                        normalizers[layer],
                    )
            else:
                source = token_input if self.mode == "input_both" else x
                global_k, global_v, global_summary, global_normalizer = self._write(
                    source,
                    global_k,
                    global_v,
                    global_summary,
                    global_normalizer,
                )
        return torch.stack(logits, dim=1)


class GRUControl(nn.Module):
    """Purpose-built recurrent positive control for benchmark learnability."""

    def __init__(
        self, vocab: int, values: int, d_model: int, layers: int
    ) -> None:
        super().__init__()
        self.layers = layers
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab, d_model)
        self.recurrent = nn.GRU(
            d_model, d_model, num_layers=layers, batch_first=True
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, values, bias=False)

    def state_elements_per_sequence(self) -> int:
        return self.layers * self.d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.recurrent(self.embedding(tokens))
        return self.output(self.final_norm(hidden))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, float | int]:
    model.eval()
    correct = torch.zeros(4, device=args.device)
    total = torch.zeros(4, device=args.device)
    label_counts = torch.zeros(4, args.values, device=args.device)
    loss_sum = torch.zeros((), device=args.device)
    loss_count = 0
    for _ in range(args.eval_batches):
        tokens, targets, categories = make_batch(
            args.batch,
            args.sequence_length,
            args.keys,
            args.values,
            args.task,
            args.state_modulus,
            args.ring_size,
            args.device,
            generator,
        )
        logits = model(tokens)
        mask = targets.ne(-100)
        loss_sum += F.cross_entropy(logits[mask], targets[mask], reduction="sum")
        loss_count += int(mask.sum())
        predicted = logits.argmax(dim=-1)
        for category in (1, 2, 3):
            selected = categories.eq(category)
            correct[category] += (predicted[selected] == targets[selected]).sum()
            total[category] += selected.sum()
            label_counts[category] += torch.bincount(
                targets[selected], minlength=args.values
            )
    names = {1: "recent_get", 2: "old_get", 3: "state"}
    result: dict[str, float | int] = {
        "loss": float(loss_sum / loss_count),
        "target_count": loss_count,
    }
    for category, name in names.items():
        result[name] = float(correct[category] / total[category].clamp_min(1))
        result[f"{name}_count"] = int(total[category])
        result[f"{name}_majority_baseline"] = float(
            label_counts[category].max() / total[category].clamp_min(1)
        )
    active_names = [
        name for category, name in names.items() if int(total[category]) > 0
    ]
    result["macro_accuracy"] = sum(
        float(result[name]) for name in active_names
    ) / len(active_names)
    return result


def train_one(args: argparse.Namespace, mode: str, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    train_gen = torch.Generator(device=args.device).manual_seed(seed * 10_000 + 1)
    eval_gen = torch.Generator(device=args.device).manual_seed(seed * 10_000 + 2)
    layout = token_layout(args.keys, args.values)
    if mode == "gru_control":
        model: nn.Module = GRUControl(
            layout["vocab"], args.values, args.d_model, args.layers
        ).to(args.device)
    else:
        model = MemoryModel(
            mode,
            layout["vocab"],
            args.values,
            args.d_model,
            args.layers,
            args.ring_size,
            args.memory_width,
        ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device)
    train_started = time.perf_counter()
    model.train()
    for step in range(args.train_steps):
        tokens, targets, _ = make_batch(
            args.batch,
            args.sequence_length,
            args.keys,
            args.values,
            args.task,
            args.state_modulus,
            args.ring_size,
            args.device,
            train_gen,
        )
        logits = model(tokens)
        mask = targets.ne(-100)
        loss = F.cross_entropy(logits[mask], targets[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device)
    train_seconds = time.perf_counter() - train_started
    eval_started = time.perf_counter()
    metrics = evaluate(model, args, eval_gen)
    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device)
    eval_seconds = time.perf_counter() - eval_started
    return {
        "mode": mode,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_elements_per_sequence": model.state_elements_per_sequence(),
        "writer_commits_per_token": (
            args.layers if mode in {"layer_local_both", "gru_control"} else 1
        ),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "metrics": metrics,
    }


def self_test(args: argparse.Namespace) -> None:
    generator = torch.Generator(device=args.device).manual_seed(123)
    tokens, targets, categories = make_batch(
        4,
        12,
        args.keys,
        args.values,
        args.task,
        args.state_modulus,
        args.ring_size,
        args.device,
        generator,
    )
    layout = token_layout(args.keys, args.values)
    counts = []
    for mode in MODES:
        model = MemoryModel(
            mode,
            layout["vocab"],
            args.values,
            16,
            2,
            3,
            8,
        ).to(args.device)
        logits = model(tokens)
        F.cross_entropy(logits[targets.ne(-100)], targets[targets.ne(-100)]).backward()
        assert logits.shape == (4, 12, args.values)
        assert torch.isfinite(logits).all()
        counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts)) == 1, counts
    assert categories.gt(0).any()

    # A unit-normalized exact ring must still be capable of a sharp lookup.
    probe = MemoryModel("top_both", layout["vocab"], args.values, 16, 2, 3, 8).to(
        args.device
    )
    q = torch.zeros(1, 8, device=args.device)
    q[:, 0] = 1.0
    probe_k = torch.zeros(1, 3, 8, device=args.device)
    probe_k[:, 0, 0] = 1.0
    probe_k[:, 1, 0] = -1.0
    probe_k[:, 2, 1] = 1.0
    probe_v = torch.zeros(1, 3, 16, device=args.device)
    probe_v[:, 0, 0] = 1.0
    zero_summary = torch.zeros(1, 8, 16, device=args.device)
    zero_normalizer = torch.zeros(1, 8, device=args.device)
    exact, _ = probe._read(q, probe_k, probe_v, zero_summary, zero_normalizer)
    assert exact[0, 0] > 0.9, exact[0, 0]

    # The state target must not retain the former constant-zero shortcut.
    audit_gen = torch.Generator(device=args.device).manual_seed(456)
    _, audit_targets, audit_categories = make_batch(
        256,
        48,
        args.keys,
        args.values,
        "state",
        args.state_modulus,
        args.ring_size,
        args.device,
        audit_gen,
    )
    state_targets = audit_targets[audit_categories.eq(3)]
    state_prior = torch.bincount(state_targets, minlength=args.values).max()
    state_prior = state_prior / state_targets.numel()
    assert state_targets.numel() > 0
    assert state_prior < (1.0 / args.state_modulus + 0.1), state_prior

    gru = GRUControl(layout["vocab"], args.values, 16, 2).to(args.device)
    gru_logits = gru(tokens)
    F.cross_entropy(
        gru_logits[targets.ne(-100)], targets[targets.ne(-100)]
    ).backward()
    assert gru_logits.shape == (4, 12, args.values)
    assert torch.isfinite(gru_logits).all()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--keys", type=int, default=16)
    parser.add_argument("--values", type=int, default=16)
    parser.add_argument(
        "--task", choices=("mixed", "binding", "state"), default="mixed"
    )
    parser.add_argument("--state-modulus", type=int, default=4)
    parser.add_argument("--ring-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--memory-width", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=str)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args)
        print("self-test passed")
        return
    results = [train_one(args, mode, seed) for mode in args.modes for seed in args.seeds]
    payload = {
        "device": str(args.device),
        "torch": torch.__version__,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"device", "output", "self_test"}
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
