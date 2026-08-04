#!/usr/bin/env python3
"""Development smoke for T70's causal state-token operator.

This is not the preregistered cross-family experiment. It tests only whether
an appended causal write slot can learn to carry a random label across a hard
segment boundary, and whether detaching the boundary removes that learning
signal.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StateTokenLearner(nn.Module):
    """Tiny causal Transformer with explicit read/event/write token order."""

    def __init__(
        self,
        *,
        num_keys: int,
        num_values: int,
        state_slots: int = 2,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.state_slots = state_slots
        self.d_model = d_model
        self.key_embedding = nn.Embedding(num_keys, d_model)
        self.value_embedding = nn.Embedding(num_values + 1, d_model)
        self.phase_embedding = nn.Embedding(2, d_model)
        self.token_type_embedding = nn.Embedding(4, d_model)
        self.initial_state = nn.Parameter(torch.randn(state_slots, d_model) * 0.02)
        self.write_queries = nn.Parameter(torch.randn(state_slots, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.state_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, num_values)
        self.missing_value_id = num_values

    def fresh_state(self, batch_size: int) -> Tensor:
        return self.initial_state.unsqueeze(0).expand(batch_size, -1, -1)

    def forward_segment(
        self,
        state: Tensor,
        key: Tensor,
        value: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = key.shape[0]
        is_query = value is None
        phase_id = 1 if is_query else 0

        read_type = self.token_type_embedding.weight[0]
        key_type = self.token_type_embedding.weight[1]
        value_type = self.token_type_embedding.weight[2]
        write_type = self.token_type_embedding.weight[3]
        phase = self.phase_embedding.weight[phase_id]

        read_tokens = state + read_type
        key_token = self.key_embedding(key) + key_type + phase
        if value is None:
            value_ids = torch.full(
                (batch_size,),
                self.missing_value_id,
                dtype=torch.long,
                device=key.device,
            )
        else:
            value_ids = value
        value_token = self.value_embedding(value_ids) + value_type + phase
        write_tokens = self.write_queries.unsqueeze(0).expand(batch_size, -1, -1)
        write_tokens = write_tokens + write_type

        sequence = torch.cat(
            [read_tokens, key_token[:, None, :], value_token[:, None, :], write_tokens],
            dim=1,
        )
        length = sequence.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequence.device),
            diagonal=1,
        )
        hidden = self.transformer(sequence, mask=causal_mask)
        query_hidden = hidden[:, self.state_slots + 1, :]
        next_state = self.state_norm(hidden[:, -self.state_slots :, :])
        return self.output(query_hidden), next_state


@dataclass(frozen=True)
class SmokeResult:
    seed: int
    candidate_accuracy: float
    detached_accuracy: float
    reset_accuracy: float
    independent_shuffle_accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def sample_batch(
    batch_size: int,
    num_keys: int,
    num_values: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    keys = torch.randint(num_keys, (batch_size,), device=device)
    values = torch.randint(num_values, (batch_size,), device=device)
    return keys, values


def train_one(
    *,
    seed: int,
    detach_boundary: bool,
    device: torch.device,
    steps: int,
    batch_size: int,
    num_keys: int,
    num_values: int,
) -> StateTokenLearner:
    set_seed(seed)
    model = StateTokenLearner(num_keys=num_keys, num_values=num_values).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        keys, values = sample_batch(batch_size, num_keys, num_values, device)
        state = model.fresh_state(batch_size)
        _, written_state = model.forward_segment(state, keys, values)
        if detach_boundary:
            written_state = written_state.detach()
        logits, _ = model.forward_segment(written_state, keys, None)
        loss = F.cross_entropy(logits, values)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(
    model: StateTokenLearner,
    *,
    seed: int,
    device: torch.device,
    batches: int,
    batch_size: int,
    num_keys: int,
    num_values: int,
) -> dict[str, float]:
    set_seed(seed)
    model.eval()
    correct = {"factual": 0, "reset": 0, "independent_shuffle": 0}
    total = batches * batch_size
    for _ in range(batches):
        keys, values = sample_batch(batch_size, num_keys, num_values, device)
        state = model.fresh_state(batch_size)
        _, written_state = model.forward_segment(state, keys, values)

        factual_logits, _ = model.forward_segment(written_state, keys, None)
        reset_logits, _ = model.forward_segment(state, keys, None)
        permutation = torch.randperm(batch_size, device=device)
        shuffled_logits, _ = model.forward_segment(written_state[permutation], keys, None)

        correct["factual"] += (factual_logits.argmax(-1) == values).sum().item()
        correct["reset"] += (reset_logits.argmax(-1) == values).sum().item()
        correct["independent_shuffle"] += (
            shuffled_logits.argmax(-1) == values
        ).sum().item()
    return {name: count / total for name, count in correct.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    results: list[SmokeResult] = []
    for seed in args.seeds:
        candidate = train_one(
            seed=seed,
            detach_boundary=False,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            num_keys=args.num_keys,
            num_values=args.num_values,
        )
        detached = train_one(
            seed=seed,
            detach_boundary=True,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            num_keys=args.num_keys,
            num_values=args.num_values,
        )
        candidate_metrics = evaluate(
            candidate,
            seed=seed + 10_000,
            device=device,
            batches=args.eval_batches,
            batch_size=args.batch_size,
            num_keys=args.num_keys,
            num_values=args.num_values,
        )
        detached_metrics = evaluate(
            detached,
            seed=seed + 10_000,
            device=device,
            batches=args.eval_batches,
            batch_size=args.batch_size,
            num_keys=args.num_keys,
            num_values=args.num_values,
        )
        results.append(
            SmokeResult(
                seed=seed,
                candidate_accuracy=candidate_metrics["factual"],
                detached_accuracy=detached_metrics["factual"],
                reset_accuracy=candidate_metrics["reset"],
                independent_shuffle_accuracy=candidate_metrics[
                    "independent_shuffle"
                ],
            )
        )

    means = {
        field: sum(getattr(result, field) for result in results) / len(results)
        for field in (
            "candidate_accuracy",
            "detached_accuracy",
            "reset_accuracy",
            "independent_shuffle_accuracy",
        )
    }
    chance = 1.0 / args.num_values
    passed = (
        means["candidate_accuracy"] >= args.candidate_floor
        and means["detached_accuracy"] <= chance + args.chance_tolerance
        and means["reset_accuracy"] <= chance + args.chance_tolerance
        and means["independent_shuffle_accuracy"] <= chance + args.chance_tolerance
    )
    return {
        "status": "pass" if passed else "fail",
        "scope": "operator development smoke only; not T70 cross-family admission",
        "device": str(device),
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "num_keys": args.num_keys,
            "num_values": args.num_values,
            "candidate_floor": args.candidate_floor,
            "chance_tolerance": args.chance_tolerance,
        },
        "chance": chance,
        "means": means,
        "seeds": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--num-keys", type=int, default=16)
    parser.add_argument("--num-values", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[701, 709, 719])
    parser.add_argument("--candidate-floor", type=float, default=0.90)
    parser.add_argument("--chance-tolerance", type=float, default=0.08)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
