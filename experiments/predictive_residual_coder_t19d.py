#!/usr/bin/env python3
"""T19d: finite top-k predictive residual coder and exact physical ledger."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import self_latent_payload_t19c as t19c


PREREGISTRATION = ROOT / "results/predictive-residual-coder-t19d-preregistration.md"
OUTPUT = ROOT / "results/predictive-residual-coder-t19d.json"
TOP_K = 256
FREQUENCY_TOTAL = 65_536
STATE_BITS = 32
VOCAB = t19c.t19b.t19a.t10.core.small.VOCAB
CONTEXT = 128
BATCH_SIZE = 16
TITLE_WRITES = 132_800
ADDRESS_WRITES = 81_770
SENTINEL_WRITES = 2_405
HARD_WRITE_BUDGET = 743_734
ADMISSION_WRITE_BUDGET = 693_734
MAX_PAYLOAD_CELLS = 350


class ArithmeticEncoder:
    def __init__(self) -> None:
        self.maximum = (1 << STATE_BITS) - 1
        self.half = 1 << (STATE_BITS - 1)
        self.quarter = self.half >> 1
        self.three_quarters = self.quarter * 3
        self.low = 0
        self.high = self.maximum
        self.pending = 0
        self.bits: list[int] = []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def encode_interval(self, cumulative_low: int, cumulative_high: int, total: int) -> None:
        if not 0 <= cumulative_low < cumulative_high <= total:
            raise ValueError("invalid arithmetic interval")
        width = self.high - self.low + 1
        self.high = self.low + (width * cumulative_high // total) - 1
        self.low = self.low + (width * cumulative_low // total)
        while True:
            if self.high < self.half:
                self._emit(0)
            elif self.low >= self.half:
                self._emit(1)
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.pending += 1
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) + 1

    def encode_frequencies(self, symbol: int, frequencies: np.ndarray) -> None:
        cumulative = np.concatenate(([0], np.cumsum(frequencies, dtype=np.int64)))
        self.encode_interval(
            int(cumulative[symbol]), int(cumulative[symbol + 1]), int(cumulative[-1])
        )

    def encode_uniform(self, symbol: int, alphabet: int) -> None:
        self.encode_interval(symbol, symbol + 1, alphabet)

    def finish(self) -> tuple[int, ...]:
        self.pending += 1
        self._emit(0 if self.low < self.quarter else 1)
        return tuple(self.bits)


class ArithmeticDecoder:
    def __init__(self, bits: tuple[int, ...]) -> None:
        self.bits = bits
        self.cursor = 0
        self.maximum = (1 << STATE_BITS) - 1
        self.half = 1 << (STATE_BITS - 1)
        self.quarter = self.half >> 1
        self.three_quarters = self.quarter * 3
        self.low = 0
        self.high = self.maximum
        self.value = 0
        for _ in range(STATE_BITS):
            self.value = (self.value << 1) | self._read()

    def _read(self) -> int:
        if self.cursor >= len(self.bits):
            return 0
        bit = self.bits[self.cursor]
        self.cursor += 1
        return bit

    def _scaled(self, total: int) -> int:
        width = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // width

    def _update(self, cumulative_low: int, cumulative_high: int, total: int) -> None:
        width = self.high - self.low + 1
        self.high = self.low + (width * cumulative_high // total) - 1
        self.low = self.low + (width * cumulative_low // total)
        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.value -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.low -= self.quarter
                self.high -= self.quarter
                self.value -= self.quarter
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) + 1
            self.value = (self.value << 1) | self._read()

    def decode_frequencies(self, frequencies: np.ndarray) -> int:
        cumulative = np.concatenate(([0], np.cumsum(frequencies, dtype=np.int64)))
        total = int(cumulative[-1])
        scaled = self._scaled(total)
        symbol = bisect.bisect_right(cumulative, scaled) - 1
        self._update(int(cumulative[symbol]), int(cumulative[symbol + 1]), total)
        return symbol

    def decode_uniform(self, alphabet: int) -> int:
        symbol = self._scaled(alphabet)
        if symbol >= alphabet:
            raise RuntimeError("uniform arithmetic symbol out of range")
        self._update(symbol, symbol + 1, alphabet)
        return symbol


def quantize_mass_rows(top_probabilities: np.ndarray) -> np.ndarray:
    if top_probabilities.ndim != 2 or top_probabilities.shape[1] != TOP_K:
        raise ValueError("top-probability matrix shape mismatch")
    escape = np.maximum(1.0 - top_probabilities.sum(axis=1, keepdims=True), 0.0)
    masses = np.concatenate((top_probabilities, escape), axis=1).astype(np.float64)
    masses /= masses.sum(axis=1, keepdims=True)
    remaining = FREQUENCY_TOTAL - masses.shape[1]
    scaled = masses * remaining
    floored = np.floor(scaled).astype(np.int64)
    frequencies = floored + 1
    leftovers = FREQUENCY_TOTAL - frequencies.sum(axis=1)
    fractional = scaled - floored
    order = np.argsort(-fractional, axis=1, kind="stable")
    for row, leftover in enumerate(leftovers.tolist()):
        if leftover < 0 or leftover > masses.shape[1]:
            raise RuntimeError(f"invalid frequency remainder {leftover}")
        frequencies[row, order[row, :leftover]] += 1
    if not np.all(frequencies > 0) or not np.all(
        frequencies.sum(axis=1) == FREQUENCY_TOTAL
    ):
        raise RuntimeError("integer frequency contract failed")
    return frequencies


def pack_bits(bits: tuple[int, ...]) -> bytes:
    packed = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        packed[index // 8] |= int(bit) << (7 - index % 8)
    return bytes(packed)


@dataclass(frozen=True)
class EncodedDocument:
    bits: int
    payload_cells: int
    escapes: int
    tokens: int
    ideal_bits: float
    roundtrip_exact: bool
    frequency_contract: bool
    bitstream: bytes


def encode_document(
    tokens: np.ndarray,
    top_indices: np.ndarray,
    frequencies: np.ndarray,
    ideal_bits: float,
) -> EncodedDocument:
    if len(tokens) < 1 or len(top_indices) != len(tokens) - 1:
        raise ValueError("document prediction shape mismatch")
    encoder = ArithmeticEncoder()
    encoder.encode_uniform(int(tokens[0]), VOCAB)
    records: list[tuple[np.ndarray, np.ndarray]] = []
    escapes = 0
    frequency_contract = True
    for position, target in enumerate(tokens[1:]):
        ids = top_indices[position]
        row_frequencies = frequencies[position]
        frequency_contract &= bool(
            np.all(row_frequencies > 0)
            and int(row_frequencies.sum()) == FREQUENCY_TOTAL
        )
        matches = np.flatnonzero(ids == target)
        if len(matches):
            symbol = int(matches[0])
        else:
            symbol = TOP_K
            escapes += 1
        encoder.encode_frequencies(symbol, row_frequencies)
        if symbol == TOP_K:
            encoder.encode_uniform(int(target), VOCAB)
        records.append((ids, row_frequencies))
    bits = encoder.finish()

    decoder = ArithmeticDecoder(bits)
    decoded = [decoder.decode_uniform(VOCAB)]
    for ids, row_frequencies in records:
        symbol = decoder.decode_frequencies(row_frequencies)
        if symbol == TOP_K:
            decoded.append(decoder.decode_uniform(VOCAB))
        else:
            decoded.append(int(ids[symbol]))
    exact = np.array_equal(np.asarray(decoded, dtype=np.int64), tokens)
    return EncodedDocument(
        bits=len(bits),
        payload_cells=(len(bits) + 3) // 4,
        escapes=escapes,
        tokens=len(tokens),
        ideal_bits=ideal_bits,
        roundtrip_exact=exact,
        frequency_contract=frequency_contract,
        bitstream=pack_bits(bits),
    )


def run_unit_contracts() -> dict[str, bool]:
    contracts: dict[str, bool] = {}

    def roundtrip(symbols: list[int], rows: list[np.ndarray]) -> bool:
        encoder = ArithmeticEncoder()
        for symbol, frequencies in zip(symbols, rows, strict=True):
            encoder.encode_frequencies(symbol, frequencies)
        bits = encoder.finish()
        decoder = ArithmeticDecoder(bits)
        decoded = [decoder.decode_frequencies(frequencies) for frequencies in rows]
        return decoded == symbols

    uniform = np.ones(17, dtype=np.int64)
    contracts["uniform"] = roundtrip(list(range(17)) * 3, [uniform] * 51)
    skewed = np.asarray([1, 2, 7, 31, 503], dtype=np.int64)
    contracts["skewed"] = roundtrip([4, 4, 3, 4, 2, 0, 4], [skewed] * 7)
    generator = np.random.default_rng(19_041)
    random_rows: list[np.ndarray] = []
    random_symbols: list[int] = []
    for _ in range(256):
        row = generator.integers(1, 1_000, size=33, dtype=np.int64)
        random_rows.append(row)
        random_symbols.append(int(generator.integers(0, len(row))))
    contracts["randomized"] = roundtrip(random_symbols, random_rows)

    encoder = ArithmeticEncoder()
    encoder.encode_frequencies(2, np.asarray([9, 3, 1], dtype=np.int64))
    encoder.encode_uniform(48_777, VOCAB)
    bits = encoder.finish()
    decoder = ArithmeticDecoder(bits)
    contracts["escape_uniform"] = (
        decoder.decode_frequencies(np.asarray([9, 3, 1], dtype=np.int64)) == 2
        and decoder.decode_uniform(VOCAB) == 48_777
    )
    return contracts


def verify_inputs() -> dict[str, bool]:
    checks = {
        "checkpoint": t19c.sha256_file(t19c.CHECKPOINT) == t19c.CHECKPOINT_SHA256,
        "candidate_corpus": t19c.sha256_file(t19c.t19b.t19a.CANDIDATE_CORPUS)
        == t19c.t19b.t19a.CANDIDATE_SHA256,
        "preregistration_exists": PREREGISTRATION.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"input integrity failed: {checks}")
    return checks


@torch.no_grad()
def run(device: torch.device) -> dict[str, object]:
    input_checks = verify_inputs()
    unit_contracts = run_unit_contracts()
    if not all(unit_contracts.values()):
        raise RuntimeError(f"arithmetic coder unit failure: {unit_contracts}")
    documents = t19c.t19b.t19a.load_raw_documents(
        t19c.t19b.t19a.CANDIDATE_CORPUS
    )
    tokenizer = AutoTokenizer.from_pretrained(
        t19c.t19b.t19a.TOKENIZER,
        revision=t19c.t19b.t19a.TOKENIZER_REVISION,
    )
    model, metadata = t19c.load_writer(device)
    results: list[EncodedDocument] = []
    bitstream_digest = hashlib.sha256()
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        token_rows, lengths = t19c.padded_tokens(
            [document["title"] + "\n" + document["text"] for document in batch],
            tokenizer,
        )
        tokens_device = token_rows.to(device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(tokens_device[:, :-1])
        log_partition = torch.logsumexp(logits.float(), dim=-1)
        top_values, top_indices = logits.topk(TOP_K, dim=-1)
        top_probabilities = torch.exp(top_values.float() - log_partition[..., None])
        target_values = logits.gather(
            -1, tokens_device[:, 1:, None]
        ).squeeze(-1).float()
        ideal_step_bits = (log_partition - target_values) / math.log(2.0)
        top_indices_cpu = top_indices.cpu().numpy()
        top_probabilities_cpu = top_probabilities.cpu().numpy().astype(np.float64)
        ideal_step_bits_cpu = ideal_step_bits.cpu().numpy()
        token_rows_numpy = token_rows.numpy()
        for row in range(len(batch)):
            length = int(lengths[row].item())
            prediction_count = length - 1
            probabilities = top_probabilities_cpu[row, :prediction_count]
            frequencies = quantize_mass_rows(probabilities)
            ideal_bits = math.log2(VOCAB) + float(
                ideal_step_bits_cpu[row, :prediction_count].sum()
            )
            encoded = encode_document(
                token_rows_numpy[row, :length],
                top_indices_cpu[row, :prediction_count],
                frequencies,
                ideal_bits,
            )
            results.append(encoded)
            bitstream_digest.update(encoded.bits.to_bytes(4, "big"))
            bitstream_digest.update(encoded.bitstream)
        print(
            json.dumps(
                {
                    "documents_encoded": len(results),
                    "last_batch_start": start,
                    "maximum_payload_cells": max(result.payload_cells for result in results),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del logits, log_partition, top_values, top_indices, top_probabilities

    payload_cells = sum(result.payload_cells for result in results)
    total_writes = TITLE_WRITES + ADDRESS_WRITES + SENTINEL_WRITES + payload_cells
    total_bits = sum(result.bits for result in results)
    ideal_bits = sum(result.ideal_bits for result in results)
    total_tokens = sum(result.tokens for result in results)
    total_escapes = sum(result.escapes for result in results)
    maximum_cells = max(result.payload_cells for result in results)
    gates = {
        "input_integrity": all(input_checks.values()),
        "checkpoint_metadata": metadata
        == {
            "schema": t19c.CHECKPOINT_SCHEMA,
            "model_seed": t19c.CHECKPOINT_MODEL_SEED,
            "steps": t19c.CHECKPOINT_STEPS,
        },
        "raw_only_compiler_interface": tuple(
            inspect.signature(encode_document).parameters
        )
        == ("tokens", "top_indices", "frequencies", "ideal_bits"),
        "unit_contracts": all(unit_contracts.values()),
        "all_documents_roundtrip_exact": len(results) == 2_405
        and all(result.roundtrip_exact for result in results),
        "all_frequency_contracts": all(
            result.frequency_contract for result in results
        ),
        "decoded_tokens_in_range": all(
            result.tokens >= 1 for result in results
        ),
        "maximum_payload_at_most_350_cells": maximum_cells
        <= MAX_PAYLOAD_CELLS,
        "complete_writes_leave_50k_decoder_entries": total_writes
        <= ADMISSION_WRITE_BUDGET,
        "complete_writes_within_hard_cap": total_writes <= HARD_WRITE_BUDGET,
        "finite_length_accounting": math.isfinite(ideal_bits)
        and total_bits > 0
        and payload_cells == sum((result.bits + 3) // 4 for result in results),
    }
    return {
        "schema": "predictive-residual-coder-t19d-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "device": str(device),
        "configuration": {
            "documents": len(documents),
            "context": CONTEXT,
            "top_k": TOP_K,
            "frequency_total": FREQUENCY_TOTAL,
            "state_bits": STATE_BITS,
            "vocabulary": VOCAB,
            "hard_write_budget": HARD_WRITE_BUDGET,
            "admission_write_budget": ADMISSION_WRITE_BUDGET,
            "maximum_payload_cells": MAX_PAYLOAD_CELLS,
        },
        "unit_contracts": unit_contracts,
        "coder": {
            "documents": len(results),
            "tokens": total_tokens,
            "escapes": total_escapes,
            "escape_rate_after_first_token": total_escapes
            / max(total_tokens - len(results), 1),
            "ideal_bits": ideal_bits,
            "finite_bits": total_bits,
            "finite_over_ideal": total_bits / ideal_bits,
            "payload_cells": payload_cells,
            "mean_payload_cells": payload_cells / len(results),
            "maximum_payload_cells": maximum_cells,
            "bitstream_sha256": bitstream_digest.hexdigest(),
        },
        "physical_write_ledger": {
            "title_writes": TITLE_WRITES,
            "address_writes": ADDRESS_WRITES,
            "payload_writes": payload_cells,
            "sentinel_writes": SENTINEL_WRITES,
            "total_writes": total_writes,
            "hard_cap_headroom": HARD_WRITE_BUDGET - total_writes,
            "decoder_reserved_entries": HARD_WRITE_BUDGET - total_writes,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "integrity": {
            "source_sha256": t19c.sha256_file(Path(__file__)),
            "preregistration_sha256": t19c.sha256_file(PREREGISTRATION),
            "checkpoint_sha256": t19c.sha256_file(t19c.CHECKPOINT),
            "candidate_corpus_sha256": t19c.sha256_file(
                t19c.t19b.t19a.CANDIDATE_CORPUS
            ),
        },
        "claim_boundary": (
            "This proves only finite residual coding and cached-CDF roundtrip. "
            "It does not provide an in-model parallel decoder or QA capability."
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
                "coder": result["coder"],
                "physical_write_ledger": result["physical_write_ledger"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
