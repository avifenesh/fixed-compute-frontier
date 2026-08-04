#!/usr/bin/env python3
"""T32 natural-information oracle over exact full records and raw local windows."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from experiments.scale_referenced_whole_record_t32_stage0 import (
    TOKEN_SLOTS,
    pack_record,
    unpack_record,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
EVALUATOR = ROOT / "data/hotpot-real-prose-t12/sealed-evaluator.jsonl"
STAGE0 = ROOT / "results/scale-referenced-whole-record-t32-stage0.json"
PREREGISTRATION = ROOT / "results/scale-referenced-whole-record-t32-information-oracle-preregistration.md"

CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
EVALUATOR_SHA256 = "5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b"
STAGE0_SHA256 = "cc4882785b27ce7e19ea443783dbeadc22e859a06e286804199d0210f656e232"
PACK_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
PACK_TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
READER_MODEL = "Qwen/Qwen3.5-9B"
READER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SHUFFLE_OFFSET = 997
LOCAL_WINDOW = 48
DOCUMENT_COUNT = 2_405
DF_CUTOFF = 0.25 * DOCUMENT_COUNT
YES_CONTINUATION = (9_405, 248_046, 198)
NO_CONTINUATION = (2_083, 248_046, 198)
CONDITIONS = (
    "question_only",
    "correct_full128",
    "shuffled_full128",
    "correct_local48",
    "shuffled_local48",
)
SYSTEM_MESSAGE = (
    "Use only the supplied evidence. Answer the comparison question with exactly "
    "yes or no. Entity names have been replaced consistently. Do not use outside "
    "knowledge and do not explain."
)


@dataclass(frozen=True)
class StoredRecord:
    title: str
    ids: tuple[int, ...]
    length: int
    decoded: str


@dataclass(frozen=True)
class Selection:
    start: int
    ids: tuple[int, ...]
    score: float
    matched_ids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nonoverlapping_title_pair(
    question: str, titles: Sequence[str]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    folded = question.casefold()
    candidates: list[tuple[int, int, int]] = []
    for index, title in enumerate(titles):
        surface = title.casefold()
        start = folded.find(surface)
        if start >= 0:
            candidates.append((start, start + len(surface), index))
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
    selected: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if all(
            candidate[1] <= existing[0] or candidate[0] >= existing[1]
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise RuntimeError(f"expected two nonoverlapping titles, got {len(selected)}")
    selected.sort()
    first, second = selected
    return (first[0], first[1]), (second[0], second[1]), (first[2], second[2])


def replace_spans(text: str, spans: Sequence[tuple[int, int]], replacements: Sequence[str]) -> str:
    if len(spans) != len(replacements):
        raise ValueError("span/replacement count mismatch")
    result = text
    for (start, end), replacement in sorted(zip(spans, replacements), reverse=True):
        result = result[:start] + replacement + result[end:]
    return result


def alias_question(question: str, spans: Sequence[tuple[int, int]]) -> str:
    return replace_spans(question, spans, ("Entity A", "Entity B"))


def selector_question(question: str, spans: Sequence[tuple[int, int]]) -> str:
    return replace_spans(question, spans, ("", ""))


def replace_title(text: str, source_title: str, alias: str) -> str:
    return re.sub(re.escape(source_title), alias, text, flags=re.IGNORECASE)


def shuffled_title_map(titles: Sequence[str]) -> dict[str, str]:
    ordered = tuple(sorted(titles))
    if len(ordered) != DOCUMENT_COUNT:
        raise ValueError(f"expected {DOCUMENT_COUNT} titles, got {len(ordered)}")
    mapping = {
        title: ordered[(index + SHUFFLE_OFFSET) % len(ordered)]
        for index, title in enumerate(ordered)
    }
    if len(set(mapping.values())) != len(mapping) or any(k == v for k, v in mapping.items()):
        raise RuntimeError("shuffle is not a derangement bijection")
    return mapping


def build_user_message(question: str, evidence_a: str, evidence_b: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence for Entity A:\n{evidence_a}\n\n"
        f"Evidence for Entity B:\n{evidence_b}\n\n"
        "Answer (yes or no):"
    )


def chat_messages(question: str, evidence_a: str, evidence_b: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(question, evidence_a, evidence_b)},
    ]


def build_records(
    corpus_rows: Sequence[dict[str, Any]], tokenizer: Any
) -> tuple[dict[str, StoredRecord], Counter[int], int]:
    records: dict[str, StoredRecord] = {}
    document_frequency: Counter[int] = Counter()
    exact_checks = 0
    for row in corpus_rows:
        if set(row) != {"document_id", "title", "text"}:
            raise RuntimeError(f"unexpected corpus schema: {sorted(row)}")
        title = str(row["title"])
        raw = tuple(
            int(value)
            for value in tokenizer.encode(
                title + "\n" + str(row["text"]), add_special_tokens=False
            )
        )
        length = min(len(raw), TOKEN_SLOTS)
        padded = raw[:length] + (0,) * (TOKEN_SLOTS - length)
        recovered, recovered_length = unpack_record(pack_record(padded, length))
        if recovered != padded or recovered_length != length:
            raise RuntimeError(f"T32 codec mismatch for {title}")
        ids = recovered[:recovered_length]
        decoded = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        original_decoded = tokenizer.decode(
            raw[:length],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != original_decoded:
            raise RuntimeError(f"decoded-text mismatch for {title}")
        if title in records:
            raise RuntimeError(f"duplicate title: {title}")
        records[title] = StoredRecord(title, ids, length, decoded)
        document_frequency.update(set(ids))
        exact_checks += 1
    if len(records) != DOCUMENT_COUNT:
        raise RuntimeError(f"expected {DOCUMENT_COUNT} records, got {len(records)}")
    return records, document_frequency, exact_checks


def select_local_window(
    record_ids: Sequence[int], query_ids: Sequence[int], document_frequency: Counter[int]
) -> Selection:
    unique_query = tuple(
        sorted(
            {
                int(token)
                for token in query_ids
                if int(token) != 0 and document_frequency[int(token)] < DF_CUTOFF
            }
        )
    )
    idf = {
        token: math.log((DOCUMENT_COUNT + 1) / (document_frequency[token] + 1))
        for token in unique_query
    }
    ids = tuple(int(token) for token in record_ids)
    width = min(LOCAL_WINDOW, len(ids))
    if width == 0:
        return Selection(0, (), 0.0, ())
    best_start = 0
    best_score = -1.0
    best_matches: tuple[int, ...] = ()
    for start in range(len(ids) - width + 1):
        present = set(ids[start : start + width])
        matches = tuple(token for token in unique_query if token in present)
        score = sum(idf[token] for token in matches)
        if score > best_score:
            best_start = start
            best_score = score
            best_matches = matches
    return Selection(
        best_start,
        ids[best_start : best_start + width],
        best_score,
        best_matches,
    )


def summarize(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "median": ordered[int(0.50 * (len(ordered) - 1))],
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def paired_changes(
    ordered_ids: Sequence[str],
    labels: dict[str, str],
    predictions: dict[tuple[str, str], str],
    candidate: str,
    control: str,
) -> dict[str, int]:
    gained = lost = unchanged = 0
    for row_id in ordered_ids:
        candidate_ok = predictions[(row_id, candidate)] == labels[row_id]
        control_ok = predictions[(row_id, control)] == labels[row_id]
        if candidate_ok and not control_ok:
            gained += 1
        elif control_ok and not candidate_ok:
            lost += 1
        else:
            unchanged += 1
    return {"gained": gained, "lost": lost, "unchanged": unchanged}


def path_has_snapshot_revision(path: str | Path, revision: str) -> bool:
    parts = Path(path).parts
    return any(
        parts[index : index + 2] == ("snapshots", revision)
        for index in range(len(parts) - 1)
    )


def resolve_cached_file(repo: str, revision: str, filename: str) -> str:
    from huggingface_hub import try_to_load_from_cache

    value = try_to_load_from_cache(repo, filename, revision=revision)
    if not isinstance(value, str):
        raise RuntimeError(f"missing cached artifact {repo}/{filename}@{revision}")
    path = Path(value).absolute()
    if not path.is_file() or not path_has_snapshot_revision(path, revision):
        raise RuntimeError(f"wrong cached revision path: {path}")
    return str(path)


class PowerSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            watts = float(completed.stdout.strip().splitlines()[0])
            self.samples.append((time.perf_counter(), watts))
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            energy += (t1 - t0) * (p0 + p1) / 2.0
        return {
            "samples": len(self.samples),
            "mean_power_watts": (
                sum(power for _, power in self.samples) / len(self.samples)
                if self.samples
                else float("nan")
            ),
            "energy_joules_trapezoid": energy,
        }


def score_prompts(
    prompt_records: Sequence[dict[str, Any]], reader_tokenizer: Any, batch_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForImageTextToText

    scorer_rows: list[dict[str, Any]] = []
    maximum_prompt_tokens = 0
    for record in prompt_records:
        messages = record["messages"]
        prompt_encoding = reader_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        )
        prompt_ids = tuple(int(value) for value in prompt_encoding["input_ids"])
        maximum_prompt_tokens = max(maximum_prompt_tokens, len(prompt_ids))
        for answer, expected in (("yes", YES_CONTINUATION), ("no", NO_CONTINUATION)):
            full_encoding = reader_tokenizer.apply_chat_template(
                list(messages) + [{"role": "assistant", "content": answer}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
                return_dict=True,
            )
            full_ids = tuple(int(value) for value in full_encoding["input_ids"])
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError("rendered prompt is not a full-chat prefix")
            continuation = full_ids[len(prompt_ids) :]
            if continuation != expected:
                raise RuntimeError(
                    f"continuation mismatch for {answer}: {continuation} != {expected}"
                )
            scorer_rows.append(
                {
                    "row_id": record["row_id"],
                    "condition": record["condition"],
                    "answer": answer,
                    "prompt_length": len(prompt_ids),
                    "input_ids": full_ids,
                    "continuation": continuation,
                }
            )

    sampler = PowerSampler()
    sampler.start()
    gpu_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        READER_MODEL,
        revision=READER_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    load_seconds = time.perf_counter() - gpu_started

    scored: list[dict[str, Any]] = []
    pad_id = int(reader_tokenizer.pad_token_id)
    scoring_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(scorer_rows), batch_size):
            batch = scorer_rows[start : start + batch_size]
            maximum = max(len(row["input_ids"]) for row in batch)
            input_ids = torch.full(
                (len(batch), maximum), pad_id, dtype=torch.long, device="cuda"
            )
            attention_mask = torch.zeros_like(input_ids)
            offsets: list[int] = []
            for index, row in enumerate(batch):
                values = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
                offset = maximum - len(values)
                offsets.append(offset)
                input_ids[index, offset:] = values
                attention_mask[index, offset:] = 1
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
            for index, (row, offset) in enumerate(zip(batch, offsets)):
                first_prediction = offset + int(row["prompt_length"]) - 1
                total = 0.0
                token_logps: list[float] = []
                for continuation_index, target in enumerate(row["continuation"]):
                    value = float(
                        log_probabilities[
                            index, first_prediction + continuation_index, int(target)
                        ].item()
                    )
                    total += value
                    token_logps.append(value)
                if not math.isfinite(total):
                    raise RuntimeError(f"non-finite score: {row['row_id']} {row['condition']}")
                scored.append(
                    {
                        "row_id": row["row_id"],
                        "condition": row["condition"],
                        "answer": row["answer"],
                        "log_likelihood": total,
                        "token_log_likelihoods": token_logps,
                    }
                )
            del logits, log_probabilities, input_ids, attention_mask
    torch.cuda.synchronize()
    scoring_seconds = time.perf_counter() - scoring_started
    power = sampler.stop()
    return scored, {
        "model_load_seconds": load_seconds,
        "scoring_seconds": scoring_seconds,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "power": power,
    }


def run_oracle(batch_size: int = 8) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoTokenizer

    started = time.perf_counter()
    expected_hashes = {
        "corpus": CORPUS_SHA256,
        "evaluator": EVALUATOR_SHA256,
        "stage0": STAGE0_SHA256,
    }
    observed_hashes = {
        "corpus": sha256_file(CORPUS),
        "evaluator": sha256_file(EVALUATOR),
        "stage0": sha256_file(STAGE0),
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"artifact hash mismatch: {observed_hashes}")
    if transformers.__version__ != "5.14.1" or torch.__version__ != "2.11.0+cu128":
        raise RuntimeError(
            f"runtime mismatch: transformers={transformers.__version__} torch={torch.__version__}"
        )
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("frozen H100 unavailable")

    provenance = {
        "pack_tokenizer_config": resolve_cached_file(
            PACK_TOKENIZER, PACK_TOKENIZER_REVISION, "tokenizer_config.json"
        ),
        "reader_config": resolve_cached_file(READER_MODEL, READER_REVISION, "config.json"),
        "reader_tokenizer_config": resolve_cached_file(
            READER_MODEL, READER_REVISION, "tokenizer_config.json"
        ),
        "reader_weight_index": resolve_cached_file(
            READER_MODEL, READER_REVISION, "model.safetensors.index.json"
        ),
    }

    corpus_rows = load_jsonl(CORPUS)
    evaluator_rows = load_jsonl(EVALUATOR)
    evaluator_schema = {
        "id",
        "question",
        "answer",
        "supporting_sentence_ids",
        "supporting_titles",
    }
    if any(set(row) != evaluator_schema for row in evaluator_rows):
        raise RuntimeError("unexpected evaluator schema")

    pack_tokenizer = AutoTokenizer.from_pretrained(
        PACK_TOKENIZER, revision=PACK_TOKENIZER_REVISION, local_files_only=True
    )
    reader_tokenizer = AutoTokenizer.from_pretrained(
        READER_MODEL, revision=READER_REVISION, local_files_only=True
    )
    reader_tokenizer.padding_side = "left"
    if reader_tokenizer.pad_token_id is None:
        reader_tokenizer.pad_token = reader_tokenizer.eos_token

    records, document_frequency, exact_checks = build_records(corpus_rows, pack_tokenizer)
    titles = tuple(records)
    shuffle = shuffled_title_map(titles)
    prompt_records: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    route_titles: dict[str, tuple[str, str]] = {}
    selection_details: list[dict[str, Any]] = []
    selected_digest = hashlib.sha256()

    # Deliberately do not access row["answer"] anywhere in prompt construction.
    for row in evaluator_rows:
        row_id = str(row["id"])
        question = str(row["question"])
        first_span, second_span, indices = nonoverlapping_title_pair(question, titles)
        spans = (first_span, second_span)
        row_titles = (titles[indices[0]], titles[indices[1]])
        if row_titles[0] == row_titles[1]:
            raise RuntimeError(f"duplicate routed title for {row_id}")
        ordered_ids.append(row_id)
        route_titles[row_id] = row_titles
        aliased = alias_question(question, spans)
        query_text = selector_question(question, spans)
        query_ids = tuple(pack_tokenizer.encode(query_text, add_special_tokens=False))
        wrong_titles = (shuffle[row_titles[0]], shuffle[row_titles[1]])
        if wrong_titles[0] == wrong_titles[1]:
            raise RuntimeError(f"shuffle collapsed pair for {row_id}")

        evidence: dict[str, tuple[str, str]] = {"question_only": ("[omitted]", "[omitted]")}
        for prefix, source_titles in (
            ("correct", row_titles),
            ("shuffled", wrong_titles),
        ):
            full_texts: list[str] = []
            local_texts: list[str] = []
            for slot, (title, alias) in enumerate(zip(source_titles, ("Entity A", "Entity B"))):
                record = records[title]
                full_texts.append(replace_title(record.decoded, title, alias))
                selection = select_local_window(record.ids, query_ids, document_frequency)
                local_decoded = pack_tokenizer.decode(
                    selection.ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                local_texts.append(replace_title(local_decoded, title, alias))
                selected_digest.update(row_id.encode())
                selected_digest.update(prefix.encode())
                selected_digest.update(bytes([slot]))
                for token in selection.ids:
                    selected_digest.update(int(token).to_bytes(2, "little"))
                selection_details.append(
                    {
                        "row_id": row_id,
                        "source": prefix,
                        "slot": slot,
                        "start": selection.start,
                        "length": len(selection.ids),
                        "score": selection.score,
                        "matched_token_count": len(selection.matched_ids),
                    }
                )
            evidence[f"{prefix}_full128"] = (full_texts[0], full_texts[1])
            evidence[f"{prefix}_local48"] = (local_texts[0], local_texts[1])

        for condition in CONDITIONS:
            evidence_a, evidence_b = evidence[condition]
            prompt_records.append(
                {
                    "row_id": row_id,
                    "condition": condition,
                    "messages": chat_messages(aliased, evidence_a, evidence_b),
                }
            )

    scored, reader_runtime = score_prompts(prompt_records, reader_tokenizer, batch_size)

    # This is the first answer-value access in the run.
    labels = {str(row["id"]): str(row["answer"]).strip().casefold() for row in evaluator_rows}
    if set(labels.values()) != {"yes", "no"}:
        raise RuntimeError(f"expected yes/no labels, got {set(labels.values())}")

    scores: dict[tuple[str, str], dict[str, float]] = {}
    for item in scored:
        key = (item["row_id"], item["condition"])
        scores.setdefault(key, {})[item["answer"]] = float(item["log_likelihood"])
    predictions: dict[tuple[str, str], str] = {}
    margins: dict[tuple[str, str], float] = {}
    for key, candidates in scores.items():
        if set(candidates) != {"yes", "no"}:
            raise RuntimeError(f"missing candidate score: {key}")
        margin = candidates["yes"] - candidates["no"]
        if not math.isfinite(margin) or margin == 0.0:
            raise RuntimeError(f"invalid forced-string margin: {key} {margin}")
        margins[key] = margin
        predictions[key] = "yes" if margin > 0.0 else "no"

    metrics: dict[str, Any] = {}
    prediction_strings: dict[str, str] = {}
    for condition in CONDITIONS:
        correct = sum(
            predictions[(row_id, condition)] == labels[row_id] for row_id in ordered_ids
        )
        condition_margins = [margins[(row_id, condition)] for row_id in ordered_ids]
        metrics[condition] = {
            "correct": correct,
            "total": len(ordered_ids),
            "accuracy_percent": 100.0 * correct / len(ordered_ids),
            "margin_summary": summarize(condition_margins),
        }
        prediction_strings[condition] = "".join(
            "Y" if predictions[(row_id, condition)] == "yes" else "N"
            for row_id in ordered_ids
        )

    full = metrics["correct_full128"]["accuracy_percent"]
    question_only = metrics["question_only"]["accuracy_percent"]
    shuffled_full = metrics["shuffled_full128"]["accuracy_percent"]
    local = metrics["correct_local48"]["accuracy_percent"]
    shuffled_local = metrics["shuffled_local48"]["accuracy_percent"]
    selection_starts = [float(item["start"]) for item in selection_details]
    selection_scores = [float(item["score"]) for item in selection_details]
    selection_lengths = [float(item["length"]) for item in selection_details]

    gates = {
        "artifact_hashes": observed_hashes == expected_hashes,
        "runtime_and_hardware": transformers.__version__ == "5.14.1"
        and torch.__version__ == "2.11.0+cu128"
        and "H100" in torch.cuda.get_device_name(0),
        "cached_revisions": all(
            path_has_snapshot_revision(
                path,
                PACK_TOKENIZER_REVISION if key == "pack_tokenizer_config" else READER_REVISION,
            )
            for key, path in provenance.items()
        ),
        "routing_and_shuffle": len(ordered_ids) == len(evaluator_rows)
        and len(set(ordered_ids)) == len(ordered_ids)
        and all(source != target for source, target in shuffle.items())
        and all(shuffle[a] != shuffle[b] for a, b in route_titles.values()),
        "record_and_decode_exactness": exact_checks == DOCUMENT_COUNT,
        "selector_boundary": all(0 <= item["length"] <= LOCAL_WINDOW for item in selection_details),
        "label_boundary": set(labels.values()) == {"yes", "no"},
        "complete_finite_nonzero_scores": len(scores) == len(evaluator_rows) * len(CONDITIONS)
        and all(math.isfinite(value) and value != 0.0 for value in margins.values()),
        "correct_full_absolute": full >= 80.0,
        "correct_full_causal": full - question_only >= 15.0
        and full - shuffled_full >= 15.0,
        "correct_local_absolute_and_retention": local >= 75.0 and abs(local - full) <= 5.0,
        "correct_local_causal": local - shuffled_local >= 10.0,
        "energy_recorded": reader_runtime["power"]["samples"] >= 2
        and math.isfinite(reader_runtime["power"]["energy_joules_trapezoid"]),
        "zero_retries": True,
    }

    detail_digest = hashlib.sha256(
        json.dumps(scored, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "experiment": "scale-referenced-whole-record-t32-information-oracle",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "reader": {
            "model": READER_MODEL,
            "revision": READER_REVISION,
            "dtype": "bfloat16",
            "score": "sum FP32 log-softmax over complete assistant continuation",
            "yes_continuation": YES_CONTINUATION,
            "no_continuation": NO_CONTINUATION,
            "batch_size": batch_size,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
            **reader_runtime,
        },
        "records": {
            "tokenizer": PACK_TOKENIZER,
            "revision": PACK_TOKENIZER_REVISION,
            "record_tokens": TOKEN_SLOTS,
            "record_exact_checks": exact_checks,
        },
        "selector": {
            "window_tokens": LOCAL_WINDOW,
            "document_frequency_cutoff": DF_CUTOFF,
            "selection_count": len(selection_details),
            "zero_score_count": sum(score == 0.0 for score in selection_scores),
            "start_summary": summarize(selection_starts),
            "length_summary": summarize(selection_lengths),
            "score_summary": summarize(selection_scores),
            "selected_tokens_sha256": selected_digest.hexdigest(),
        },
        "rows": {
            "count": len(ordered_ids),
            "ordered_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
            "labels": "".join("Y" if labels[row_id] == "yes" else "N" for row_id in ordered_ids),
            "predictions": prediction_strings,
            "scored_details_sha256": detail_digest,
        },
        "metrics": metrics,
        "causal_deltas_points": {
            "full_minus_question_only": full - question_only,
            "full_minus_shuffled_full": full - shuffled_full,
            "local_minus_question_only": local - question_only,
            "local_minus_shuffled_local": local - shuffled_local,
            "local_minus_full": local - full,
        },
        "paired_changes": {
            "full_vs_question_only": paired_changes(
                ordered_ids, labels, predictions, "correct_full128", "question_only"
            ),
            "full_vs_shuffled_full": paired_changes(
                ordered_ids, labels, predictions, "correct_full128", "shuffled_full128"
            ),
            "local_vs_question_only": paired_changes(
                ordered_ids, labels, predictions, "correct_local48", "question_only"
            ),
            "local_vs_shuffled_local": paired_changes(
                ordered_ids, labels, predictions, "correct_local48", "shuffled_local48"
            ),
        },
        "gates": gates,
        "integrity": {
            "observed_artifact_hashes": observed_hashes,
            "cached_artifact_paths": provenance,
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "test_sha256": sha256_file(
                ROOT / "tests/test_scale_referenced_whole_record_t32_information_oracle.py"
            ),
            "evaluator_prompt_fields": ["id", "question"],
            "evaluator_scoring_fields": ["id", "answer"],
            "support_field_reads": 0,
            "retries": 0,
        },
        "wall_seconds_total": time.perf_counter() - started,
        "claim_boundary": (
            "Strong-reader decoded-information oracle only; no physical small-model reader, "
            "training, language-quality, latency, or smarter-production-model claim."
        ),
    }
    return result


def main() -> None:
    print(json.dumps(run_oracle(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
