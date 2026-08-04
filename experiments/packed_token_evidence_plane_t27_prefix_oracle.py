from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
EVALUATOR_SHA256 = "5a0413f53bfc0fc73b5e66c941c708077624908095934af740c84d234b9fea6b"
STAGE0_SHA256 = "6d24c18fbb21ec95e41fff87f547067f93cee4fe46dcdb2a5f499e564dbef0a8"
PACK_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
PACK_TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
READER_MODEL = "Qwen/Qwen3.5-9B"
READER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PREFIX_TOKENS = 55
PAD_TOKEN_ID = 0
SHUFFLE_OFFSET = 997
YES_TOKEN_ID = 9405
NO_TOKEN_ID = 2083

SYSTEM_MESSAGE = (
    "Use only the supplied evidence. Answer the comparison question with exactly "
    "yes or no. Entity names have been replaced consistently. Do not use outside "
    "knowledge and do not explain."
)


@dataclass(frozen=True)
class RoutedRow:
    row_id: str
    question: str
    answer: str
    titles: tuple[str, str]


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
        raise RuntimeError(
            f"expected two nonoverlapping raw titles, found {len(selected)}: {question}"
        )
    selected.sort()
    first, second = selected
    return (first[0], first[1]), (second[0], second[1]), (first[2], second[2])


def alias_question(question: str, spans: Sequence[tuple[int, int]]) -> str:
    if len(spans) != 2:
        raise ValueError("expected exactly two spans")
    result = question
    for (start, end), alias in sorted(zip(spans, ("Entity A", "Entity B")), reverse=True):
        result = result[:start] + alias + result[end:]
    return result


def replace_title(text: str, source_title: str, alias: str) -> str:
    return re.sub(re.escape(source_title), alias, text, flags=re.IGNORECASE)


def shuffled_title_map(titles: Sequence[str]) -> dict[str, str]:
    ordered = tuple(sorted(titles))
    if not 0 < SHUFFLE_OFFSET < len(ordered):
        raise ValueError("shuffle offset must be nonzero and smaller than title count")
    mapping = {
        title: ordered[(index + SHUFFLE_OFFSET) % len(ordered)]
        for index, title in enumerate(ordered)
    }
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("fixed shuffle is not a derangement")
    return mapping


def build_user_message(question: str, evidence_a: str, evidence_b: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence for Entity A:\n{evidence_a}\n\n"
        f"Evidence for Entity B:\n{evidence_b}\n\n"
        "Answer (yes or no):"
    )


def render_prompt(reader_tokenizer: Any, question: str, evidence_a: str, evidence_b: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(question, evidence_a, evidence_b)},
    ]
    return reader_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def decoded_prefix(pack_tokenizer: Any, text: str) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    ids = tuple(pack_tokenizer.encode(text, add_special_tokens=False)[:PREFIX_TOKENS])
    padded = ids + (PAD_TOKEN_ID,) * (PREFIX_TOKENS - len(ids))
    decoded = pack_tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    reencoded = tuple(pack_tokenizer.encode(decoded, add_special_tokens=False))
    if reencoded != ids:
        raise RuntimeError("decoded prefix does not re-encode to identical IDs")
    return ids, padded, decoded


def summarize(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    return {
        "min": ordered[0],
        "p25": ordered[int(0.25 * (n - 1))],
        "median": ordered[int(0.50 * (n - 1))],
        "p75": ordered[int(0.75 * (n - 1))],
        "max": ordered[-1],
    }


def predictions_digest(details: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(list(details), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_oracle(batch_size: int = 8) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    corpus_path = root / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
    evaluator_path = root / "data/hotpot-real-prose-t12/sealed-evaluator.jsonl"
    stage0_path = root / "results/packed-token-evidence-plane-t27-stage0.json"
    prereg_path = root / "results/packed-token-evidence-plane-t27-prefix-oracle-preregistration.md"
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_packed_token_evidence_plane_t27_prefix_oracle.py"

    observed_hashes = {
        "corpus": sha256_file(corpus_path),
        "evaluator": sha256_file(evaluator_path),
        "stage0": sha256_file(stage0_path),
    }
    expected_hashes = {
        "corpus": CORPUS_SHA256,
        "evaluator": EVALUATOR_SHA256,
        "stage0": STAGE0_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"artifact hash mismatch: {observed_hashes}")
    if transformers.__version__ != "5.14.1":
        raise RuntimeError(f"unexpected transformers version {transformers.__version__}")
    if torch.__version__ != "2.11.0+cu128":
        raise RuntimeError(f"unexpected torch version {torch.__version__}")

    corpus_rows = load_jsonl(corpus_path)
    evaluator_rows = load_jsonl(evaluator_path)
    titles = tuple(row["title"] for row in corpus_rows)
    documents = {row["title"]: row["text"] for row in corpus_rows}
    shuffle = shuffled_title_map(titles)

    pack_tokenizer = AutoTokenizer.from_pretrained(
        PACK_TOKENIZER,
        revision=PACK_TOKENIZER_REVISION,
    )
    reader_tokenizer = AutoTokenizer.from_pretrained(
        READER_MODEL,
        revision=READER_REVISION,
    )
    if reader_tokenizer.encode("yes", add_special_tokens=False) != [YES_TOKEN_ID]:
        raise RuntimeError("frozen yes token ID changed")
    if reader_tokenizer.encode("no", add_special_tokens=False) != [NO_TOKEN_ID]:
        raise RuntimeError("frozen no token ID changed")
    reader_tokenizer.padding_side = "left"
    if reader_tokenizer.pad_token_id is None:
        reader_tokenizer.pad_token = reader_tokenizer.eos_token

    routed: list[RoutedRow] = []
    prompt_records: list[dict[str, Any]] = []
    prefix_roundtrips = 0
    padded_shapes = 0
    for row in evaluator_rows:
        first_span, second_span, indices = nonoverlapping_title_pair(row["question"], titles)
        row_titles = (titles[indices[0]], titles[indices[1]])
        aliased_question = alias_question(row["question"], (first_span, second_span))
        routed.append(
            RoutedRow(
                row_id=row["id"],
                question=aliased_question,
                answer=row["answer"].strip().casefold(),
                titles=row_titles,
            )
        )

        condition_evidence: dict[str, tuple[str, str]] = {}
        for condition in ("correct_prefix", "full_document"):
            texts: list[str] = []
            for title, alias in zip(row_titles, ("Entity A", "Entity B")):
                if condition == "correct_prefix":
                    ids, padded, decoded = decoded_prefix(pack_tokenizer, documents[title])
                    prefix_roundtrips += int(tuple(pack_tokenizer.encode(decoded, add_special_tokens=False)) == ids)
                    padded_shapes += int(len(padded) == PREFIX_TOKENS)
                    text = decoded
                else:
                    text = documents[title]
                texts.append(replace_title(text, title, alias))
            condition_evidence[condition] = (texts[0], texts[1])

        wrong_titles = (shuffle[row_titles[0]], shuffle[row_titles[1]])
        wrong_texts: list[str] = []
        for title, alias in zip(wrong_titles, ("Entity A", "Entity B")):
            ids, padded, decoded = decoded_prefix(pack_tokenizer, documents[title])
            prefix_roundtrips += int(tuple(pack_tokenizer.encode(decoded, add_special_tokens=False)) == ids)
            padded_shapes += int(len(padded) == PREFIX_TOKENS)
            wrong_texts.append(replace_title(decoded, title, alias))
        condition_evidence["shuffled_prefix"] = (wrong_texts[0], wrong_texts[1])
        condition_evidence["omitted"] = ("[omitted]", "[omitted]")

        for condition in ("correct_prefix", "shuffled_prefix", "omitted", "full_document"):
            evidence_a, evidence_b = condition_evidence[condition]
            prompt_records.append(
                {
                    "row_id": row["id"],
                    "condition": condition,
                    "prompt": render_prompt(
                        reader_tokenizer,
                        aliased_question,
                        evidence_a,
                        evidence_b,
                    ),
                }
            )

    labels = {row.row_id: row.answer for row in routed}
    if set(labels.values()) != {"yes", "no"}:
        raise RuntimeError(f"expected both answer classes, got {set(labels.values())}")
    expected_prefix_checks = len(evaluator_rows) * 4
    if prefix_roundtrips != expected_prefix_checks or padded_shapes != expected_prefix_checks:
        raise RuntimeError(
            f"prefix integrity failed: roundtrips={prefix_roundtrips}, shapes={padded_shapes}, "
            f"expected={expected_prefix_checks}"
        )

    model = AutoModelForImageTextToText.from_pretrained(
        READER_MODEL,
        revision=READER_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    details: list[dict[str, Any]] = []
    max_prompt_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(prompt_records), batch_size):
            batch = prompt_records[start : start + batch_size]
            inputs = reader_tokenizer(
                [item["prompt"] for item in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            max_prompt_tokens = max(max_prompt_tokens, int(inputs["attention_mask"].sum(dim=1).max().item()))
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            logits = model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, YES_TOKEN_ID].float().cpu()
            no_logits = logits[:, NO_TOKEN_ID].float().cpu()
            for item, yes_logit, no_logit in zip(batch, yes_logits, no_logits):
                margin = float(yes_logit.item() - no_logit.item())
                if not math.isfinite(margin) or margin == 0.0:
                    raise RuntimeError(f"invalid yes/no margin for {item['row_id']} {item['condition']}")
                details.append(
                    {
                        "row_id": item["row_id"],
                        "condition": item["condition"],
                        "prediction": "yes" if margin > 0 else "no",
                        "label": labels[item["row_id"]],
                        "margin": margin,
                    }
                )

    conditions = ("correct_prefix", "shuffled_prefix", "omitted", "full_document")
    metrics: dict[str, Any] = {}
    prediction_strings: dict[str, str] = {}
    ordered_ids = [row.row_id for row in routed]
    for condition in conditions:
        selected = [item for item in details if item["condition"] == condition]
        by_id = {item["row_id"]: item for item in selected}
        correct = sum(by_id[row_id]["prediction"] == labels[row_id] for row_id in ordered_ids)
        metrics[condition] = {
            "correct": correct,
            "total": len(ordered_ids),
            "accuracy_percent": 100.0 * correct / len(ordered_ids),
            "margin_summary": summarize([by_id[row_id]["margin"] for row_id in ordered_ids]),
        }
        prediction_strings[condition] = "".join(
            "Y" if by_id[row_id]["prediction"] == "yes" else "N" for row_id in ordered_ids
        )

    correct_accuracy = metrics["correct_prefix"]["accuracy_percent"]
    shuffled_accuracy = metrics["shuffled_prefix"]["accuracy_percent"]
    omitted_accuracy = metrics["omitted"]["accuracy_percent"]
    full_accuracy = metrics["full_document"]["accuracy_percent"]
    gates = {
        "artifact_hashes": observed_hashes == expected_hashes,
        "routing_and_shuffle": len(routed) == len(evaluator_rows)
        and all(source != target for source, target in shuffle.items())
        and all(shuffle[a] != shuffle[b] for a, b in (row.titles for row in routed)),
        "prefix_roundtrip_and_shape": prefix_roundtrips == expected_prefix_checks
        and padded_shapes == expected_prefix_checks,
        "label_boundary": set(labels.values()) == {"yes", "no"},
        "single_token_finite_scores": len(details) == len(evaluator_rows) * len(conditions),
        "correct_prefix_absolute": correct_accuracy >= 80.0
        and correct_accuracy - 60.5769 >= 15.0,
        "correct_prefix_causal": correct_accuracy - shuffled_accuracy >= 15.0
        and correct_accuracy - omitted_accuracy >= 15.0,
        "full_document_sanity": full_accuracy >= 80.0,
        "zero_retries": True,
    }

    paired = {}
    detail_by_key = {(item["row_id"], item["condition"]): item for item in details}
    for control in ("shuffled_prefix", "omitted"):
        gained = lost = unchanged = 0
        for row_id in ordered_ids:
            candidate_ok = detail_by_key[(row_id, "correct_prefix")]["prediction"] == labels[row_id]
            control_ok = detail_by_key[(row_id, control)]["prediction"] == labels[row_id]
            if candidate_ok and not control_ok:
                gained += 1
            elif control_ok and not candidate_ok:
                lost += 1
            else:
                unchanged += 1
        paired[control] = {"gained": gained, "lost": lost, "unchanged": unchanged}

    result = {
        "experiment": "packed-token-evidence-plane-t27-decoded-prefix-oracle",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "reader": {
            "model": READER_MODEL,
            "revision": READER_REVISION,
            "dtype": "bfloat16",
            "score": "final-position logit(yes=9405) > logit(no=2083)",
            "batch_size": batch_size,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
            "max_prompt_tokens": max_prompt_tokens,
        },
        "packing": {
            "tokenizer": PACK_TOKENIZER,
            "revision": PACK_TOKENIZER_REVISION,
            "prefix_tokens": PREFIX_TOKENS,
            "plane_cells": 220,
            "logical_bits": 880,
            "prefix_checks": expected_prefix_checks,
        },
        "rows": {
            "count": len(ordered_ids),
            "ordered_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
            "labels": "".join("Y" if labels[row_id] == "yes" else "N" for row_id in ordered_ids),
            "predictions": prediction_strings,
            "details_sha256": predictions_digest(details),
        },
        "metrics": metrics,
        "causal_deltas_points": {
            "correct_minus_shuffled": correct_accuracy - shuffled_accuracy,
            "correct_minus_omitted": correct_accuracy - omitted_accuracy,
            "full_minus_correct": full_accuracy - correct_accuracy,
        },
        "paired_changes": paired,
        "gates": gates,
        "integrity": {
            "observed_artifact_hashes": observed_hashes,
            "preregistration_sha256": sha256_file(prereg_path),
            "source_sha256": sha256_file(source_path),
            "test_sha256": sha256_file(test_path),
        },
        "claim_boundary": (
            "Strong-reader information oracle only; the reader is not part of the compiler, "
            "candidate, training method, or served model."
        ),
    }
    return result


def main() -> None:
    print(json.dumps(run_oracle(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
