#!/usr/bin/env python3
"""Local optimistic upper bound on representable T38 COPY events."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROSE_PATH = ROOT / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
TOKENIZER_PATH = (
    ROOT / "data/equivariant-template-engram-t38/tokenizer/tokenizer.json"
)
PREREGISTRATION_PATH = (
    ROOT
    / "results/equivariant-template-engram-t38-local-copy-upper-bound-preregistration.md"
)
OUTPUT_PATH = (
    ROOT / "results/equivariant-template-engram-t38-local-copy-upper-bound.json"
)

TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
LITERAL_COUNT = 8192
MAX_TOKENS = 1024
WINDOWS = (8, 16, 32)
CODE_SUFFIXES = frozenset({".py", ".c", ".cc", ".cpp", ".cu", ".h"})
EXCLUDED_MATH_FILES = frozenset(
    {
        "equivariant-template-engram-t38-local-copy-upper-bound-preregistration.md",
        "equivariant-template-engram-t38-fetch-failure.md",
        "equivariant-template-engram-t38r-fetch-failure.md",
    }
)


@dataclass(frozen=True)
class RawDocument:
    stratum: str
    document_id: str
    text: str
    sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_document(stratum: str, document_id: str, text: str) -> RawDocument:
    return RawDocument(stratum, document_id, text, sha256_bytes(text.encode("utf-8")))


def collect_prose() -> list[RawDocument]:
    documents = []
    with PROSE_PATH.open() as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            text = str(row["text"])
            document_id = str(row.get("document_id") or f"row-{index}")
            documents.append(raw_document("prose", document_id, text))
    return documents


def collect_math() -> list[RawDocument]:
    documents = []
    for path in sorted((ROOT / "results").glob("*.md")):
        if path.name in EXCLUDED_MATH_FILES or path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(errors="replace")
        documents.append(raw_document("math", path.relative_to(ROOT).as_posix(), text))
    return documents


def collect_code() -> list[RawDocument]:
    paths = []
    for directory in (ROOT / "experiments", ROOT / "tests"):
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix in CODE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    documents = []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        text = path.read_text(errors="replace")
        documents.append(raw_document("code", path.relative_to(ROOT).as_posix(), text))
    return documents


def oracle_counts(
    tokens: Sequence[int], literal_ids: frozenset[int]
) -> dict[str, int]:
    counts = {"events": 0, "unrestricted_repeat_32": 0}
    counts.update({f"copyable_{window}": 0 for window in WINDOWS})
    for position in range(min(WINDOWS), len(tokens)):
        target = tokens[position]
        counts["events"] += 1
        if target in tokens[max(0, position - 32) : position]:
            counts["unrestricted_repeat_32"] += 1
        if target in literal_ids:
            continue
        for window in WINDOWS:
            if target in tokens[max(0, position - window) : position]:
                counts[f"copyable_{window}"] += 1
    return counts


def merge_counts(rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        result.update(row)
    return dict(result)


def add_fractions(counts: Mapping[str, int]) -> dict[str, int | float]:
    result: dict[str, int | float] = dict(counts)
    events = counts["events"]
    result["unrestricted_repeat_32_fraction"] = (
        counts["unrestricted_repeat_32"] / events if events else 0.0
    )
    for window in WINDOWS:
        result[f"copyable_{window}_fraction"] = (
            counts[f"copyable_{window}"] / events if events else 0.0
        )
    return result


def self_check() -> dict[str, Any]:
    tokens = (1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 7, 6)
    literal_ids = frozenset({1, 2, 3, 4, 5, 7})
    counts = oracle_counts(tokens, literal_ids)
    if counts != {
        "events": 4,
        "unrestricted_repeat_32": 2,
        "copyable_8": 1,
        "copyable_16": 1,
        "copyable_32": 1,
    }:
        raise RuntimeError(f"T38 local oracle self-check failed: {counts}")
    all_eligible = oracle_counts(tokens, frozenset())
    if all_eligible["copyable_32"] != all_eligible["unrestricted_repeat_32"]:
        raise RuntimeError("T38 unrestricted oracle identity failed")
    if not (
        counts["copyable_8"]
        <= counts["copyable_16"]
        <= counts["copyable_32"]
        <= counts["unrestricted_repeat_32"]
    ):
        raise RuntimeError("T38 oracle window monotonicity failed")
    return {"pass": True, "toy_events": counts["events"]}


def write_atomic(payload: Mapping[str, Any]) -> None:
    temporary = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, OUTPUT_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("T38 local upper bound already exists; refusing overwrite")
    check = self_check()

    try:
        from tokenizers import Tokenizer
        import tokenizers
    except ImportError as error:
        raise RuntimeError("run with tokenizers==0.22.2") from error
    if tokenizers.__version__ != "0.22.2":
        raise RuntimeError(f"expected tokenizers 0.22.2, got {tokenizers.__version__}")
    tokenizer_sha = sha256_bytes(TOKENIZER_PATH.read_bytes())
    if tokenizer_sha != TOKENIZER_SHA256:
        raise RuntimeError("sealed tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    raw_documents = [*collect_prose(), *collect_math(), *collect_code()]
    tokenized: list[tuple[RawDocument, tuple[int, ...]]] = []
    frequency: Counter[int] = Counter()
    for document in raw_documents:
        tokens = tuple(
            tokenizer.encode(document.text, add_special_tokens=False).ids[:MAX_TOKENS]
        )
        tokenized.append((document, tokens))
        frequency.update(tokens)

    vocab_size = tokenizer.get_vocab_size()
    literal_ranking = sorted(
        range(vocab_size), key=lambda token: (-frequency[token], token)
    )
    literal_ids = frozenset(literal_ranking[:LITERAL_COUNT])

    by_stratum_rows: dict[str, list[dict[str, int]]] = defaultdict(list)
    for document, tokens in tokenized:
        by_stratum_rows[document.stratum].append(oracle_counts(tokens, literal_ids))
    by_stratum = {
        stratum: add_fractions(merge_counts(rows))
        for stratum, rows in sorted(by_stratum_rows.items())
    }
    math_code_upper_bound = (
        float(by_stratum["math"]["copyable_32_fraction"])
        + float(by_stratum["code"]["copyable_32_fraction"])
    ) / 2.0

    result = {
        "schema": "equivariant-template-engram-t38-local-copy-upper-bound-v1",
        "status": "closed" if math_code_upper_bound < 0.03 else "inconclusive",
        "claim_boundary": (
            "Optimistic local oracle COPY upper bound only; no table, control, "
            "model, benchmark, serving, remote corpus, or GPU claim."
        ),
        "preregistration_sha256": sha256_bytes(PREREGISTRATION_PATH.read_bytes()),
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "self_check": check,
        "tokenizer": {
            "sha256": tokenizer_sha,
            "tokenizers_version": tokenizers.__version__,
            "vocab_size": vocab_size,
        },
        "constants": {
            "literal_count": LITERAL_COUNT,
            "max_tokens_per_document": MAX_TOKENS,
            "windows": WINDOWS,
            "close_threshold": 0.03,
        },
        "documents": {
            stratum: sum(1 for document in raw_documents if document.stratum == stratum)
            for stratum in ("prose", "math", "code")
        },
        "document_manifest": [
            {
                "stratum": document.stratum,
                "document_id": document.document_id,
                "sha256": document.sha256,
                "tokens": len(tokens),
            }
            for document, tokens in tokenized
        ],
        "by_stratum": by_stratum,
        "equal_math_code_copyable_32_fraction": math_code_upper_bound,
        "gate": {
            "upper_bound_at_least_3pct": math_code_upper_bound >= 0.03,
        },
    }
    write_atomic(result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "equal_math_code_copyable_32_fraction": math_code_upper_bound,
                "output": str(OUTPUT_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
