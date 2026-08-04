#!/usr/bin/env python3
"""Frozen CPU opportunity census for T38 equivariant template memory.

This is an exact, non-neural reference.  It compiles only raw next-token counts,
compares equal record caps, and writes a decision artifact.  It does not train a
model or use a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/equivariant-template-engram-t38"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
TOKENIZER_PATH = DATA_DIR / "tokenizer/tokenizer.json"
OUTPUT_PATH = ROOT / "results/equivariant-template-engram-t38-cpu-census.json"
ADJUDICATION_OPENED_PATH = (
    ROOT / "results/equivariant-template-engram-t38-adjudication-opened.json"
)
ADJUDICATION_FINALIZED_PATH = (
    ROOT / "results/equivariant-template-engram-t38-adjudication-finalized.json"
)

CORPUS_SEED = "t38-census-v1"
SPLIT_SEED = "t38-split-v1"
WINDOWS = (8, 16, 32)
DOCS_PER_STRATUM = 768
BLOCKS_PER_STRATUM = 32
ROWS_PER_BLOCK = 32
MAX_TOKENS = 1024
LITERAL_COUNT = 8192
RECORD_CAP = 65536
MIN_SUPPORT = 16
MIN_DOCUMENTS = 4
T38_TRAIN_PURITY = 0.98
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 38003
SHUFFLE_PAYLOAD_SEED = 38001
SHUFFLE_BINDING_SEED = 38002
WILSON_Z = 1.959963984540054

TOKENIZER_REPO = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
CORPUS_SCHEMA = "equivariant-template-engram-t38-corpus-v1"


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str
    revision: str
    rows: int
    text_field: str
    id_fields: tuple[str, ...]


SOURCES = (
    Source(
        "prose",
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        "3ba9d605774198c5868892d7a8deda78031a781f",
        190_168_005,
        "text",
        ("id",),
    ),
    Source(
        "math",
        "open-web-math/open-web-math",
        "default",
        "fde8ef8de2300f5e778f56261843dab89f230815",
        6_315_233,
        "text",
        ("url",),
    ),
    Source(
        "code",
        "transformersbook/codeparrot",
        "default",
        "1525880546992f12c04c5ae4cf5c4d1e80ca04a4",
        477_249,
        "content",
        ("repo_name", "path"),
    ),
)


@dataclass
class RawDocument:
    source: str
    document_id: str
    text: str
    text_sha256: str
    rank_sha256: str
    row_index: int


@dataclass
class Document:
    source: str
    document_id: str
    split: str
    tokens: tuple[int, ...]


@dataclass
class KeyStats:
    total: int
    winner: int
    winner_count: int
    documents: set[str] = field(default_factory=set)
    bindings: set[tuple[int, ...]] = field(default_factory=set)

    @property
    def purity(self) -> float:
        return self.winner_count / self.total


@dataclass
class Record:
    n: int
    key: tuple[int, ...]
    outcome: int
    total: int
    winner_count: int
    confidence_lcb: float
    order_digest: str
    fingerprint: str
    train_bindings: frozenset[tuple[int, ...]] = frozenset()


@dataclass
class Prediction:
    token: int
    record: Record
    is_copy: bool
    binding: tuple[int, ...]


class OutcomeCounts:
    """Memory-light exact outcome counter, homogeneous until a conflict."""

    __slots__ = ("_state",)

    def __init__(self, outcome: int) -> None:
        self._state: tuple[int, int] | dict[int, int] = (outcome, 1)

    def add(self, outcome: int) -> None:
        state = self._state
        if isinstance(state, tuple):
            value, count = state
            if value == outcome:
                self._state = (value, count + 1)
            else:
                self._state = {value: count, outcome: 1}
            return
        state[outcome] = state.get(outcome, 0) + 1

    def summary(self) -> tuple[int, int, int]:
        state = self._state
        if isinstance(state, tuple):
            outcome, count = state
            return count, outcome, count
        total = sum(state.values())
        winner, winner_count = min(
            state.items(), key=lambda item: (-item[1], item[0])
        )
        return total, winner, winner_count


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def key_bytes(key: tuple[int, ...]) -> bytes:
    return canonical_json(key)


def key_order_digest(key: tuple[int, ...]) -> str:
    return sha256_bytes(key_bytes(key))


def key_fingerprint(key: tuple[int, ...]) -> str:
    return hashlib.blake2b(
        key_bytes(key),
        digest_size=16,
        key=b"t38-census-key-v1",
        person=b"t38-key-v1",
    ).hexdigest()


def split_for(source: str, document_id: str, text_sha256: str) -> str:
    material = f"{SPLIT_SEED}|{source}|{document_id}|{text_sha256}".encode()
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 20
    if bucket < 14:
        return "train"
    if bucket < 17:
        return "development"
    return "adjudication"


def block_offsets(source: Source) -> list[int]:
    modulus = source.rows - ROWS_PER_BLOCK + 1
    offsets = []
    for index in range(BLOCKS_PER_STRATUM):
        material = f"{CORPUS_SEED}|{source.revision}|{index}".encode()
        offsets.append(
            int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            % modulus
        )
    return offsets


def http_json(url: str, attempts: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "fixed-compute-frontier-t38/1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception as error:  # pragma: no cover - network path
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def hub_sha(dataset: str) -> str:
    encoded = urllib.parse.quote(dataset, safe="/")
    payload = http_json(f"https://huggingface.co/api/datasets/{encoded}")
    return str(payload["sha"])


def model_revision_sha(model: str, revision: str) -> str:
    encoded_model = urllib.parse.quote(model, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    payload = http_json(
        f"https://huggingface.co/api/models/{encoded_model}/revision/{encoded_revision}"
    )
    return str(payload["sha"])


def fetch_source(source: Source) -> list[RawDocument]:
    current = hub_sha(source.dataset)
    if current != source.revision:
        raise RuntimeError(
            f"{source.dataset} moved: expected {source.revision}, got {current}"
        )

    candidates: list[RawDocument] = []
    for offset in block_offsets(source):
        query = urllib.parse.urlencode(
            {
                "dataset": source.dataset,
                "config": source.config,
                "split": "train",
                "offset": offset,
                "length": ROWS_PER_BLOCK,
            }
        )
        payload = http_json(f"https://datasets-server.huggingface.co/rows?{query}")
        if int(payload.get("num_rows_total", -1)) != source.rows:
            raise RuntimeError(
                f"{source.name}: viewer row total changed from {source.rows} "
                f"to {payload.get('num_rows_total')}"
            )
        if payload.get("partial") is not False:
            raise RuntimeError(f"{source.name}: viewer returned a partial block")
        rows = payload.get("rows", [])
        if len(rows) != ROWS_PER_BLOCK:
            raise RuntimeError(
                f"{source.name} offset {offset}: expected {ROWS_PER_BLOCK} rows, got {len(rows)}"
            )
        for wrapper in rows:
            row = wrapper["row"]
            row_index = int(wrapper["row_idx"])
            if not offset <= row_index < offset + ROWS_PER_BLOCK:
                raise RuntimeError(
                    f"{source.name}: returned row {row_index} outside requested block {offset}"
                )
            text = str(row.get(source.text_field) or "")
            if not text:
                continue
            id_components = [str(row.get(field) or "") for field in source.id_fields]
            if not all(id_components):
                raise RuntimeError(f"{source.name}: empty canonical document ID")
            document_id = ":".join(id_components)
            text_digest = sha256_text(text)
            rank = sha256_text(
                f"{source.name}|{document_id}|{text_digest}"
            )
            candidates.append(
                RawDocument(
                    source.name,
                    document_id,
                    text,
                    text_digest,
                    rank,
                    row_index,
                )
            )

    current_after = hub_sha(source.dataset)
    if current_after != source.revision:
        raise RuntimeError(
            f"{source.dataset} moved during fetch: {current_after}"
        )
    return candidates


def verify_existing_corpus() -> dict[str, Any]:
    if not CORPUS_PATH.exists() or not MANIFEST_PATH.exists():
        raise RuntimeError("T38 corpus/manifest pair is incomplete; refusing replacement")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise RuntimeError("unexpected T38 corpus schema")
    if manifest.get("documents") != len(SOURCES) * DOCS_PER_STRATUM:
        raise RuntimeError("unexpected T38 manifest document count")
    if manifest.get("documents_per_stratum") != DOCS_PER_STRATUM:
        raise RuntimeError("unexpected T38 documents-per-stratum")
    if manifest.get("blocks_per_stratum") != BLOCKS_PER_STRATUM:
        raise RuntimeError("unexpected T38 block count")
    if manifest.get("rows_per_block") != ROWS_PER_BLOCK:
        raise RuntimeError("unexpected T38 rows-per-block")

    actual_corpus_sha = sha256_bytes(CORPUS_PATH.read_bytes())
    if actual_corpus_sha != manifest.get("corpus_sha256"):
        raise RuntimeError("existing T38 corpus hash does not match manifest")

    source_by_name = {source.name: source for source in SOURCES}
    manifest_sources = manifest.get("sources", {})
    if set(manifest_sources) != set(source_by_name):
        raise RuntimeError("T38 manifest source set changed")
    for name, source in source_by_name.items():
        row = manifest_sources[name]
        expected = {
            "dataset": source.dataset,
            "config": source.config,
            "revision": source.revision,
            "rows": source.rows,
            "offsets": block_offsets(source),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise RuntimeError(f"T38 manifest {name}.{key} changed")

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_identities: set[tuple[str, str]] = set()
    seen_texts: set[str] = set()
    ordering: list[tuple[int, str]] = []
    source_order = {source.name: index for index, source in enumerate(SOURCES)}
    rows_seen = 0
    with CORPUS_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            required = {
                "source",
                "document_id",
                "text",
                "text_sha256",
                "rank_sha256",
                "row_index",
                "split",
            }
            if set(row) != required:
                raise RuntimeError("T38 cached corpus row schema changed")
            source = source_by_name.get(row["source"])
            if source is None:
                raise RuntimeError("unknown T38 cached source")
            text_sha = sha256_text(row["text"])
            if text_sha != row["text_sha256"]:
                raise RuntimeError("T38 cached text hash mismatch")
            rank = sha256_text(f"{source.name}|{row['document_id']}|{text_sha}")
            if rank != row["rank_sha256"]:
                raise RuntimeError("T38 cached rank mismatch")
            expected_split = split_for(source.name, row["document_id"], text_sha)
            if expected_split != row["split"]:
                raise RuntimeError("T38 cached split mismatch")
            row_index = int(row["row_index"])
            if not any(
                offset <= row_index < offset + ROWS_PER_BLOCK
                for offset in block_offsets(source)
            ):
                raise RuntimeError("T38 cached row lies outside frozen blocks")
            if not row["document_id"]:
                raise RuntimeError("T38 cached corpus has empty document ID")
            identity = (source.name, row["document_id"])
            if identity in seen_identities or text_sha in seen_texts:
                raise RuntimeError("T38 cached corpus contains a duplicate")
            seen_identities.add(identity)
            seen_texts.add(text_sha)
            counts[source.name][row["split"]] += 1
            ordering.append((source_order[source.name], rank))
            rows_seen += 1

    if rows_seen != len(SOURCES) * DOCS_PER_STRATUM:
        raise RuntimeError("T38 cached corpus row count changed")
    if ordering != sorted(ordering):
        raise RuntimeError("T38 cached corpus order changed")
    for source in SOURCES:
        if sum(counts[source.name].values()) != DOCS_PER_STRATUM:
            raise RuntimeError(f"T38 cached {source.name} count changed")
        if dict(counts[source.name]) != manifest_sources[source.name].get(
            "split_documents"
        ):
            raise RuntimeError(f"T38 cached {source.name} split counts changed")
    return manifest


def materialize_corpus() -> dict[str, Any]:
    if CORPUS_PATH.exists() or MANIFEST_PATH.exists():
        return verify_existing_corpus()

    all_candidates = {source.name: fetch_source(source) for source in SOURCES}
    selected: dict[str, list[RawDocument]] = {source.name: [] for source in SOURCES}
    seen_texts: set[str] = set()
    unique_identity: set[tuple[str, str]] = set()

    merged = sorted(
        (doc for docs in all_candidates.values() for doc in docs),
        key=lambda doc: (doc.rank_sha256, doc.source, doc.document_id),
    )
    for doc in merged:
        identity = (doc.source, doc.document_id)
        if identity in unique_identity:
            continue
        unique_identity.add(identity)
        if doc.text_sha256 in seen_texts:
            continue
        if len(selected[doc.source]) >= DOCS_PER_STRATUM:
            continue
        selected[doc.source].append(doc)
        seen_texts.add(doc.text_sha256)

    for source in SOURCES:
        count = len(selected[source.name])
        if count != DOCS_PER_STRATUM:
            raise RuntimeError(
                f"{source.name}: selected {count}, expected {DOCS_PER_STRATUM}; no resample allowed"
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    split_counts: dict[str, Counter[str]] = {}
    for source in SOURCES:
        counts: Counter[str] = Counter()
        for doc in sorted(selected[source.name], key=lambda item: item.rank_sha256):
            split = split_for(doc.source, doc.document_id, doc.text_sha256)
            counts[split] += 1
            lines.append(
                json.dumps(
                    {
                        "source": doc.source,
                        "document_id": doc.document_id,
                        "text": doc.text,
                        "text_sha256": doc.text_sha256,
                        "rank_sha256": doc.rank_sha256,
                        "row_index": doc.row_index,
                        "split": split,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        split_counts[source.name] = counts

    CORPUS_PATH.write_text("\n".join(lines) + "\n")
    manifest: dict[str, Any] = {
        "schema": CORPUS_SCHEMA,
        "corpus_sha256": sha256_bytes(CORPUS_PATH.read_bytes()),
        "documents": len(lines),
        "documents_per_stratum": DOCS_PER_STRATUM,
        "blocks_per_stratum": BLOCKS_PER_STRATUM,
        "rows_per_block": ROWS_PER_BLOCK,
        "sources": {
            source.name: {
                "dataset": source.dataset,
                "config": source.config,
                "revision": source.revision,
                "rows": source.rows,
                "offsets": block_offsets(source),
                "candidate_rows": len(all_candidates[source.name]),
                "split_documents": dict(split_counts[source.name]),
            }
            for source in SOURCES
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_existing_corpus()


def load_documents() -> tuple[list[Document], dict[str, Any]]:
    try:
        from tokenizers import Tokenizer
        import tokenizers
    except ImportError as error:  # pragma: no cover - environment path
        raise RuntimeError("run with tokenizers==0.22.2") from error

    if tokenizers.__version__ != "0.22.2":
        raise RuntimeError(f"expected tokenizers 0.22.2, got {tokenizers.__version__}")
    if not TOKENIZER_PATH.exists():
        raise RuntimeError(f"missing pinned tokenizer at {TOKENIZER_PATH}")
    remote_revision = model_revision_sha(TOKENIZER_REPO, TOKENIZER_REVISION)
    if remote_revision != TOKENIZER_REVISION:
        raise RuntimeError(
            f"tokenizer revision mismatch: expected {TOKENIZER_REVISION}, got {remote_revision}"
        )
    tokenizer_sha = sha256_bytes(TOKENIZER_PATH.read_bytes())
    if tokenizer_sha != TOKENIZER_SHA256:
        raise RuntimeError(
            f"tokenizer content mismatch: expected {TOKENIZER_SHA256}, got {tokenizer_sha}"
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    documents: list[Document] = []
    stats: dict[str, Any] = {
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizers_version": tokenizers.__version__,
        "documents": defaultdict(lambda: Counter()),
        "tokens": defaultdict(lambda: Counter()),
    }
    with CORPUS_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            actual = sha256_text(row["text"])
            if actual != row["text_sha256"]:
                raise RuntimeError(f"text hash mismatch for {row['document_id']}")
            token_ids = tuple(
                tokenizer.encode(row["text"], add_special_tokens=False).ids[:MAX_TOKENS]
            )
            document = Document(
                row["source"], row["document_id"], row["split"], token_ids
            )
            documents.append(document)
            stats["documents"][document.source][document.split] += 1
            stats["tokens"][document.source][document.split] += len(token_ids)

    stats["documents"] = {
        source: dict(counts) for source, counts in stats["documents"].items()
    }
    stats["tokens"] = {
        source: dict(counts) for source, counts in stats["tokens"].items()
    }
    return documents, stats


def canonicalize(
    context: Sequence[int], literal_ids: frozenset[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Positive values are tagged literals; negative values are VAR indices."""
    template: list[int] = []
    bindings: list[int] = []
    binding_index: dict[int, int] = {}
    for token in context:
        if token in literal_ids:
            template.append(token + 1)
            continue
        index = binding_index.get(token)
        if index is None:
            index = len(bindings)
            binding_index[token] = index
            bindings.append(token)
        template.append(-(index + 1))
    return tuple(template), tuple(bindings)


def encode_operation(
    target: int,
    bindings: Sequence[int],
    literal_ids: frozenset[int],
    vocab_size: int,
) -> int:
    if target in literal_ids:
        return target
    try:
        return vocab_size + bindings.index(target)
    except ValueError:
        return -1


def decode_operation(
    operation: int,
    bindings: Sequence[int],
    literal_ids: frozenset[int],
    vocab_size: int,
) -> int | None:
    if operation < 0:
        return None
    if operation < vocab_size:
        return operation if operation in literal_ids else None
    index = operation - vocab_size
    if not 0 <= index < len(bindings):
        return None
    return bindings[index]


def wilson_lower(successes: int, total: int, z: float = WILSON_Z) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def quantized_confidence(successes: int, total: int) -> float:
    return math.floor(255.0 * wilson_lower(successes, total)) / 255.0


def iter_events(documents: Iterable[Document], n: int) -> Iterator[tuple[Document, int, tuple[int, ...], int]]:
    for document in documents:
        tokens = document.tokens
        for position in range(n, len(tokens)):
            yield document, position, tokens[position - n : position], tokens[position]


def add_outcome(
    counts: dict[tuple[int, ...], OutcomeCounts],
    key: tuple[int, ...],
    outcome: int,
) -> None:
    existing = counts.get(key)
    if existing is None:
        counts[key] = OutcomeCounts(outcome)
    else:
        existing.add(outcome)


def first_pass_counts(
    documents: Sequence[Document],
    n: int,
    mode: str,
    literal_ids: frozenset[int],
    vocab_size: int,
) -> dict[tuple[int, ...], OutcomeCounts]:
    counts: dict[tuple[int, ...], OutcomeCounts] = {}
    for _document, _position, context, target in iter_events(documents, n):
        template, bindings = canonicalize(context, literal_ids)
        if mode == "t38":
            key = template
            outcome = encode_operation(target, bindings, literal_ids, vocab_size)
        elif mode == "lexical":
            key = context
            outcome = target
        elif mode == "static_template":
            key = template
            outcome = target
        elif mode == "concrete_relative":
            key = context
            outcome = encode_operation(target, bindings, literal_ids, vocab_size)
        else:
            raise ValueError(mode)
        add_outcome(counts, key, outcome)
    return counts


def candidate_summaries(
    counts: Mapping[tuple[int, ...], OutcomeCounts], mode: str
) -> dict[tuple[int, ...], KeyStats]:
    summaries: dict[tuple[int, ...], KeyStats] = {}
    for key, counter in counts.items():
        total, winner, winner_count = counter.summary()
        if winner < 0:
            continue
        if mode in {"t38", "static_template"} and total < MIN_SUPPORT:
            continue
        stats = KeyStats(total, winner, winner_count)
        if mode == "t38" and stats.purity < T38_TRAIN_PURITY:
            continue
        summaries[key] = stats
    return summaries


def collect_candidate_metadata(
    documents: Sequence[Document],
    n: int,
    mode: str,
    candidates: Mapping[tuple[int, ...], KeyStats],
    literal_ids: frozenset[int],
) -> None:
    if not candidates:
        return
    for document, _position, context, _target in iter_events(documents, n):
        template, bindings = canonicalize(context, literal_ids)
        key = template if mode in {"t38", "static_template"} else context
        stats = candidates.get(key)
        if stats is None:
            continue
        stats.documents.add(f"{document.source}:{document.document_id}")
        if mode == "t38":
            stats.bindings.add(bindings)


def admit_records(
    summaries: Mapping[tuple[int, ...], KeyStats],
    n: int,
    mode: str,
) -> dict[tuple[int, ...], Record]:
    eligible = []
    for key, stats in summaries.items():
        if mode in {"t38", "static_template"} and len(stats.documents) < MIN_DOCUMENTS:
            continue
        eligible.append((key, stats))
    eligible.sort(
        key=lambda item: (
            -item[1].winner_count,
            -item[1].total,
            key_order_digest(item[0]),
        )
    )
    admitted: dict[tuple[int, ...], Record] = {}
    seen_fingerprints: dict[str, tuple[int, ...]] = {}
    for key, stats in eligible[:RECORD_CAP]:
        if stats.total > 0xFFFFFFFF:
            raise RuntimeError(f"32-bit support overflow in {mode} n={n}")
        fingerprint = key_fingerprint(key)
        prior = seen_fingerprints.get(fingerprint)
        if prior is not None and prior != key:
            raise RuntimeError(f"128-bit retained-key collision in {mode} n={n}")
        seen_fingerprints[fingerprint] = key
        admitted[key] = Record(
            n=n,
            key=key,
            outcome=stats.winner,
            total=stats.total,
            winner_count=stats.winner_count,
            confidence_lcb=quantized_confidence(stats.winner_count, stats.total),
            order_digest=key_order_digest(key),
            fingerprint=fingerprint,
            train_bindings=frozenset(stats.bindings),
        )
    return admitted


def compile_head(
    train_documents: Sequence[Document],
    n: int,
    mode: str,
    literal_ids: frozenset[int],
    vocab_size: int,
) -> dict[tuple[int, ...], Record]:
    counts = first_pass_counts(train_documents, n, mode, literal_ids, vocab_size)
    summaries = candidate_summaries(counts, mode)
    del counts
    collect_candidate_metadata(train_documents, n, mode, summaries, literal_ids)
    return admit_records(summaries, n, mode)


def compile_systems(
    train_documents: Sequence[Document],
    literal_ids: frozenset[int],
    vocab_size: int,
) -> dict[str, dict[int, dict[tuple[int, ...], Record]]]:
    systems: dict[str, dict[int, dict[tuple[int, ...], Record]]] = {
        mode: {} for mode in ("t38", "lexical", "static_template", "concrete_relative")
    }
    for n in WINDOWS:
        for mode in systems:
            systems[mode][n] = compile_head(
                train_documents, n, mode, literal_ids, vocab_size
            )
    systems["shuffled_payload"] = shuffle_payloads(systems["t38"], vocab_size)
    systems["shuffled_binding"] = systems["t38"]
    return systems


def shuffle_payloads(
    heads: Mapping[int, Mapping[tuple[int, ...], Record]], vocab_size: int
) -> dict[int, dict[tuple[int, ...], Record]]:
    shuffled: dict[int, dict[tuple[int, ...], Record]] = {}
    rng = random.Random(SHUFFLE_PAYLOAD_SEED)
    for n in WINDOWS:
        ordered = sorted(heads[n].values(), key=lambda record: record.order_digest)
        grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
        outcomes = [record.outcome for record in ordered]
        for index, record in enumerate(ordered):
            arity = max((-component for component in record.key if component < 0), default=0)
            group = (
                ("copy", arity)
                if record.outcome >= vocab_size
                else ("literal", 0)
            )
            grouped_indices[group].append(index)
        shuffled_outcomes = outcomes.copy()
        for indices in grouped_indices.values():
            group_outcomes = [outcomes[index] for index in indices]
            rng.shuffle(group_outcomes)
            for index, outcome in zip(indices, group_outcomes, strict=True):
                shuffled_outcomes[index] = outcome
        shuffled[n] = {
            record.key: Record(
                n=record.n,
                key=record.key,
                outcome=outcome,
                total=record.total,
                winner_count=record.winner_count,
                confidence_lcb=record.confidence_lcb,
                order_digest=record.order_digest,
                fingerprint=record.fingerprint,
                train_bindings=record.train_bindings,
            )
            for record, outcome in zip(ordered, shuffled_outcomes, strict=True)
        }
    return shuffled


def permute_bindings(
    bindings: tuple[int, ...], document: Document, position: int
) -> tuple[int, ...]:
    if len(bindings) < 2:
        return bindings
    material = (
        f"{SHUFFLE_BINDING_SEED}|{document.source}|{document.document_id}|{position}"
    ).encode()
    offset = 1 + int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (
        len(bindings) - 1
    )
    return bindings[offset:] + bindings[:offset]


def head_prediction(
    mode: str,
    record: Record,
    bindings: tuple[int, ...],
    document: Document,
    position: int,
    literal_ids: frozenset[int],
    vocab_size: int,
) -> Prediction | None:
    used_bindings = bindings
    if mode == "shuffled_binding":
        used_bindings = permute_bindings(bindings, document, position)
    if mode in {"t38", "concrete_relative", "shuffled_payload", "shuffled_binding"}:
        token = decode_operation(
            record.outcome, used_bindings, literal_ids, vocab_size
        )
        if token is None:
            return None
        return Prediction(token, record, record.outcome >= vocab_size, bindings)
    return Prediction(record.outcome, record, False, bindings)


def predict(
    mode: str,
    heads: Mapping[int, Mapping[tuple[int, ...], Record]],
    tokens: tuple[int, ...],
    position: int,
    document: Document,
    literal_ids: frozenset[int],
    vocab_size: int,
) -> Prediction | None:
    options: list[Prediction] = []
    for n in WINDOWS:
        if position < n:
            continue
        context = tokens[position - n : position]
        template, bindings = canonicalize(context, literal_ids)
        key = template if mode in {"t38", "static_template", "shuffled_payload", "shuffled_binding"} else context
        record = heads[n].get(key)
        if record is None:
            continue
        prediction = head_prediction(
            mode, record, bindings, document, position, literal_ids, vocab_size
        )
        if prediction is not None:
            options.append(prediction)
    if not options:
        return None
    return min(
        options,
        key=lambda prediction: (
            -prediction.record.confidence_lcb,
            -prediction.record.total,
            -prediction.record.n,
        ),
    )


def new_metric() -> Counter[str]:
    return Counter(
        {
            "events": 0,
            "correct": 0,
            "wrong": 0,
            "abstain": 0,
            "copy_correct": 0,
            "copy_predictions": 0,
            "unseen_binding_events": 0,
            "unseen_binding_correct": 0,
            "bind_sensitive_copy_predictions": 0,
            "bind_sensitive_copy_correct": 0,
        }
    )


def evaluate_systems(
    documents: Sequence[Document],
    systems: Mapping[str, Mapping[int, Mapping[tuple[int, ...], Record]]],
    literal_ids: frozenset[int],
    vocab_size: int,
) -> tuple[
    dict[str, dict[str, Counter[str]]],
    dict[str, dict[str, tuple[int, int]]],
    dict[str, Counter[str]],
]:
    metrics: dict[str, dict[str, Counter[str]]] = {
        mode: {source.name: new_metric() for source in SOURCES} for mode in systems
    }
    document_scores: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    cell_scores: dict[str, Counter[str]] = defaultdict(Counter)

    for document in documents:
        per_mode_correct: Counter[str] = Counter()
        events = 0
        for position in range(min(WINDOWS), len(document.tokens)):
            target = document.tokens[position]
            events += 1
            for mode, heads in systems.items():
                metric = metrics[mode][document.source]
                metric["events"] += 1
                prediction = predict(
                    mode,
                    heads,
                    document.tokens,
                    position,
                    document,
                    literal_ids,
                    vocab_size,
                )
                if prediction is None:
                    metric["abstain"] += 1
                    continue
                correct = prediction.token == target
                metric["correct" if correct else "wrong"] += 1
                per_mode_correct[mode] += int(correct)
                if prediction.is_copy:
                    metric["copy_predictions"] += 1
                    metric["copy_correct"] += int(correct)
                    if len(prediction.binding) >= 2:
                        metric["bind_sensitive_copy_predictions"] += 1
                        metric["bind_sensitive_copy_correct"] += int(correct)
                if mode == "t38":
                    cell_id = f"{prediction.record.n}:{prediction.record.fingerprint}"
                    cell_scores[cell_id]["events"] += 1
                    cell_scores[cell_id]["correct"] += int(correct)
                    unseen = prediction.binding not in prediction.record.train_bindings
                    metric["unseen_binding_events"] += int(unseen)
                    metric["unseen_binding_correct"] += int(unseen and correct)
        doc_key = f"{document.source}:{document.document_id}"
        for mode in systems:
            document_scores[mode][doc_key] = (per_mode_correct[mode], events)
    return metrics, document_scores, cell_scores


def score_advantage(
    candidate: Mapping[str, Counter[str]],
    baseline: Mapping[str, Counter[str]],
) -> tuple[dict[str, float], float]:
    by_source: dict[str, float] = {}
    for source in (item.name for item in SOURCES):
        events = candidate[source]["events"]
        if events != baseline[source]["events"] or events <= 0:
            raise RuntimeError(f"event mismatch for {source}")
        by_source[source] = (
            candidate[source]["correct"] - baseline[source]["correct"]
        ) / events
    return by_source, statistics.fmean(by_source.values())


def bootstrap_advantage(
    adjudication_documents: Sequence[Document],
    document_scores: Mapping[str, Mapping[str, tuple[int, int]]],
    candidate_mode: str = "t38",
    baseline_mode: str = "lexical",
) -> tuple[float, float]:
    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for document in adjudication_documents:
        key = f"{document.source}:{document.document_id}"
        cand_correct, cand_events = document_scores[candidate_mode][key]
        base_correct, base_events = document_scores[baseline_mode][key]
        if cand_events != base_events:
            raise RuntimeError("document event mismatch")
        by_source[document.source].append((cand_correct - base_correct, cand_events))

    rng = random.Random(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAPS):
        source_scores = []
        for source in (item.name for item in SOURCES):
            rows = by_source[source]
            difference = 0
            events = 0
            for _index in range(len(rows)):
                row_difference, row_events = rows[rng.randrange(len(rows))]
                difference += row_difference
                events += row_events
            source_scores.append(difference / events)
        samples.append(statistics.fmean(source_scores))
    samples.sort()
    lower = samples[max(0, math.ceil(0.025 * BOOTSTRAPS) - 1)]
    upper = samples[math.ceil(0.975 * BOOTSTRAPS) - 1]
    return lower, upper


def quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def serialise_metrics(
    metrics: Mapping[str, Mapping[str, Counter[str]]]
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for mode, sources in metrics.items():
        result[mode] = {}
        for source, counter in sources.items():
            row: dict[str, float | int] = dict(counter)
            events = counter["events"]
            matches = counter["correct"] + counter["wrong"]
            row["all_event_accuracy"] = counter["correct"] / events if events else 0.0
            row["matched_purity"] = counter["correct"] / matches if matches else 0.0
            row["coverage"] = matches / events if events else 0.0
            unseen = counter["unseen_binding_events"]
            row["unseen_binding_purity"] = (
                counter["unseen_binding_correct"] / unseen if unseen else 0.0
            )
            result[mode][source] = row
    return result


def evaluate_head_diagnostics(
    documents: Sequence[Document],
    systems: Mapping[str, Mapping[int, Mapping[tuple[int, ...], Record]]],
    literal_ids: frozenset[int],
    vocab_size: int,
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    counters: dict[str, dict[str, dict[int, Counter[str]]]] = {
        mode: {
            source.name: {n: new_metric() for n in WINDOWS} for source in SOURCES
        }
        for mode in systems
    }
    matched_cells: dict[tuple[str, str, int], set[str]] = defaultdict(set)

    for document in documents:
        for position in range(min(WINDOWS), len(document.tokens)):
            target = document.tokens[position]
            for n in WINDOWS:
                if position < n:
                    continue
                context = document.tokens[position - n : position]
                template, bindings = canonicalize(context, literal_ids)
                for mode, heads in systems.items():
                    metric = counters[mode][document.source][n]
                    metric["events"] += 1
                    key = (
                        template
                        if mode
                        in {
                            "t38",
                            "static_template",
                            "shuffled_payload",
                            "shuffled_binding",
                        }
                        else context
                    )
                    record = heads[n].get(key)
                    if record is None:
                        metric["abstain"] += 1
                        continue
                    prediction = head_prediction(
                        mode,
                        record,
                        bindings,
                        document,
                        position,
                        literal_ids,
                        vocab_size,
                    )
                    if prediction is None:
                        metric["abstain"] += 1
                        continue
                    matched_cells[(mode, document.source, n)].add(record.fingerprint)
                    correct = prediction.token == target
                    metric["correct" if correct else "wrong"] += 1
                    if prediction.is_copy:
                        metric["copy_predictions"] += 1
                        metric["copy_correct"] += int(correct)
                        if len(prediction.binding) >= 2:
                            metric["bind_sensitive_copy_predictions"] += 1
                            metric["bind_sensitive_copy_correct"] += int(correct)
                    if mode == "t38":
                        unseen = bindings not in record.train_bindings
                        metric["unseen_binding_events"] += int(unseen)
                        metric["unseen_binding_correct"] += int(unseen and correct)

    result: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for mode, sources in counters.items():
        result[mode] = {}
        for source, heads in sources.items():
            result[mode][source] = {}
            for n, counter in heads.items():
                events = counter["events"]
                matches = counter["correct"] + counter["wrong"]
                unseen = counter["unseen_binding_events"]
                row: dict[str, float | int] = dict(counter)
                row.update(
                    {
                        "matched_cells": len(matched_cells[(mode, source, n)]),
                        "all_event_accuracy": counter["correct"] / events
                        if events
                        else 0.0,
                        "matched_purity": counter["correct"] / matches
                        if matches
                        else 0.0,
                        "coverage": matches / events if events else 0.0,
                        "unseen_binding_purity": counter["unseen_binding_correct"]
                        / unseen
                        if unseen
                        else 0.0,
                    }
                )
                result[mode][source][str(n)] = row
    return result


def record_diagnostics(
    systems: Mapping[str, Mapping[int, Mapping[tuple[int, ...], Record]]]
) -> tuple[dict[str, dict[str, dict[str, float | int]]], list[dict[str, Any]]]:
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    t38_records: list[dict[str, Any]] = []
    for mode, heads in systems.items():
        summaries[mode] = {}
        for n in WINDOWS:
            records = list(heads[n].values())
            total = sum(record.total for record in records)
            winners = sum(record.winner_count for record in records)
            summaries[mode][str(n)] = {
                "records": len(records),
                "train_occurrences": total,
                "train_winner_occurrences": winners,
                "aggregate_train_purity": winners / total if total else 0.0,
                "at_record_cap": int(len(records) == RECORD_CAP),
            }
            if mode == "t38":
                for record in sorted(records, key=lambda item: item.order_digest):
                    t38_records.append(
                        {
                            "n": n,
                            "fingerprint": record.fingerprint,
                            "train_occurrences": record.total,
                            "train_winner_occurrences": record.winner_count,
                            "train_purity": record.winner_count / record.total,
                            "train_binding_tuples": len(record.train_bindings),
                        }
                    )
    return summaries, t38_records


def run_algebra_self_check() -> dict[str, int | bool]:
    contexts_checked = 0
    operations_checked = 0
    bottom_operations_checked = 0
    converse_pairs_checked = 0
    partitions_checked = 0

    for vocab_size in range(1, 5):
        vocabulary = tuple(range(vocab_size))
        for literal_mask in range(1 << vocab_size):
            literal_ids = frozenset(
                token
                for token in vocabulary
                if literal_mask & (1 << token)
            )
            eligible = tuple(token for token in vocabulary if token not in literal_ids)
            permutations = [
                dict(zip(eligible, values, strict=True))
                for values in itertools.permutations(eligible)
            ]
            for eligible_token in eligible:
                if (
                    decode_operation(
                        eligible_token, (), literal_ids, vocab_size
                    )
                    is not None
                ):
                    raise RuntimeError("T38 invalid-literal self-check failed")
            partitions_checked += 1
            for length in range(1, 5):
                contexts = tuple(itertools.product(vocabulary, repeat=length))
                templates = {
                    context: canonicalize(context, literal_ids)[0]
                    for context in contexts
                }
                for permutation in permutations:
                    for context in contexts:
                        template, bindings = canonicalize(context, literal_ids)
                        renamed_context = tuple(
                            permutation.get(token, token) for token in context
                        )
                        renamed_template, renamed_bindings = canonicalize(
                            renamed_context, literal_ids
                        )
                        expected_bindings = tuple(
                            permutation.get(token, token) for token in bindings
                        )
                        if (
                            renamed_template != template
                            or renamed_bindings != expected_bindings
                        ):
                            raise RuntimeError(
                                "T38 canonicalization self-check failed"
                            )
                        contexts_checked += 1
                        for target in vocabulary:
                            operation = encode_operation(
                                target, bindings, literal_ids, vocab_size
                            )
                            renamed_target = permutation.get(target, target)
                            renamed_operation = encode_operation(
                                renamed_target,
                                renamed_bindings,
                                literal_ids,
                                vocab_size,
                            )
                            if renamed_operation != operation:
                                raise RuntimeError(
                                    "T38 operation covariance self-check failed"
                                )
                            if operation < 0:
                                if (
                                    decode_operation(
                                        operation,
                                        renamed_bindings,
                                        literal_ids,
                                        vocab_size,
                                    )
                                    is not None
                                ):
                                    raise RuntimeError(
                                        "T38 bottom-operation self-check failed"
                                    )
                                bottom_operations_checked += 1
                                continue
                            if (
                                decode_operation(
                                    operation,
                                    renamed_bindings,
                                    literal_ids,
                                    vocab_size,
                                )
                                != renamed_target
                            ):
                                raise RuntimeError(
                                    "T38 equivariance self-check failed"
                                )
                            operations_checked += 1
                for left in contexts:
                    for right in contexts:
                        same_template = templates[left] == templates[right]
                        same_orbit = any(
                            tuple(
                                permutation.get(token, token) for token in left
                            )
                            == right
                            for permutation in permutations
                        )
                        if same_template != same_orbit:
                            raise RuntimeError(
                                "T38 converse-orbit self-check failed"
                            )
                        converse_pairs_checked += 1

    literal_ids = frozenset({0})

    deterministic_candidates = {
        (1,): KeyStats(20, 7, 20, {"a", "b", "c", "d"}),
        (2,): KeyStats(30, 8, 25, {"a", "b", "c", "d"}),
    }
    first = admit_records(deterministic_candidates, 8, "static_template")
    second = admit_records(
        dict(reversed(tuple(deterministic_candidates.items()))),
        8,
        "static_template",
    )
    if list(first) != list(second):
        raise RuntimeError("T38 deterministic-admission self-check failed")

    witness_documents = []
    for doc_index, first_token in enumerate((1, 4, 7, 10)):
        context = (
            first_token,
            0,
            first_token + 1,
            0,
            first_token + 2,
            0,
            first_token,
            0,
        )
        witness_documents.append(
            Document(
                "code",
                f"witness-{doc_index}",
                "train",
                tuple((*context, first_token) * 4),
            )
        )
    compiled_a = compile_head(witness_documents, 8, "t38", literal_ids, 64)
    compiled_b = compile_head(
        tuple(reversed(witness_documents)), 8, "t38", literal_ids, 64
    )
    replay_a = [
        (key, record.outcome, record.total, record.winner_count, record.fingerprint)
        for key, record in compiled_a.items()
    ]
    replay_b = [
        (key, record.outcome, record.total, record.winner_count, record.fingerprint)
        for key, record in compiled_b.items()
    ]
    if replay_a != replay_b:
        raise RuntimeError("T38 deterministic-compiler replay failed")

    candidate = {
        "prose": Counter(events=100, correct=15),
        "math": Counter(events=200, correct=40),
        "code": Counter(events=50, correct=5),
    }
    baseline = {
        "prose": Counter(events=100, correct=10),
        "math": Counter(events=200, correct=30),
        "code": Counter(events=50, correct=5),
    }
    by_source, aggregate = score_advantage(candidate, baseline)
    if by_source != {"prose": 0.05, "math": 0.05, "code": 0.0}:
        raise RuntimeError("T38 repair-minus-harm source score failed")
    if abs(aggregate - 1.0 / 30.0) > 1e-15:
        raise RuntimeError("T38 repair-minus-harm aggregate failed")

    return {
        "pass": True,
        "contexts_checked": contexts_checked,
        "operations_checked": operations_checked,
        "bottom_operations_checked": bottom_operations_checked,
        "converse_pairs_checked": converse_pairs_checked,
        "literal_partitions_checked": partitions_checked,
        "deterministic_admission_checked": True,
        "deterministic_compiler_replay_checked": True,
        "repair_minus_harm_checked": True,
    }


def run_test_receipt() -> dict[str, Any]:
    pytest_version = importlib.metadata.version("pytest")
    if pytest_version != "9.0.2":
        raise RuntimeError(f"expected pytest 9.0.2, got {pytest_version}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(ROOT / "tests/test_equivariant_template_engram_t38_census.py"),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    receipt = {
        "command": command,
        "returncode": completed.returncode,
        "pytest_version": pytest_version,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"sealed T38 pytest failed: {receipt}")
    return receipt


def create_adjudication_seal() -> dict[str, Any]:
    preregistration = (
        ROOT / "results/equivariant-template-engram-t38-cpu-census-preregistration.md"
    )
    implementation = Path(__file__).resolve()
    test_file = ROOT / "tests/test_equivariant_template_engram_t38_census.py"
    payload = {
        "schema": "equivariant-template-engram-t38-adjudication-open-v1",
        "opened_unix_seconds": time.time(),
        "preregistration_sha256": sha256_bytes(preregistration.read_bytes()),
        "implementation_sha256": sha256_bytes(implementation.read_bytes()),
        "tests_sha256": sha256_bytes(test_file.read_bytes()),
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(ADJUDICATION_OPENED_PATH, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return payload


def finalize_adjudication_seal(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "equivariant-template-engram-t38-adjudication-final-v1",
        "opening_seal_sha256": sha256_bytes(ADJUDICATION_OPENED_PATH.read_bytes()),
        "corpus_sha256": manifest["corpus_sha256"],
        "manifest_sha256": sha256_bytes(MANIFEST_PATH.read_bytes()),
        "finalized_unix_seconds": time.time(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(ADJUDICATION_FINALIZED_PATH, flags, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def write_result_atomic(payload: Mapping[str, Any]) -> None:
    temporary = OUTPUT_PATH.with_name(
        f".{OUTPUT_PATH.name}.tmp-{os.getpid()}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o644)
    linked = False
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, OUTPUT_PATH)
        linked = True
        directory = os.open(OUTPUT_PATH.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not linked:
        raise RuntimeError("failed to publish T38 result atomically")


def run_census() -> dict[str, Any]:
    if (
        ADJUDICATION_OPENED_PATH.exists()
        or ADJUDICATION_FINALIZED_PATH.exists()
        or OUTPUT_PATH.exists()
        or CORPUS_PATH.exists()
        or MANIFEST_PATH.exists()
    ):
        raise RuntimeError(
            "T38 corpus/adjudication/result already exists; refusing unsealed reuse, rerun, or overwrite"
        )
    self_check = run_algebra_self_check()
    test_receipt = run_test_receipt()
    seal = create_adjudication_seal()
    manifest = materialize_corpus()
    finalized_seal = finalize_adjudication_seal(manifest)
    documents, tokenization = load_documents()
    documents_by_split = {
        split: [document for document in documents if document.split == split]
        for split in ("train", "development", "adjudication")
    }
    if any(not split_documents for split_documents in documents_by_split.values()):
        raise RuntimeError("empty train, development, or adjudication split")
    train = documents_by_split["train"]
    adjudication = documents_by_split["adjudication"]

    train_frequency: Counter[int] = Counter(
        token for document in train for token in document.tokens
    )
    vocab_size = int(tokenization["tokenizer_vocab_size"])
    ranked_tokens = sorted(range(vocab_size), key=lambda token: (-train_frequency[token], token))
    literal_ids = frozenset(ranked_tokens[:LITERAL_COUNT])

    started = time.perf_counter()
    systems = compile_systems(train, literal_ids, vocab_size)
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    split_metrics: dict[str, dict[str, dict[str, Counter[str]]]] = {}
    split_serialised: dict[str, Any] = {}
    split_head_metrics: dict[str, Any] = {}
    for split in ("train", "development"):
        split_result, _document_scores, _cell_scores = evaluate_systems(
            documents_by_split[split], systems, literal_ids, vocab_size
        )
        split_metrics[split] = split_result
        split_serialised[split] = serialise_metrics(split_result)
        split_head_metrics[split] = evaluate_head_diagnostics(
            documents_by_split[split], systems, literal_ids, vocab_size
        )

    metrics, document_scores, cell_scores = evaluate_systems(
        adjudication, systems, literal_ids, vocab_size
    )
    split_metrics["adjudication"] = metrics
    split_serialised["adjudication"] = serialise_metrics(metrics)
    split_head_metrics["adjudication"] = evaluate_head_diagnostics(
        adjudication, systems, literal_ids, vocab_size
    )
    evaluate_seconds = time.perf_counter() - started

    advantages: dict[str, Any] = {}
    for mode in systems:
        if mode == "lexical":
            continue
        by_source, aggregate = score_advantage(metrics[mode], metrics["lexical"])
        advantages[mode] = {"by_source": by_source, "equal_stratum": aggregate}

    bootstrap_lower, bootstrap_upper = bootstrap_advantage(
        adjudication, document_scores
    )

    t38_records = [
        record for n in WINDOWS for record in systems["t38"][n].values()
    ]
    binding_counts = [len(record.train_bindings) for record in t38_records]
    active_cells = [
        score for score in cell_scores.values() if score["events"] >= 20
    ]
    active_cell_min_purity = min(
        (score["correct"] / score["events"] for score in active_cells),
        default=0.0,
    )

    record_summaries, t38_record_details = record_diagnostics(systems)
    t38_matches = sum(
        row["correct"] + row["wrong"] for row in metrics["t38"].values()
    )
    t38_correct = sum(row["correct"] for row in metrics["t38"].values())
    unseen_events = sum(
        row["unseen_binding_events"] for row in metrics["t38"].values()
    )
    unseen_correct = sum(
        row["unseen_binding_correct"] for row in metrics["t38"].values()
    )
    copy_coverages = []
    for source in ("math", "code"):
        row = metrics["t38"][source]
        copy_coverages.append(row["copy_correct"] / row["events"])
    copy_coverage = statistics.fmean(copy_coverages)

    t38_advantage = advantages["t38"]["equal_stratum"]
    shuffled_payload_advantage = advantages["shuffled_payload"]["equal_stratum"]
    payload_advantage_loss = (
        1.0 - max(0.0, shuffled_payload_advantage) / t38_advantage
        if t38_advantage > 0.0
        else 0.0
    )
    bind_sensitive_t38_correct = sum(
        row["bind_sensitive_copy_correct"] for row in metrics["t38"].values()
    )
    bind_sensitive_shuffled_correct = sum(
        row["bind_sensitive_copy_correct"]
        for row in metrics["shuffled_binding"].values()
    )
    binding_correct_loss = (
        1.0 - bind_sensitive_shuffled_correct / bind_sensitive_t38_correct
        if bind_sensitive_t38_correct > 0
        else 0.0
    )
    shuffle_losses = {
        "shuffled_payload_aggregate_advantage_loss": payload_advantage_loss,
        "shuffled_binding_bind_sensitive_correct_loss": binding_correct_loss,
        "bind_sensitive_t38_correct": bind_sensitive_t38_correct,
        "bind_sensitive_shuffled_correct": bind_sensitive_shuffled_correct,
    }

    gates = {
        "advantage_at_least_3pp": t38_advantage >= 0.03,
        "bootstrap_lower_above_2pp": bootstrap_lower > 0.02,
        "aggregate_purity_at_least_97pct": (
            t38_correct / t38_matches >= 0.97 if t38_matches else False
        ),
        "unseen_binding_purity_at_least_97pct": (
            unseen_correct / unseen_events >= 0.97 if unseen_events else False
        ),
        "active_cell_min_purity_at_least_95pct": (
            bool(active_cells) and active_cell_min_purity >= 0.95
        ),
        "median_bindings_at_least_8": quantile(binding_counts, 0.5) >= 8.0,
        "p75_bindings_at_least_16": quantile(binding_counts, 0.75) >= 16.0,
        "math_code_copy_coverage_at_least_3pct": copy_coverage >= 0.03,
        "shuffled_payload_loses_90pct": payload_advantage_loss >= 0.90,
        "shuffled_binding_loses_90pct_of_bind_sensitive_correct": (
            binding_correct_loss >= 0.90
        ),
        "integrity_and_algebra_checks_pass": (
            bool(self_check["pass"])
            and test_receipt["returncode"] == 0
            and manifest["schema"] == CORPUS_SCHEMA
        ),
    }

    result: dict[str, Any] = {
        "schema": "equivariant-template-engram-t38-cpu-census-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "claim_boundary": (
            "CPU decoded-memory opportunity census only; no model capability, "
            "benchmark, physical serving, or GPU claim."
        ),
        "adjudication_seal": seal,
        "adjudication_finalization": finalized_seal,
        "self_check": self_check,
        "test_receipt": test_receipt,
        "preregistration": str(
            ROOT / "results/equivariant-template-engram-t38-cpu-census-preregistration.md"
        ),
        "manifest": manifest,
        "tokenization": tokenization,
        "constants": {
            "windows": WINDOWS,
            "literal_count": LITERAL_COUNT,
            "record_cap_per_head": RECORD_CAP,
            "logical_record_bytes": 24,
            "logical_total_bytes": len(WINDOWS) * RECORD_CAP * 24,
            "min_support": MIN_SUPPORT,
            "min_documents": MIN_DOCUMENTS,
            "t38_train_purity": T38_TRAIN_PURITY,
            "bootstraps": BOOTSTRAPS,
            "fingerprint_union_bound_at_1e12_queries": (
                len(WINDOWS) * RECORD_CAP * 1_000_000_000_000 / 2**128
            ),
        },
        "records": {
            mode: {str(n): len(heads[n]) for n in WINDOWS}
            for mode, heads in systems.items()
        },
        "record_diagnostics": record_summaries,
        "t38_record_details": t38_record_details,
        "metrics_by_split": split_serialised,
        "head_metrics_by_split": split_head_metrics,
        "adjudication_t38_cell_metrics": [
            {
                "cell": cell,
                "events": score["events"],
                "correct": score["correct"],
                "purity": score["correct"] / score["events"],
            }
            for cell, score in sorted(cell_scores.items())
        ],
        "advantages_vs_lexical": advantages,
        "bootstrap_95pct": {
            "lower": bootstrap_lower,
            "upper": bootstrap_upper,
        },
        "t38": {
            "aggregate_matched_purity": t38_correct / t38_matches if t38_matches else 0.0,
            "unseen_binding_events": unseen_events,
            "unseen_binding_purity": unseen_correct / unseen_events if unseen_events else 0.0,
            "active_cells_at_least_20_events": len(active_cells),
            "active_cell_min_purity": active_cell_min_purity,
            "binding_tuple_median": quantile(binding_counts, 0.5),
            "binding_tuple_p75": quantile(binding_counts, 0.75),
            "math_code_correct_copy_coverage": copy_coverage,
            "shuffle_advantage_losses": shuffle_losses,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "timing_seconds": {
            "compile": compile_seconds,
            "evaluate": evaluate_seconds,
        },
    }
    write_result_atomic(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run_census()
    print(
        json.dumps(
            {
                "status": result["status"],
                "all_gates_pass": result["all_gates_pass"],
                "failed_gates": [
                    name for name, passed in result["gates"].items() if not passed
                ],
                "output": str(OUTPUT_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
