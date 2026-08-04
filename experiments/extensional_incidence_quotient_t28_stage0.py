from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CORPUS_SHA256 = "a02d2bdc714bae0c6e42d8527333de01506120845095aec936a301634fb7d00a"
MAX_VALUE_TOKENS = 4
MAX_VALUE_DOCUMENT_FRACTION = 20
MAX_FACTS_PER_DOCUMENT = 24
MAX_RELATIONS = 4_096
MAX_VALUES = 65_536
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_RE = re.compile(r"\w", flags=re.UNICODE)
SENTENCE_END = frozenset({".", "?", "!"})

Pattern = tuple[str, ...]
Value = tuple[str, ...]
TupleKey = tuple[str, Value]
Edge = tuple[TupleKey, Pattern]


@dataclass(frozen=True)
class Witness:
    document_index: int
    title: str
    sentence_index: int
    value_start: int
    value_end: int
    subject_spans: tuple[tuple[int, int], ...]
    source_tokens: tuple[str, ...]

    def ordering_key(self) -> tuple[Any, ...]:
        return (
            self.title.casefold(),
            self.sentence_index,
            self.value_start,
            self.value_end,
            self.source_tokens,
        )


@dataclass(frozen=True)
class CorpusDocument:
    original_index: int
    title: str
    title_tokens: tuple[str, ...]
    sentences: tuple[tuple[str, ...], ...]


class UnionFind:
    def __init__(self, items: Iterable[Pattern]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: Pattern) -> Pattern:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: Pattern, right: Pattern) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_RE.findall(text))


def is_lexical(token: str) -> bool:
    return WORD_RE.search(token) is not None


def split_sentences(tokens: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    sentences: list[tuple[str, ...]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token in SENTENCE_END:
            if index + 1 > start:
                sentences.append(tuple(tokens[start : index + 1]))
            start = index + 1
    if start < len(tokens):
        sentences.append(tuple(tokens[start:]))
    return tuple(sentence for sentence in sentences if sentence)


def all_lexical_spans(tokens: Sequence[str]) -> Iterable[tuple[int, int, Value]]:
    for start in range(len(tokens)):
        if not is_lexical(tokens[start]):
            continue
        for width in range(1, MAX_VALUE_TOKENS + 1):
            end = start + width
            if end > len(tokens):
                break
            span = tuple(tokens[start:end])
            if not all(is_lexical(token) for token in span):
                break
            yield start, end, span


def nonoverlapping_exact_occurrences(
    tokens: Sequence[str], needle: Sequence[str]
) -> tuple[tuple[int, int], ...]:
    if not needle:
        return ()
    spans: list[tuple[int, int]] = []
    index = 0
    width = len(needle)
    while index + width <= len(tokens):
        if tuple(tokens[index : index + width]) == tuple(needle):
            spans.append((index, index + width))
            index += width
        else:
            index += 1
    return tuple(spans)


def overlaps(span: tuple[int, int], others: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in others)


def canonical_pattern(
    tokens: Sequence[str],
    subject_spans: Sequence[tuple[int, int]],
    value_span: tuple[int, int],
) -> Pattern:
    subject_by_start = {start: end for start, end in subject_spans}
    output: list[str] = []
    index = 0
    while index < len(tokens):
        if index in subject_by_start:
            output.append("<S>")
            index = subject_by_start[index]
        elif index == value_span[0]:
            output.append("<V>")
            index = value_span[1]
        else:
            output.append(tokens[index])
            index += 1
    return tuple(output)


def load_documents(path: Path) -> tuple[CorpusDocument, ...]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    documents: list[CorpusDocument] = []
    for index, row in enumerate(rows):
        if not {"title", "text"}.issubset(row):
            raise RuntimeError(f"required raw fields missing at row {index}: {sorted(row)}")
        title = row["title"]
        text = row["text"]
        if not isinstance(title, str) or not isinstance(text, str):
            raise RuntimeError(f"non-string raw field at row {index}")
        documents.append(
            CorpusDocument(
                original_index=index,
                title=title,
                title_tokens=tokenize(title),
                sentences=split_sentences(tokenize(text)),
            )
        )
    if len({document.title for document in documents}) != len(documents):
        raise RuntimeError("document titles are not unique")
    return tuple(documents)


def value_document_frequency(documents: Sequence[CorpusDocument]) -> Counter[Value]:
    counts: Counter[Value] = Counter()
    for document in documents:
        observed: set[Value] = set()
        for sentence in document.sentences:
            observed.update(span for _, _, span in all_lexical_spans(sentence))
        counts.update(observed)
    return counts


def enumerate_edges(
    documents: Sequence[CorpusDocument],
    frequencies: Counter[Value],
) -> tuple[dict[Edge, Witness], dict[str, int]]:
    maximum_df = math.floor(len(documents) / MAX_VALUE_DOCUMENT_FRACTION)
    edges: dict[Edge, Witness] = {}
    diagnostics = {
        "sentences": 0,
        "tokens": 0,
        "eligible_subject_sentences": 0,
        "raw_emitted_edges": 0,
        "witness_regeneration_failures": 0,
    }
    for document in documents:
        for sentence_index, sentence in enumerate(document.sentences):
            diagnostics["sentences"] += 1
            diagnostics["tokens"] += len(sentence)
            subject_spans = nonoverlapping_exact_occurrences(
                sentence, document.title_tokens
            )
            if not subject_spans:
                continue
            diagnostics["eligible_subject_sentences"] += 1
            for start, end, value in all_lexical_spans(sentence):
                if not 2 <= frequencies[value] <= maximum_df:
                    continue
                value_span = (start, end)
                if overlaps(value_span, subject_spans):
                    continue
                pattern = canonical_pattern(sentence, subject_spans, value_span)
                tuple_key: TupleKey = (document.title.casefold(), value)
                edge: Edge = (tuple_key, pattern)
                witness = Witness(
                    document_index=document.original_index,
                    title=document.title,
                    sentence_index=sentence_index,
                    value_start=start,
                    value_end=end,
                    subject_spans=subject_spans,
                    source_tokens=sentence,
                )
                diagnostics["raw_emitted_edges"] += 1
                if tuple(sentence[start:end]) != value or any(
                    tuple(sentence[left:right]) != document.title_tokens
                    for left, right in subject_spans
                ):
                    diagnostics["witness_regeneration_failures"] += 1
                previous = edges.get(edge)
                if previous is None or witness.ordering_key() < previous.ordering_key():
                    edges[edge] = witness
    return edges, diagnostics


def inverted_graph(
    edges: Iterable[Edge],
) -> tuple[dict[TupleKey, set[Pattern]], set[Pattern], set[tuple[Pattern, Pattern]]]:
    tuple_to_patterns: dict[TupleKey, set[Pattern]] = defaultdict(set)
    pattern_to_tuples: dict[Pattern, set[TupleKey]] = defaultdict(set)
    for tuple_key, pattern in edges:
        tuple_to_patterns[tuple_key].add(pattern)
        pattern_to_tuples[pattern].add(tuple_key)
    nodes = {
        pattern for pattern, neighbors in pattern_to_tuples.items() if len(neighbors) >= 2
    }
    graph_edges: set[tuple[Pattern, Pattern]] = set()
    for patterns in tuple_to_patterns.values():
        eligible = sorted(pattern for pattern in patterns if pattern in nodes)
        graph_edges.update(itertools.combinations(eligible, 2))
    return tuple_to_patterns, nodes, graph_edges


def set_intersection_graph(
    edges: Iterable[Edge],
) -> tuple[set[Pattern], set[tuple[Pattern, Pattern]]]:
    pattern_to_tuples: dict[Pattern, set[TupleKey]] = defaultdict(set)
    for tuple_key, pattern in edges:
        pattern_to_tuples[pattern].add(tuple_key)
    nodes = {
        pattern for pattern, neighbors in pattern_to_tuples.items() if len(neighbors) >= 2
    }
    graph_edges: set[tuple[Pattern, Pattern]] = set()
    for left, right in itertools.combinations(sorted(nodes), 2):
        if pattern_to_tuples[left].intersection(pattern_to_tuples[right]):
            graph_edges.add((left, right))
    return nodes, graph_edges


def connected_components(
    nodes: Iterable[Pattern], graph_edges: Iterable[tuple[Pattern, Pattern]]
) -> tuple[tuple[Pattern, ...], ...]:
    node_set = set(nodes)
    union_find = UnionFind(node_set)
    for left, right in graph_edges:
        union_find.union(left, right)
    groups: dict[Pattern, list[Pattern]] = defaultdict(list)
    for node in sorted(node_set):
        groups[union_find.find(node)].append(node)
    return tuple(sorted((tuple(sorted(group)) for group in groups.values())))


def component_assignments(
    components: Sequence[Sequence[Pattern]],
) -> tuple[dict[Pattern, int], tuple[Pattern, ...]]:
    keys = tuple(sorted(component[0] for component in components))
    id_by_key = {key: index for index, key in enumerate(keys)}
    assignment: dict[Pattern, int] = {}
    for component in components:
        component_id = id_by_key[component[0]]
        for pattern in component:
            assignment[pattern] = component_id
    return assignment, keys


def percentile_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0}
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[int(fraction * (len(ordered) - 1))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": ordered[-1],
    }


def stable_components_digest(components: Sequence[Sequence[Pattern]]) -> str:
    payload = json.dumps(components, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def analyze_documents(documents: Sequence[CorpusDocument]) -> dict[str, Any]:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    frequencies = value_document_frequency(documents)
    edges, diagnostics = enumerate_edges(documents, frequencies)
    tuple_to_patterns, nodes, graph_edges = inverted_graph(edges)
    components = connected_components(nodes, graph_edges)
    assignment, component_keys = component_assignments(components)

    pattern_to_tuples: dict[Pattern, set[TupleKey]] = defaultdict(set)
    for tuple_key, pattern in edges:
        pattern_to_tuples[pattern].add(tuple_key)

    qualifying_component_ids: set[int] = set()
    qualifying_metadata: list[dict[str, int]] = []
    for component in components:
        component_id = assignment[component[0]]
        tuples = set().union(*(pattern_to_tuples[pattern] for pattern in component))
        subjects = {tuple_key[0] for tuple_key in tuples}
        if len(component) >= 2 and len(subjects) >= 4 and len(tuples) >= 4:
            qualifying_component_ids.add(component_id)
            qualifying_metadata.append(
                {
                    "component_id": component_id,
                    "patterns": len(component),
                    "subjects": len(subjects),
                    "tuples": len(tuples),
                }
            )

    bridged_tuples = {
        tuple_key for tuple_key, patterns in tuple_to_patterns.items() if len(patterns) >= 2
    }
    all_tuple_keys = set(tuple_to_patterns)
    bridged_documents = {tuple_key[0] for tuple_key in bridged_tuples}

    facts_by_document: dict[str, set[tuple[int, Value]]] = defaultdict(set)
    values: set[Value] = set()
    for tuple_key, pattern in edges:
        component_id = assignment.get(pattern)
        if component_id not in qualifying_component_ids:
            continue
        subject, value = tuple_key
        facts_by_document[subject].add((component_id, value))
        values.add(value)

    all_document_keys = {document.title.casefold() for document in documents}
    qualifying_documents = set(facts_by_document)
    fact_counts = [len(facts_by_document.get(title, set())) for title in sorted(all_document_keys)]
    covered_fact_counts = [count for count in fact_counts if count > 0]
    overflow_documents = sum(count > MAX_FACTS_PER_DOCUMENT for count in covered_fact_counts)
    fit_fraction = (
        1.0 - overflow_documents / len(covered_fact_counts)
        if covered_fact_counts
        else 0.0
    )

    frequency_values = list(frequencies.values())
    component_sizes = [len(component) for component in components]
    edge_payload = [
        [list(tuple_key[1]), tuple_key[0], list(pattern)]
        for tuple_key, pattern in sorted(edges)
    ]
    graph_payload = [
        [list(left), list(right)] for left, right in sorted(graph_edges)
    ]
    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    metrics = {
        "documents": len(documents),
        "sentences": diagnostics["sentences"],
        "tokens": diagnostics["tokens"],
        "eligible_subject_sentences": diagnostics["eligible_subject_sentences"],
        "distinct_value_spans_all_df": len(frequencies),
        "value_document_frequency": percentile_summary(frequency_values),
        "raw_emitted_edges": diagnostics["raw_emitted_edges"],
        "deduplicated_edges": len(edges),
        "distinct_tuples": len(all_tuple_keys),
        "distinct_patterns": len({pattern for _, pattern in edges}),
        "bridged_tuples": len(bridged_tuples),
        "bridged_tuple_fraction": len(bridged_tuples) / len(all_tuple_keys) if all_tuple_keys else 0.0,
        "bridged_documents": len(bridged_documents),
        "bridged_document_fraction": len(bridged_documents) / len(documents),
        "graph_nodes": len(nodes),
        "graph_edges": len(graph_edges),
        "graph_components": len(components),
        "component_pattern_sizes": percentile_summary(component_sizes),
        "qualifying_components": len(qualifying_component_ids),
        "qualifying_component_metadata": sorted(
            qualifying_metadata, key=lambda item: item["component_id"]
        ),
        "qualifying_documents": len(qualifying_documents),
        "qualifying_document_fraction": len(qualifying_documents) / len(documents),
        "facts_per_all_document": percentile_summary(fact_counts),
        "facts_per_covered_document": percentile_summary(covered_fact_counts),
        "documents_over_24_facts": overflow_documents,
        "covered_documents_fit_24_fraction": fit_fraction,
        "retained_value_dictionary": len(values),
        "witness_regeneration_failures": diagnostics["witness_regeneration_failures"],
        "logical_edge_json_bytes": len(
            json.dumps(edge_payload, separators=(",", ":"), ensure_ascii=False).encode()
        ),
        "logical_graph_json_bytes": len(
            json.dumps(graph_payload, separators=(",", ":"), ensure_ascii=False).encode()
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "peak_rss_kib": peak_rss_kib,
        "components_sha256": stable_components_digest(components),
        "component_key_count": len(component_keys),
    }
    return {
        "metrics": metrics,
        "components": components,
        "edge_set": frozenset(edges),
    }


def run_census() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    corpus_path = root / "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl"
    prereg_path = root / "results/extensional-incidence-quotient-t28-stage0-preregistration.md"
    source_path = Path(__file__).resolve()
    test_path = root / "tests/test_extensional_incidence_quotient_t28_stage0.py"
    observed_hash = sha256_file(corpus_path)
    if observed_hash != CORPUS_SHA256:
        raise RuntimeError(f"raw corpus hash mismatch: {observed_hash}")
    documents = load_documents(corpus_path)
    forward = analyze_documents(documents)
    reverse = analyze_documents(tuple(reversed(documents)))
    order_agreement = (
        forward["edge_set"] == reverse["edge_set"]
        and forward["components"] == reverse["components"]
    )
    metrics = forward["metrics"]
    gates = {
        "input_and_field_boundary": observed_hash == CORPUS_SHA256,
        "witness_and_constructor_integrity": metrics["witness_regeneration_failures"] == 0
        and order_agreement,
        "bridged_tuple_fraction": metrics["bridged_tuple_fraction"] >= 0.25,
        "bridged_document_fraction": metrics["bridged_document_fraction"] >= 0.50,
        "qualifying_component_count": metrics["qualifying_components"] >= 64,
        "qualifying_document_fraction": metrics["qualifying_document_fraction"] >= 0.50,
        "relation_and_value_cardinality": metrics["qualifying_components"] <= MAX_RELATIONS
        and metrics["retained_value_dictionary"] <= MAX_VALUES,
        "fact_budget": metrics["covered_documents_fit_24_fraction"] >= 0.95,
        "cpu_only_finite": all(
            math.isfinite(float(metrics[key]))
            for key in ("cpu_seconds", "wall_seconds", "peak_rss_kib")
        ),
    }
    return {
        "experiment": "extensional-incidence-quotient-t28-stage0-census",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "input": {
            "path": "data/hotpot-real-prose-t12/candidate-raw-prose.jsonl",
            "sha256": observed_hash,
            "fields": ["title", "text"],
        },
        "frozen_rules": {
            "tokenizer": r"\w+|[^\w\s] casefolded",
            "value_tokens": [1, MAX_VALUE_TOKENS],
            "value_document_frequency": [2, math.floor(len(documents) / MAX_VALUE_DOCUMENT_FRACTION)],
            "subject": "casefold-exact own-title occurrence",
            "bridge": "same exact (title,value) tuple under distinct patterns",
            "max_facts_per_document": MAX_FACTS_PER_DOCUMENT,
        },
        "metrics": metrics,
        "integrity": {
            "reverse_document_order_agreement": order_agreement,
            "preregistration_sha256": sha256_file(prereg_path),
            "source_sha256": sha256_file(source_path),
            "test_sha256": sha256_file(test_path),
        },
        "gates": gates,
        "claim_boundary": (
            "Raw-only bridge-existence census. No semantic-purity, natural-QA, "
            "model-training, physical-plane, or production claim."
        ),
    }


def main() -> None:
    print(json.dumps(run_census(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
