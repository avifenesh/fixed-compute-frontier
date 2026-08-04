#!/usr/bin/env python3
"""Stage 0: recover a latent fact tensor from noisy permuted raw prose views."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/consensus-permutation-raw-prose-stage0b.json"
PREREGISTRATION = ROOT / "results/consensus-permutation-raw-prose-stage0b-preregistration.md"

TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\w\s]")
SLOT = "*"


class NonIdentifiable(RuntimeError):
    """Raised when raw observations do not determine one latent alignment."""


@dataclass(frozen=True, order=True)
class Frame:
    first_slot: int
    second_slot: int
    tokens: tuple[str, ...]

    def serial(self) -> str:
        return f"{self.first_slot}:{self.second_slot}:" + " ".join(self.tokens)


@dataclass(frozen=True)
class SurfaceTemplate:
    tokens: tuple[str, ...]
    entity_position: int
    value_position: int
    view: int
    relation: int

    @property
    def frame(self) -> Frame:
        first, second = sorted((self.entity_position, self.value_position))
        masked = list(self.tokens)
        masked[first] = SLOT
        masked[second] = SLOT
        return Frame(first, second, tuple(masked))


@dataclass(frozen=True)
class GeneratedCorpus:
    sentences: tuple[str, ...]
    facts_by_view: tuple[tuple[tuple[int, ...], ...], ...]
    alias_latent: dict[str, int]
    frame_relation: dict[Frame, int]
    value_index: dict[str, int]
    expected_frames: int
    expected_views: int
    corruption_rate: float
    repeats: int


@dataclass(frozen=True)
class TemplateTable:
    frame: Frame
    entity_position: int
    value_position: int
    mapping: dict[str, str]
    mentions: int
    contradictory_mentions: int
    minimum_vote_fraction: float


@dataclass(frozen=True)
class CompileResult:
    canonical_entities: tuple[str, ...]
    canonical_relation_count: int
    canonical_values: tuple[tuple[str, ...], ...]
    alias_to_canonical: dict[str, str]
    frame_to_relation: dict[Frame, int]
    induced_frames: tuple[Frame, ...]
    inferred_view_count: int
    input_sentences: int
    input_tokens: int
    contradictory_mentions: int
    minimum_vote_fraction: float
    relation_permutations_tested: int


SKELETONS = (
    "According to the {A} ledger , {E} is marked {V} .",
    "The {A} record for {E} reports {V} .",
    "{V} appears beside {E} in the {A} register .",
    "Archivists classify {E} as {V} for {A} .",
    "For {E} , {A} documentation lists {V} .",
    "Under {A} , the entry attached to {E} is {V} .",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(sentence: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(sentence))


def opaque_words(count: int, rng: random.Random, used: set[str]) -> list[str]:
    consonants = "bcdfghjklmnprstvwxyz"
    vowels = "aeiou"
    words: list[str] = []
    while len(words) < count:
        word = "".join(
            rng.choice(consonants) + rng.choice(vowels) for _ in range(4)
        )
        if word not in used:
            used.add(word)
            words.append(word)
    return words


def instantiate_template(
    skeleton: str,
    anchor: str,
    entity_placeholder: str = "ENTITYPLACEHOLDER",
    value_placeholder: str = "VALUEPLACEHOLDER",
) -> tuple[tuple[str, ...], int, int]:
    text = skeleton.format(A=anchor, E=entity_placeholder, V=value_placeholder)
    tokens = tokenize(text)
    return tokens, tokens.index(entity_placeholder), tokens.index(value_placeholder)


def make_corpus(
    *,
    seed: int = 5_200_043,
    entities: int = 128,
    relations: int = 6,
    values: int = 16,
    views: int = 3,
    paraphrases: int = 2,
    repeats: int = 63,
    corruption_rate: float = 0.15,
    independent_views: bool = False,
    duplicate_entity_signature: bool = False,
    constant_zero_relation: bool = False,
) -> GeneratedCorpus:
    if paraphrases > len(SKELETONS):
        raise ValueError("not enough frozen prose skeletons")
    if repeats % 2 != 1:
        raise ValueError("repeats must be odd")
    rng = random.Random(seed)
    used: set[str] = set()
    value_words = opaque_words(values, rng, used)
    aliases = [opaque_words(entities, rng, used) for _ in range(views)]
    anchors = opaque_words(views * relations * paraphrases, rng, used)
    anchor_cursor = 0

    base_facts = tuple(
        tuple(rng.randrange(values) for _ in range(relations))
        for _ in range(entities)
    )
    if constant_zero_relation:
        base_facts = tuple((0,) + row[1:] for row in base_facts)
    if duplicate_entity_signature:
        mutable = [list(row) for row in base_facts]
        mutable[1] = list(mutable[0])
        base_facts = tuple(tuple(row) for row in mutable)
    facts_by_view: list[tuple[tuple[int, ...], ...]] = []
    for view in range(views):
        if independent_views and view:
            table = tuple(
                tuple(rng.randrange(values) for _ in range(relations))
                for _ in range(entities)
            )
        else:
            table = base_facts
        facts_by_view.append(table)

    templates: list[SurfaceTemplate] = []
    frame_relation: dict[Frame, int] = {}
    for view in range(views):
        for relation in range(relations):
            for paraphrase in range(paraphrases):
                anchor = anchors[anchor_cursor]
                anchor_cursor += 1
                skeleton_index = (view + relation + paraphrase) % len(SKELETONS)
                tokens, entity_position, value_position = instantiate_template(
                    SKELETONS[skeleton_index], anchor
                )
                template = SurfaceTemplate(
                    tokens, entity_position, value_position, view, relation
                )
                templates.append(template)
                frame_relation[template.frame] = relation

    sentences: list[str] = []
    for template in templates:
        table = facts_by_view[template.view]
        for entity in range(entities):
            correct_value = table[entity][template.relation]
            for _ in range(repeats):
                observed_value = correct_value
                if rng.random() < corruption_rate:
                    observed_value = rng.randrange(values - 1)
                    if observed_value >= correct_value:
                        observed_value += 1
                tokens = list(template.tokens)
                tokens[template.entity_position] = aliases[template.view][entity]
                tokens[template.value_position] = value_words[observed_value]
                sentences.append(" ".join(tokens))
    rng.shuffle(sentences)

    alias_latent = {
        alias: entity
        for view_aliases in aliases
        for entity, alias in enumerate(view_aliases)
    }
    return GeneratedCorpus(
        sentences=tuple(sentences),
        facts_by_view=tuple(facts_by_view),
        alias_latent=alias_latent,
        frame_relation=frame_relation,
        value_index={word: index for index, word in enumerate(value_words)},
        expected_frames=len(templates),
        expected_views=views,
        corruption_rate=corruption_rate,
        repeats=repeats,
    )


def mask_frame(tokens: tuple[str, ...], first: int, second: int) -> Frame:
    masked = list(tokens)
    masked[first] = SLOT
    masked[second] = SLOT
    return Frame(first, second, tuple(masked))


def all_frames(tokens: tuple[str, ...]) -> Iterable[Frame]:
    for first in range(len(tokens)):
        for second in range(first + 1, len(tokens)):
            yield mask_frame(tokens, first, second)


def induce_frames(
    tokenized: Sequence[tuple[str, ...]], sample_limit: int = 12_000
) -> tuple[dict[Frame, list[tuple[str, str]]], tuple[Frame, ...]]:
    sample_counts: Counter[Frame] = Counter()
    for tokens in tokenized[:sample_limit]:
        sample_counts.update(all_frames(tokens))
    maximum_by_length: dict[int, int] = defaultdict(int)
    for frame, count in sample_counts.items():
        maximum_by_length[len(frame.tokens)] = max(
            maximum_by_length[len(frame.tokens)], count
        )
    candidates = {
        frame
        for frame, count in sample_counts.items()
        if count >= max(8, math.floor(0.35 * maximum_by_length[len(frame.tokens)]))
    }
    by_length: dict[int, set[Frame]] = defaultdict(set)
    for frame in candidates:
        by_length[len(frame.tokens)].add(frame)

    observations: dict[Frame, list[tuple[int, str, str]]] = defaultdict(list)
    for sentence_index, tokens in enumerate(tokenized):
        matches: list[Frame] = []
        for frame in all_frames(tokens):
            if frame in by_length[len(tokens)]:
                matches.append(frame)
        # True frames have two high-cardinality slots.  Sampling can also retain
        # masks containing one true slot and one fixed word; discard those here.
        if len(matches) > 1:
            variable_matches = []
            for frame in matches:
                if tokens[frame.first_slot] != tokens[frame.second_slot]:
                    variable_matches.append(frame)
            matches = variable_matches
        if not matches:
            raise NonIdentifiable(f"sentence {sentence_index} has no induced frame")
        # Defer the cardinality filter until all observations are collected.
        for frame in matches:
            observations[frame].append(
                (
                    sentence_index,
                    tokens[frame.first_slot],
                    tokens[frame.second_slot],
                )
            )

    admissible: dict[Frame, list[tuple[int, str, str]]] = {}
    for frame, rows in observations.items():
        first_unique = len({row[1] for row in rows})
        second_unique = len({row[2] for row in rows})
        small = min(first_unique, second_unique)
        large = max(first_unique, second_unique)
        if small > 1 and large >= 2 * small:
            admissible[frame] = rows

    # A false mask can hide an entity and a fixed frame word while leaving one
    # frequent value literal in place.  Such a pattern crosses several real
    # templates, but covers only a subset of each.  Real two-slot frames are a
    # disjoint exact cover and have the largest remaining support.  Greedy
    # maximum uncovered support therefore selects the literal templates without
    # knowing their slot locations or count.
    uncovered = set(range(len(tokenized)))
    filtered: dict[Frame, list[tuple[str, str]]] = {}
    while uncovered:
        ranked: list[tuple[int, str, Frame]] = []
        for frame, rows in admissible.items():
            gain = sum(sentence_index in uncovered for sentence_index, _, _ in rows)
            if gain:
                ranked.append((gain, frame.serial(), frame))
        if not ranked:
            raise NonIdentifiable("induced frames do not cover the corpus")
        _, _, best = max(ranked, key=lambda item: (item[0], item[1]))
        rows = admissible[best]
        selected_indices = {
            sentence_index
            for sentence_index, _, _ in rows
            if sentence_index in uncovered
        }
        filtered[best] = [
            (first, second)
            for sentence_index, first, second in rows
            if sentence_index in selected_indices
        ]
        uncovered.difference_update(selected_indices)

    # After exact-cover selection every sentence must belong to one frame.
    coverage = 0
    for tokens in tokenized:
        hits = sum(1 for frame in all_frames(tokens) if frame in filtered)
        if hits != 1:
            raise NonIdentifiable(f"a sentence has {hits} admissible frames")
        coverage += 1
    if coverage != len(tokenized):
        raise AssertionError("frame coverage accounting failed")
    return filtered, tuple(sorted(filtered))


def build_template_table(
    frame: Frame, pairs: Sequence[tuple[str, str]]
) -> TemplateTable:
    first_unique = len({pair[0] for pair in pairs})
    second_unique = len({pair[1] for pair in pairs})
    if first_unique == second_unique:
        raise NonIdentifiable("cannot orient equal-cardinality frame slots")
    entity_is_first = first_unique > second_unique
    entity_position = frame.first_slot if entity_is_first else frame.second_slot
    value_position = frame.second_slot if entity_is_first else frame.first_slot
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for first, second in pairs:
        entity, value = (first, second) if entity_is_first else (second, first)
        votes[entity][value] += 1
    mapping: dict[str, str] = {}
    contradictory = 0
    minimum_fraction = 1.0
    for entity, counts in votes.items():
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            raise NonIdentifiable("tied categorical vote")
        value, count = ranked[0]
        mapping[entity] = value
        contradictory += sum(counts.values()) - count
        minimum_fraction = min(minimum_fraction, count / sum(counts.values()))
    return TemplateTable(
        frame,
        entity_position,
        value_position,
        mapping,
        len(pairs),
        contradictory,
        minimum_fraction,
    )


def group_relation_tables(
    tables: Sequence[TemplateTable],
) -> list[tuple[dict[str, str], tuple[Frame, ...]]]:
    grouped: dict[tuple[tuple[str, str], ...], list[Frame]] = defaultdict(list)
    mappings: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for table in tables:
        key = tuple(sorted(table.mapping.items()))
        grouped[key].append(table.frame)
        mappings[key] = table.mapping
    groups = [
        (mappings[key], tuple(sorted(frames))) for key, frames in grouped.items()
    ]
    return sorted(groups, key=lambda group: tuple(f.serial() for f in group[1]))


def unique_signatures(
    aliases: Iterable[str], relation_tables: Sequence[dict[str, str]]
) -> dict[tuple[str, ...], str]:
    signatures: dict[tuple[str, ...], str] = {}
    for alias in aliases:
        signature = tuple(table[alias] for table in relation_tables)
        if signature in signatures:
            raise NonIdentifiable("duplicate complete entity signature")
        signatures[signature] = alias
    return signatures


def relation_histogram(table: dict[str, str]) -> tuple[tuple[str, int], ...]:
    """Alias-permutation invariant used to align large relation sets cheaply."""
    return tuple(sorted(Counter(table.values()).items()))


def compile_corpus(sentences: Sequence[str]) -> CompileResult:
    """Compile raw strings only; no latent IDs or generator state are accepted."""
    if not sentences:
        raise NonIdentifiable("empty corpus")
    tokenized = tuple(tokenize(sentence) for sentence in sentences)
    observations, induced = induce_frames(tokenized)
    tables = [build_template_table(frame, observations[frame]) for frame in induced]

    by_alias_set: dict[frozenset[str], list[TemplateTable]] = defaultdict(list)
    for table in tables:
        by_alias_set[frozenset(table.mapping)].append(table)
    if len(by_alias_set) < 2:
        raise NonIdentifiable("need at least two independently aliased views")
    view_keys = sorted(by_alias_set, key=lambda key: tuple(sorted(key)))
    relation_groups = [group_relation_tables(by_alias_set[key]) for key in view_keys]
    relation_counts = {len(groups) for groups in relation_groups}
    if len(relation_counts) != 1:
        raise NonIdentifiable("views infer different relation counts")
    relation_count = relation_counts.pop()
    if relation_count < 2:
        raise NonIdentifiable("need at least two relations for permutation recovery")

    reference_groups = relation_groups[0]
    reference_tables = [group[0] for group in reference_groups]
    reference_aliases = tuple(sorted(view_keys[0]))
    reference_signatures = unique_signatures(reference_aliases, reference_tables)

    alias_to_canonical = {alias: alias for alias in reference_aliases}
    frame_to_relation: dict[Frame, int] = {}
    for relation, (_, frames) in enumerate(reference_groups):
        for frame in frames:
            frame_to_relation[frame] = relation

    permutations_tested = 0
    for view_index in range(1, len(view_keys)):
        aliases = tuple(sorted(view_keys[view_index]))
        groups = relation_groups[view_index]
        valid: list[tuple[tuple[int, ...], dict[tuple[str, ...], str]]] = []
        reference_histograms = [relation_histogram(table) for table in reference_tables]
        current_by_histogram: dict[
            tuple[tuple[str, int], ...], list[int]
        ] = defaultdict(list)
        for index, (table, _) in enumerate(groups):
            current_by_histogram[relation_histogram(table)].append(index)
        if (
            len(set(reference_histograms)) == relation_count
            and all(len(current_by_histogram[histogram]) == 1 for histogram in reference_histograms)
        ):
            permutation = tuple(
                current_by_histogram[histogram][0]
                for histogram in reference_histograms
            )
            permutations_tested += relation_count * relation_count
            ordered = [groups[index][0] for index in permutation]
            signatures = unique_signatures(aliases, ordered)
            if signatures.keys() == reference_signatures.keys():
                valid.append((permutation, signatures))
        elif relation_count <= 8:
            for permutation in itertools.permutations(range(relation_count)):
                permutations_tested += 1
                ordered = [groups[index][0] for index in permutation]
                try:
                    signatures = unique_signatures(aliases, ordered)
                except NonIdentifiable:
                    continue
                if signatures.keys() == reference_signatures.keys():
                    valid.append((permutation, signatures))
        else:
            raise NonIdentifiable(
                "large relation set lacks unique permutation-invariant histograms"
            )
        if len(valid) != 1:
            raise NonIdentifiable(
                f"view {view_index} has {len(valid)} admissible alignments"
            )
        permutation, signatures = valid[0]
        for signature, alias in signatures.items():
            alias_to_canonical[alias] = reference_signatures[signature]
        for reference_relation, current_index in enumerate(permutation):
            current_mapping, frames = groups[current_index]
            for frame in frames:
                frame_to_relation[frame] = reference_relation
            for alias, value in current_mapping.items():
                canonical = alias_to_canonical[alias]
                if value != reference_tables[reference_relation][canonical]:
                    raise NonIdentifiable("aligned view disagrees on a recovered fact")

    canonical_values = tuple(
        tuple(table[alias] for table in reference_tables)
        for alias in reference_aliases
    )
    return CompileResult(
        canonical_entities=reference_aliases,
        canonical_relation_count=relation_count,
        canonical_values=canonical_values,
        alias_to_canonical=alias_to_canonical,
        frame_to_relation=frame_to_relation,
        induced_frames=induced,
        inferred_view_count=len(view_keys),
        input_sentences=len(sentences),
        input_tokens=sum(len(tokens) for tokens in tokenized),
        contradictory_mentions=sum(table.contradictory_mentions for table in tables),
        minimum_vote_fraction=min(table.minimum_vote_fraction for table in tables),
        relation_permutations_tested=permutations_tested,
    )


def evaluate_recovery(
    corpus: GeneratedCorpus, compiled: CompileResult
) -> dict[str, float | int]:
    canonical_entity_latent = {
        alias: corpus.alias_latent[alias] for alias in compiled.canonical_entities
    }
    relation_latent: dict[int, int] = {}
    relation_correct = 0
    for frame, canonical_relation in compiled.frame_to_relation.items():
        latent = corpus.frame_relation[frame]
        if canonical_relation in relation_latent:
            relation_correct += int(relation_latent[canonical_relation] == latent)
        else:
            relation_latent[canonical_relation] = latent
            relation_correct += 1

    alias_correct = sum(
        corpus.alias_latent[alias] == canonical_entity_latent[canonical]
        for alias, canonical in compiled.alias_to_canonical.items()
    )
    direct_correct = 0
    direct_total = 0
    equality_correct = 0
    equality_total = 0
    for entity_index, canonical_alias in enumerate(compiled.canonical_entities):
        latent_entity = canonical_entity_latent[canonical_alias]
        for canonical_relation in range(compiled.canonical_relation_count):
            latent_relation = relation_latent[canonical_relation]
            predicted = corpus.value_index[
                compiled.canonical_values[entity_index][canonical_relation]
            ]
            target = corpus.facts_by_view[0][latent_entity][latent_relation]
            direct_correct += int(predicted == target)
            direct_total += 1
        for first in range(compiled.canonical_relation_count):
            for second in range(compiled.canonical_relation_count):
                predicted_first = compiled.canonical_values[entity_index][first]
                predicted_second = compiled.canonical_values[entity_index][second]
                target_first = corpus.facts_by_view[0][latent_entity][
                    relation_latent[first]
                ]
                target_second = corpus.facts_by_view[0][latent_entity][
                    relation_latent[second]
                ]
                equality_correct += int(
                    (predicted_first == predicted_second)
                    == (target_first == target_second)
                )
                equality_total += 1
    return {
        "direct_fact_accuracy": direct_correct / direct_total,
        "direct_fact_queries": direct_total,
        "entity_alias_accuracy": alias_correct / len(compiled.alias_to_canonical),
        "entity_aliases": len(compiled.alias_to_canonical),
        "relation_paraphrase_accuracy": relation_correct
        / len(compiled.frame_to_relation),
        "relation_paraphrases": len(compiled.frame_to_relation),
        "compiler_derived_equality_accuracy": equality_correct / equality_total,
        "compiler_derived_equality_queries": equality_total,
    }


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def swiglu_product_error(beta: float = 8.0) -> float:
    errors = []
    for first in (-1.0, 1.0):
        for second in (-1.0, 1.0):
            estimate = (
                silu(beta * first) - silu(-beta * first)
            ) * second / beta
            errors.append(abs(estimate - first * second))
    return max(errors)


def corpus_sha256(sentences: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sentence in sentences:
        digest.update(sentence.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def rejection_control(**kwargs: object) -> tuple[bool, str]:
    corpus = make_corpus(**kwargs)
    try:
        compile_corpus(corpus.sentences)
    except NonIdentifiable as error:
        return True, str(error)
    return False, "compiler forced an alignment"


def run() -> dict[str, object]:
    started = time.perf_counter()
    corpus = make_corpus()
    generated_seconds = time.perf_counter() - started
    compile_started = time.perf_counter()
    compiled = compile_corpus(corpus.sentences)
    compile_seconds = time.perf_counter() - compile_started
    metrics = evaluate_recovery(corpus, compiled)

    independent_rejected, independent_reason = rejection_control(
        seed=5_300_047,
        entities=48,
        relations=5,
        views=2,
        paraphrases=2,
        repeats=31,
        corruption_rate=0.05,
        independent_views=True,
    )
    ambiguous_rejected, ambiguous_reason = rejection_control(
        seed=5_400_049,
        entities=48,
        relations=5,
        views=2,
        paraphrases=2,
        repeats=31,
        corruption_rate=0.05,
        duplicate_entity_signature=True,
    )
    epsilon = corpus.corruption_rate
    per_fact_bound = math.exp(-2 * corpus.repeats * (0.5 - epsilon) ** 2)
    opportunities = (
        len(corpus.alias_latent)
        * (corpus.expected_frames // corpus.expected_views)
    )
    union_bound = opportunities * per_fact_bound
    product_error = swiglu_product_error()
    signature = inspect.signature(compile_corpus)
    raw_only_signature = tuple(signature.parameters) == ("sentences",)

    gates = {
        "all_sentences_assigned_once": compiled.input_sentences
        == len(corpus.sentences),
        "surface_frame_count_exact": len(compiled.induced_frames)
        == corpus.expected_frames,
        "alias_view_count_exact": compiled.inferred_view_count
        == corpus.expected_views,
        "direct_facts_exact": metrics["direct_fact_accuracy"] == 1.0,
        "entity_aliases_exact": metrics["entity_alias_accuracy"] == 1.0,
        "relation_paraphrases_exact": metrics["relation_paraphrase_accuracy"]
        == 1.0,
        "all_held_out_equality_compositions_exact": metrics[
            "compiler_derived_equality_accuracy"
        ]
        == 1.0,
        "swiglu_product_identity_below_1e_12": product_error < 1e-12,
        "independent_view_null_rejected": independent_rejected,
        "duplicate_signature_null_rejected": ambiguous_rejected,
        "majority_error_union_bound_below_1pct": union_bound < 0.01,
        "compiler_api_accepts_raw_strings_only": raw_only_signature,
    }
    return {
        "schema": "consensus-permutation-raw-prose-stage0b-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "world": {
            "entities": 128,
            "relations": 6,
            "values": 16,
            "alias_views": 3,
            "paraphrases_per_relation_view": 2,
            "mentions_per_fact_frame": corpus.repeats,
            "contradiction_probability": corpus.corruption_rate,
            "raw_sentences": len(corpus.sentences),
            "raw_tokens": compiled.input_tokens,
        },
        "metrics": metrics,
        "extractor": {
            "induced_frames": len(compiled.induced_frames),
            "inferred_views": compiled.inferred_view_count,
            "inferred_relations": compiled.canonical_relation_count,
            "contradictory_mentions_after_majority": compiled.contradictory_mentions,
            "observed_contradiction_fraction": compiled.contradictory_mentions
            / len(corpus.sentences),
            "minimum_fact_vote_fraction": compiled.minimum_vote_fraction,
            "relation_permutations_tested": compiled.relation_permutations_tested,
            "compiler_signature": str(signature),
        },
        "mathematics": {
            "entity_signature_collision_union_bound_per_view": 128
            * 127
            / (2 * 16**6),
            "majority_error_bound_per_fact_frame": per_fact_bound,
            "majority_error_union_bound": union_bound,
            "swiglu_bipolar_product_max_abs_error": product_error,
        },
        "null_controls": {
            "independent_views": {
                "rejected": independent_rejected,
                "reason": independent_reason,
            },
            "duplicate_entity_signature": {
                "rejected": ambiguous_rejected,
                "reason": ambiguous_reason,
            },
        },
        "resource_ledger": {
            "generation_seconds": generated_seconds,
            "compiler_seconds": compile_seconds,
            "gpu_seconds": 0.0,
            "relation_permutations_tested": compiled.relation_permutations_tested,
            "failed_or_retried_compilations": 0,
        },
        "sealing": {
            "source_sha256": sha256_file(Path(__file__)),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "raw_corpus_sha256": corpus_sha256(corpus.sentences),
        },
        "claim_boundary": (
            "This proves exact raw-sequence recovery only in a controlled "
            "identifiable repeated-frame regime. Equality is invariant to the "
            "opaque value-label permutation. The SwiGLU identity is proved separately, "
            "but deployed-model integration is not yet tested."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
