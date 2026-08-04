from __future__ import annotations

from collections import defaultdict

from experiments import consensus_permutation_raw_prose as raw
from experiments import raw_prose_equality_plane_t10 as t10
from experiments import raw_prose_equality_plane_t10_train as train


def fake_layout(compiled: raw.CompileResult) -> t10.ProseLayout:
    anchors = t10.discover_frame_anchors(compiled)
    cursor = 0
    aliases = {
        word: index for index, word in enumerate(sorted(compiled.alias_to_canonical))
    }
    cursor += len(aliases)
    frame_tokens = {frame: cursor + index for index, frame in enumerate(sorted(anchors))}
    cursor += len(frame_tokens)
    values = sorted({value for row in compiled.canonical_values for value in row})
    value_tokens = {value: cursor + index for index, value in enumerate(values)}
    cursor += len(value_tokens)
    grouped: dict[int, list[raw.Frame]] = defaultdict(list)
    for frame, relation in compiled.frame_to_relation.items():
        grouped[relation].append(frame)
    roles: dict[raw.Frame, str] = {}
    for frames in grouped.values():
        ordered = sorted(frames)
        midpoint = len(ordered) // 2
        roles.update({frame: "a" for frame in ordered[:midpoint]})
        roles.update({frame: "b" for frame in ordered[midpoint:]})
    return t10.ProseLayout(
        aliases,
        anchors,
        frame_tokens,
        roles,
        value_tokens,
        cursor,
        cursor + 1,
        cursor + 2,
        cursor + 3,
        {},
    )


def test_query_split_holds_out_aliases_surfaces_and_compositions() -> None:
    corpus = raw.make_corpus(
        seed=101,
        entities=t10.ENTITIES,
        relations=t10.RELATIONS,
        values=t10.VALUES,
        views=t10.ALIAS_VIEWS,
        paraphrases=t10.PARAPHRASES,
        repeats=1,
        corruption_rate=0.0,
    )
    compiled = raw.compile_corpus(corpus.sentences)
    split = train.build_query_split(compiled, fake_layout(compiled))
    assert len(split.direct_train_tokens) == t10.ENTITIES * t10.RELATIONS
    assert len(split.equality_train_tokens) == (t10.ENTITIES // 4) * 32
    assert len(split.direct_heldout_tokens) > len(split.direct_train_tokens)
    assert len(split.equality_heldout_tokens) > 50 * len(split.equality_train_tokens)
    assert not (
        {tuple(row) for row in split.direct_train_tokens.tolist()}
        & {tuple(row) for row in split.direct_heldout_tokens.tolist()}
    )
    assert not (
        {tuple(row) for row in split.equality_train_tokens.tolist()}
        & {tuple(row) for row in split.equality_heldout_tokens.tolist()}
    )
