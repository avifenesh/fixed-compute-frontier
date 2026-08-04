from experiments import raw_self_query_functional_record_stage0 as stage0


def test_frequency_thresholds_are_scale_derived() -> None:
    assert stage0.frequency_thresholds(2_405) == {
        "anchor_min_document_frequency": 25,
        "anchor_max_document_frequency": 1_202,
        "target_min_document_frequency": 1,
        "target_max_document_frequency": 121,
    }


def test_relation_skeleton_deletes_target_and_rare_values() -> None:
    words = ("rare_name", "was", "born", "in", "rare_year")
    skeleton = stage0.relation_skeleton(
        words, 4, frozenset({"was", "born", "in"}), radius=4
    )
    assert skeleton == ((-3, "was"), (-2, "born"), (-1, "in"))
    assert all(word != "rare_year" for _, word in skeleton)


def test_document_split_is_stable_disjoint_and_nonempty() -> None:
    probes = [
        stage0.Probe(0, "doc", index, f"target-{index}", ((-1, "was"),))
        for index in range(8)
    ]
    first = stage0.split_document_probes(probes)
    second = stage0.split_document_probes(list(reversed(probes)))
    assert first == second
    assert {probe.split for probe in first} == {"train", "evaluation"}
    assert len({probe.identity for probe in first}) == len(first)


def test_collision_statistics_require_document_and_target_ambiguity() -> None:
    skeleton = ((-1, "born"), (1, "in"))
    probes = [
        stage0.Probe(0, "a", 0, "1970", skeleton, split="train"),
        stage0.Probe(1, "b", 0, "1970", skeleton, split="train"),
        stage0.Probe(2, "c", 0, "1980", skeleton, split="train"),
    ]
    result = stage0.collision_statistics(probes)
    assert result["collision_groups"] == 1
    assert result["equality_groups"] == 1
    assert result["positive_cross_document_pairs"] == 1
    assert result["negative_cross_document_pairs"] == 2
    assert result["conditional_target_entropy_bits"] > 0


def test_collision_statistics_exclude_multivalued_document_skeletons() -> None:
    skeleton = ((-1, "born"), (1, "in"))
    probes = [
        stage0.Probe(0, "a", 0, "1970", skeleton, split="train"),
        stage0.Probe(0, "a", 1, "1980", skeleton, split="train"),
        stage0.Probe(1, "b", 0, "1970", skeleton, split="train"),
        stage0.Probe(2, "c", 0, "1980", skeleton, split="train"),
    ]
    result = stage0.collision_statistics(probes)
    assert result["ambiguous_document_skeletons_excluded"] == 1
    assert result["collision_probe_pairs"] == 2
    assert result["equality_groups"] == 0
