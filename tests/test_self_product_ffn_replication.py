import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from experiments.self_product_ffn_replication import (
    BASE_INTEGRITY,
    BASE_SOURCE,
    DISCOVERY,
    batch_clustered_interval,
    sha256_file,
    symmetric_envelope,
)


def test_discovery_failed_only_asymmetric_envelope_and_passes_symmetric_safety():
    discovery = json.loads(DISCOVERY.read_text())
    gates = discovery["decision"]["gates"]
    assert not gates["candidate_activation_outside_4x_baseline_rms_at_most_1_percent"]
    assert all(value for key, value in gates.items() if key != "candidate_activation_outside_4x_baseline_rms_at_most_1_percent")
    assert all(symmetric_envelope(discovery).values())


def test_discovery_artifact_is_complete_and_seed_1907():
    discovery = json.loads(Path("results/self-product-ffn-lm-screen.json").read_text())
    assert discovery["decision"]["complete"]
    assert discovery["experiment_protocol"]["actual"]["seed"] == 1907
    assert discovery["folded_h100_feasibility_pass"]
    assert discovery["source_sha256"] == sha256_file(BASE_SOURCE)
    assert discovery["integrity_manifest_sha256"] == sha256_file(BASE_INTEGRITY)


def test_batch_clustered_interval_does_not_double_count_repeated_batches():
    discovery = json.loads(DISCOVERY.read_text())
    interval = batch_clustered_interval(discovery, discovery, "parallel_swiglu")
    assert interval["validation_batch_clusters"] == 128
    assert interval["seeds_per_cluster"] == 2
