from __future__ import annotations

import copy
import unittest

from frontier_g0 import validate_manifest


HASH = "a" * 64


def valid_manifest() -> dict:
    artifact = {
        "checkpoint_sha256": HASH,
        "tokenizer_sha256": HASH,
        "learned_bytes": {
            "core": 100,
            "auxiliary": 0,
            "metadata": 4,
            "total": 104,
        },
    }
    return {
        "schema_version": 1,
        "experiment_id": "example-001",
        "claim": {"scope": "serve-only", "primary_edge": "A", "edges": ["A"]},
        "artifacts": {"baseline": copy.deepcopy(artifact), "candidate": artifact},
        "hardware": {
            "gpu_stratum": "H100_PCIE",
            "gpu_pci_id": "0000:01:00.0",
            "container_digest": "sha256:example",
            "server_commit": "deadbeef",
        },
        "workload": {
            "trace_sha256": HASH,
            "frozen": True,
            "cells": [{
                "name": "normal-chat",
                "input_tokens": 512,
                "output_tokens": 128,
                "concurrency": 1,
            }],
        },
        "quality": {
            "protected_slices": ["P1", "P2", "P3", "P4", "P5"],
            "noninferiority_margins": {
                "P1": 0.01, "P2": 0.01, "P3": 0.01, "P4": 0.01, "P5": 0.01,
            },
        },
        "planned_training_seeds": [11, 22, 33],
    }


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_passes(self) -> None:
        self.assertEqual(validate_manifest(valid_manifest()), [])

    def test_unknown_edge_and_primary_mismatch_fail(self) -> None:
        manifest = valid_manifest()
        manifest["claim"] = {"scope": "serve-only", "primary_edge": "B", "edges": ["A", "Z"]}
        errors = validate_manifest(manifest)
        self.assertTrue(any("unknown edges" in error for error in errors))
        self.assertTrue(any("must also appear" in error for error in errors))

    def test_learned_byte_components_must_reconcile(self) -> None:
        manifest = valid_manifest()
        manifest["artifacts"]["candidate"]["learned_bytes"]["total"] = 100
        errors = validate_manifest(manifest)
        self.assertTrue(any("expected 104" in error for error in errors))

    def test_all_protected_slices_are_required(self) -> None:
        manifest = valid_manifest()
        manifest["quality"]["protected_slices"].remove("P5")
        errors = validate_manifest(manifest)
        self.assertTrue(any("expected exactly" in error for error in errors))

    def test_placeholder_hash_is_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["workload"]["trace_sha256"] = "TODO"
        errors = validate_manifest(manifest)
        self.assertTrue(any("lowercase SHA-256" in error for error in errors))

    def test_zero_learned_bytes_are_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["artifacts"]["candidate"]["learned_bytes"] = {
            "core": 0, "auxiliary": 0, "metadata": 0, "total": 0,
        }
        errors = validate_manifest(manifest)
        self.assertTrue(any("must be greater than zero" in error for error in errors))

    def test_hardware_placeholders_are_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["hardware"]["container_digest"] = "TO_BE_CAPTURED"
        errors = validate_manifest(manifest)
        self.assertTrue(any("captured non-placeholder" in error for error in errors))

    def test_three_unique_seeds_are_required(self) -> None:
        manifest = valid_manifest()
        manifest["planned_training_seeds"] = [1, 1, 2]
        errors = validate_manifest(manifest)
        self.assertTrue(any("3 unique seeds" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
