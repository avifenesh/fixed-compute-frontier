from __future__ import annotations

from experiments import xor_functional_record_t24_sufficiency as oracle


def test_typed_feature_roles_are_distinct() -> None:
    collisions = {
        ((-1, "born"), (1, "in")): {(0, "paris"), (1, "rome")},
    }
    rows = oracle.typed_feature_counts(collisions, 2)
    assert rows[0]["anchor:born"] == 1
    assert rows[0]["anchor:in"] == 1
    assert rows[0]["target:paris"] == 1
    assert "target:born" not in rows[0]


def test_preregistration_and_stage0_are_sealed() -> None:
    assert (
        oracle.sha256_file(oracle.PREREGISTRATION)
        == oracle.PREREGISTRATION_SHA256
    )
    assert oracle.sha256_file(oracle.T24_RESULT) == oracle.T24_RESULT_SHA256
