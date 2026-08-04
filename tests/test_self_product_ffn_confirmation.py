import pytest

pytest.importorskip("torch")

from experiments.self_product_ffn_confirmation import (
    BASE_INTEGRITY,
    BASE_SOURCE,
    DISCOVERY,
    OUTPUT,
    PREREGISTRATION,
    REPLICATION,
    SEED,
)


def test_confirmation_is_a_distinct_frozen_seed_and_output():
    assert SEED == 31415
    assert DISCOVERY != REPLICATION != OUTPUT
    assert PREREGISTRATION.name == "self-product-ffn-confirmation-preregistration.md"
    assert BASE_SOURCE.exists()
    assert BASE_INTEGRITY.exists()
