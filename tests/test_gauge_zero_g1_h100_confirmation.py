import json

from experiments.gauge_zero_g1_h100_confirmation import (
    FORMAL_V1,
    FORMAL_V1_SHA256,
    ROWS,
    SEED,
    TRIALS,
    WARMUP,
    sha256_file,
    validate_v1,
)


def test_confirmation_contract_is_narrow_and_v1_is_immutable():
    assert ROWS == 1
    assert TRIALS == 4000
    assert WARMUP == 50
    assert SEED == 6829
    assert sha256_file(FORMAL_V1) == FORMAL_V1_SHA256
    checks = validate_v1()
    assert all(checks.values())
    payload = json.loads(FORMAL_V1.read_text())
    assert payload["cells"][0]["candidate_over_control"]["upper_95"] > 1.02
