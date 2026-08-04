from __future__ import annotations

import numpy as np

from experiments import predictive_residual_coder_t19d as t19d


def test_arithmetic_unit_contracts() -> None:
    assert all(t19d.run_unit_contracts().values())


def test_mass_quantization_is_positive_and_exact() -> None:
    generator = np.random.default_rng(71)
    values = generator.random((19, t19d.TOP_K))
    values *= (0.93 / values.sum(axis=1))[:, None]
    frequencies = t19d.quantize_mass_rows(values)
    assert frequencies.shape == (19, t19d.TOP_K + 1)
    assert np.all(frequencies > 0)
    assert np.all(frequencies.sum(axis=1) == t19d.FREQUENCY_TOTAL)


def test_small_document_escape_roundtrips() -> None:
    tokens = np.asarray([7, 11, 48_777, 13], dtype=np.int64)
    top_indices = np.tile(np.arange(t19d.TOP_K, dtype=np.int64), (3, 1))
    probabilities = np.full((3, t19d.TOP_K), 0.9 / t19d.TOP_K)
    frequencies = t19d.quantize_mass_rows(probabilities)
    result = t19d.encode_document(
        tokens, top_indices, frequencies, ideal_bits=41.0
    )
    assert result.roundtrip_exact
    assert result.escapes == 1
    assert result.frequency_contract


def test_physical_fixed_writes() -> None:
    assert t19d.TITLE_WRITES + t19d.ADDRESS_WRITES + t19d.SENTINEL_WRITES == 216_975
    assert t19d.ADMISSION_WRITE_BUDGET < t19d.HARD_WRITE_BUDGET


def test_frozen_hashes() -> None:
    assert t19d.t19c.sha256_file(t19d.t19c.CHECKPOINT) == t19d.t19c.CHECKPOINT_SHA256
    assert (
        t19d.t19c.sha256_file(t19d.t19c.t19b.t19a.CANDIDATE_CORPUS)
        == t19d.t19c.t19b.t19a.CANDIDATE_SHA256
    )
