from __future__ import annotations

from experiments import raw_prose_equality_plane_t11_replicate as replicate


def test_replication_uses_three_unseen_seeds() -> None:
    assert replicate.MODEL_SEEDS == (6_703, 6_997, 7_307)
    assert 6_401 not in replicate.MODEL_SEEDS
    assert len(set(replicate.MODEL_SEEDS)) == 3


def test_relative_nll_delta_direction() -> None:
    assert replicate.relative_nll_delta(1.005, 1.0) == 0.004999999999999893
    assert replicate.relative_nll_delta(0.9, 1.0) < 0.0
