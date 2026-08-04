from experiments.phase_elastic_v2_50m_replication import SEED, STEPS


def test_replication_uses_fresh_seed_and_long_horizon():
    assert SEED == 3719
    assert STEPS == 1525
    assert 1525 * 2 * 32 * 512 == 49_971_200

