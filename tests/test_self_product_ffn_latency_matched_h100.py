from pathlib import Path


def test_preregistered_width_is_single_aligned_point():
    source = Path("experiments/self_product_ffn_latency_matched_h100.py").read_text()
    assert "LATENCY_MATCHED_WIDE = 2560" in source
    assert "range(2560" not in source
    assert "for width" not in source


def test_candidate_has_less_dense_ffn_cost_but_more_elementwise_work():
    hidden, baseline, candidate = 640, 1792, 2560
    assert 2 * hidden * candidate < 3 * hidden * baseline
    assert candidate > baseline
    assert (2 * hidden * candidate) / (3 * hidden * baseline) == 20 / 21


def test_preregistration_freezes_required_gates():
    text = Path("results/self-product-ffn-latency-matched-h100-preregistration.md").read_text()
    for phrase in ("median ratio `<= 1.00`", "95% upper bound `<= 1.02`", "strictly lower", "No other width"):
        assert phrase in text
