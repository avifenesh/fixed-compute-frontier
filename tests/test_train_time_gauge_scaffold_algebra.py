from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.train_time_gauge_scaffold_algebra import run


def test_training_gain_folds_exactly_for_every_chart_and_consumer():
    result = run()
    for errors in result["multiple_consumer_export_max_errors"].values():
        assert max(errors) < 1e-14
    assert result["served_norm_parameters"] == result["baseline_norm_parameters"]
    assert result["matrix_shapes_unchanged"]
