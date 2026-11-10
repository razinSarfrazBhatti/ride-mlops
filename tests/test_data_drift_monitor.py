import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_drift_monitor import summarize_feature_drift


def test_summarize_feature_drift_detects_stability_and_shift():
    stable = summarize_feature_drift([1, 2, 3, 4], [1, 2, 3, 4], "trip_distance")
    assert stable["ks_stat"] == 0
    assert stable["p_value"] == 1.0
    assert stable["mean_delta"] == 0.0

    shifted = summarize_feature_drift([1, 1, 1, 1], [5, 5, 5, 5], "trip_distance")
    assert shifted["mean_delta"] > 0
    assert shifted["ks_stat"] > 0.5
    assert shifted["feature"] == "trip_distance"
