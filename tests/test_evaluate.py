import numpy as np
import pandas as pd

from patchrisk import evaluate as V


def test_ece_is_small_for_calibrated_scores(rng):
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert V.ece(y, p) < 0.02


def test_ece_is_large_for_constant_wrong_scores():
    y = np.array([1] * 50 + [0] * 50)
    assert V.ece(y, np.full(100, 0.05) + np.linspace(0, 1e-6, 100)) > 0.4


def test_effort_curve_endpoints():
    y = np.array([1, 0, 1, 0])
    e, r = V.effort_curve(y, np.array([4, 3, 2, 1.0]), np.array([10, 10, 10, 10.0]))
    assert e[0] == 0 and r[0] == 0 and e[-1] == 1 and r[-1] == 1
    assert (np.diff(e) >= 0).all() and (np.diff(r) >= 0).all()


def test_recall_at_effort_rewards_a_good_order():
    y = np.array([1, 1, 0, 0, 0])
    effort = np.ones(5)
    good = V.recall_at_effort(y, np.array([5, 4, 3, 2, 1.0]), effort, 0.4)
    bad = V.recall_at_effort(y, np.array([1, 2, 3, 4, 5.0]), effort, 0.4)
    assert good == 1.0 and bad == 0.0


def test_popt_is_one_for_the_optimal_order():
    y = np.array([1, 0, 1, 0, 0, 1])
    effort = np.array([5, 20, 3, 50, 8, 2.0])
    assert abs(V.popt(y, y / effort, effort) - 1.0) < 1e-9


def test_initial_false_alarms():
    y = np.array([0, 0, 1, 1])
    assert V.initial_false_alarms(y, np.array([0.9, 0.8, 0.7, 0.1])) == 2
    assert V.initial_false_alarms(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 3


def test_cluster_bootstrap_brackets_the_point_estimate(rng):
    df = pd.DataFrame({"ts": np.repeat(np.arange(30) * 2.7e6 + 1.6e9, 20), "repo": "r", "x": rng.normal(size=600)})
    df = V.add_blocks(df)
    lo, hi = V.cluster_bootstrap(df, lambda d: d["x"].mean(), n=200)
    assert lo < df["x"].mean() < hi


def test_summarise_has_expected_keys(rng):
    y = rng.integers(0, 2, 200)
    out = V.summarise(y, rng.random(200), np.ones(200))
    assert {"auc", "avg_precision", "log_loss", "ece", "recall_at_20pct_loc", "popt", "ifa"} <= set(out)
