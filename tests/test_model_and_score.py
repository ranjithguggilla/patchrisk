import numpy as np
import pandas as pd

from patchrisk import app, score
from patchrisk import config as C
from patchrisk import model as M


def test_candidates_fit_and_predict(data):
    commits, _ = data
    df = commits.sample(600, random_state=0).copy()
    df["bug"] = (df["churn"] > df["churn"].median()).astype(int)
    for name, factory in M.CANDIDATES.items():
        m = factory()
        m.fit(df[C.FEATURES], df["bug"])
        p = M.predict(m, df)
        assert p.shape == (600,) and ((p >= 0) & (p <= 1)).all(), name


def test_signed_log_keeps_sign():
    out = M._signed_log([-9, 0, 9])
    assert out[0] < 0 < out[2] and out[1] == 0


def test_one_se_prefers_the_simpler_model(data):
    commits, _ = data
    df = commits.sample(800, random_state=2).copy()
    df["bug"] = (df["churn"] > 40).astype(int)
    models = {
        "logistic": M.make_logistic().fit(df[C.FEATURES], df["bug"]),
        "boosting": M.make_boosting(iters=40).fit(df[C.FEATURES], df["bug"]),
    }
    pick, losses, ses = M.one_se_pick(df, models, ["logistic", "boosting"])
    best = min(losses, key=losses.get)
    assert losses[pick] <= losses[best] + ses[best]


def test_reasons_are_plain_sentences(small_bundle):
    row = pd.Series({**small_bundle["background"], "la": 900.0, "ld": 400.0, "churn": 1300.0})
    out = M.reasons(small_bundle["model"], row, small_bundle["background_frame"])
    assert len(out) == 3 and all("risk" in r for r in out)


def test_big_change_scores_higher_than_a_tiny_one(small_bundle):
    tiny = score.score_commit({"la": 2, "ld": 1, "nf": 1, "n_code_files": 1}, small_bundle)
    big = score.score_commit({"la": 800, "ld": 300, "nf": 25, "n_code_files": 25, "nd": 6}, small_bundle)
    assert big["probability"] > tiny["probability"]
    assert big["band"] in {"medium", "high"}


def test_missing_features_fall_back_to_the_median(small_bundle):
    res = score.score_commit({}, small_bundle)
    assert 0 < res["probability"] < 1


def test_rank_queue_orders_by_risk_per_line(small_bundle, data):
    commits, _ = data
    q = score.rank_queue(commits.head(300), small_bundle, top=10)
    assert len(q) == 10
    assert q["risk_per_100_lines"].is_monotonic_decreasing


def test_app_evaluate_returns_text(small_bundle):
    head, why = app.evaluate(small_bundle, la=50, ld=5, nf=2, nd=1, author_exp=0, files_prior_changes=3, touches_tests=0, is_fix_msg=0)
    assert "Estimated chance" in head and why.startswith("- ")


def test_band_thresholds():
    c = {"high": 0.5, "medium": 0.2}
    assert score.band(0.6, c) == "high" and score.band(0.3, c) == "medium" and score.band(0.1, c) == "low"


def test_rank_queue_handles_zero_line_commit(small_bundle, data):
    commits, _ = data
    row = commits.head(5).copy()
    row.loc[row.index[0], "churn"] = 0
    out = score.rank_queue(row, small_bundle, top=5)
    assert np.isfinite(out["risk_per_100_lines"]).all()


def test_gradio_app_builds(small_bundle, tmp_path, monkeypatch):
    import joblib

    path = tmp_path / "bundle.joblib"
    joblib.dump(small_bundle, path)
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame({"repo": ["flask"], "sha": ["a"], "subject": ["s"], "churn": [10], "risk": [0.2]}).to_csv(
        tables / "top_risk_test.csv", index=False
    )
    monkeypatch.setattr(C, "OUT_DIR", tmp_path)
    demo = app.build(path)
    assert demo is not None
