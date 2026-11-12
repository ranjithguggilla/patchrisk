import pandas as pd

from patchrisk import config as C
from patchrisk import labels as L


def _toy():
    commits = pd.DataFrame({"repo": ["r"] * 4, "sha": list("abcd"), "ts": [1e9, 1.1e9, 1.2e9, 1.3e9]})
    links = pd.DataFrame(
        {"repo": ["r", "r"], "fix_sha": ["x", "y"], "fix_ts": [1.15e9, 1.35e9], "intro_sha": ["a", "b"], "path": ["f.py", "f.py"]}
    )
    return commits, links


def test_hindsight_labels_every_link():
    c, l_ = _toy()
    assert L.label_as_of(c, l_).tolist() == [1, 1, 0, 0]


def test_as_of_ignores_later_fixes():
    c, l_ = _toy()
    assert L.label_as_of(c, l_, as_of=1.2e9).tolist() == [1, 0, 0, 0]


def test_days_to_first_fix():
    c, l_ = _toy()
    d = L.days_to_first_fix(c, l_)
    assert round(d.iloc[0]) == round(0.15e9 / 86400)
    assert d.iloc[2] != d.iloc[2]  # NaN for a commit nobody blamed


def test_split_is_forward_and_non_overlapping(data):
    commits, links = data
    train, val, test = L.split_frames(commits, links)
    assert train["ts"].max() < val["ts"].min() <= val["ts"].max() < test["ts"].min()
    assert test["ts"].max() < L.to_ts(C.TEST_END)


def test_dropped_variant_removes_young_tail(data):
    commits, links = data
    dropped, _, _ = L.split_frames(commits, links, "known_dropped")
    kept, _, _ = L.split_frames(commits, links, "known_kept")
    cutoff = L.to_ts(C.TRAIN_END) - C.LABEL_WINDOW_DAYS * 86400
    assert dropped["ts"].max() <= cutoff
    assert len(kept) > len(dropped)


def test_known_labels_are_a_subset_of_hindsight(data):
    commits, links = data
    known, _, _ = L.split_frames(commits, links, "known_kept")
    hind, _, _ = L.split_frames(commits, links, "hindsight")
    assert (known["bug"] <= hind["bug"]).all()
    assert known["bug"].sum() < hind["bug"].sum()
