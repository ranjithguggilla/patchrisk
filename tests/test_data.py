import pandas as pd

from patchrisk import config as C


def test_columns_and_no_missing_values(data):
    commits, links = data
    assert set(C.FEATURES) <= set(commits.columns)
    assert commits[C.FEATURES].isna().sum().sum() == 0
    assert {"repo", "fix_sha", "fix_ts", "intro_sha", "path"} <= set(links.columns)


def test_every_link_points_at_a_known_earlier_commit(data):
    commits, links = data
    ts = commits.set_index(["repo", "sha"])["ts"]
    keys = list(zip(links["repo"], links["intro_sha"], strict=False))
    known = pd.Index(keys).isin(ts.index)
    assert known.mean() > 0.97
    intro_ts = ts.reindex(pd.MultiIndex.from_arrays([links["repo"], links["intro_sha"]])).to_numpy()
    ok = ~pd.isna(intro_ts)
    assert (intro_ts[ok] <= links["fix_ts"].to_numpy()[ok]).all()


def test_pinned_repositories_are_full_shas():
    for url, sha in C.REPOS.values():
        assert url.startswith("https://github.com/") and len(sha) == 40


def test_hand_audit_matches_the_mined_data(data):
    commits, _ = data
    audit = pd.read_csv(C.DATA_DIR / "fix_message_audit.csv")
    assert len(audit) == 40
    merged = audit.merge(commits[["repo", "sha", "is_fix_msg"]], on=["repo", "sha"])
    assert len(merged) == 40 and (merged["is_fix_msg"] == 1).all()
    assert audit["reads_as_bug_fix"].mean() == 0.6
