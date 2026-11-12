"""Turn SZZ links into labels, and keep track of what was knowable when."""

from __future__ import annotations

import pandas as pd

from . import config as C

DAY = 86400.0


def to_ts(date: str) -> float:
    return pd.Timestamp(date, tz="UTC").timestamp()


def label_as_of(commits: pd.DataFrame, links: pd.DataFrame, as_of: float | None = None) -> pd.Series:
    """1 when a later fix, committed on or before `as_of`, blamed this commit.

    `as_of=None` means hindsight: every fix in the mined history counts.
    """
    use = links if as_of is None else links[links["fix_ts"] <= as_of]
    guilty = set(zip(use["repo"], use["intro_sha"], strict=False))
    keys = zip(commits["repo"], commits["sha"], strict=False)
    return pd.Series([int(k in guilty) for k in keys], index=commits.index, name="bug")


def days_to_first_fix(commits: pd.DataFrame, links: pd.DataFrame) -> pd.Series:
    first = links.groupby(["repo", "intro_sha"])["fix_ts"].min()
    idx = pd.MultiIndex.from_arrays([commits["repo"], commits["sha"]])
    got = first.reindex(idx).to_numpy()
    return pd.Series((got - commits["ts"].to_numpy()) / DAY, index=commits.index, name="days_to_fix")


def split_frames(commits: pd.DataFrame, links: pd.DataFrame, train_labels: str = "known_dropped"):
    """Return train, validation, test frames with a `bug` column.

    Validation and test labels are mature (all fixes known), because those commits are
    over a year old. Training labels follow `train_labels`:

    known_dropped  fixes up to the train cutoff only, and commits younger than the label
                   window at the cutoff are dropped (the honest, deployable setting)
    known_kept     same labels, but the young tail stays and is mostly mislabelled clean
    hindsight      labels use every fix ever mined, including ones after the cutoff
    """
    t_end, v_end, e_end = (to_ts(d) for d in (C.TRAIN_END, C.VAL_END, C.TEST_END))
    c = commits.sort_values("ts").reset_index(drop=True)
    train = c[c["ts"] < t_end].copy()
    val = c[(c["ts"] >= t_end) & (c["ts"] < v_end)].copy()
    test = c[(c["ts"] >= v_end) & (c["ts"] < e_end)].copy()
    val["bug"] = label_as_of(val, links)
    test["bug"] = label_as_of(test, links)
    if train_labels == "hindsight":
        train["bug"] = label_as_of(train, links)
    else:
        train["bug"] = label_as_of(train, links, as_of=t_end)
        if train_labels == "known_dropped":
            train = train[train["ts"] <= t_end - C.LABEL_WINDOW_DAYS * DAY].copy()
    return train, val, test
