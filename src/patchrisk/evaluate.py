"""Metrics: ranking, calibration, and effort-aware review curves with clustered bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1, 2
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def effort_curve(y: np.ndarray, score: np.ndarray, effort: np.ndarray):
    """Cumulative share of effort (lines) against cumulative share of buggy commits found."""
    order = np.argsort(-score, kind="stable")
    e = np.cumsum(effort[order]) / effort.sum()
    r = np.cumsum(y[order]) / max(y.sum(), 1)
    return np.concatenate([[0], e]), np.concatenate([[0], r])


def recall_at_effort(y, score, effort, budget: float = 0.2) -> float:
    e, r = effort_curve(y, score, effort)
    return float(np.interp(budget, e, r))


def popt(y, score, effort) -> float:
    """Kamei's P_opt: 1 - normalised area between the model curve and the optimal curve."""
    e, r = effort_curve(y, score, effort)
    area = np.trapezoid(r, e)
    eo, ro = effort_curve(y, y / np.maximum(effort, 1), effort)
    ew, rw = effort_curve(y, -y / np.maximum(effort, 1), effort)
    a_opt, a_worst = np.trapezoid(ro, eo), np.trapezoid(rw, ew)
    return float((area - a_worst) / (a_opt - a_worst)) if a_opt != a_worst else float("nan")


def initial_false_alarms(y, score) -> int:
    """Clean commits a reviewer reads before the first buggy one."""
    order = np.argsort(-score, kind="stable")
    hits = np.flatnonzero(y[order] == 1)
    return int(hits[0]) if len(hits) else len(y)


def summarise(y, p, effort) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, pc)),
        "ece": ece(y, pc),
        "recall_at_20pct_loc": recall_at_effort(y, p, np.asarray(effort), 0.2),
        "popt": popt(y, p, np.asarray(effort)),
        "ifa": initial_false_alarms(y, p),
    }


def cluster_bootstrap(df: pd.DataFrame, fn, cluster: str = "block", n: int = 400, seed: int = 7):
    """Resample whole clusters (repo-months) so correlated commits move together."""
    rng = np.random.default_rng(seed)
    groups = {k: g.index.to_numpy() for k, g in df.groupby(cluster)}
    keys = list(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        try:
            vals.append(fn(df.loc[idx]))
        except ValueError:
            continue
    arr = np.asarray(vals, dtype=float)
    return float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def add_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    month = pd.to_datetime(df["ts"], unit="s", utc=True).dt.strftime("%Y-%m")
    df["block"] = df["repo"] + "|" + month
    return df.reset_index(drop=True)
