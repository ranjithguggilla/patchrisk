"""Candidate models, one-standard-error selection and plain-language reasons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config as C

# Columns with a long right tail get log1p before scaling so the linear model is not
# dominated by a handful of huge refactors.
HEAVY = [
    "la",
    "ld",
    "churn",
    "nf",
    "nd",
    "msg_len",
    "n_code_files",
    "author_exp",
    "author_recent",
    "author_age_days",
    "files_prior_changes",
    "files_age_days",
    "files_prior_devs",
    "lines_per_file",
    "cc_max",
]


def _signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def make_logistic(c: float = 0.3, features: list[str] | None = None) -> Pipeline:
    features = features or C.FEATURES
    heavy = [f for f in features if f in HEAVY]
    other = [f for f in features if f not in HEAVY]
    pre = ColumnTransformer(
        [
            ("heavy", Pipeline([("log", FunctionTransformer(_signed_log)), ("sc", StandardScaler())]), heavy),
            ("other", StandardScaler(), other),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=c, max_iter=2000))])


def make_boosting(depth: int = 3, lr: float = 0.05, iters: int = 250, seed: int = C.SEED, features: list[str] | None = None) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        max_depth=depth, learning_rate=lr, max_iter=iters, l2_regularization=1.0, min_samples_leaf=40, random_state=seed
    )
    return Pipeline([("clf", clf)])


def make_forest(seed: int = C.SEED, features: list[str] | None = None) -> Pipeline:
    return Pipeline([("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=15, max_features=0.5, n_jobs=2, random_state=seed))])


CANDIDATES = {
    "logistic": make_logistic,
    "boosting": make_boosting,
    "forest": make_forest,
}


def fit_candidates(train: pd.DataFrame) -> dict[str, Pipeline]:
    X, y = train[C.FEATURES], train["bug"]
    out = {}
    for name, factory in CANDIDATES.items():
        m = factory()
        m.fit(X, y)
        out[name] = m
    return out


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[C.FEATURES])[:, 1]


def one_se_pick(val: pd.DataFrame, models: dict[str, Pipeline], order: list[str]):
    """Simplest model whose validation log loss is within one SE of the best."""
    y = val["bug"].to_numpy()
    losses, ses = {}, {}
    for name, m in models.items():
        p = np.clip(predict(m, val), 1e-6, 1 - 1e-6)
        per = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        losses[name], ses[name] = float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per)))
    best = min(losses, key=losses.get)
    limit = losses[best] + ses[best]
    pick = next(n for n in order if losses[n] <= limit)
    return pick, losses, ses


# ---- plain-language reasons ------------------------------------------------------------

PHRASES = {
    "la": ("adds {v:.0f} lines", "adds few lines"),
    "ld": ("deletes {v:.0f} lines", "deletes little"),
    "churn": ("touches {v:.0f} lines in total", "is a small change"),
    "nf": ("edits {v:.0f} Python files", "edits only a file or two"),
    "nd": ("spans {v:.0f} directories", "stays in one directory"),
    "entropy": ("spreads its edits unevenly over many files", "keeps its edits focused"),
    "msg_len": ("has a long commit message", "has a terse commit message"),
    "is_fix_msg": ("is itself a bug fix", "is not a bug fix"),
    "touches_tests": ("changes tests", "does not change any tests"),
    "touches_docs": ("also changes docs", "does not touch docs"),
    "n_code_files": ("edits {v:.0f} code files", "edits few code files"),
    "cc_delta": ("raises cyclomatic complexity by {v:.0f}", "does not add complexity"),
    "cc_max": ("touches a file with complexity {v:.0f}", "touches only simple files"),
    "author_exp": ("comes from an author with {v:.0f} earlier commits", "comes from a newer contributor"),
    "author_recent": ("comes from someone active in the last 90 days", "comes from someone who was inactive lately"),
    "author_age_days": ("comes from a long-standing contributor", "comes from a recent contributor"),
    "files_prior_changes": ("edits files changed {v:.0f} times before", "edits rarely changed files"),
    "files_age_days": ("edits files untouched for {v:.0f} days", "edits files changed recently"),
    "files_prior_devs": ("edits files with {v:.0f} earlier authors", "edits files with few earlier authors"),
    "hour": ("was committed at an unusual hour", "was committed in the usual hours"),
    "weekday": ("was committed on a weekend", "was committed on a weekday"),
    "new_author": ("is the author's first commit here", "is not a first-time author"),
    "lines_per_file": ("averages {v:.0f} changed lines per file", "averages few changed lines per file"),
    "test_ratio": ("is mostly test code", "has little test code"),
}


def reasons(model, row: pd.Series, background: pd.DataFrame, top: int = 3) -> list[str]:
    """Occlusion reasons: swap each feature for its background median, see the log-odds move."""
    x = pd.DataFrame([row[C.FEATURES].astype(float)])
    base = float(model.predict_proba(x)[0, 1])
    med = background[C.FEATURES].median()
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    deltas = {}
    for f in C.FEATURES:
        alt = x.copy()
        alt[f] = med[f]
        deltas[f] = logit(base) - logit(float(model.predict_proba(alt)[0, 1]))
    ranked = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:top]
    out = []
    for f, d in ranked:
        hi, lo = PHRASES[f]
        text = hi.format(v=float(x[f].iloc[0])) if d > 0 else lo
        out.append(f"{'raises' if d > 0 else 'lowers'} risk: this commit {text}")
    return out
