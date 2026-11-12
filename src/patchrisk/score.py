"""Score one commit from its feature values; used by the app and the command line."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import model as M

DEFAULT_BUNDLE = C.OUT_DIR / "bundle.joblib"


def load_bundle(path=DEFAULT_BUNDLE) -> dict:
    return joblib.load(path)


def band(p: float, cutoffs: dict) -> str:
    if p >= cutoffs["high"]:
        return "high"
    if p >= cutoffs["medium"]:
        return "medium"
    return "low"


def score_commit(values: dict, bundle: dict) -> dict:
    """Return probability, risk band and top reasons for one commit.

    Missing features fall back to the training median, so a partial form still scores.
    """
    row = {f: float(values.get(f, bundle["background"][f])) for f in bundle["features"]}
    if "churn" not in values and ("la" in values or "ld" in values):
        row["churn"] = row["la"] + row["ld"]
    if "lines_per_file" not in values and row["n_code_files"] > 0:
        row["lines_per_file"] = row["churn"] / row["n_code_files"]
    frame = pd.DataFrame([row])
    raw = float(bundle["model"].predict_proba(frame[bundle["features"]])[0, 1])
    iso = bundle.get("isotonic")
    prob = float(iso.predict([raw])[0]) if iso is not None else raw
    series = pd.Series(row)
    return {
        "probability": prob,
        "raw_probability": raw,
        "band": band(raw, bundle["risk_cutoffs"]),
        "risk_per_100_lines": 100 * raw / max(row["churn"], 1.0),
        "reasons": M.reasons(bundle["model"], series, bundle["background_frame"]),
    }


def rank_queue(frame: pd.DataFrame, bundle: dict, top: int = 20) -> pd.DataFrame:
    """Order commits for review: highest bug probability per changed line first."""
    p = bundle["model"].predict_proba(frame[bundle["features"]])[:, 1]
    out = frame[["repo", "sha", "subject", "churn"]].copy()
    out["risk"] = np.round(p, 3)
    out["risk_per_100_lines"] = np.round(100 * p / frame["churn"].clip(lower=1), 2)
    return out.sort_values("risk_per_100_lines", ascending=False).head(top).reset_index(drop=True)
