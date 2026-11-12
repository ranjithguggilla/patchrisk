"""End-to-end study: splits, models, label-maturity and split-protocol audits, economics."""

from __future__ import annotations

import json
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from . import config as C
from . import economics as E
from . import evaluate as V
from . import labels as L
from . import model as M

ORDER = ["logistic", "boosting", "forest"]

GROUPS = {
    "size": ["la", "ld", "churn", "nf", "nd", "entropy", "lines_per_file", "n_code_files"],
    "file history": ["files_prior_changes", "files_age_days", "files_prior_devs"],
    "author": ["author_exp", "author_recent", "author_age_days", "new_author"],
    "complexity": ["cc_delta", "cc_max"],
    "process": ["msg_len", "is_fix_msg", "touches_tests", "touches_docs", "test_ratio", "hour", "weekday"],
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(C.COMMITS_CSV), pd.read_csv(C.FIXES_CSV)


def _metrics(model, df, p=None):
    p = M.predict(model, df) if p is None else p
    return V.summarise(df["bug"], p, df["churn"].clip(lower=1))


def _ci(df, fn):
    return V.cluster_bootstrap(V.add_blocks(df), fn, n=300)


def run_study(commits: pd.DataFrame, links: pd.DataFrame) -> dict:
    out: dict = {}
    c = commits.copy()
    c["bug"] = L.label_as_of(c, links)
    c["year"] = pd.to_datetime(c["ts"], unit="s", utc=True).dt.year

    # ---- data summary --------------------------------------------------------------
    train, val, test = L.split_frames(commits, links, "known_dropped")
    out["data"] = {
        "commits": int(len(c)),
        "repos": int(c["repo"].nunique()),
        "fix_commits_by_message": int(c["is_fix_msg"].sum()),
        "szz_links": int(len(links)),
        "bug_rate_all": float(c["bug"].mean()),
        "per_repo": {r: {"commits": int(len(g)), "bug_rate": float(g["bug"].mean())} for r, g in c.groupby("repo")},
        "split": {
            k: {"commits": int(len(d)), "bug_rate": float(d["bug"].mean()), "lines": int(d["churn"].sum())}
            for k, d in {"train": train, "validation": val, "test": test}.items()
        },
    }
    by_year = c.groupby("year").agg(commits=("bug", "size"), bug_rate=("bug", "mean"))
    by_year.to_csv(C.OUT_DIR / "tables" / "bug_rate_by_year.csv")

    # ---- model comparison ------------------------------------------------------------
    models = M.fit_candidates(train)
    pick, vloss, vse = M.one_se_pick(val, models, ORDER)
    shipped = models[pick]
    comp = {}
    for name, m in models.items():
        comp[name] = {"val_log_loss": vloss[name], "val_se": vse[name], **_metrics(m, test)}
    rng = np.random.default_rng(C.SEED)
    base = {
        "random": rng.random(len(test)),
        "size_rule (churn)": test["churn"].to_numpy(dtype=float),
        "file_history_rule": (test["files_prior_changes"]).to_numpy(dtype=float),
    }
    rate = float(train["bug"].mean())
    for name, s in base.items():
        sc = pd.Series(s).rank(pct=True).to_numpy()
        row = V.summarise(test["bug"], sc, test["churn"].clip(lower=1))
        row["log_loss"] = float("nan")
        row["ece"] = float("nan")
        comp[name] = row
    prior = np.full(len(test), rate)
    comp["train_base_rate"] = {**V.summarise(test["bug"], prior + 1e-9 * rng.random(len(test)), test["churn"].clip(lower=1))}
    size_cols = GROUPS["size"]
    m_size = M.make_logistic(features=size_cols)
    m_size.fit(train[size_cols], train["bug"])
    ps_size = m_size.predict_proba(test[size_cols])[:, 1]
    comp["size_features_only_logistic"] = V.summarise(test["bug"], ps_size, test["churn"].clip(lower=1))
    out["models"] = comp
    out["shipped_model"] = pick

    tb = V.add_blocks(test)
    ps = M.predict(shipped, tb)
    tb["p"] = ps
    auc_ci = V.cluster_bootstrap(tb, lambda d: V.summarise(d["bug"], d["p"], d["churn"].clip(lower=1))["auc"])
    rec_ci = V.cluster_bootstrap(
        tb, lambda d: V.recall_at_effort(d["bug"].to_numpy(), d["p"].to_numpy(), d["churn"].clip(lower=1).to_numpy())
    )

    def size_rec(d):
        return V.recall_at_effort(d["bug"].to_numpy(), d["churn"].to_numpy(dtype=float), d["churn"].clip(lower=1).to_numpy())

    gap_ci = V.cluster_bootstrap(
        tb, lambda d: V.recall_at_effort(d["bug"].to_numpy(), d["p"].to_numpy(), d["churn"].clip(lower=1).to_numpy()) - size_rec(d)
    )
    tb["churn1"] = tb["churn"].clip(lower=1)
    tb["dens"] = tb["p"] / tb["churn1"]
    from sklearn.metrics import roc_auc_score

    auc_gap_ci = V.cluster_bootstrap(tb, lambda d: roc_auc_score(d["bug"], d["p"]) - roc_auc_score(d["bug"], d["churn"]))

    def dens_rec(d):
        return V.recall_at_effort(d["bug"].to_numpy(), d["dens"].to_numpy(), d["churn1"].to_numpy())

    dens_gap_ci = V.cluster_bootstrap(
        tb, lambda d: dens_rec(d) - V.recall_at_effort(d["bug"].to_numpy(), d["churn"].to_numpy(dtype=float), d["churn1"].to_numpy())
    )
    rand_gap_ci = V.cluster_bootstrap(tb, lambda d: dens_rec(d) - 0.2)
    out["shipped_ci"] = {
        "auc": auc_ci,
        "recall_at_20pct_loc_by_probability": rec_ci,
        "auc_gap_vs_size_rule": auc_gap_ci,
        "recall_at_20pct_loc_by_risk_density": [dens_rec(tb), *dens_gap_ci],
        "density_minus_random_20pct": [dens_rec(tb) - 0.2, *rand_gap_ci],
        "recall_gap_probability_vs_size": gap_ci,
    }
    out["models"]["risk_density (p per changed line)"] = V.summarise(test["bug"], tb["dens"].to_numpy(), tb["churn1"])
    out["models"]["risk_density (p per changed line)"]["log_loss"] = float("nan")
    out["models"]["risk_density (p per changed line)"]["ece"] = float("nan")

    # calibration: isotonic is judged out of sample inside validation (fit on the earlier half,
    # score the later half), then refit on all of validation if it helps
    pv = M.predict(shipped, val)
    half = len(val) // 2
    iso_half = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv[:half], val["bug"].to_numpy()[:half])
    yv2, pv2 = val["bug"].to_numpy()[half:], pv[half:]
    ece_raw_v, ece_iso_v = V.ece(yv2, pv2), V.ece(yv2, iso_half.predict(pv2))
    use_iso = ece_iso_v < ece_raw_v
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, val["bug"])
    out["calibration"] = {
        "val_second_half_ece_raw": ece_raw_v,
        "val_second_half_ece_isotonic": ece_iso_v,
        "isotonic_shipped": bool(use_iso),
        "test_ece_raw": V.ece(test["bug"].to_numpy(), ps),
        "test_ece_isotonic": V.ece(test["bug"].to_numpy(), iso.predict(ps)),
        "mean_prediction_test": float(ps.mean()),
        "observed_test": float(test["bug"].mean()),
        "validation_bug_rate": float(val["bug"].mean()),
        "train_bug_rate": float(train["bug"].mean()),
    }

    # ---- label maturity study ----------------------------------------------------------
    maturity = {}
    for variant in ("known_dropped", "known_kept", "hindsight"):
        tr, _, te = L.split_frames(commits, links, variant)
        m = M.CANDIDATES[pick]()
        m.fit(tr[C.FEATURES], tr["bug"])
        maturity[variant] = {"train_rows": int(len(tr)), "train_bug_rate": float(tr["bug"].mean()), **_metrics(m, te)}
    out["label_maturity"] = maturity
    age = (L.to_ts(C.TRAIN_END) - train["ts"]) / 86400
    young = c[(c["ts"] < L.to_ts(C.TRAIN_END)) & (c["ts"] >= L.to_ts(C.TRAIN_END) - 365 * 86400)]
    out["label_maturity"]["young_tail"] = {
        "commits": int(len(young)),
        "bug_rate_known_at_cutoff": float(L.label_as_of(young, links, as_of=L.to_ts(C.TRAIN_END)).mean()),
        "bug_rate_with_hindsight": float(L.label_as_of(young, links).mean()),
        "oldest_train_age_days": float(age.max()),
    }
    curve = []
    ts_end = L.to_ts(C.TRAIN_END)
    for lo in range(0, 3000, 90):
        sel = c[(c["ts"] < ts_end - lo * 86400) & (c["ts"] >= ts_end - (lo + 90) * 86400)]
        if len(sel) > 20:
            curve.append(
                {
                    "age_days": lo + 45,
                    "commits": len(sel),
                    "known": float(L.label_as_of(sel, links, as_of=ts_end).mean()),
                    "hindsight": float(L.label_as_of(sel, links).mean()),
                }
            )
    pd.DataFrame(curve).to_csv(C.OUT_DIR / "tables" / "label_maturity_curve.csv", index=False)

    # ---- split protocol study ----------------------------------------------------------
    mature = c[c["ts"] < L.to_ts(C.TEST_END)].reset_index(drop=True)
    tr_r, te_r = train_test_split(mature, test_size=0.3, random_state=C.SEED, stratify=mature["bug"])
    m_r = M.CANDIDATES[pick]()
    m_r.fit(tr_r[C.FEATURES], tr_r["bug"])
    proto = {"random_rows": _metrics(m_r, te_r)}
    proto["forward_in_time"] = _metrics(shipped, test)
    # repository-held-out versus within-repository, scored on each repo's test period
    loro_rows = []
    for repo in sorted(c["repo"].unique()):
        te_repo = test[test["repo"] == repo]
        if te_repo["bug"].sum() < 15:
            continue
        tr_other = train[train["repo"] != repo]
        m_o = M.CANDIDATES[pick]()
        m_o.fit(tr_other[C.FEATURES], tr_other["bug"])
        within = M.predict(shipped, te_repo)
        cross = M.predict(m_o, te_repo)
        loro_rows.append(
            {
                "repo": repo,
                "test_commits": len(te_repo),
                "test_bugs": int(te_repo["bug"].sum()),
                "auc_trained_on_all_repos": V.summarise(te_repo["bug"], within, te_repo["churn"].clip(lower=1))["auc"],
                "auc_repo_held_out": V.summarise(te_repo["bug"], cross, te_repo["churn"].clip(lower=1))["auc"],
            }
        )
    loro = pd.DataFrame(loro_rows)
    loro.to_csv(C.OUT_DIR / "tables" / "repo_held_out.csv", index=False)
    proto["repo_held_out_mean_auc"] = float(loro["auc_repo_held_out"].mean())
    proto["all_repos_mean_auc"] = float(loro["auc_trained_on_all_repos"].mean())
    out["split_protocols"] = proto

    # ---- feature-group ablation (retrain without a group, score on test) -------------------
    ablation = {}
    full = _metrics(shipped, test)
    for g, cols in GROUPS.items():
        keep = [f for f in C.FEATURES if f not in cols]
        m = M.CANDIDATES[pick](features=keep)
        m.fit(train[keep], train["bug"])
        p = m.predict_proba(test[keep])[:, 1]
        r = V.summarise(test["bug"], p, test["churn"].clip(lower=1))
        ablation[g] = {"auc": r["auc"], "auc_drop": full["auc"] - r["auc"], "recall_at_20pct_loc": r["recall_at_20pct_loc"]}
    out["feature_groups"] = ablation

    # ---- economics ----------------------------------------------------------------------
    a = E.Assumptions()
    budgets = [round(x, 2) for x in np.arange(0.05, 1.0001, 0.05)]
    vscore = M.predict(shipped, val) / val["churn"].clip(lower=1).to_numpy()
    vs = E.sweep(val["bug"], vscore, val["churn"], budgets, a)
    best = max(vs, key=lambda r: r["net"])
    chosen = best["budget"]
    econ = {"assumptions": {**asdict(a), "cost_per_loc": a.cost_per_loc}, "chosen_budget": chosen}
    policies = {
        "model": ps,
        "largest_first": test["churn"].to_numpy(dtype=float),
        "random": np.random.default_rng(C.SEED).random(len(test)),
        "risk_density": ps / test["churn"].clip(lower=1).to_numpy(),
        "smallest_first": 1.0 / test["churn"].clip(lower=1).to_numpy(),
    }
    sweeps = {}
    for name, s in policies.items():
        rows = E.sweep(test["bug"], s, test["churn"], budgets, a)
        sweeps[name] = rows
        r = E.net_benefit(test["bug"], s, test["churn"], chosen, a)
        econ[name] = {**r, "net_per_1000_commits": r["net"] / len(test) * 1000}
    pd.DataFrame([{"policy": k, **r} for k, rows in sweeps.items() for r in rows]).to_csv(
        C.OUT_DIR / "tables" / "budget_sweep.csv", index=False
    )

    def net_gap(d, other="churn1"):
        return (
            (
                E.net_benefit(d["bug"], d["dens"], d["churn1"], chosen, a)["net"]
                - E.net_benefit(d["bug"], d[other], d["churn1"], chosen, a)["net"]
            )
            / len(d)
            * 1000
        )

    econ["density_minus_largest_first_per_1000"] = {"point": net_gap(tb), "ci": V.cluster_bootstrap(tb, net_gap)}
    tb["rnd"] = np.random.default_rng(C.SEED).random(len(tb))
    econ["density_minus_random_per_1000"] = {"point": net_gap(tb, "rnd"), "ci": V.cluster_bootstrap(tb, lambda d: net_gap(d, "rnd"))}
    tb["small"] = 1.0 / tb["churn1"]

    def small_rec(d):
        return V.recall_at_effort(d["bug"].to_numpy(), (1.0 / d["churn1"]).to_numpy(), d["churn1"].to_numpy())

    out["shipped_ci"]["smallest_first_recall_at_20pct_loc"] = small_rec(tb)
    out["shipped_ci"]["density_minus_smallest_first_recall_20pct"] = [
        dens_rec(tb) - small_rec(tb),
        *V.cluster_bootstrap(tb, lambda d: dens_rec(d) - small_rec(d)),
    ]
    econ["density_minus_smallest_first_per_1000"] = {
        "point": net_gap(tb, "small"),
        "ci": V.cluster_bootstrap(tb, lambda d: net_gap(d, "small")),
    }
    # commit-count capacity: review the top share of commits whatever their size
    lift = {}
    yv = test["bug"].to_numpy()
    for name, sc in {
        "probability": ps,
        "largest_first": test["churn"].to_numpy(dtype=float),
        "smallest_first": tb["small"].to_numpy(),
        "random": tb["rnd"].to_numpy(),
    }.items():
        order = np.argsort(-sc, kind="stable")
        lift[name] = {f"top_{int(q * 100)}pct": float(yv[order[: int(round(q * len(yv)))]].sum() / yv.sum()) for q in (0.1, 0.2, 0.3)}
    econ["commit_count_recall"] = lift

    def top20(d, col):
        k = int(round(0.2 * len(d)))
        o = np.argsort(-d[col].to_numpy(dtype=float), kind="stable")[:k]
        return d["bug"].to_numpy()[o].sum() / d["bug"].sum()

    econ["top20pct_commits_prob_minus_largest"] = [
        top20(tb, "p") - top20(tb, "churn"),
        *V.cluster_bootstrap(tb, lambda d: top20(d, "p") - top20(d, "churn")),
    ]
    econ["top20pct_commits_prob_minus_random"] = [
        top20(tb, "p") - top20(tb, "rnd"),
        *V.cluster_bootstrap(tb, lambda d: top20(d, "p") - top20(d, "rnd")),
    ]
    pk = np.argsort(-ps, kind="stable")[: int(round(0.2 * len(yv)))]
    lk = np.argsort(-test["churn"].to_numpy(dtype=float), kind="stable")[: int(round(0.2 * len(yv)))]
    econ["top20pct_commits_lines_share"] = {
        "probability": float(test["churn"].to_numpy()[pk].sum() / test["churn"].sum()),
        "largest_first": float(test["churn"].to_numpy()[lk].sum() / test["churn"].sum()),
    }
    econ["break_even_escape_cost_density"] = E.break_even_escape_cost(test["bug"], tb["dens"].to_numpy(), test["churn"], chosen, a)
    econ["break_even_escape_cost_random"] = E.break_even_escape_cost(test["bug"], tb["rnd"].to_numpy(), test["churn"], chosen, a)
    econ["smallest_first_bugs_in_list"] = E.net_benefit(test["bug"], tb["small"].to_numpy(), test["churn"], chosen, a)["bugs_in_list"]
    econ["sensitivity_density"] = E.grid(
        test["bug"], tb["dens"].to_numpy(), test["churn"], chosen, [500, 1000, 2000, 5000], [0.15, 0.30, 0.50], a
    )
    econ["review_everything_per_1000"] = E.net_benefit(test["bug"], tb["dens"].to_numpy(), test["churn"], 1.0, a)["net"] / len(test) * 1000
    out["economics"] = econ

    # ---- curves for charts ---------------------------------------------------------------
    curves = {}
    eff = test["churn"].clip(lower=1).to_numpy()
    for name, s in {
        "risk density": tb["dens"].to_numpy(),
        "probability": ps,
        "largest first": test["churn"].to_numpy(dtype=float),
        "random": np.random.default_rng(1).random(len(test)),
        "optimal": test["bug"].to_numpy() / eff,
    }.items():
        e, r = V.effort_curve(test["bug"].to_numpy(), s, eff)
        curves[name] = (e, r)

    # ---- final bundle: refit on train + validation, mature labels --------------------------
    final_train = pd.concat([train, val], ignore_index=True)
    final = M.CANDIDATES[pick]()
    final.fit(final_train[C.FEATURES], final_train["bug"])
    bundle = {
        "model": final,
        "features": list(C.FEATURES),
        "background": final_train[C.FEATURES].median().to_dict(),
        "background_frame": final_train[C.FEATURES].sample(min(1500, len(final_train)), random_state=C.SEED),
        "model_name": pick,
        "budget": chosen,
        "isotonic": iso if use_iso else None,
        "risk_cutoffs": {"high": float(np.quantile(ps, 0.9)), "medium": float(np.quantile(ps, 0.7))},
    }
    joblib.dump(bundle, C.OUT_DIR / "bundle.joblib")
    test_scored = test[["repo", "sha", "subject", "churn", "bug"]].copy()
    test_scored["risk"] = ps
    test_scored["risk_density"] = tb["dens"].to_numpy()
    test_scored.sort_values("risk", ascending=False).head(500).to_csv(C.OUT_DIR / "tables" / "top_risk_test.csv", index=False)
    return {"metrics": out, "curves": curves, "sweeps": sweeps, "test": tb, "loro": loro, "iso": iso, "by_year": by_year}


def save_metrics(metrics: dict) -> None:
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    (C.OUT_DIR / "metrics.json").write_text(json.dumps(clean(metrics), indent=2))
