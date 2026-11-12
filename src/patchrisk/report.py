"""Charts for the README and the overview document."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

INK, BLUE, ORANGE, GREY, GREEN, RED = "#1f2933", "#2b6cb0", "#dd6b20", "#8a94a0", "#2f855a", "#c53030"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.dpi": 130,
    }
)


def _save(fig, name):
    C.ASSET_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.ASSET_DIR / name, dpi=150)
    plt.close(fig)


def make_all(res: dict) -> None:
    m = res["metrics"]
    by_year = res["by_year"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(by_year.index, by_year["bug_rate"] * 100, color=BLUE)
    for lo, _hi, _lab in ((2006.5, 2020.5, "train"), (2020.5, 2022.5, "validation"), (2022.5, 2026.5, "test")):
        ax.axvline(lo, color=GREY, lw=0.8, ls=":")
    ax.set_ylabel("commits later blamed for a bug (%)")
    ax.set_title("Bug rate by commit year, labels as known today")
    _save(fig, "bug_rate_by_year.png")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    styles = {
        "risk density": (BLUE, "-"),
        "probability": (GREEN, "-"),
        "largest first": (RED, "--"),
        "random": (GREY, ":"),
        "optimal": (INK, "-."),
    }
    for name, (e, r) in res["curves"].items():
        col, ls = styles[name]
        ax.plot(e * 100, r * 100, color=col, ls=ls, lw=2 if name == "risk density" else 1.4, label=name)
    ax.axvline(20, color=GREY, lw=0.6)
    ax.set_xlabel("share of changed lines reviewed (%)")
    ax.set_ylabel("bug-introducing commits reached (%)")
    ax.set_title("Review order on the test period")
    ax.legend(frameon=False)
    _save(fig, "effort_curves.png")

    mat = m["label_maturity"]
    cur = pd.read_csv(C.OUT_DIR / "tables" / "label_maturity_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(cur["age_days"], cur["known"] * 100, color=ORANGE, marker="o", ms=3, label="known at the cutoff")
    axes[0].plot(cur["age_days"], cur["hindsight"] * 100, color=BLUE, marker="o", ms=3, label="known today")
    axes[0].axvline(365, color=GREY, ls=":")
    axes[0].set_xlabel("commit age at the training cutoff (days)")
    axes[0].set_ylabel("labelled buggy (%)")
    axes[0].set_xlim(0, 1500)
    axes[0].set_title("Young commits look cleaner than they are")
    axes[0].legend(frameon=False)
    names = ["known_dropped", "known_kept", "hindsight"]
    vals = [mat[n]["auc"] for n in names]
    axes[1].bar(["young tail\ndropped", "young tail\nkept", "hindsight\nlabels"], vals, color=[BLUE, ORANGE, GREY])
    axes[1].set_ylim(0.7, 0.85)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.003, f"{v:.3f}", ha="center")
    axes[1].set_ylabel("test AUC")
    axes[1].set_title("Training-label variants, same test set")
    _save(fig, "label_maturity.png")

    sp = m["split_protocols"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = ["random rows", "forward in time", "all repos\n(per-repo)", "repo held out\n(per-repo)"]
    vals = [sp["random_rows"]["auc"], sp["forward_in_time"]["auc"], sp["all_repos_mean_auc"], sp["repo_held_out_mean_auc"]]
    ax.bar(labels, vals, color=[GREY, BLUE, BLUE, GREEN])
    ax.set_ylim(0.7, 0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center")
    ax.set_ylabel("AUC")
    ax.set_title("Validation protocol barely moves the score")
    _save(fig, "split_protocols.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, rows in res["sweeps"].items():
        if name == "random" or name in ("risk_density", "smallest_first", "largest_first"):
            df = pd.DataFrame(rows)
            ax.plot(
                df["budget"] * 100,
                df["net"] / len(res["test"]) * 1000 / 1000,
                label=name.replace("_", " "),
                color={"risk_density": BLUE, "smallest_first": ORANGE, "largest_first": RED, "random": GREY}[name],
            )
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(m["economics"]["chosen_budget"] * 100, color=GREY, ls=":")
    ax.set_xlabel("review budget (% of changed lines)")
    ax.set_ylabel("net benefit per 1,000 commits ($ thousand)")
    ax.set_title("Net benefit by review order")
    ax.legend(frameon=False)
    _save(fig, "net_benefit.png")

    t = res["test"]
    iso = res["iso"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    bins = np.quantile(t["p"], np.linspace(0, 1, 9))
    idx = np.clip(np.digitize(t["p"], bins[1:-1]), 0, 7)
    g = pd.DataFrame({"p": t["p"], "iso": iso.predict(t["p"]), "y": t["bug"], "b": idx}).groupby("b").mean()
    ax.plot([0, 0.8], [0, 0.8], color=GREY, ls=":")
    ax.plot(g["p"], g["y"], marker="o", color=ORANGE, label="raw")
    ax.plot(g["iso"], g["y"], marker="o", color=BLUE, label="isotonic (shipped)")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed bug rate")
    ax.set_title("Reliability on the test period")
    ax.legend(frameon=False)
    _save(fig, "reliability.png")

    fg = m["feature_groups"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    items = sorted(fg.items(), key=lambda kv: kv[1]["auc_drop"])
    ax.barh([k for k, _ in items], [v["auc_drop"] for _, v in items], color=[BLUE if v["auc_drop"] > 0.005 else GREY for _, v in items])
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("test AUC lost when the group is removed")
    ax.set_title("Which features carry the model")
    _save(fig, "feature_groups.png")
