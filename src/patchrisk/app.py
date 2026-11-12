"""Gradio front end: score a commit by hand, or browse the review queue."""

from __future__ import annotations

import pandas as pd

from . import config as C
from . import score as S

FIELDS = [
    ("la", "Lines added", 40, 0, 5000),
    ("ld", "Lines deleted", 10, 0, 5000),
    ("nf", "Python files edited", 2, 1, 100),
    ("nd", "Directories touched", 1, 1, 50),
    ("author_exp", "Author's earlier commits in this repo", 60, 0, 5000),
    ("files_prior_changes", "Average earlier changes to these files", 12, 0, 500),
    ("touches_tests", "Changes tests (0 or 1)", 1, 0, 1),
    ("is_fix_msg", "Commit message reads as a bug fix (0 or 1)", 0, 0, 1),
]


def evaluate(bundle, **kw) -> tuple[str, str]:
    """Pure function behind the form, so tests can call it without a browser."""
    values = {k: float(v) for k, v in kw.items()}
    values["nf"] = max(values.get("nf", 1), 1)
    values["n_code_files"] = values["nf"]
    values["churn"] = values.get("la", 0) + values.get("ld", 0)
    values["new_author"] = float(values.get("author_exp", 0) == 0)
    res = S.score_commit(values, bundle)
    headline = (
        f"Estimated chance this commit is later blamed for a bug: {res['probability']:.0%} "
        f"(band: {res['band']}). About {res['risk_per_100_lines']:.1f} risk points per 100 changed lines."
    )
    return headline, "\n".join(f"- {r}" for r in res["reasons"])


def build(bundle_path=S.DEFAULT_BUNDLE):
    import gradio as gr

    bundle = S.load_bundle(bundle_path)
    queue = pd.read_csv(C.OUT_DIR / "tables" / "top_risk_test.csv").head(25)

    def fn(*vals):
        return evaluate(bundle, **{f[0]: v for f, v in zip(FIELDS, vals, strict=False)})

    with gr.Blocks(title="patchrisk") as demo:
        gr.Markdown("# patchrisk\nBug-introduction risk for a code change, learned from six open-source Python projects.")
        with gr.Tab("Score a commit"):
            inputs = [gr.Number(label=label, value=val, minimum=lo, maximum=hi) for _, label, val, lo, hi in FIELDS]
            btn = gr.Button("Score")
            out1 = gr.Textbox(label="Result")
            out2 = gr.Markdown()
            btn.click(fn, inputs, [out1, out2])
        with gr.Tab("Review queue (test period)"):
            gr.Dataframe(queue[["repo", "subject", "churn", "risk"]], interactive=False)
    return demo


def main():
    build().launch()
