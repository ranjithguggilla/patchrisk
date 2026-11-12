"""Command line entry point: mine, run, app."""

from __future__ import annotations

import argparse

from . import config as C


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="patchrisk", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="clone the pinned repositories and rebuild data/*.csv (slow)")
    m.add_argument("--repos", nargs="*", default=None)
    sub.add_parser("run", help="train, evaluate, write outputs/ and assets/")
    sub.add_parser("app", help="launch the Gradio app")
    args = ap.parse_args(argv)

    if args.cmd == "mine":
        from .mining import mine_all

        commits, links = mine_all(args.repos)
        C.DATA_DIR.mkdir(exist_ok=True)
        commits.to_csv(C.COMMITS_CSV, index=False)
        links.to_csv(C.FIXES_CSV, index=False)
        print(f"wrote {len(commits)} commits and {len(links)} links")
    elif args.cmd == "run":
        from . import pipeline, report

        (C.OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
        commits, links = pipeline.load()
        res = pipeline.run_study(commits, links)
        pipeline.save_metrics(res["metrics"])
        report.make_all(res)
        print("shipped model:", res["metrics"]["shipped_model"])
    else:
        from .app import main as app_main

        app_main()


if __name__ == "__main__":
    main()
