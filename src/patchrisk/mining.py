"""Mine commit-level features and SZZ bug-introducing links from a local git clone."""

from __future__ import annotations

import math
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pydriller import Repository
from radon.complexity import cc_visit

from . import config as C

warnings.filterwarnings("ignore", category=SyntaxWarning)


def clone_repo(name: str, dest: Path | None = None) -> Path:
    """Clone `name` into the data directory and check out the pinned revision."""
    url, sha = C.REPOS[name]
    dest = dest or (C.CLONE_DIR / name)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check=True)
    return dest


def is_fix_message(msg: str) -> bool:
    return bool(C.FIX_RE.search(msg)) and not C.NOT_FIX_RE.search(msg.split("\n")[0])


def _is_test(path: str) -> bool:
    parts = path.lower().replace("\\", "/").split("/")
    return any(p.startswith(C.TEST_MARKERS) for p in parts)


def _complexity(source: str | None) -> int:
    if not source:
        return 0
    try:
        return sum(b.complexity for b in cc_visit(source))
    except Exception:  # radon raises on syntax it cannot parse (py2 era files)
        return 0


def change_entropy(per_file_changes: list[int]) -> float:
    """Shannon entropy of how a commit's changed lines spread over its files."""
    total = sum(per_file_changes)
    if total <= 0 or len(per_file_changes) < 2:
        return 0.0
    probs = [c / total for c in per_file_changes if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def mine_repo(name: str, repo_path: Path, szz: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk non-merge commits oldest first and return (commit rows, fix links)."""
    rows: list[dict] = []
    links: list[dict] = []
    file_state: dict[str, dict] = defaultdict(lambda: {"n": 0, "last": None, "devs": set()})
    author_state: dict[str, dict] = defaultdict(lambda: {"n": 0, "first": None, "recent": []})

    repo = Repository(str(repo_path), only_no_merge=True, order="reverse")
    commits = list(repo.traverse_commits())
    commits.reverse()  # oldest first so history features only see the past

    for c in commits:
        ts = c.committer_date.timestamp()
        files = [m for m in c.modified_files if m.new_path or m.old_path]
        code_files = [m for m in files if (m.new_path or m.old_path or "").endswith(C.CODE_SUFFIXES)]
        author = (c.author.email or c.author.name or "").lower()
        a = author_state[author]

        if code_files:
            paths = [m.new_path or m.old_path for m in code_files]
            la = sum(m.added_lines for m in code_files)
            ld = sum(m.deleted_lines for m in code_files)
            per_file = [m.added_lines + m.deleted_lines for m in code_files]
            n_test = sum(_is_test(p) for p in paths)
            dirs = {p.rsplit("/", 1)[0] if "/" in p else "." for p in paths}
            cc_after = cc_before = cc_max = 0
            for m in code_files[: C.MAX_PY_FILES_COMPLEXITY]:
                after = _complexity(m.source_code)
                before = _complexity(m.source_code_before)
                cc_after += after
                cc_before += before
                cc_max = max(cc_max, after)
            prior = [file_state[p] for p in paths]
            prior_changes = sum(s["n"] for s in prior) / len(prior)
            ages = [(ts - s["last"]) / 86400 for s in prior if s["last"] is not None]
            files_age = sum(ages) / len(ages) if ages else -1.0
            devs = set().union(*(s["devs"] for s in prior))
            msg = c.msg or ""
            rows.append(
                {
                    "repo": name,
                    "sha": c.hash,
                    "ts": ts,
                    "la": la,
                    "ld": ld,
                    "churn": la + ld,
                    "nf": len(code_files),
                    "nd": len(dirs),
                    "entropy": change_entropy(per_file),
                    "msg_len": len(msg),
                    "is_fix_msg": int(is_fix_message(msg)),
                    "touches_tests": int(n_test > 0),
                    "touches_docs": int(any((m.new_path or m.old_path or "").endswith(C.DOC_SUFFIXES) for m in files)),
                    "n_code_files": len(code_files),
                    "cc_delta": cc_after - cc_before,
                    "cc_max": cc_max,
                    "author_exp": a["n"],
                    "author_recent": sum(1 for t in a["recent"] if ts - t < 90 * 86400),
                    "author_age_days": 0.0 if a["first"] is None else (ts - a["first"]) / 86400,
                    "files_prior_changes": prior_changes,
                    "files_age_days": files_age,
                    "files_prior_devs": len(devs - {author}),
                    "hour": c.committer_date.hour,
                    "weekday": c.committer_date.weekday(),
                    "new_author": int(a["n"] == 0),
                    "lines_per_file": (la + ld) / len(code_files),
                    "test_ratio": n_test / len(code_files),
                    "subject": msg.split("\n")[0][:120],
                }
            )
            if szz and rows[-1]["is_fix_msg"] and len(code_files) <= C.MAX_FILES_FOR_SZZ:
                try:
                    guilty = Git_last_modified(repo_path, c)
                except Exception:
                    guilty = {}
                for path, shas in guilty.items():
                    if not path.endswith(C.CODE_SUFFIXES):
                        continue
                    for sha in shas:
                        if sha != c.hash:
                            links.append({"repo": name, "fix_sha": c.hash, "fix_ts": ts, "intro_sha": sha, "path": path})
            for p in paths:
                s = file_state[p]
                s["n"] += 1
                s["last"] = ts
                s["devs"].add(author)
        a["n"] += 1
        a["first"] = ts if a["first"] is None else a["first"]
        a["recent"].append(ts)
        a["recent"] = [t for t in a["recent"] if ts - t < 90 * 86400]
    return pd.DataFrame(rows), pd.DataFrame(links)


def Git_last_modified(repo_path: Path, commit) -> dict[str, set[str]]:
    """SZZ: blame the lines a fix removed or changed, ignoring whitespace-only edits."""
    from pydriller import Git

    g = Git(str(repo_path))
    try:
        out = g.get_commits_last_modified_lines(commit)
    finally:
        g.clear()
    return out


def mine_all(names: list[str] | None = None, szz: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    commits, links = [], []
    for name in names or list(C.REPOS):
        path = clone_repo(name)
        c, l_ = mine_repo(name, path, szz=szz)
        commits.append(c)
        links.append(l_)
    return pd.concat(commits, ignore_index=True), pd.concat(links, ignore_index=True)
