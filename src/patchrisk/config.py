"""Project-wide constants: repositories, pinned revisions, labelling rules."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
ASSET_DIR = ROOT / "assets"
CLONE_DIR = DATA_DIR / "repos"  # gitignored, re-created by `patchrisk mine`
COMMITS_CSV = DATA_DIR / "commits.csv"  # one row per commit, features plus raw label ingredients
FIXES_CSV = DATA_DIR / "fix_links.csv"  # one row per (fix commit -> bug-introducing commit) link

# Each repository is pinned to the revision that was mined, so a rerun sees the same history.
REPOS = {
    "click": ("https://github.com/pallets/click", "2247b35ea1c47c727d7a06e51fa280e12a863ff6"),
    "flask": ("https://github.com/pallets/flask", "49943b5a94413a530761350f04e896f76e8d620e"),
    "httpx": ("https://github.com/encode/httpx", "b5addb64f0161ff6bfe94c124ef76f6a1fba5254"),
    "jinja": ("https://github.com/pallets/jinja", "5ef70112a1ff19c05324ff889dd30405b1002044"),
    "requests": ("https://github.com/psf/requests", "3c7baaa05ab4aa49cc819ce3a819d260de19c157"),
    "werkzeug": ("https://github.com/pallets/werkzeug", "5f11cd39b40ac03296ab7db446bfb0a737d77590"),
}

# A commit counts as a bug fix when its message matches FIX and does not match NOT_FIX.
FIX_RE = re.compile(
    r"\b(fix(e[sd]|ing)?|bug(s|fix)?|crash(es)?|regression|broken|incorrect(ly)?|"
    r"wrong(ly)?|exception|traceback|fail(s|ed|ure)?|resolve[sd]?\s+#?\d+|closes?\s+#\d+)\b",
    re.IGNORECASE,
)
NOT_FIX_RE = re.compile(
    r"\b(typos?|docs?|documentation|changelog|release|version|bump|lint|format(ting)?|"
    r"style|pre-commit|ci|readme|spelling|merge)\b",
    re.IGNORECASE,
)

CODE_SUFFIXES = (".py",)
TEST_MARKERS = ("test", "tests", "conftest")
DOC_SUFFIXES = (".rst", ".md", ".txt")
MAX_FILES_FOR_SZZ = 20  # fix commits touching more files than this are skipped (usually refactors)
MAX_PY_FILES_COMPLEXITY = 15

# Mature-label rule: a commit only gets a label once this many days of later history exist.
LABEL_WINDOW_DAYS = 365
SEED = 7

FEATURES = [
    "la",
    "ld",
    "churn",
    "nf",
    "nd",
    "entropy",
    "msg_len",
    "is_fix_msg",
    "touches_tests",
    "touches_docs",
    "n_code_files",
    "cc_delta",
    "cc_max",
    "author_exp",
    "author_recent",
    "author_age_days",
    "files_prior_changes",
    "files_age_days",
    "files_prior_devs",
    "hour",
    "weekday",
    "new_author",
    "lines_per_file",
    "test_ratio",
]

# Forward-in-time split. Test commits are all at least LABEL_WINDOW_DAYS old when the data was mined.
TRAIN_END = "2021-01-01"
VAL_END = "2022-07-01"
TEST_END = "2025-11-01"
