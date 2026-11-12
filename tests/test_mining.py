import subprocess
from pathlib import Path

from patchrisk import mining


def test_fix_message_rules():
    assert mining.is_fix_message("Fix crash when the header is empty")
    assert mining.is_fix_message("handle failure in redirect loop")
    assert not mining.is_fix_message("Fix typo in docs")
    assert not mining.is_fix_message("Add streaming helper")
    assert not mining.is_fix_message("Bump version to 2.0")


def test_change_entropy():
    assert mining.change_entropy([10]) == 0.0
    assert abs(mining.change_entropy([5, 5]) - 1.0) < 1e-9
    assert mining.change_entropy([9, 1]) < 1.0
    assert mining.change_entropy([]) == 0.0


def _git(repo: Path, *args, **env):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "PATH": __import__("os").environ["PATH"],
            **env,
        },
    )


def test_szz_links_a_fix_to_the_commit_that_wrote_the_line(tmp_path):
    repo = tmp_path / "toy"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    f = repo / "mod.py"
    f.write_text("def area(r):\n    return 3 * r * r\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add area helper")
    f.write_text("def area(r):\n    return 3.14159 * r * r\n")
    _git(repo, "commit", "-q", "-am", "fix wrong constant in area")
    commits, links = mining.mine_repo("toy", repo)
    assert len(commits) == 2
    assert commits["is_fix_msg"].tolist() == [0, 1]
    assert len(links) == 1
    first_sha = commits.iloc[0]["sha"]
    assert links.iloc[0]["intro_sha"] == first_sha
    assert commits.iloc[1]["author_exp"] == 1
    assert commits.iloc[0]["new_author"] == 1


def test_commits_without_python_files_are_skipped(tmp_path):
    repo = tmp_path / "docs_only"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "readme")
    commits, links = mining.mine_repo("docs_only", repo)
    assert commits.empty and links.empty
