"""One end-to-end run on the committed data, written into a temporary output folder."""

import json

import pytest

from patchrisk import cli
from patchrisk import config as C


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "OUT_DIR", out / "outputs")
    mp.setattr(C, "ASSET_DIR", out / "assets")
    cli.main(["run"])
    yield out
    mp.undo()


def test_run_writes_metrics_charts_and_bundle(run_dir):
    assert (run_dir / "outputs" / "metrics.json").exists()
    assert (run_dir / "outputs" / "bundle.joblib").exists()
    assert len(list((run_dir / "assets").glob("*.png"))) == 7


def test_metrics_are_sane(run_dir):
    m = json.loads((run_dir / "outputs" / "metrics.json").read_text())
    assert m["shipped_model"] in {"logistic", "boosting", "forest"}
    assert 0.7 < m["models"][m["shipped_model"]]["auc"] < 0.9
    assert m["models"]["random"]["auc"] < 0.6
    lo, hi = m["shipped_ci"]["auc"]
    assert lo < m["models"][m["shipped_model"]]["auc"] < hi
    assert m["data"]["split"]["test"]["commits"] > 500


def test_metrics_match_the_committed_file(run_dir):
    new = json.loads((run_dir / "outputs" / "metrics.json").read_text())
    old = json.loads((C.ROOT / "outputs" / "metrics.json").read_text())
    assert new["models"]["logistic"]["auc"] == pytest.approx(old["models"]["logistic"]["auc"], abs=1e-9)
    assert new["economics"]["chosen_budget"] == old["economics"]["chosen_budget"]
