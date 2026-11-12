import numpy as np
import pandas as pd
import pytest

from patchrisk import config as C
from patchrisk import model as M


@pytest.fixture(scope="session")
def data():
    return pd.read_csv(C.COMMITS_CSV), pd.read_csv(C.FIXES_CSV)


@pytest.fixture(scope="session")
def small_bundle(data):
    from patchrisk import labels as L

    commits, links = data
    train, _, _ = L.split_frames(commits, links)
    train = train.sample(2500, random_state=1)
    model = M.make_logistic()
    model.fit(train[C.FEATURES], train["bug"])
    return {
        "model": model,
        "features": list(C.FEATURES),
        "background": train[C.FEATURES].median().to_dict(),
        "background_frame": train[C.FEATURES].head(300),
        "isotonic": None,
        "risk_cutoffs": {"high": 0.5, "medium": 0.25},
    }


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
