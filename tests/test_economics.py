import numpy as np

from patchrisk import economics as E

A = E.Assumptions(loc_per_hour=100, overhead_hours=0.0, reviewer_cost_hour=100, escape_cost=1000, deep_catch_rate=0.5)


def test_cost_per_line_and_commit():
    assert A.cost_per_loc == 1.0
    assert E.Assumptions(overhead_hours=0.5, reviewer_cost_hour=80).cost_per_commit == 40.0


def test_net_benefit_by_hand():
    y = np.array([1, 0, 1, 0])
    loc = np.array([10, 10, 10, 10.0])
    r = E.net_benefit(y, np.array([4, 3, 2, 1.0]), loc, 0.5, A)
    assert r["commits"] == 2 and r["bugs_in_list"] == 1
    assert r["defects_prevented"] == 0.5
    assert r["review_cost"] == 20.0 and r["value"] == 500.0 and r["net"] == 480.0


def test_zero_budget_reviews_nothing():
    r = E.net_benefit(np.array([1, 1]), np.array([1.0, 2.0]), np.array([5, 5.0]), 0.0, A)
    assert r["commits"] == 0 and r["net"] == 0


def test_full_budget_reviews_everything():
    r = E.net_benefit(np.array([1, 0, 1]), np.array([1.0, 2.0, 3.0]), np.array([5, 5, 5.0]), 1.0, A)
    assert r["commits"] == 3 and r["bugs_in_list"] == 2


def test_break_even_cost_makes_net_zero():
    y = np.array([1, 0, 1, 0])
    loc = np.full(4, 10.0)
    s = np.array([4, 3, 2, 1.0])
    be = E.break_even_escape_cost(y, s, loc, 0.5, A)
    b = E.Assumptions(100, 0.0, 100, be, 0.5)
    assert abs(E.net_benefit(y, s, loc, 0.5, b)["net"]) < 1e-9


def test_grid_shape():
    rows = E.grid(np.array([1, 0, 1, 0]), np.array([4, 3, 2, 1.0]), np.full(4, 10.0), 0.5, [500, 2000], [0.3, 0.6])
    assert len(rows) == 4
