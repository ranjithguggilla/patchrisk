"""Review-budget economics under openly stated assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Assumptions:
    """Every number here is an assumption, not a measurement. Change them in config or the CLI."""

    loc_per_hour: float = 200.0  # careful review speed, changed lines per reviewer hour
    overhead_hours: float = 0.25  # fixed reviewer time per commit (context load, discussion)
    reviewer_cost_hour: float = 90.0  # loaded cost of one reviewer hour, USD
    escape_cost: float = 2000.0  # triage, fix, release and support cost of a defect found later, USD
    deep_catch_rate: float = 0.30  # share of defects a deep review prevents in a reviewed commit

    @property
    def cost_per_loc(self) -> float:
        return self.reviewer_cost_hour / self.loc_per_hour

    @property
    def cost_per_commit(self) -> float:
        return self.reviewer_cost_hour * self.overhead_hours


def net_benefit(y, score, loc, budget: float, a: Assumptions = Assumptions()) -> dict:
    """Deep-review the highest-ranked commits until `budget` of all changed lines is used."""
    order = np.argsort(-np.asarray(score), kind="stable")
    loc = np.asarray(loc, dtype=float)
    y = np.asarray(y)
    cum = np.cumsum(loc[order])
    k = int(np.searchsorted(cum, budget * loc.sum(), side="right"))
    picked = order[:k]
    caught = float(y[picked].sum()) * a.deep_catch_rate
    cost = float(loc[picked].sum()) * a.cost_per_loc + k * a.cost_per_commit
    value = caught * a.escape_cost
    return {
        "budget": budget,
        "commits": k,
        "bugs_in_list": int(y[picked].sum()),
        "defects_prevented": caught,
        "review_cost": cost,
        "value": value,
        "net": value - cost,
    }


def sweep(y, score, loc, budgets, a: Assumptions = Assumptions()) -> list[dict]:
    return [net_benefit(y, score, loc, b, a) for b in budgets]


def grid(y, score, loc, budget: float, escape_costs, catch_rates, a: Assumptions = Assumptions()):
    """Net benefit per 1,000 commits over a grid of the two assumptions that matter most."""
    rows = []
    for ec in escape_costs:
        for cr in catch_rates:
            b = Assumptions(a.loc_per_hour, a.overhead_hours, a.reviewer_cost_hour, ec, cr)
            r = net_benefit(y, score, loc, budget, b)
            rows.append({"escape_cost": ec, "catch_rate": cr, "net_per_1000": r["net"] / len(y) * 1000})
    return rows


def break_even_escape_cost(y, score, loc, budget: float, a: Assumptions = Assumptions()) -> float:
    r = net_benefit(y, score, loc, budget, a)
    if r["defects_prevented"] == 0:
        return float("inf")
    return r["review_cost"] / r["defects_prevented"]
