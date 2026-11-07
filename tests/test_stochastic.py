"""Two-stage reinforcement program: the switchable recourse and Benders against brute force on small grids."""

import itertools

import numpy as np
import pandas as pd
import pytest

from stormgrid import flow, stochastic


def _ring():
    """Generator at A, loads at B, C and D on a ring A-B-C-D-A with a chord A-C; limits make failures costly."""
    buses = pd.Index(["A", "B", "C", "D"])
    br = pd.DataFrame({"bus0": ["A", "B", "C", "D", "A"], "bus1": ["B", "C", "D", "A", "C"],
                       "b": [1000.0, 800.0, 1200.0, 900.0, 600.0], "limit": [250.0, 150.0, 200.0, 250.0, 180.0]})
    return flow.ShedModel(buses, br, np.array([600.0, 0, 0, 0]), np.array([0, 150.0, 200.0, 100.0]))


FAILURE_SETS = [frozenset(s) for s in ([0], [3], [4], [0, 4], [3, 4], [0, 3], [1, 2], [0, 1, 4], [2, 3, 4])]


@pytest.mark.parametrize("failed", FAILURE_SETS)
def test_recourse_is_exact_at_binary_x(failed):
    m = _ring()
    rec = stochastic.Recourse(m)
    cand = {0: 0, 3: 1, 4: 2}
    for bits in itertools.product([0, 1], repeat=3):
        x = dict(enumerate(bits))
        kept = {l for l, i in cand.items() if x[i] == 1}
        q, _ = rec.solve(failed, cand, x)
        assert q == pytest.approx(m.solve(failed - kept).sum(), abs=1e-5)


@pytest.mark.parametrize("failed", FAILURE_SETS)
def test_cuts_never_cut_off_a_binary_point(failed):
    m = _ring()
    rec = stochastic.Recourse(m)
    cand = {0: 0, 3: 1, 4: 2}
    points = list(itertools.product([0, 1], repeat=3))
    exact = {p: m.solve(failed - {l for l, i in cand.items() if p[i]}).sum() for p in points}
    for p in points:
        q, g = rec.solve(failed, cand, dict(enumerate(p)))
        for z in points:
            cut = q + sum(g.get(i, 0.0) * (z[i] - p[i]) for i in range(3))
            assert cut <= exact[z] + 1e-5


def test_benders_matches_brute_force():
    m = _ring()
    base = m.solve().sum()
    scen = [stochastic.Scenario(failed=f, case=i % 3, weight=w)
            for i, (f, w) in enumerate(zip(FAILURE_SETS, [0.05, 0.03, 0.04, 0.02, 0.02, 0.01, 0.03, 0.01, 0.02]))]
    cand_pos, km = [0, 3, 4], [50.0, 80.0, 30.0]
    prior = np.array([0.25, 0.5, 0.25])
    for c in (0.5, 5.0, 50.0, 500.0):
        bd = stochastic.Benders(m, base, [stochastic.Scenario(s.failed, s.case, s.weight) for s in scen], cand_pos, km, log=lambda h: None)
        res = bd.solve(c, prior)
        best = min(
            c * np.dot(km, bits) + sum(prior[s.case] * s.weight * stochastic.LOSS_UNIT *
                                       (m.solve(s.failed - {cand_pos[i] for i in range(3) if bits[i]}).sum() - base) for s in scen)
            for bits in itertools.product([0, 1], repeat=3))
        assert res["upper"] == pytest.approx(best, rel=1e-6, abs=1e-6)
        assert res["lower"] <= res["upper"] + 1e-6
        assert res["gap"] <= stochastic.GAP + 1e-9
