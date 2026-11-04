"""Value of information and correlated sampling, against answers worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from stormgrid import flow, value


def test_evpi_of_a_two_by_two_decision():
    # Act costs 3 in both states; no action costs 0 or 10. Prior 0.5: acting is best at 3, knowing the state gives 1.5
    cost = np.array([[3.0, 3.0], [0.0, 10.0]])
    assert value.evpi(cost, np.array([0.5, 0.5])) == pytest.approx(1.5)


def test_evpi_is_zero_when_one_action_dominates():
    cost = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 3.0]])
    assert value.evpi(cost, np.array([0.25, 0.5, 0.25])) == pytest.approx(0.0)


def test_evsi_is_bounded_by_evpi_and_reaches_it_at_full_accuracy():
    cost = np.array([[3.0, 3.0, 3.0], [0.0, 2.0, 10.0]])
    prior = np.array([0.25, 0.5, 0.25])
    assert value.evsi(cost, prior, 1.0) == pytest.approx(value.evpi(cost, prior))
    values = [value.evsi(cost, prior, q) for q in (1 / 3, 0.6, 0.8, 1.0)]
    assert values[0] == pytest.approx(0.0, abs=1e-12)     # a signal at chance level is worth nothing
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))


def test_correlated_sampling_keeps_marginals_and_adds_joint_failures():
    br = pd.DataFrame({"geometry": ["LINESTRING (7.0 50.0, 7.01 50.0)", "LINESTRING (7.02 50.0, 7.03 50.0)"]})
    prob = np.array([0.2, 0.2])
    rng = np.random.default_rng(2)
    sets = value.sample_correlated(prob, 20000, rng, value.correlation_factor(br, 200.0))
    p0 = np.mean([0 in s for s in sets])
    both = np.mean([s == frozenset({0, 1}) for s in sets])
    assert p0 == pytest.approx(0.2, abs=0.01)
    assert both > 0.15                                     # nearly identical location, nearly always together
    assert value.correlation_factor(br, 0.0) is None


def _model():
    buses = pd.Index(["A", "B", "C"])
    br = pd.DataFrame({"bus0": ["A", "A"], "bus1": ["B", "C"], "b": [1000.0, 1000.0], "limit": [1000.0, 1000.0]})
    return flow.ShedModel(buses, br, np.array([500.0, 0, 0]), np.array([0, 100.0, 50.0]))


def test_simulator_reinforcing_a_line_removes_its_loss():
    m = _model()
    prob = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = value.Simulator(m, m.solve().sum(), prob, seed=1, samples=5)
    assert sim.by_event() == pytest.approx([100.0, 50.0])
    assert sim.by_event(frozenset({0})) == pytest.approx([0.0, 50.0])
    assert sim.ranking().tolist() == [0, 1]
    assert sim.tail(75.0) == pytest.approx(1 / value.WINTERS)
