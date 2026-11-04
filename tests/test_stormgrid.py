"""Offline tests on small synthetic grids with answers known by hand."""

import numpy as np
import pandas as pd
import pytest

from stormgrid import flow, hazard, network


def _grid(limit=1000.0):
    """Generator at A, load at C, two equal parallel paths A-B-C and A-C through D."""
    buses = pd.Index(["A", "B", "C"])
    branches = pd.DataFrame({"bus0": ["A", "B", "A"], "bus1": ["B", "C", "C"], "b": [1000.0, 1000.0, 1000.0],
                             "limit": [limit, limit, limit]})
    return flow.ShedModel(buses, branches, np.array([500.0, 0.0, 0.0]), np.array([0.0, 0.0, 300.0]))


def test_intact_grid_serves_the_load():
    assert _grid().solve().sum() == pytest.approx(0.0, abs=1e-6)


def test_redundancy_covers_one_failure():
    assert _grid().solve({2}).sum() == pytest.approx(0.0, abs=1e-6)


def test_cutting_all_paths_sheds_the_load_at_the_right_bus():
    shed = _grid().solve({1, 2})
    assert shed[2] == pytest.approx(300.0)
    assert shed.sum() == pytest.approx(300.0)


def test_flow_limit_binds_through_the_impedance_split():
    # 300 MW splits 1/3 over A-B-C and 2/3 over A-C; a limit of 150 MW on A-C allows 225 MW in total
    assert _grid(limit=150.0).solve().sum() == pytest.approx(75.0, abs=1e-4)


def test_flow_limit_on_a_single_path_after_failure():
    assert _grid(limit=200.0).solve({2}).sum() == pytest.approx(100.0, abs=1e-4)


def test_generation_shortage_is_shed():
    buses = pd.Index(["A", "B"])
    br = pd.DataFrame({"bus0": ["A"], "bus1": ["B"], "b": [1000.0], "limit": [1000.0]})
    assert flow.ShedModel(buses, br, np.array([100.0, 0.0]), np.array([0.0, 250.0])).solve().sum() == pytest.approx(150.0)


def test_results_do_not_depend_on_susceptance_scale():
    buses = pd.Index(["A", "B", "C"])
    br = pd.DataFrame({"bus0": ["A", "B", "A"], "bus1": ["B", "C", "C"], "b": [5e4, 5e4, 5e4], "limit": [150.0] * 3})
    m = flow.ShedModel(buses, br, np.array([500.0, 0, 0]), np.array([0, 0, 300.0]))
    assert m.solve().sum() == pytest.approx(75.0, abs=1e-4)


def test_parse_wkt_and_distance():
    xy = network.parse_wkt("LINESTRING (6.5 51.0, 7.5 51.0)")
    assert xy.tolist() == [[6.5, 51.0], [7.5, 51.0]]
    assert network.haversine_km(6.5, 51.0, 7.5, 51.0) == pytest.approx(70.0, abs=0.5)
    assert network.haversine_km(10.0, 50.0, 10.0, 51.0) == pytest.approx(111.2, abs=0.2)


def test_largest_component():
    keep = network.largest_component(list("abcde"), np.array(["a", "b", "d"]), np.array(["b", "c", "e"]))
    assert keep == {"a", "b", "c"}


def test_spread_load_conserves_the_total():
    st = pd.DataFrame({"lon": [7.0, 12.0], "lat": [50.0, 52.0], "weight": [0.25, 0.75]})
    buses = pd.DataFrame({"x": [7.0, 7.1, 7.2, 12.0, 12.1, 12.2], "y": [50.0] * 3 + [52.0] * 3})
    load = network.spread_load(st, buses, 1000.0)
    assert load.sum() == pytest.approx(1000.0)
    assert load[:3].sum() == pytest.approx(250.0)


def test_station_km_splits_a_line_between_stations():
    st = pd.DataFrame({"lon": [7.0, 8.0], "lat": [50.0, 50.0]})
    br = pd.DataFrame({"geometry": ["LINESTRING (7.0 50.0, 7.2 50.0, 8.0 50.0)", ""], "overhead": [True, False]})
    km = hazard.station_km(br, st)
    assert km[1].sum() == 0
    assert km[0].sum() == pytest.approx(network.haversine_km(7.0, 50.0, 8.0, 50.0), rel=1e-3)
    assert km[0, 0] > 0 and km[0, 1] > 0


def test_fill_missing_takes_the_nearest_station():
    st = pd.DataFrame({"lon": [7.0, 7.1, 12.0], "lat": [50.0, 50.0, 50.0]})
    ratio = pd.DataFrame([[np.nan, 1.5, 3.0]])
    assert hazard.fill_missing(ratio, st)[0, 0] == pytest.approx(1.5)


def test_fragility_is_half_at_the_median_and_increasing():
    p = hazard.fragility(np.array([1.0, 2.0, 3.0]), median=2.0)
    assert p[1] == pytest.approx(0.5)
    assert p[0] < p[1] < p[2]


def test_line_probability_of_two_segments():
    ratio = np.array([[2.0]])
    km = np.array([[2 * hazard.SEGMENT_KM]])
    assert hazard.line_probabilities(ratio, km, median=2.0)[0, 0] == pytest.approx(0.75)


def test_median_reaches_the_target_and_higher_target_means_lower_median():
    rng = np.random.default_rng(1)
    ratio, km = rng.uniform(1.0, 1.5, 5), rng.uniform(0, 30, (200, 5))
    m5 = hazard.median_for_target(ratio, km, 5.0)
    m15 = hazard.median_for_target(ratio, km, 15.0)
    assert hazard.line_probabilities(ratio[None, :], km, m5).sum() == pytest.approx(5.0, abs=0.01)
    assert m15 < m5


def test_sample_outages_frequency():
    prob = np.array([0.0, 0.5, 1.0])
    sets = flow.sample_outages(prob, 4000, np.random.default_rng(0))
    assert all(2 in s and 0 not in s for s in sets)
    assert np.mean([1 in s for s in sets]) == pytest.approx(0.5, abs=0.03)
