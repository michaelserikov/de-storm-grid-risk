"""The German transmission grid from open data: buses, lines, transformers, plants and load.

Network: PyPSA-Eur prebuilt OpenStreetMap network (220 kV and above, Xiong et al. 2025). Germany as an island.
Plants: powerplantmatching. Load: station population weights of de-windstorm-risk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRANSFORMER_X_PU = 0.1
LOAD_BUSES = 3
EARTH_KM = 6371.0
# Availability of installed capacity in a winter storm hour. Assumptions, varied in the sensitivities.
AVAILABILITY = {"Hard Coal": 0.9, "Lignite": 0.9, "Natural Gas": 0.9, "Oil": 0.9, "Hydro": 0.5, "Biogas": 0.8,
                "Solid Biomass": 0.8, "Waste": 0.8, "Geothermal": 0.8, "Other": 0.8, "Onshore": 0.55, "Offshore": 0.7}


@dataclass
class Grid:
    buses: pd.DataFrame       # index bus_id; x, y
    branches: pd.DataFrame    # bus0, bus1, b (MW per rad), limit (MW), length_km, overhead, voltage, kind
    gen: np.ndarray           # available generation per bus, MW
    load: np.ndarray          # load per bus, MW, sums to the requested total


def parse_wkt(text: str) -> np.ndarray:
    """Coordinates of a WKT LINESTRING as an array of (lon, lat)."""
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text)
    return np.asarray(nums, dtype=float).reshape(-1, 2)


def haversine_km(lon0, lat0, lon1, lat1):
    lon0, lat0, lon1, lat1 = map(np.radians, (lon0, lat0, lon1, lat1))
    a = np.sin((lat1 - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin((lon1 - lon0) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def nearest(lon: np.ndarray, lat: np.ndarray, plon: np.ndarray, plat: np.ndarray) -> np.ndarray:
    """Index of the nearest point (plon, plat) for each (lon, lat)."""
    out = np.zeros(len(lon), dtype=int)
    for i, (a, b) in enumerate(zip(lon, lat)):
        out[i] = int(np.argmin(haversine_km(a, b, plon, plat)))
    return out


def largest_component(buses: list, bus0: np.ndarray, bus1: np.ndarray) -> set:
    parent = {b: b for b in buses}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(bus0, bus1):
        parent[find(a)] = find(b)
    roots = pd.Series([find(b) for b in buses], index=buses)
    return set(roots[roots == roots.value_counts().idxmax()].index)


def read_branches(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(raw / "buses.csv", quotechar="'")
    b = b[(b["country"] == "DE") & (b["dc"] == "f") & (b["under_construction"] == "f")].set_index("bus_id")
    ln = pd.read_csv(raw / "lines.csv", quotechar="'")
    ln = ln[ln["bus0"].isin(b.index) & ln["bus1"].isin(b.index) & (ln["under_construction"] == "f")]
    tr = pd.read_csv(raw / "transformers.csv", quotechar="'")
    tr = tr[tr["bus0"].isin(b.index) & tr["bus1"].isin(b.index)]
    lines = pd.DataFrame({
        "id": ln["line_id"], "bus0": ln["bus0"], "bus1": ln["bus1"], "b": ln["voltage"] ** 2 / ln["x"],
        "limit": ln["s_nom"], "length_km": ln["length"] / 1000.0, "overhead": ln["underground"] == "f",
        "voltage": ln["voltage"], "kind": "line", "geometry": ln["geometry"]})
    trafo = pd.DataFrame({
        "id": tr["transformer_id"], "bus0": tr["bus0"], "bus1": tr["bus1"], "b": tr["s_nom"] / TRANSFORMER_X_PU,
        "limit": tr["s_nom"], "length_km": 0.0, "overhead": False, "voltage": tr["voltage_bus0"],
        "kind": "transformer", "geometry": ""})
    br = pd.concat([lines, trafo], ignore_index=True)
    keep = largest_component(list(b.index), br["bus0"].to_numpy(), br["bus1"].to_numpy())
    br = br[br["bus0"].isin(keep) & br["bus1"].isin(keep)].reset_index(drop=True)
    return b.loc[sorted(keep)][["x", "y", "voltage"]], br


def available_generation(raw: Path, buses: pd.DataFrame, availability: dict = AVAILABILITY, year: int = 2026) -> np.ndarray:
    p = pd.read_csv(raw / "powerplants.csv", low_memory=False)
    p = p[(p["Country"] == "Germany") & p["lat"].notna()]
    p = p[(p["DateOut"].isna() | (p["DateOut"] > year)) & (p["DateIn"].isna() | (p["DateIn"] <= year))]
    key = np.where(p["Fueltype"] == "Wind", p["Technology"], p["Fueltype"])
    mw = p["Capacity"].to_numpy() * pd.Series(key).map(availability).fillna(0.0).to_numpy()
    p, mw = p[mw > 0], mw[mw > 0]
    # Group plants on a 0.1 degree grid first: over 100,000 entries, and the nearest bus barely changes within a cell
    cell = pd.DataFrame({"lon": p["lon"].round(1).to_numpy(), "lat": p["lat"].round(1).to_numpy(), "mw": mw})
    cell = cell.groupby(["lon", "lat"], as_index=False)["mw"].sum()
    idx = nearest(cell["lon"].to_numpy(), cell["lat"].to_numpy(), buses["x"].to_numpy(), buses["y"].to_numpy())
    return np.bincount(idx, weights=cell["mw"].to_numpy(), minlength=len(buses))


def spread_load(stations: pd.DataFrame, buses: pd.DataFrame, total_mw: float) -> np.ndarray:
    """Population weight of each station, split equally between its nearest buses."""
    load = np.zeros(len(buses))
    x, y = buses["x"].to_numpy(), buses["y"].to_numpy()
    w = stations["weight"] / stations["weight"].sum()
    for lon, lat, share in zip(stations["lon"], stations["lat"], w):
        order = np.argsort(haversine_km(lon, lat, x, y))[:LOAD_BUSES]
        load[order] += share / LOAD_BUSES
    return load * total_mw


def build(raw: Path, stations: pd.DataFrame, total_load_mw: float = 70000.0, availability: dict = AVAILABILITY) -> Grid:
    buses, branches = read_branches(raw)
    return Grid(buses, branches, available_generation(raw, buses, availability), spread_load(stations, buses, total_load_mw))
