"""From event gust footprints to line failure probabilities.

Each event footprint gives the maximum gust at each DWD station. Every kilometre of overhead line takes the gust of
its nearest station as a ratio to that station's 98th percentile, since lines, like buildings, are built for their
local wind climate. A lognormal fragility in this ratio gives the failure probability per 10 km.
No public failure data exist for German transmission lines, so the median is set from an assumed number of expected
line failures in the largest storm, and that number is varied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .network import haversine_km, nearest, parse_wkt

SEGMENT_KM = 10.0
BETA = 0.2


def station_km(branches: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    """Kilometres of each overhead branch closest to each station (branches x stations)."""
    lon, lat = stations["lon"].to_numpy(), stations["lat"].to_numpy()
    km = np.zeros((len(branches), len(stations)))
    for i, (geom, overhead) in enumerate(zip(branches["geometry"], branches["overhead"])):
        if not overhead or not geom:
            continue
        xy = parse_wkt(geom)
        mid = (xy[:-1] + xy[1:]) / 2
        seg = haversine_km(xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1])
        np.add.at(km[i], nearest(mid[:, 0], mid[:, 1], lon, lat), seg)
    return km


def fill_missing(ratio: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    """Events x stations ratios; a missing station takes the value of the nearest reporting one."""
    r = ratio.to_numpy(dtype=float).copy()
    lon, lat = stations["lon"].to_numpy(), stations["lat"].to_numpy()
    for j in np.where(np.isnan(r).any(axis=0))[0]:
        order = np.argsort(haversine_km(lon[j], lat[j], lon, lat))
        missing = np.where(np.isnan(r[:, j]))[0]
        for e in missing:
            values = r[e, order]
            valid = values[~np.isnan(values)]
            r[e, j] = valid[0] if len(valid) else 0.0
    return r


def fragility(ratio: np.ndarray, median: float, beta: float = BETA) -> np.ndarray:
    """Failure probability per segment, zero at or below a ratio of zero."""
    out = np.zeros_like(ratio, dtype=float)
    pos = ratio > 0
    out[pos] = norm.cdf(np.log(ratio[pos] / median) / beta)
    return out


def line_probabilities(ratio: np.ndarray, km: np.ndarray, median: float, beta: float = BETA) -> np.ndarray:
    """Events x branches failure probabilities; segments fail independently."""
    log_survive = np.log1p(-np.minimum(fragility(ratio, median, beta), 1 - 1e-12))
    return -np.expm1(log_survive @ (km.T / SEGMENT_KM))


def median_for_target(ratio_event: np.ndarray, km: np.ndarray, target_failures: float, beta: float = BETA) -> float:
    """Fragility median at which the given event has the target expected number of failed lines."""
    def gap(m):
        return line_probabilities(ratio_event[None, :], km, m, beta).sum() - target_failures
    return float(brentq(gap, 0.5, 20.0, xtol=1e-4))
