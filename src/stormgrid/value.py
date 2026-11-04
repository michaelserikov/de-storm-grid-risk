"""Correlated failures (A2) and the value of information for line reinforcement (V1).

The decision: which overhead branches to reinforce before the storm season, while the fragility of the lines is
uncertain. Reinforcing makes a branch immune. Rankings come from one Monte Carlo sample and are evaluated on another,
so that a line is not credited with the luck of the sample that selected it.

Value of information follows the thesis logic: EVPI = min_a E[cost(a, theta)] - E[min_a cost(a, theta)], and the
expected value of an imperfect signal about theta (EVSI) by Bayesian updating.

    python -m stormgrid.value
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import flow, hazard, network
from .run import CENTRAL, HAZ, MIN_EXPECTED, OUT, RAW, SEED, TARGETS, WINTERS, load_hazard

PRIOR = np.array([0.25, 0.5, 0.25])     # on the three fragility cases in TARGETS
SAMPLES = 100
BOOT = 500
TOP_K = (1, 2, 3, 5, 8, 12, 20, 30)
LENGTHS_KM = (0.0, 50.0, 200.0)
HOURS = 24.0                # assumed restoration time, h
VOLL = 10000.0              # EUR/MWh, illustrative
COST_GRID = np.round(np.logspace(0, 2.5, 26), 2)      # annual cost of reinforcing, thousand EUR per km
SIGNAL_ACCURACY = (0.6, 0.8, 1.0)


def correlation_factor(branches: pd.DataFrame, length_km: float) -> np.ndarray | None:
    """Cholesky factor of an exponential correlation between branch midpoints; None for independence."""
    if length_km <= 0:
        return None
    mids = np.zeros((len(branches), 2))
    for i, g in enumerate(branches["geometry"]):
        mids[i] = network.parse_wkt(g).mean(axis=0) if g else np.nan
    mids = np.nan_to_num(mids, nan=np.nanmean(mids))
    d = network.haversine_km(mids[:, 0][:, None], mids[:, 1][:, None], mids[:, 0][None, :], mids[:, 1][None, :])
    return np.linalg.cholesky(np.exp(-d / length_km) + 1e-6 * np.eye(len(mids)))


def sample_correlated(prob: np.ndarray, n: int, rng: np.random.Generator, factor: np.ndarray | None) -> list[frozenset]:
    """Failure sets with the given marginal probabilities, through a Gaussian copula."""
    if factor is None:
        return flow.sample_outages(prob, n, rng)
    z = rng.standard_normal((n, len(prob))) @ factor.T
    hits = norm.cdf(z) < prob
    return [frozenset(np.flatnonzero(h).tolist()) for h in hits]


class Simulator:
    """Failure samples for all events of one fragility case, and lost load with any set of reinforced branches."""

    def __init__(self, model, base, prob, seed, factor=None, samples=SAMPLES):
        rng = np.random.default_rng(seed)
        self.model, self.base, self.samples = model, base, samples
        self.active = np.flatnonzero(prob.sum(axis=1) >= MIN_EXPECTED)
        self.n_events = prob.shape[0]
        self.sets = {e: sample_correlated(prob[e], samples, rng, factor) for e in self.active}
        self.cache = {frozenset(): 0.0}
        self.lost0 = {e: np.array([self.lost(s) for s in self.sets[e]]) for e in self.active}

    def lost(self, out: frozenset) -> float:
        if out not in self.cache:
            self.cache[out] = max(self.model.solve(out).sum() - self.base, 0.0)
        return self.cache[out]

    def by_event(self, reinforced: frozenset = frozenset()) -> np.ndarray:
        """Mean lost load per event. A sample with no loss stays without loss when branches are immune."""
        out = np.zeros(self.n_events)
        for e in self.active:
            values = self.lost0[e].copy()
            if reinforced:
                for k in np.flatnonzero(values > 0):
                    s = self.sets[e][k]
                    if s & reinforced:
                        values[k] = self.lost(s - reinforced)
            out[e] = values.mean()
        return out

    def ranking(self) -> np.ndarray:
        """Branches by their share of the lost load, attributing a sample's loss to each failed branch."""
        score = np.zeros(self.model.b.size)
        for e in self.active:
            for s, v in zip(self.sets[e], self.lost0[e]):
                if v > 0:
                    score[list(s)] += v / len(s)
        return np.argsort(-score)[: int((score > 0).sum())]

    def tail(self, level: float) -> float:
        return sum((self.lost0[e] >= level).mean() for e in self.active) / WINTERS


def evpi(cost: np.ndarray, prior: np.ndarray) -> float:
    """cost: actions x states. Expected value of perfect information."""
    return float((cost @ prior).min() - (prior * cost.min(axis=0)).sum())


def evsi(cost: np.ndarray, prior: np.ndarray, accuracy: float) -> float:
    """Value of a signal naming the true state with the given accuracy, otherwise another state at equal odds."""
    n = len(prior)
    like = np.full((n, n), (1 - accuracy) / (n - 1))
    np.fill_diagonal(like, accuracy)
    without = (cost @ prior).min()
    with_signal = 0.0
    for s in range(n):
        joint = prior * like[:, s]
        with_signal += (cost @ joint).min()
    return float(without - with_signal)


def correlated_failures(model, base, prob, branches) -> pd.DataFrame:
    rows = []
    for length in LENGTHS_KM:
        sim = Simulator(model, base, prob, SEED, correlation_factor(branches, length))
        rows.append({"correlation_length_km": length, "expected_lost_mw_per_winter": sim.by_event().sum() / WINTERS,
                     "events_per_winter_above_500_mw": sim.tail(500.0), "events_per_winter_above_1000_mw": sim.tail(1000.0),
                     "max_sampled_lost_mw": max(v.max() for v in sim.lost0.values()), "lps": len(sim.cache)})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def value_of_information(model, base, probs, branches, events, length_km) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    factor = correlation_factor(branches, length_km)
    train = [Simulator(model, base, p, SEED + 100 + i, factor) for i, p in enumerate(probs)]
    test = [Simulator(model, base, p, SEED + 200 + i, factor) for i, p in enumerate(probs)]
    actions = {"none": frozenset()}
    for t, sim in zip(TARGETS, train):
        order = sim.ranking()
        for k in TOP_K:
            actions[f"top {k}, ranked at {t:g} failures"] = frozenset(order[:k].tolist())
    km = branches["length_km"].to_numpy()
    names, kms, event_lost = [], [], []
    seen = set()
    for name, r in actions.items():
        if r in seen:
            continue
        seen.add(r)
        names.append(name)
        kms.append(km[list(r)].sum() if r else 0.0)
        event_lost.append(np.stack([s.by_event(r) for s in test], axis=1))
        print(name, round(kms[-1]), np.round(event_lost[-1].sum(axis=0) / WINTERS, 2), flush=True)
    kms, event_lost = np.array(kms), np.stack(event_lost)      # actions x events x states
    annual = event_lost.sum(axis=1) / WINTERS                 # MW per winter, actions x states
    table = pd.DataFrame(annual, columns=[f"lost_mw_per_winter_at_{t:g}" for t in TARGETS]).assign(action=names, km=kms)
    loss_unit = HOURS * VOLL / 1000.0                          # thousand EUR per MW of lost load

    def cost_matrix(c, ann=annual):
        return c * kms[:, None] + loss_unit * ann

    rows = []
    for c in COST_GRID:
        m = cost_matrix(c)
        best = np.argmin(m @ PRIOR)
        rows.append({"cost_keur_per_km_year": c, "best_action": names[best], "best_km": kms[best],
                     "expected_cost_meur": (m @ PRIOR).min() / 1000, "evpi_meur": evpi(m, PRIOR) / 1000,
                     **{f"evsi_{q:g}_meur": evsi(m, PRIOR, q) / 1000 for q in SIGNAL_ACCURACY},
                     "acts_differ_by_state": len(set(np.argmin(m, axis=0).tolist())) > 1})
    curve = pd.DataFrame(rows)
    peak = curve.loc[curve["evpi_meur"].idxmax()]
    # Whole winters resampled: how firm is the peak of the value of information?
    seasons = events["season"].to_numpy()
    unique = np.unique(seasons)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(BOOT):
        w = np.zeros(len(seasons))
        for y in rng.choice(unique, len(unique)):
            w[seasons == y] += 1
        ann = (event_lost * w[None, :, None]).sum(axis=1) / WINTERS
        boot.append(evpi(cost_matrix(peak["cost_keur_per_km_year"], ann), PRIOR) / 1000)
    info = {"correlation_length_km": length_km, "prior": PRIOR.tolist(), "hours": HOURS, "voll_eur_per_mwh": VOLL,
            "peak_cost_keur_per_km_year": float(peak["cost_keur_per_km_year"]), "peak_evpi_meur_per_year": float(peak["evpi_meur"]),
            "peak_evpi_share_of_cost": float(peak["evpi_meur"] / peak["expected_cost_meur"]),
            "peak_evpi_boot_p05": float(np.quantile(boot, 0.05)), "peak_evpi_boot_p95": float(np.quantile(boot, 0.95)),
            "expected_cost_no_action_meur": float((cost_matrix(0.0)[0] @ PRIOR) / 1000)}
    return table, curve, info


def main() -> dict:
    st, events, ratio = load_hazard()
    grid = network.build(RAW, st)
    b = grid.branches
    km = hazard.station_km(b, st)
    model = flow.ShedModel(grid.buses.index, b, grid.gen, grid.load)
    base = model.solve().sum()
    biggest = int(np.argmax(events["loss_index"].to_numpy()))
    probs = [hazard.line_probabilities(ratio, km, hazard.median_for_target(ratio[biggest], km, t)) for t in TARGETS]
    corr = correlated_failures(model, base, probs[TARGETS.index(CENTRAL)], b)
    corr.to_csv(OUT / "correlated_failures.csv", index=False, float_format="%.4f")
    out = {}
    for length in (0.0, 50.0):
        table, curve, info = value_of_information(model, base, probs, b, events, length)
        tag = f"{int(length)}km"
        table.to_csv(OUT / f"voi_actions_{tag}.csv", index=False, float_format="%.3f")
        curve.to_csv(OUT / f"voi_curve_{tag}.csv", index=False, float_format="%.4f")
        out[tag] = info
        print(json.dumps(info), flush=True)
    (OUT / "voi_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
