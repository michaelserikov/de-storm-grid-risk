"""Windstorms on the German transmission grid: line failures and lost load for 861 events.

    python -m stormgrid.run
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import flow, hazard, network

ROOT = Path(__file__).resolve().parents[2]
RAW, HAZ, OUT = ROOT / "data" / "raw", ROOT / "data" / "hazard", ROOT / "results"
SAMPLES = 200
MIN_EXPECTED = 0.01     # events below this expected number of failures count as no failure
SEED = 11
TARGETS = (2.0, 5.0, 15.0)
CENTRAL = 5.0
LOADS = (55000.0, 70000.0, 80000.0)
WINTERS = 35


def load_hazard():
    st = pd.read_csv(HAZ / "stations.csv").set_index("station_id")
    gust = pd.read_csv(HAZ / "event_max_gust.csv", index_col=0)
    gust.columns = gust.columns.astype(int)
    st = st.loc[gust.columns]
    events = pd.read_csv(HAZ / "events.csv")
    ratio = gust.div(st["v98"], axis=1)
    return st, events, hazard.fill_missing(ratio, st)


def simulate(model, base_shed, prob, cache, rng, samples=SAMPLES):
    """Per event: lost load in each sample, above the shed of the intact grid."""
    rows, by_line_sum, by_line_n = [], np.zeros(prob.shape[1]), np.zeros(prob.shape[1])
    all_lost = []
    for e in range(prob.shape[0]):
        if prob[e].sum() < MIN_EXPECTED:
            all_lost.append(np.zeros(samples))
            continue
        lost = np.zeros(samples)
        for k, out in enumerate(flow.sample_outages(prob[e], samples, rng)):
            if out:
                if out not in cache:
                    cache[out] = max(model.solve(out).sum() - base_shed, 0.0)
                lost[k] = cache[out]
                idx = list(out)
                by_line_sum[idx] += lost[k]
                by_line_n[idx] += 1
        all_lost.append(lost)
    return np.array(all_lost), by_line_sum, by_line_n


def main() -> dict:
    OUT.mkdir(exist_ok=True)
    st, events, ratio = load_hazard()
    grid = network.build(RAW, st)
    b = grid.branches
    km = hazard.station_km(b, st)
    model = flow.ShedModel(grid.buses.index, b, grid.gen, grid.load)
    base = model.solve()
    print("buses", len(grid.buses), "branches", len(b), "gen GW", round(grid.gen.sum() / 1e3, 1),
          "base shed MW", round(base.sum(), 1), flush=True)
    biggest = int(np.argmax(events["loss_index"].to_numpy()))
    rows, keep = [], {}
    for target in TARGETS:
        median = hazard.median_for_target(ratio[biggest], km, target)
        prob = hazard.line_probabilities(ratio, km, median)
        cache = {}
        lost, s, n = simulate(model, base.sum(), prob, cache, np.random.default_rng(SEED))
        keep[target] = (median, prob, lost, s, n)
        mean_event = lost.mean(axis=1)
        rows.append({"expected_failures_largest_storm": target, "fragility_median": median,
                     "events_with_failures": int((prob.sum(axis=1) >= MIN_EXPECTED).sum()), "lps_solved": len(cache),
                     "failed_lines_per_winter": prob.sum() / WINTERS,
                     "lost_load_largest_storm_mw": mean_event[biggest],
                     "p_any_loss_largest_storm": (lost[biggest] > 0).mean(),
                     "max_sampled_lost_load_mw": lost.max(), "expected_lost_mw_per_winter": mean_event.sum() / WINTERS,
                     "kendall_event_loss_index_vs_lost_load": kendalltau(events["loss_index"], mean_event)[0]})
        print(rows[-1], flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary_by_fragility.csv", index=False, float_format="%.4f")

    median, prob, lost, s, n = keep[CENTRAL]
    ev = events.assign(expected_failures=prob.sum(axis=1), expected_lost_mw=lost.mean(axis=1),
                       p_any_loss=(lost > 0).mean(axis=1), max_lost_mw=lost.max(axis=1))
    ev.sort_values("expected_lost_mw", ascending=False).head(30).to_csv(OUT / "top_events.csv", index=False, float_format="%.4f")
    curve = pd.DataFrame({"lost_mw": [100, 250, 500, 1000, 2000, 5000]})
    curve["events_per_winter"] = [(lost >= x).mean(axis=1).sum() / WINTERS for x in curve["lost_mw"]]
    curve["return_period_winters"] = 1 / curve["events_per_winter"].replace(0, np.nan)
    curve.to_csv(OUT / "exceedance.csv", index=False, float_format="%.4f")
    lines = b.drop(columns=["geometry"]).assign(
        mean_failure_per_winter=prob.sum(axis=0) / WINTERS, samples_failed=n,
        mean_lost_when_failed_mw=np.where(n > 0, s / np.maximum(n, 1), 0.0), share_of_lost=s / max(s.sum(), 1e-9))
    lines.sort_values("share_of_lost", ascending=False).head(30).to_csv(OUT / "critical_branches.csv", index=False, float_format="%.4f")

    # Sensitivity of the central case: load level and an N-1-style reduction of the flow limits
    sens = []
    cases = [(f"load {int(x / 1000)} GW", network.spread_load(st, grid.buses, x), b["limit"].to_numpy()) for x in LOADS]
    cases.append(("flow limits x 0.7", grid.load, 0.7 * b["limit"].to_numpy()))
    top = np.argsort(-prob.sum(axis=1))[:40]
    for name, load, limit in cases:
        branches = b.assign(limit=limit)
        m = flow.ShedModel(grid.buses.index, branches, grid.gen, load)
        b0 = m.solve().sum()
        l2, _, _ = simulate(m, b0, prob[top], {}, np.random.default_rng(SEED), samples=100)
        sens.append({"case": name, "intact_grid_shed_mw": b0, "expected_lost_mw_top_40_events": l2.mean(axis=1).sum() / WINTERS})
        print(sens[-1], flush=True)
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False, float_format="%.2f")
    out = {"buses": len(grid.buses), "branches": len(b), "lines": int((b["kind"] == "line").sum()),
           "overhead_km": float(b.loc[b["overhead"], "length_km"].sum()), "available_generation_gw": grid.gen.sum() / 1e3,
           "intact_grid_shed_mw": float(base.sum()), "largest_storm": events.iloc[biggest].to_dict()}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, default=str))
    return out


if __name__ == "__main__":
    main()
