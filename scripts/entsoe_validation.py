"""Do real transmission outages follow the storms the model rates most dangerous? ENTSO-E, September 2020 to 2025.

Counts forced (unplanned, business type A54) unavailabilities of German transmission assets that start
inside each storm window (start day to end day + 1), compares them with the forced-outage rate on
ordinary winter days, and ranks the 2015-2025 storms by real outages against the model's expected line
failures (central fragility, 5 failures in the largest storm) and the hazard index.

Caveat: the Transparency Platform covers only outages relevant to cross-zonal capacity, a small subset
of the grid, and its asset names do not map to the open OSM lines, so the check is at event level.

    python scripts/entsoe_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, poisson, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stormgrid import hazard, network  # noqa: E402
from stormgrid.run import CENTRAL, RAW, TARGETS, load_hazard  # noqa: E402

OUTAGES = ROOT / "data" / "outages" / "entsoe_transmission_outages.csv"
OUT = ROOT / "results"
# Forced outages (A54) appear on the platform only from September 2020; earlier years hold planned works only.
FIRST, LAST = "2020-09-01", "2025-12-31"
WINTER = (10, 11, 12, 1, 2, 3)
LAG_DAYS = 1          # outages that start on the day after the storm still count


def forced_outages() -> pd.DataFrame:
    o = pd.read_csv(OUTAGES)
    o = o[o["business_type"] == "A54"].copy()
    o["start"] = pd.to_datetime(o["start"], utc=True).dt.tz_convert(None)
    o["end"] = pd.to_datetime(o["end"], utc=True).dt.tz_convert(None)
    # one row per document: a forced outage of several assets is one event on the grid, assets counted apart
    doc = o.groupby("mrid").agg(start=("start", "first"), end=("end", "first"), assets=("asset_mrid", "nunique"),
                                reason=("reason", "first"), names=("asset_name", lambda s: "; ".join(sorted(set(map(str, s))))),
                                psr=("psr_type", lambda s: ";".join(sorted(set(map(str, s))))), pair=("pair", "first"))
    doc = doc[(doc["start"] >= FIRST) & (doc["start"] <= LAST)]
    doc["day"] = doc["start"].dt.normalize()
    return doc.reset_index()


def main():
    OUT.mkdir(exist_ok=True)
    st, ev, ratio = load_hazard()
    grid = network.build(RAW, st)
    km = hazard.station_km(grid.branches, st)
    big = int(np.argmax(ev["loss_index"].to_numpy()))
    for t in TARGETS:
        ev[f"ef_{int(t)}"] = hazard.line_probabilities(ratio, km, hazard.median_for_target(ratio[big], km, t)).sum(axis=1)
    ev["ef"] = ev[f"ef_{int(CENTRAL)}"]
    ev["start"], ev["end"] = pd.to_datetime(ev["start"]), pd.to_datetime(ev["end"])
    e = ev[(ev["start"] >= FIRST) & (ev["end"] <= LAST)].copy()

    doc = forced_outages()
    days = pd.Series(0, index=pd.date_range(FIRST, LAST, freq="D"))
    days = days.add(doc.groupby("day").size(), fill_value=0)
    assets = pd.Series(0, index=days.index).add(doc.groupby("day")["assets"].sum(), fill_value=0)
    window = pd.Series(False, index=days.index)
    for s, t in zip(e["start"], e["end"]):
        window.loc[s:t + pd.Timedelta(days=LAG_DAYS)] = True
    e["forced_docs"] = [days.loc[s:t + pd.Timedelta(days=LAG_DAYS)].sum() for s, t in zip(e["start"], e["end"])]
    e["forced_assets"] = [assets.loc[s:t + pd.Timedelta(days=LAG_DAYS)].sum() for s, t in zip(e["start"], e["end"])]
    e["window_days"] = (e["end"] - e["start"]).dt.days + 1 + LAG_DAYS

    winter = days.index.month.isin(WINTER)
    base_rate = days[winter & ~window.to_numpy()].mean()
    e["expected_by_chance"] = base_rate * e["window_days"]
    e["p_value"] = poisson.sf(e["forced_docs"] - 1, e["expected_by_chance"])

    preds = ["loss_index", "stations_above_v98", "max_gust", "ef_2", "ef", "ef_15"]
    rows = []
    for sample, d in [("all events Sep 2020-2025", e), ("top 30 by loss index", e.nlargest(30, "loss_index")),
                      ]:
        for p in preds:
            tau, ptau = kendalltau(d[p], d["forced_docs"])
            rho, _ = spearmanr(d[p], d["forced_docs"])
            rows.append({"sample": sample, "predictor": p, "events": len(d), "kendall_tau": tau, "p_tau": ptau,
                         "spearman": rho})
    corr = pd.DataFrame(rows)

    named = e[e["storm"].notna()]
    top_model = set(e.nlargest(5, "ef").index)
    top_real = set(e.nlargest(5, "forced_docs").index)
    summary = {
        "period": f"{FIRST} to {LAST}",
        "forced_outage_documents": int(len(doc)),
        "forced_outage_assets": int(doc["assets"].sum()),
        "events": int(len(e)),
        "base_rate_per_ordinary_winter_day": float(base_rate),
        "share_of_forced_outages_in_storm_windows": float(days[window.to_numpy()].sum() / max(days.sum(), 1)),
        "share_of_days_in_storm_windows": float(window.mean()),
        "named_storms": {r.storm: {"start": str(r.start.date()), "forced_docs": int(r.forced_docs),
                                   "expected_by_chance": round(float(r.expected_by_chance), 2),
                                   "p_value": float(r.p_value), "model_rank_ef": int((e["ef"] > r.ef).sum() + 1),
                                   "real_rank": int((e["forced_docs"] > r.forced_docs).sum() + 1)}
                         for r in named.itertuples()},
        "overlap_top5_model_vs_real": len(top_model & top_real),
    }
    cols = ["start", "end", "storm", "loss_index", "ef", "forced_docs", "forced_assets", "expected_by_chance", "p_value"]
    e.sort_values("ef", ascending=False)[cols].to_csv(OUT / "entsoe_events.csv", index=False, float_format="%.4f")
    corr.to_csv(OUT / "entsoe_rank_correlation.csv", index=False, float_format="%.4f")
    in_storm = doc[window.reindex(doc["day"]).to_numpy()]
    in_storm[["start", "end", "assets", "reason", "psr", "pair", "names"]].sort_values("start").to_csv(
        OUT / "entsoe_forced_in_storms.csv", index=False)
    (OUT / "entsoe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(corr.round(3).to_string())
    print(e.nlargest(12, "forced_docs")[cols].round(3).to_string())
    print(e.nlargest(12, "ef")[cols].round(3).to_string())


if __name__ == "__main__":
    main()
