"""Are the same branches critical in different storms?

Central fragility case, independent failures. Branches are ranked for each of the worst storms on a training sample.
Rankings are compared between storms, and tested on an independent sample: reinforce the top k from all other events
(leave one storm out) and measure the lost load left in the storm that was left out. The storm's own ranking gives the
upper reference, and randomly chosen overhead lines the lower one.

    PYTHONPATH=src python scripts/storm_stability.py
"""

from __future__ import annotations

import itertools
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from stormgrid import flow, hazard, network
from stormgrid.run import CENTRAL, OUT, RAW, SEED, load_hazard
from stormgrid.value import Simulator

STORMS = 10
TRAIN_SAMPLES = 200
TEST_SAMPLES = 200
TOP_K = (5, 10, 20, 30)
RANDOM_DRAWS = 3
CACHE = RAW / "storm_stability_lp_cache.pkl"
SAVE_EVERY = 50


class Stored:
    """Lost load per set of failed branches, kept on disk so that an interrupted run resumes without solving again."""

    def __init__(self, path):
        self.path = path
        self.data = pickle.loads(path.read_bytes()) if path.exists() else {}
        self.new = 0
        print("cached LPs on disk", len(self.data), flush=True)

    def add(self, key, value):
        self.data[key] = value
        self.new += 1
        if self.new % SAVE_EVERY == 0:
            self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self.data))
        os.replace(tmp, self.path)


STORE = Stored(CACHE)


class ResumableSimulator(Simulator):
    """The same simulator, with every solved LP shared through the disk store. Results are identical: the sampled
    failure sets depend only on the seed, and a set's lost load does not depend on the event it was drawn in."""

    def lost(self, out: frozenset) -> float:
        if out not in self.cache and out in STORE.data:
            self.cache[out] = STORE.data[out]
        if out not in self.cache:
            value = super().lost(out)
            STORE.add(out, value)
            return value
        return self.cache[out]


def event_scores(sim: Simulator) -> dict[int, np.ndarray]:
    """Per event: share of the lost load per branch, a sample's loss split equally between its failed branches."""
    out = {}
    for e in sim.active:
        score = np.zeros(sim.model.b.size)
        for s, v in zip(sim.sets[e], sim.lost0[e]):
            if v > 0:
                score[list(s)] += v / len(s)
        out[int(e)] = score
    return out


def top(score: np.ndarray, k: int) -> frozenset:
    order = np.argsort(-score)
    return frozenset(order[: min(k, int((score > 0).sum()))].tolist())


def main() -> dict:
    t0 = time.time()
    st, events, ratio = load_hazard()
    grid = network.build(RAW, st)
    b = grid.branches
    km = hazard.station_km(b, st)
    model = flow.ShedModel(grid.buses.index, b, grid.gen, grid.load)
    base = model.solve().sum()
    biggest = int(np.argmax(events["loss_index"].to_numpy()))
    median = hazard.median_for_target(ratio[biggest], km, CENTRAL)
    prob = hazard.line_probabilities(ratio, km, median)

    train = ResumableSimulator(model, base, prob, SEED + 300, samples=TRAIN_SAMPLES)
    STORE.save()
    print("train done", round(time.time() - t0), "s, LPs", len(train.cache), flush=True)
    mean_lost = train.by_event()
    storms = np.argsort(-mean_lost)[:STORMS].tolist()
    scores = event_scores(train)
    total = sum(scores.values())

    # Overlap between storms
    rows = []
    for k in TOP_K:
        sets = [top(scores[e], k) for e in storms]
        jac = [len(x & y) / max(len(x | y), 1) for x, y in itertools.combinations(sets, 2)]
        rows.append({"k": k, "mean_pairwise_jaccard": float(np.mean(jac)), "median_pairwise_jaccard": float(np.median(jac)),
                     "share_of_pairs_with_no_overlap": float(np.mean([j == 0 for j in jac]))})
    overlap = pd.DataFrame(rows)

    # Independent test on the storms
    test = ResumableSimulator(model, base, prob[storms], SEED + 400, samples=TEST_SAMPLES)
    STORE.save()
    no_action = test.by_event()
    overhead = np.flatnonzero(b["overhead"].to_numpy())
    rng = np.random.default_rng(SEED + 500)
    rows = []
    for k in TOP_K:
        draws = [frozenset(rng.choice(overhead, k, replace=False).tolist()) for _ in range(RANDOM_DRAWS)]
        random_left = np.mean([test.by_event(r) for r in draws], axis=0)
        for i, e in enumerate(storms):
            own = test.by_event(top(scores[e], k))[i]
            others = test.by_event(top(total - scores[e], k))[i]
            rows.append({"k": k, "event": e, "storm": events["storm"].iloc[e], "start": events["start"].iloc[e],
                         "lost_no_action_mw": no_action[i], "lost_own_ranking_mw": own, "lost_others_ranking_mw": others,
                         "lost_random_mw": random_left[i]})
        STORE.save()
        print("k", k, round(time.time() - t0), "s, LPs", len(test.cache), flush=True)
    per_storm = pd.DataFrame(rows)
    grouped = per_storm.groupby("k")[["lost_no_action_mw", "lost_own_ranking_mw", "lost_others_ranking_mw", "lost_random_mw"]].sum()
    for name in ("own_ranking", "others_ranking", "random"):
        grouped[f"reduction_{name}"] = 1 - grouped[f"lost_{name}_mw"] / grouped["lost_no_action_mw"]
    grouped = grouped.reset_index()

    overlap.to_csv(OUT / "storm_stability_overlap.csv", index=False, float_format="%.3f")
    per_storm.to_csv(OUT / "storm_stability_by_storm.csv", index=False, float_format="%.2f")
    grouped.to_csv(OUT / "storm_stability_summary.csv", index=False, float_format="%.3f")
    info = {"storms": STORMS, "train_samples": TRAIN_SAMPLES, "test_samples": TEST_SAMPLES, "random_draws": RANDOM_DRAWS,
            "fragility_case": CENTRAL, "selection": "worst storms by mean lost load on the training sample",
            "lines_overhead": int(len(overhead)), "seconds": round(time.time() - t0)}
    (OUT / "storm_stability_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(overlap.to_string())
    print(grouped.to_string())
    print(per_storm[per_storm["k"] == 20].to_string())
    return info


if __name__ == "__main__":
    main()
