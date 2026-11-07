"""Line reinforcement as a two-stage stochastic mixed-integer program, solved by Benders decomposition (O1).

First stage: which candidate branches to reinforce (binary), at a cost per km and year. Second stage: for every
sampled failure set, a DC power flow that minimises load shedding. A failed candidate branch stays in service if it
was reinforced; a failed branch that is not a candidate is lost. Scenarios cover the three fragility cases with their
prior, so the program weighs the reinforcement against the expected lost load under uncertain fragility.

    min_x  sum_l c km_l x_l + sum_w p_w VoLL T Q_w(x)
    Q_w(x) = min shed - base   s.t. DC flow; failed candidate l in service iff x_l = 1 (big-M on the flow law)

The big-M of a failed candidate is its susceptance times the shortest path between its ends over the branches that
are surely in service, with weights limit / susceptance: no feasible angle difference can exceed it, so the relaxed
constraint never binds and Q_w is exact at binary x. Q_w is convex in the relaxed x, so the LP duals give valid
optimality cuts (multi-cut Benders, one cut set per scenario). Cuts are kept in a pool and reused across problems.

    python -m stormgrid.stochastic
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.csgraph import dijkstra

from . import flow, hazard, lp, network
from .run import CENTRAL, OUT, RAW, SEED, TARGETS, WINTERS, load_hazard
from .value import HOURS, PRIOR, VOLL, Simulator

LOSS_UNIT = HOURS * VOLL / 1000.0          # thousand EUR per MW of lost load
COSTS = (2.5, 4.0, 6.3, 10.0, 16.0)        # thousand EUR per km and year, inside the window of the VoI study
N_CANDIDATES = 40                          # from the central ranking, plus the top of the other cases
N_EXTRA = 15
SAMPLES = 100
GAP = 1e-3
MAX_ITER = 60
CACHE = RAW / "stochastic_cache.pkl"


class Recourse:
    """Second-stage LP with failed candidates switchable by x; returns value and subgradient in x."""

    def __init__(self, model: flow.ShedModel):
        self.m = model
        self.n = model.n
        self.bs = model.b / model.scale                  # susceptance in scaled angle units
        self.weight = model.limit / self.bs              # largest angle difference a branch in service allows
        self.total = float(self.weight.sum())

    def big_m(self, keep: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = self.n
        g = sp.csr_matrix((np.r_[self.weight[keep], self.weight[keep]],
                           (np.r_[self.m.f[keep], self.m.t[keep]], np.r_[self.m.t[keep], self.m.f[keep]])), shape=(n, n))
        ends = np.unique(np.r_[self.m.f[d], self.m.t[d]])
        dist = dijkstra(g, directed=False, indices=ends)
        pos = {e: i for i, e in enumerate(ends)}
        out = np.empty(len(d))
        for j, l in enumerate(d):
            span = dist[pos[self.m.f[l]], self.m.t[l]]
            out[j] = self.bs[l] * (span if np.isfinite(span) else self.total)
        return out * (1 + 1e-6) + 1e-6

    def solve(self, failed: frozenset, cand: dict, x: dict) -> tuple[float, dict]:
        """failed: branch positions; cand: position -> candidate index; x: candidate index -> 0/1.
        Returns shed - base (MW) and the subgradient with respect to x of the failed candidates."""
        m, n = self.m, self.n
        d = np.array(sorted(l for l in failed if l in cand), dtype=int)
        keep = np.array([l for l in range(len(m.b)) if l not in failed], dtype=int)
        k, nd = len(keep), len(d)
        f, t, bs = m.f[keep], m.t[keep], self.bs[keep]
        rows = np.arange(k)
        flowK = sp.csr_matrix((np.r_[bs, -bs], (np.r_[rows, rows], np.r_[f, t])), shape=(k, n))
        incK = sp.csr_matrix((np.r_[np.ones(k), -np.ones(k)], (np.r_[f, t], np.r_[rows, rows])), shape=(n, k))
        eye = sp.identity(n, format="csr")
        # variables: theta (n), g (n), s (n), f_d (nd)
        if nd:
            rd = np.arange(nd)
            incD = sp.csr_matrix((np.r_[np.ones(nd), -np.ones(nd)], (np.r_[m.f[d], m.t[d]], np.r_[rd, rd])), shape=(n, nd))
            angD = sp.csr_matrix((np.r_[self.bs[d], -self.bs[d]], (np.r_[rd, rd], np.r_[m.f[d], m.t[d]])), shape=(nd, n))
        else:
            incD = sp.csr_matrix((n, 0))
            angD = sp.csr_matrix((0, n))
        A_eq = sp.hstack([incK @ flowK, -eye, -eye, incD], format="csr")
        zK = sp.csr_matrix((k, 2 * n + nd))
        idD = sp.identity(nd, format="csr")
        zD = sp.csr_matrix((nd, 2 * n))
        A_ub = sp.vstack([sp.hstack([flowK, zK[:, :2 * n + nd]]), sp.hstack([-flowK, zK[:, :2 * n + nd]]),
                          sp.hstack([-angD, zD, idD]), sp.hstack([angD, zD, -idD]),
                          sp.hstack([sp.csr_matrix((nd, 3 * n)), idD]), sp.hstack([sp.csr_matrix((nd, 3 * n)), -idD])],
                         format="csr")
        M = self.big_m(keep, d) if nd else np.zeros(0)
        xd = np.array([x[cand[l]] for l in d], dtype=float)
        lim = m.limit[keep]
        limD = m.limit[d]
        b_ub = np.r_[lim, lim, M * (1 - xd), M * (1 - xd), limD * xd, limD * xd]
        cost = np.r_[np.zeros(2 * n), np.ones(n), np.zeros(nd)]
        bounds = [(None, None)] * n + [(0.0, g) for g in m.gen] + [(0.0, dd) for dd in m.load] + [(None, None)] * nd
        r = lp.solve(cost, A_ub, b_ub, A_eq, -m.load, bounds)
        mu = r.ineq_marginals                            # d objective / d b_ub
        o = 2 * k
        grad = {}
        for j, l in enumerate(d):
            g_l = -M[j] * (mu[o + j] + mu[o + nd + j]) + limD[j] * (mu[o + 2 * nd + j] + mu[o + 3 * nd + j])
            grad[cand[l]] = float(g_l)
        return float(r.fun), grad


@dataclass
class Scenario:
    failed: frozenset
    case: int                    # index into TARGETS
    weight: float                # probability mass per winter, before the prior
    dec: tuple = ()              # candidate indices among the failed branches
    cuts: list = field(default_factory=list)       # (value at xhat, {cand: grad}, xhat restricted to dec)
    values: dict = field(default_factory=dict)     # xhat restricted to dec -> value


class Benders:
    def __init__(self, model, base, scenarios, cand_pos, km, log=print):
        self.rec = Recourse(model)
        self.base = base
        self.sc = scenarios
        self.cand = {p: i for i, p in enumerate(cand_pos)}
        self.km = np.asarray(km, dtype=float)
        self.nc = len(cand_pos)
        self.lps = 0
        self.log = log
        for s in self.sc:
            s.dec = tuple(sorted(self.cand[l] for l in s.failed if l in self.cand))

    def value(self, s: Scenario, x: np.ndarray) -> float:
        key = tuple(int(round(x[i])) for i in s.dec)
        if key not in s.values:
            q, g = self.rec.solve(s.failed, self.cand, {i: v for i, v in zip(s.dec, key)})
            self.lps += 1
            s.values[key] = q - self.base
            s.cuts.append((q - self.base, g, key))
        return s.values[key]

    def solve(self, c: float, prior: np.ndarray, start=None, time_limit=600.0) -> dict:
        """Minimise c km x + sum prior_case weight LOSS_UNIT Q. Returns plan, bounds and history."""
        active = [s for s in self.sc if prior[s.case] > 0]
        fixed = [s for s in active if not s.dec]
        dec = [s for s in active if s.dec]
        p = np.array([prior[s.case] * s.weight * LOSS_UNIT for s in dec])
        const = sum(prior[s.case] * s.weight * LOSS_UNIT * self.value(s, np.zeros(self.nc)) for s in fixed)
        x = np.zeros(self.nc) if start is None else np.asarray(start, dtype=float)
        best_ub, best_x, lb, hist = np.inf, x.copy(), -np.inf, []
        t0 = time.time()
        for s in dec:                      # seed the cuts with no action and with all candidates reinforced
            self.value(s, np.zeros(self.nc))
            self.value(s, np.ones(self.nc))
        for it in range(MAX_ITER):
            ub = c * self.km @ x + const + sum(pi * self.value(s, x) for pi, s in zip(p, dec))
            if ub < best_ub:
                best_ub, best_x = ub, x.copy()
            # master over x (nc binaries) and eta (one per scenario with a decision)
            nv = self.nc + len(dec)
            obj = np.r_[c * self.km, p]
            rows, cols, vals, lo = [], [], [], []
            r = 0
            for j, s in enumerate(dec):
                for q, g, key in s.cuts:
                    # eta_j >= q + sum_i g_i (x_i - key_i)
                    rows.append(r); cols.append(self.nc + j); vals.append(1.0)
                    rhs = q
                    for i, kv in zip(s.dec, key):
                        gi = g.get(i, 0.0)
                        if gi:
                            rows.append(r); cols.append(i); vals.append(-gi)
                            rhs -= gi * kv
                    lo.append(rhs)
                    r += 1
            A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
            eta_lo = np.full(len(dec), -self.base - 1e-6)      # shed is never negative
            res = milp(obj, constraints=LinearConstraint(A, np.array(lo), np.inf),
                       integrality=np.r_[np.ones(self.nc), np.zeros(len(dec))],
                       bounds=Bounds(np.r_[np.zeros(self.nc), eta_lo], np.r_[np.ones(self.nc), np.full(len(dec), np.inf)]),
                       options={"time_limit": max(30.0, time_limit / 4), "mip_rel_gap": GAP / 4})
            if res.x is None:
                raise RuntimeError(res.message)
            lb = max(lb, res.fun + const) if res.status == 0 else lb
            x = np.round(res.x[: self.nc])
            gap = (best_ub - lb) / max(abs(best_ub), 1e-9)
            hist.append({"iteration": it, "lower": lb, "upper": best_ub, "gap": gap, "lps": self.lps,
                         "seconds": time.time() - t0, "reinforced": int(best_x.sum())})
            self.log(hist[-1])
            if gap <= GAP or time.time() - t0 > time_limit:
                break
        return {"x": best_x, "upper": best_ub, "lower": lb, "gap": (best_ub - lb) / max(abs(best_ub), 1e-9),
                "iterations": len(hist), "history": hist, "seconds": time.time() - t0}

    def expected(self, x: np.ndarray, c: float, prior: np.ndarray) -> float:
        return c * self.km @ x + sum(prior[s.case] * s.weight * LOSS_UNIT * self.value(s, x) for s in self.sc if prior[s.case] > 0)


def scenarios_from(sims: list[Simulator]) -> list[Scenario]:
    """One scenario per distinct failure set and fragility case, weighted per winter."""
    out = []
    for case, sim in enumerate(sims):
        w = {}
        for e in sim.active:
            for s in sim.sets[e]:
                if s:
                    w[s] = w.get(s, 0.0) + 1.0 / (sim.samples * WINTERS)
        out += [Scenario(failed=s, case=case, weight=v) for s, v in w.items()]
    return out


def candidates(sims: list[Simulator], branches: pd.DataFrame) -> list[int]:
    central = sims[TARGETS.index(CENTRAL)].ranking()
    out = [int(i) for i in central[:N_CANDIDATES]]
    for i, sim in enumerate(sims):
        if TARGETS[i] != CENTRAL:
            out += [int(j) for j in sim.ranking()[:N_EXTRA] if int(j) not in out]
    return [i for i in out if branches["overhead"].iloc[i]]


def test_cost(sims_test: list[Simulator], reinforced: frozenset, km: float, c: float, prior: np.ndarray) -> float:
    """Expected annual cost on an independent sample, thousand EUR (exact lost load, as in the VoI study)."""
    lost = np.array([s.by_event(reinforced).sum() / WINTERS for s in sims_test])
    return float(c * km + LOSS_UNIT * prior @ lost)


def main() -> dict:
    t0 = time.time()
    st, events, ratio = load_hazard()
    grid = network.build(RAW, st)
    b = grid.branches
    kmat = hazard.station_km(b, st)
    model = flow.ShedModel(grid.buses.index, b, grid.gen, grid.load)
    base = model.solve().sum()
    biggest = int(np.argmax(events["loss_index"].to_numpy()))
    probs = [hazard.line_probabilities(ratio, kmat, hazard.median_for_target(ratio[biggest], kmat, t)) for t in TARGETS]
    if CACHE.exists():
        state = pickle.loads(CACHE.read_bytes())
        print("resumed cache", flush=True)
    else:
        train = [Simulator(model, base, p, SEED + 100 + i, None, SAMPLES) for i, p in enumerate(probs)]
        test = [Simulator(model, base, p, SEED + 200 + i, None, SAMPLES) for i, p in enumerate(probs)]
        cand = candidates(train, b)
        state = {"cand": cand, "scen": scenarios_from(train), "train": train, "test": test}
        CACHE.write_bytes(pickle.dumps(state))
    cand, scen, train, test = state["cand"], state["scen"], state["train"], state["test"]
    km = b["length_km"].to_numpy()[cand]
    print("candidates", len(cand), "km", round(km.sum()), "scenarios", len(scen),
          "with a candidate failed", sum(any(l in set(cand) for l in s.failed) for s in scen), flush=True)
    bd = Benders(model, base, scen, cand, km, log=lambda h: print(h, flush=True))
    ids = b.index.to_numpy()

    def plan(x):
        return frozenset(int(cand[i]) for i in np.flatnonzero(np.round(x) > 0))

    # heuristic plans of the VoI study: top k ranked under each case on the training sample
    heur = {}
    for i, sim in enumerate(train):
        order = sim.ranking()
        for k in (1, 2, 3, 5, 8, 12, 20, 30):
            heur[f"top {k}, ranked at {TARGETS[i]:g} failures"] = frozenset(int(j) for j in order[:k])
    heur["none"] = frozenset()

    rows, plans, history = [], [], []
    ev_prior = np.array([1.0 if t == CENTRAL else 0.0 for t in TARGETS])
    for c in COSTS:
        res = {"RP": bd.solve(c, PRIOR)}
        res["EV"] = bd.solve(c, ev_prior, start=res["RP"]["x"])
        for i, t in enumerate(TARGETS):
            one = np.zeros(len(TARGETS)); one[i] = 1.0
            res[f"WS {t:g}"] = bd.solve(c, one, start=res["RP"]["x"])
        rp = res["RP"]["upper"]
        eev = bd.expected(res["EV"]["x"], c, PRIOR)
        ws = sum(PRIOR[i] * res[f"WS {t:g}"]["upper"] for i, t in enumerate(TARGETS))
        # out of sample: exact lost load on the independent test sample
        all_km = b["length_km"].to_numpy()
        oos = {name: test_cost(test, plan(res[name]["x"]), all_km[list(plan(res[name]["x"]))].sum(), c, PRIOR) for name in ("RP", "EV")}
        heur_cost = {name: test_cost(test, r, all_km[list(r)].sum(), c, PRIOR) for name, r in heur.items()}
        best_h = min(heur_cost, key=heur_cost.get)
        heur_train = {name: c * all_km[list(r)].sum() + LOSS_UNIT * PRIOR @ np.array([s.by_event(r).sum() / WINTERS for s in train]) for name, r in heur.items()}
        best_h_train = min(heur_train, key=heur_train.get)
        row = {"cost_keur_per_km_year": c, "rp_meur": rp / 1000, "eev_meur": eev / 1000, "ws_meur": ws / 1000,
               "evpi_meur": (rp - ws) / 1000, "vss_meur": (eev - rp) / 1000,
               "rp_branches": int(res["RP"]["x"].sum()), "rp_km": float(km @ res["RP"]["x"]),
               "ev_branches": int(res["EV"]["x"].sum()), "ev_km": float(km @ res["EV"]["x"]),
               "rp_gap": res["RP"]["gap"], "rp_iterations": res["RP"]["iterations"], "rp_seconds": res["RP"]["seconds"],
               "test_rp_meur": oos["RP"] / 1000, "test_ev_meur": oos["EV"] / 1000,
               "test_best_heuristic_meur": heur_cost[best_h] / 1000, "best_heuristic_on_test": best_h,
               "train_best_heuristic_meur": heur_train[best_h_train] / 1000, "best_heuristic_on_train": best_h_train,
               "test_heuristic_chosen_on_train_meur": heur_cost[best_h_train] / 1000,
               "test_no_action_meur": heur_cost["none"] / 1000, "lps_total": bd.lps}
        rows.append(row)
        print(row, flush=True)
        for name in res:
            plans.append({"cost_keur_per_km_year": c, "problem": name, "branches": ";".join(str(ids[j]) for j in sorted(plan(res[name]["x"]))),
                          "n": int(res[name]["x"].sum()), "km": float(km @ res[name]["x"]), "objective_meur": res[name]["upper"] / 1000,
                          "gap": res[name]["gap"]})
            history += [{"cost_keur_per_km_year": c, "problem": name, **h} for h in res[name]["history"]]
        pd.DataFrame(rows).to_csv(OUT / "stochastic_summary.csv", index=False, float_format="%.4f")
        pd.DataFrame(plans).to_csv(OUT / "stochastic_plans.csv", index=False, float_format="%.4f")
        pd.DataFrame(history).to_csv(OUT / "stochastic_benders_history.csv", index=False, float_format="%.5f")
        CACHE.write_bytes(pickle.dumps(state))          # keeps the cut pool and solved LPs for a restart
    info = {"candidates": len(cand), "candidate_km": float(km.sum()), "scenarios": len(scen),
            "samples_per_event": SAMPLES, "prior": PRIOR.tolist(), "costs_keur_per_km_year": list(COSTS),
            "loss_unit_keur_per_mw": LOSS_UNIT, "gap_target": GAP, "lps": bd.lps, "seconds": time.time() - t0}
    (OUT / "stochastic_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":
    main()
