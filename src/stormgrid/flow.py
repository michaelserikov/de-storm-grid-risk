"""Lost load after line failures: a DC power flow that minimises load shedding (linear program, HiGHS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from . import lp


class ShedModel:
    """Variables per bus: angle, generation, shed. Balance at each bus, flow limits on each branch in service."""

    def __init__(self, buses: pd.Index, branches: pd.DataFrame, gen: np.ndarray, load: np.ndarray):
        self.n = len(buses)
        pos = pd.Series(np.arange(self.n), index=buses)
        self.f = pos[branches["bus0"]].to_numpy()
        self.t = pos[branches["bus1"]].to_numpy()
        self.b = branches["b"].to_numpy(dtype=float)
        self.limit = branches["limit"].to_numpy(dtype=float)
        self.gen, self.load = np.asarray(gen, dtype=float), np.asarray(load, dtype=float)
        self.scale = np.median(self.b)     # angles in scaled units, for conditioning
        self.cost = np.concatenate([np.zeros(2 * self.n), np.ones(self.n)])

    def solve(self, out: frozenset | set = frozenset()) -> np.ndarray:
        """Shed per bus (MW) with the given branch positions out of service."""
        keep = np.array([i for i in range(len(self.b)) if i not in out], dtype=int)
        f, t, b = self.f[keep], self.t[keep], self.b[keep] / self.scale
        m, n = len(keep), self.n
        rows = np.arange(m)
        flow = sp.csr_matrix((np.r_[b, -b], (np.r_[rows, rows], np.r_[f, t])), shape=(m, n))
        incidence = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[f, t], np.r_[rows, rows])), shape=(n, m))
        # incidence * flow - generation - shed = -load, in units of the scale so the angles stay well conditioned
        eye = sp.identity(n, format="csr")
        A_eq = sp.hstack([incidence @ flow, -eye, -eye], format="csr")
        A_ub = sp.vstack([sp.hstack([flow, sp.csr_matrix((m, 2 * n))]), sp.hstack([-flow, sp.csr_matrix((m, 2 * n))])], format="csr")
        limit = self.limit[keep]
        bounds = [(None, None)] * n + [(0.0, g) for g in self.gen] + [(0.0, d) for d in self.load]
        # A flow in MW is scale * b * angle difference, so the angle variable absorbs the scale
        r = lp.solve(self.cost, A_ub, np.r_[limit, limit], A_eq, -self.load, bounds)
        return r.x[2 * n:]


def sample_outages(prob: np.ndarray, n: int, rng: np.random.Generator) -> list[frozenset]:
    """Independent failures per branch, one set per sample."""
    hits = rng.random((n, len(prob))) < prob
    return [frozenset(np.flatnonzero(h).tolist()) for h in hits]
