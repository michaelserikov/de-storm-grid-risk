"""Linear programs through HiGHS. Uses the highspy package when installed (on some machines about 100 times faster
than the HiGHS build inside SciPy for these grid LPs), otherwise scipy.optimize.linprog. Same solver, same optimum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:
    import highspy
except ImportError:          # pragma: no cover
    highspy = None


@dataclass
class Result:
    x: np.ndarray
    fun: float
    ineq_marginals: np.ndarray       # d objective / d b_ub, as scipy's res.ineqlin.marginals


def solve(c, A_ub, b_ub, A_eq, b_eq, bounds) -> Result:
    c = np.asarray(c, dtype=float)
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], dtype=float)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], dtype=float)
    if highspy is None:
        from scipy.optimize import linprog
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lo, hi)), method="highs")
        if not r.success:
            raise RuntimeError(r.message)
        return Result(r.x, float(r.fun), np.asarray(r.ineqlin.marginals))
    m_ub = A_ub.shape[0]
    A = sp.vstack([A_ub, A_eq]).tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = len(c), A.shape[0]
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lo, hi
    lp.row_lower_ = np.r_[np.full(m_ub, -np.inf), np.asarray(b_eq, dtype=float)]
    lp.row_upper_ = np.r_[np.asarray(b_ub, dtype=float), np.asarray(b_eq, dtype=float)]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    h.passModel(lp)
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(h.modelStatusToString(status))
    sol = h.getSolution()
    return Result(np.asarray(sol.col_value), float(h.getInfo().objective_function_value), np.asarray(sol.row_dual)[:m_ub])
