"""Small, dependency-light statistics used by the experiments (paired by game seed wherever possible)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def bootstrap_mean_ci(x: np.ndarray, *, seed: int = 0, n_boot: int = 10_000, level: float = 0.95) -> list[float]:
    """Percentile bootstrap CI of the mean (seeded → reproducible)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 100 - 50 * (1 - level)])
    return [float(lo), float(hi)]


def paired_comparison(a: np.ndarray, b: np.ndarray, *, seed: int = 0) -> dict:
    """a vs b on the same game seeds: mean difference with bootstrap CI, Wilcoxon signed-rank, rank-biserial r."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equally shaped samples")
    d = a - b
    out = {"mean_diff": float(d.mean()), "mean_diff_ci95": bootstrap_mean_ci(d, seed=seed)}
    nz = d[d != 0]
    if nz.size == 0:
        return out | {"wilcoxon_p": 1.0, "rank_biserial": 0.0, "n_nonzero": 0}
    res = sps.wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="auto")
    ranks = sps.rankdata(np.abs(nz))
    r_plus, r_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return out | {
        "wilcoxon_p": float(res.pvalue),
        "rank_biserial": float((r_plus - r_minus) / ranks.sum()),  # matched-pairs rank-biserial correlation
        "n_nonzero": int(nz.size),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def survival_curve(frames: np.ndarray, crashed: np.ndarray, horizon: int, n_points: int = 101) -> dict:
    """Kaplan–Meier survival over game frames (censoring = not crashed) + restricted mean survival time."""
    frames, crashed = np.asarray(frames, dtype=np.int64), np.asarray(crashed, dtype=bool)
    order = np.argsort(frames, kind="stable")
    t, e = frames[order], crashed[order]
    at_risk, s, times, surv = len(t), 1.0, [0], [1.0]
    i = 0
    while i < len(t):
        j = i
        deaths = 0
        while j < len(t) and t[j] == t[i]:
            deaths += int(e[j])
            j += 1
        if deaths:
            s *= 1.0 - deaths / at_risk
            times.append(int(t[i]))
            surv.append(s)
        at_risk -= j - i
        i = j
    times_a, surv_a = np.asarray(times), np.asarray(surv)
    grid = np.linspace(0, horizon, n_points)
    on_grid = surv_a[np.searchsorted(times_a, grid, side="right") - 1]
    edges = np.append(times_a, horizon).clip(max=horizon)
    rmst = float(np.sum(surv_a * np.diff(edges)))
    return {"frame": grid.round(1).tolist(), "survival": on_grid.round(4).tolist(), "rmst_frames": rmst}
