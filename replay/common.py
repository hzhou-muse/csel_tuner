"""Replay data and scoring shared by replay_v1/v2/v3.py.

Replay runs the builders against recorded measurements instead of a live
cluster. The data is a dense grid of one collective (bcast) recorded with
the batch-mode tools (groundtruth/) and converted by make_data.py; see
README.md in this directory. Two files in replay/data:

- sessions.csv.gz: per (alg, N, ppn, m, session) the median per-call time
  of the request's batches (t, us) and the request's wall time (wall_us, the
  cost of measuring it). A replayed measurement draws one session.
- gt.csv.gz: the ground truth, per (alg, N, ppn, p, m) the median over all
  batches of all sessions (t), with n_sess, noise and wall_us.

The algorithms are those in the data; "auto" (MPICH's own selection, if
recorded) is used only as a reference. The allocation's node count, which
a measurement's cost is multiplied by, is the largest N in the data.
"""
import os
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
GT_FILE = os.path.join(DATA, "gt.csv.gz")
SESSIONS_FILE = os.path.join(DATA, "sessions.csv.gz")
SHORT = {"binomial": "binom", "scatter_recursive_doubling_allgather": "scat_rd",
         "scatter_ring_allgather": "scat_ring", "circ_graph": "circ", "tree": "tree",
         "pipelined_tree": "ptree", "auto": "auto"}


class NoData(RuntimeError):
    pass


def have_data():
    return os.path.exists(GT_FILE) and os.path.exists(SESSIONS_FILE)


def _need():
    if not have_data():
        raise NoData("no replay data: see replay/README.md for how to record it and run make_data.py "
                     f"(expected {GT_FILE} and {SESSIONS_FILE})")


def _header():
    """(algorithms, nodes) from the ground truth, or defaults without data."""
    if not have_data():
        return [], None
    g = pd.read_csv(GT_FILE, usecols=["alg", "N"])
    algs = sorted(g.alg.unique(), key=lambda a: (list(SHORT).index(a) if a in SHORT else len(SHORT), a))
    return algs, int(g.N.max())


ALGS, NODES = _header()
FIXED = [a for a in ALGS if a != "auto"]
for _a in ALGS:
    SHORT.setdefault(_a, _a)


def replay_one(sessions, seed, alg, n, ppn, m, k, noise=0.01):
    """The k-th replayed measurement of (alg, n, ppn, m): a recorded session
    and multiplicative noise exp(N(0, noise)), drawn from a generator seeded
    by (seed, the request, k). It does not depend on what else was measured
    or in which order, so a run cut and resumed replays the same values as
    an uninterrupted one. Returns (t_us, wall_us)."""
    rng = np.random.default_rng([seed, zlib.crc32(("%s %d %d %d" % (alg, n, ppn, m)).encode()), k])
    rows = sessions[(alg, n, ppn, m)]
    t, wall = rows[rng.integers(len(rows))]
    return t * np.exp(rng.normal(0, noise)), wall


def results_path(name):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def load_gt():
    """Ground truth: one row per (alg, N, ppn, m) with t (us)."""
    _need()
    return pd.read_csv(GT_FILE)


def load_sessions():
    """{(alg, N, ppn, m): array of (t, wall_us) rows, one per session}."""
    _need()
    s = pd.read_csv(SESSIONS_FILE)
    return {k: v[["t", "wall_us"]].to_numpy() for k, v in s.groupby(["alg", "N", "ppn", "m"])}


def table(gt):
    """Wide table: index (N, ppn, p, m), one column of T per algorithm."""
    return gt.pivot_table(index=["N", "ppn", "p", "m"], columns="alg", values="t")


def regret(w, sel):
    """Per-point regret of a selection (algorithm names in w's row order)."""
    t = w[FIXED].to_numpy()[np.arange(len(w)), [FIXED.index(a) for a in sel]]
    return pd.Series(np.asarray(t) / w[FIXED].min(axis=1).to_numpy(), index=w.index)


def score(w, r, name):
    best = w[FIXED].min(axis=1)
    return {"profile": name, "geomean": float(np.exp(np.log(r).mean())), "p95": float(r.quantile(.95)),
            "max": float(r.max()), "> 1.1": float((r > 1.1).mean()),
            "total / oracle": float((r * best).sum() / best.sum())}


def sel_oracle(w, keys):
    """Best algorithm per group of `keys` (index level names): the one with the
    smallest summed log regret over the group."""
    lr = np.log(w[FIXED].div(w[FIXED].min(axis=1), axis=0))
    pick = lr.groupby(level=keys).sum().idxmin(axis=1)
    return pick.reindex(pd.MultiIndex.from_frame(w.index.to_frame()[keys])).to_numpy()


def profiles(w):
    """name -> per-point regret of the reference selections (no sampling):
    MPICH's default auto (if recorded), the best single algorithm, the best
    selection keyed on (comm_size, message size), and the oracle."""
    lr = np.log(w[FIXED].div(w[FIXED].min(axis=1), axis=0))
    single = lr.sum().idxmin()
    out = {}
    if "auto" in w:
        out["default auto"] = w["auto"] / w[FIXED].min(axis=1)
    out[f"single ({SHORT[single]})"] = regret(w, [single] * len(w))
    out["oracle(p, m)"] = regret(w, sel_oracle(w, ["p", "m"]))
    out["oracle(N, ppn, m)"] = regret(w, w[FIXED].idxmin(axis=1).to_numpy())
    return out
