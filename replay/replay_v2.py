"""Replay builder v2 (csel_tuner/multi.py) on recorded data, scored by selection regret.

    python3 replay/replay_v2.py [--seeds 3] [--budget 4] [--tag TAG] [--v1]
        -> replay/results/v2_runs[_TAG].csv, replay/results/v2_curve[_TAG].csv
    python3 replay/replay_v2.py --report TAG  -> replay/results/v2.md

Needs replay/data (README.md). A run builds all recorded algorithms
together from a blank start (the recorded layouts allowed, the recorded
sizes as the grid). A measurement is one session's median (random session)
times exp(N(0, 1%)), costing its wall time times the allocation's nodes, as
in replay_v1.py.

Score: the model's selection (argmin of the predicted T) at every recorded
(N, ppn, m) point, against the ground truth: regret = T(selected) /
T(fastest), geomean / p95 / max and total time / oracle (common.score),
also split into measured and unmeasured layouts. The online estimate is
the builder's own (random checks). The curve file scores the model after
every 5th added layout.

--v1: the baseline of independent v1 builders (single.Builder, default
settings, one per algorithm, the same seed); its selection is the argmin of
their models and its cost the sum.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SHORT, load_gt, results_path  # noqa: E402
from csel_tuner.single import Builder, LayoutModel  # noqa: E402
from csel_tuner.multi import Builder2, OptimumModel  # noqa: E402
from common import FIXED, profiles, score, table  # noqa: E402
from common import NODES, load_sessions, replay_one  # noqa: E402

NOISE = 0.01


class Replay2:
    """measure(n, ppn, alg, [m, ...]) from the recorded sessions."""
    def __init__(self, sessions, seed):
        self.g, self.seed = sessions, seed
        self.count = {}

    def __call__(self, n, ppn, alg, ms):
        out = []
        for m in ms:
            key = (alg, n, ppn, m)
            k = self.count.get(key, 0)
            self.count[key] = k + 1
            out.append(self.one(alg, n, ppn, m, k))
        return out

    def one(self, alg, n, ppn, m, k):
        """The k-th measurement of (alg, n, ppn, m), whatever came before."""
        return replay_one(self.g, self.seed, alg, n, ppn, m, k, NOISE)


def selection(predict, layouts, sizes):
    """Series of selected algorithm names on (N, ppn, m); predict(n, ppn) -> log2 T rows (FIXED order)."""
    idx, sel = [], []
    for n, q in layouts:
        w = np.argmin(predict(n, q), axis=0)
        idx += [(n, q, int(m)) for m in sizes]
        sel += [FIXED[k] for k in w]
    return pd.Series(sel, index=pd.MultiIndex.from_tuples(idx, names=["N", "ppn", "m"]))


def evaluate(sel, w, measured):
    """Regret scores of a selection: all points, measured and unmeasured layouts."""
    wn = w.droplevel("p")
    s = sel.reindex(wn.index)
    t = wn[FIXED].to_numpy()[np.arange(len(wn)), [FIXED.index(a) for a in s]]
    r = pd.Series(t / wn[FIXED].min(axis=1).to_numpy(), index=wn.index)
    out = {}
    lay = list(zip(wn.index.get_level_values("N"), wn.index.get_level_values("ppn")))
    is_m = np.array([l in measured for l in lay])
    for name, mask in (("all", np.ones(len(r), bool)), ("meas", is_m), ("unmeas", ~is_m)):
        if mask.any():
            sc = score(wn[mask], r[mask], name)
            out.update({f"{name} {k}": v for k, v in sc.items() if k != "profile"})
    return out


def run_v2(sessions, w, sizes, layouts, seed, budget, kw):
    b = Builder2(Replay2(sessions, seed), FIXED, layouts, sizes, n_max=max(n for n, _ in layouts),
                 ppn_max=max(p for _, p in layouts), seed=seed, **kw)
    b.run(budget_node_h=budget, nodes_billed=NODES)
    m = b.model()
    row = {"builder": "v2", "seed": seed, "layouts": len(b.samplers), "meas": b.n_meas, "node_h": b.node_h,
           "stop": b.stop, "sparse": ",".join(SHORT[a] for a in sorted(b.sparse))}
    row.update({f"meas {SHORT[a]}": b.n_meas_alg[a] for a in FIXED})
    row.update(evaluate(selection(m.predict, layouts, sizes), w, set(b.samplers)))
    est = b.estimate()
    row["online gm"], row["online p95"] = est if est else (np.nan, np.nan)
    curve = []
    for j in range(4, len(b.history), 5):
        lays = [h[0] for h in b.history[:j + 1]]
        sub = {a: {l: b.curves[a][l] for l in lays if l in b.curves[a]} for a in FIXED}
        mm = OptimumModel(FIXED, sub, b.penalty)
        sc = evaluate(selection(mm.predict, layouts, sizes), w, set(lays))
        curve.append({"builder": "v2", "seed": seed, "layouts": len(lays),
                      "node_h": b.history[j][1] * 1e-6 * NODES / 3600, **sc})
    return row, curve


def run_v1(sessions, w, sizes, layouts, seed, budget):
    from replay_v1 import Replay
    models, node_h, meas, lays_all = {}, 0.0, 0, set()
    for a in FIXED:
        b = Builder(Replay(sessions, a, seed), layouts, sizes, n_max=max(n for n, _ in layouts),
                    ppn_max=max(p for _, p in layouts), seed=seed)
        b.run(budget_node_h=budget, nodes_billed=NODES)
        models[a] = b.model()
        node_h += b.node_h
        meas += b.n_meas
        lays_all |= set(b.curves)
    pred = lambda n, q: np.array([models[a].predict(n, q) for a in FIXED])
    row = {"builder": "v1 x 6", "seed": seed, "layouts": len(lays_all), "meas": meas, "node_h": node_h}
    row.update(evaluate(selection(pred, layouts, sizes), w, lays_all))
    return row


def report(tag):
    df = pd.read_csv(results_path(f"v2_runs_{tag}.csv"))
    gt = load_gt()
    wp = table(gt)
    ref = pd.DataFrame([score(wp, r, n) for n, r in profiles(wp).items() if "p, m" not in n])
    out = ["# Collective model builder v2: replay on recorded data", "",
           "Generated by `replay/replay_v2.py --report " + tag + "`. Regret = T(selected) / "
           "T(fastest) over all recorded (N, ppn, m) points; mean over seeds.", "",
           "## References (no sampling)", "",
           "| profile | geomean | p95 | max | total / oracle |", "|---|---|---|---|---|"]
    for _, r in ref.iterrows():
        out.append(f"| {r['profile']} | {r['geomean']:.3f} | {r['p95']:.2f} | {r['max']:.2f} | "
                   f"{r['total / oracle']:.3f} |")
    out += ["", "## Builders", "",
            "| builder | layouts | measurements | node-h | stop | all geomean / p95 / max | unmeasured geomean / p95 "
            "| total / oracle | online estimate geomean / p95 |", "|---|---|---|---|---|---|---|---|---|"]
    for bname, g in df.groupby("builder", sort=False):
        r = g.mean(numeric_only=True)
        stop = ",".join(sorted(set(g.stop.dropna()))) if "stop" in g else ""
        out.append(f"| {bname} | {r.layouts:.0f} | {r.meas:.0f} | {r.node_h:.2f} | {stop} | "
                   f"{r['all geomean']:.3f} / {r['all p95']:.2f} / {r['all max']:.2f} | "
                   f"{r.get('unmeas geomean', np.nan):.3f} / {r.get('unmeas p95', np.nan):.2f} | "
                   f"{r['all total / oracle']:.3f} | "
                   f"{r.get('online gm', np.nan):.3f} / {r.get('online p95', np.nan):.2f} |")
    v2 = df[df.builder.str.startswith("v2, prune 2x")] if df.builder.str.startswith("v2, prune 2x").any() else df[df.builder == "v2"]
    if len(v2):
        cols = [f"meas {SHORT[a]}" for a in FIXED]
        out += ["", "v2 measurements per algorithm (mean): " +
                ", ".join(f"{c[5:]} {v2[c].mean():.0f}" for c in cols) +
                "; algorithms flagged sparse at the end: " + ("; ".join(sorted(set(v2["sparse"].fillna("").astype(str)) - {""})) or "none"), ""]
    cp = results_path(f"v2_curve_{tag}.csv")
    if os.path.exists(cp):
        cv = pd.read_csv(cp)
        cv["bin"] = pd.cut(cv.node_h, [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5])
        t = cv.groupby("bin", observed=True)[["all geomean", "all p95"]].mean()
        out += ["## v2 regret vs. cost (model after every 5th layout)", "",
                "| node-h | geomean | p95 |", "|---|---|---|"]
        out += [f"| {b} | {r['all geomean']:.3f} | {r['all p95']:.2f} |" for b, r in t.iterrows()]
    text = "\n".join(out) + "\n"
    open(results_path("v2.md"), "w").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=4.0, help="node-hours (v1: per algorithm)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--v1", action="store_true", help="also run the six-v1 baseline")
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--dom-margin", type=float, default=1.5)
    ap.add_argument("--prune", type=float, default=2.0, help="model-based pruning factor; 0 = off")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    gt = load_gt()
    w = table(gt)
    sessions = load_sessions()
    sizes = np.array(sorted(gt.m.unique()))
    layouts = sorted({(n, p) for n, p in zip(gt.N, gt.ppn)})
    sfx = f"_{a.tag}" if a.tag else ""
    rows, curves = [], []
    for seed in range(a.seeds):
        row, curve = run_v2(sessions, w, sizes, layouts, seed, a.budget,
                            dict(margin=a.margin, dom_margin=a.dom_margin, prune=a.prune or None))
        rows.append(row)
        curves += curve
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        if a.v1:
            rows.append(run_v1(sessions, w, sizes, layouts, seed, 2.0))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(results_path(f"v2_runs{sfx}.csv"), index=False)
        pd.DataFrame(curves).to_csv(results_path(f"v2_curve{sfx}.csv"), index=False)


if __name__ == "__main__":
    main()
