"""Coarse growth classes of the algorithms, from a model's growth anchors.

    python3 -m csel_tuner.growth MODEL.json [--unit KEY=VALUE,...] [--linear-above 0.7]

The growth anchors (collmodel --growth-anchors) measure every algorithm at a
fixed set of layouts and sizes. Per algorithm and size band (latency: m <=
LAT_MAX; bandwidth: above), a least-squares fit of

    log2 t = e * log2 p + c(m) + d * [N = 1] (+ f * [p is a power of two])

with an intercept per size, a term for single-node layouts (no network) and,
where the layouts mix powers of two and others, a pof2 term; e is the
growth exponent in the communicator size. A bootstrap over the layouts gives
its 90% interval. Over p = 2..64, t ~ log2 p has a log-log slope of about 0.5
and t ~ p - 1 about 1.2; the class is

    linear       the interval lies above --linear-above (0.7)
    logarithmic  it lies below
    unclear      it contains the threshold (or the fit has too few layouts)

The classes are coarse on purpose: anchors from a few nodes cannot tell
log p from k log_k p, or anything that shows only at large p. Each class is
compared with the pattern expected from the algorithm's name (ring,
pairwise, linear and the like are expected to be linear in the latency
band); a mismatch is reported, not overridden.
"""
import argparse
import re
import sys

import numpy as np

LAT_MAX = 65536             # latency band: sizes up to this (bytes)
LINEAR_ABOVE = 0.7
EXPECT_LINEAR = re.compile(r"(^|_)(ring|pairwise|linear|naive|scattered|inplace|sendrecv_replace)(_|$)")
MIN_LAYOUTS = 4


def anchor_records(model):
    """{alg: [(N, ppn, m, median t)]} of the model's growth-anchor records."""
    out = {}
    for r in model.measurements:
        if r.get("growth"):
            out.setdefault(r["alg"], []).append((r["N"], r["ppn"], r["m"], float(np.median(r["t_us"]))))
    return out


def fit_exponent(rows, rng=None, boots=200):
    """(e, lo, hi, layouts) for rows [(N, ppn, m, t)] of one band; None if too few layouts."""
    lays = sorted({(n, q) for n, q, _, _ in rows})
    if len(lays) < MIN_LAYOUTS or len({n * q for n, q in lays}) < 3:
        return None
    sizes = sorted({m for _, _, m, _ in rows})
    pof2 = [float((n * q) & (n * q - 1) == 0) for n, q in lays]
    use_pof2 = 0 < sum(pof2) < len(pof2)

    def design(rs):
        cols = [np.log2([n * q for n, q, _, _ in rs])]
        cols += [np.array([float(m == s) for _, _, m, _ in rs]) for s in sizes]
        single = np.array([float(n == 1) for n, _, _, _ in rs])
        if 0 < single.sum() < len(rs):
            cols.append(single)
        if use_pof2:
            p2 = np.array([float((n * q) & (n * q - 1) == 0) for n, q, _, _ in rs])
            if 0 < p2.sum() < len(rs):
                cols.append(p2)
        return np.column_stack(cols), np.log2([t for *_, t in rs])

    def solve(rs):
        A, y = design(rs)
        return float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    e = solve(rows)
    rng = rng or np.random.RandomState(0)
    by_lay = {l: [r for r in rows if (r[0], r[1]) == l] for l in lays}
    es = []
    for _ in range(boots):
        pick = [lays[i] for i in rng.randint(len(lays), size=len(lays))]
        if len({n * q for n, q in pick}) < 3:
            continue
        es.append(solve([r for l in pick for r in by_lay[l]]))
    lo, hi = (float(np.quantile(es, 0.05)), float(np.quantile(es, 0.95))) if es else (e, e)
    return e, lo, hi, len(lays)


def classify(fit, linear_above=LINEAR_ABOVE):
    if fit is None:
        return "unclear"
    _, lo, hi, _ = fit
    return "linear" if lo > linear_above else "logarithmic" if hi < linear_above else "unclear"


def expected(alg):
    return "linear" if EXPECT_LINEAR.search(alg) else "logarithmic"


def classes(model, linear_above=LINEAR_ABOVE):
    """{alg: {"lat": (class, fit), "bw": (class, fit)}} from the model's growth anchors; an
    algorithm without anchors is absent."""
    out = {}
    for alg, rows in anchor_records(model).items():
        rec = {}
        for band, keep in (("lat", lambda m: m <= LAT_MAX), ("bw", lambda m: m > LAT_MAX)):
            rs = [r for r in rows if keep(r[2])]
            fit = fit_exponent(rs) if rs else None
            rec[band] = (classify(fit, linear_above), fit)
        out[alg] = rec
    return out


def table(model, cls):
    """Markdown lines of the class table."""
    lines = [f"### {model.op} ({', '.join(f'{k}={v}' for k, v in sorted(model.tags.items()) if k != 'coll')})", "",
             "| algorithm | latency e [90%] | class | bandwidth e [90%] | class | expected (latency) |",
             "|---|---|---|---|---|---|"]
    for alg in sorted(cls):
        cells = []
        for band in ("lat", "bw"):
            c, fit = cls[alg][band]
            cells += ["—" if fit is None else f"{fit[0]:.2f} [{fit[1]:.2f}, {fit[2]:.2f}] ({fit[3]} layouts)", c]
        exp = expected(alg)
        flag = "" if cls[alg]["lat"][0] in (exp, "unclear") else " **mismatch**"
        lines.append(f"| {alg} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {exp}{flag} |")
    return lines


def main(argv=None):
    from csel_tuner.model import ModelFile, tags_match
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="+")
    ap.add_argument("--unit", help="KEY=VALUE,... tags of the units to report (default: all)")
    ap.add_argument("--linear-above", type=float, default=LINEAR_ABOVE)
    a = ap.parse_args(argv)
    sel = dict(kv.split("=", 1) for kv in a.unit.split(",")) if a.unit else None
    for path in a.model:
        for u in ModelFile.load(path).units:
            if sel and not tags_match(u.tags, sel):
                continue
            cls = classes(u, a.linear_above)
            if not cls:
                print(f"{path}: {u.op}: no growth anchors", file=sys.stderr)
                continue
            print("\n".join(table(u, cls)) + "\n")


if __name__ == "__main__":
    main()
