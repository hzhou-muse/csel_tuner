"""The circ_graph param rule: chunk_size and q_len of MPICH's circ_graph collectives.

The four circ_graph collectives plus allreduce share one schedule: n blocks of chunk_size bytes in
n - 1 + q rounds, q = ceil(log2 p). bcast, ibcast and allgather are bcast-like, reduce and allreduce
reduce-like:

    chunk_size = 0 (one block)                    if P2 and p <= 2 (bcast-like: one round)
               = max(F, pow2(m / D))              if m <= M0
               = clamp(pow2(sqrt(A m)), Fb, C)    otherwise
    q_len      = Qb if m >= Mq else Q

pow2 rounds to the nearest power of two in log2, halves up. A chunk >= m and chunk 0 are the same
schedule (one block). MPICH clamps a chunk to the pool cell (MPIR_CVAR_CIRC_GRAPH_CHUNK_SIZE) when it
packs (device buffers, noncontiguous types), so there C above the cell acts as the cell.

Calibration (calibrate, fit): rows of chunk sizes at q_len Q on a few layouts, a grid search of F,
A, Fb and C over powers of two for the least p98 regret of the rows (then the mean log regret),
then a q_len sweep at the fitted chunk at m >= Mq for Qb. See docs/design.md, section 14.

The interface of a rule module (csel_tuner/params.py loads it as param_<rule>): VERSION, PARAMS,
CVAR_PARAMS, DEFAULTS, consts_for, theta, describe, check; for bin/algoparam FAMILY, add_args,
calibrate and fit.
"""
import math
import sys

import numpy as np

from .params import K, M, parse_size, pow2, score, size_str, stats

# bump whenever theta() gives other values for the same constants (tests/test_param_circ_graph.py
# pins them): maps of another version are refused, they were measured with other parameters
VERSION = 1
PARAMS = ["chunk_size", "q_len"]
CVAR_PARAMS = ["CIRC_GRAPH_CHUNK_SIZE", "CIRC_GRAPH_Q_LEN"]
# the constants a calibration fits, then the fixed structure; P2 (one block at p <= 2) is per family
DEFAULTS = {"F": 16 * K, "A": 16 * K, "Fb": 64 * K, "C": 512 * K,
            "M0": 64 * K, "D": 8, "Q": 8, "Qb": 16, "Mq": 16 * M}
FITTED = ["F", "A", "Fb", "C"]
# the fit's search grid (powers of two), Fb <= C: a calibration has about ten cells, so a wider grid
# overfits them
GRID = {"F": [1 << k for k in range(11, 16)], "A": [1 << k for k in range(10, 17)],
        "Fb": [1 << k for k in range(14, 18)], "C": [1 << k for k in range(17, 22)]}
# the collectives of a family share the rule's structure and, by default, its constants
FAMILY = {"bcast": "bcast", "ibcast": "bcast", "allgather": "bcast", "reduce": "reduce", "allreduce": "reduce"}
DEFAULT_CHUNK = 128 * K         # MPICH's default, the baseline of the fit report


def consts_for(op, **consts):
    """The default constants of op's family, with consts over them."""
    if op not in FAMILY:
        raise ValueError("%s: no circ_graph rule for this collective (%s)" % (op, ", ".join(FAMILY)))
    return dict(DEFAULTS, P2=int(FAMILY[op] == "bcast"), **consts)


def chunk(c, p, m):
    """The rule's chunk_size (vectorized over p, m)."""
    p, m = np.asarray(p, float), np.asarray(m, float)
    small = np.maximum(float(c["F"]), pow2(m / c["D"]))
    large = np.clip(pow2(np.sqrt(float(c["A"]) * m)), float(c["Fb"]), float(c["C"]))
    ch = np.where(m <= c["M0"], small, large)
    if c.get("P2"):
        ch = np.where(p <= 2, 0.0, ch)
    return ch


def qlen(c, m):
    return np.where(np.asarray(m, float) >= c["Mq"], c["Qb"], c["Q"])


def theta(c, p, m):
    """{param: int array} over the points (p, m)."""
    return {"chunk_size": chunk(c, p, m).astype(np.int64),
            "q_len": np.broadcast_to(qlen(c, m), np.shape(m)).astype(np.int64)}


def describe(c):
    p2 = "p <= 2: 0 (one block); " if c.get("P2") else ""
    return (f"chunk_size = {p2}m <= {size_str(c['M0'])}: max({size_str(c['F'])}, m/{c['D']}); "
            f"else clamp(pow2(sqrt({size_str(c['A'])} m)), {size_str(c['Fb'])}, {size_str(c['C'])}); "
            f"q_len = m >= {size_str(c['Mq'])}: {c['Qb']}, else {c['Q']}")


def check(c, cvar_values, device):
    """A warning about the server's settings, or None: on device buffers MPICH clamps a packed chunk
    to the pool cell, so a map whose cap is above the cell measures the cell instead."""
    cell = (cvar_values or {}).get("MPIR_CVAR_CIRC_GRAPH_CHUNK_SIZE")
    if device and cell and c["C"] > cell:
        return (f"the map's chunk cap C = {size_str(c['C'])} is above the pool cell "
                f"MPIR_CVAR_CIRC_GRAPH_CHUNK_SIZE = {size_str(cell)}: device chunks above the cell run as the "
                "cell (raise the cell in the launch environment to measure them)")
    return None


# ---- calibration ------------------------------------------------------------------------------

def add_args(g):
    """bin/algoparam's options of this rule."""
    g.add_argument("--rows", help="the chunk rows: LAYOUT:SIZES;... (default "
                   "1xP:16K,64K,4M,64M;2x1:16K,64K,4M,64M;NxP:256K,512K,1M; N, P: the allocation)")
    g.add_argument("--qlen", default="2,4,8,16,32", help="q_len values to sweep at the rule's chunk ('' for none)")
    g.add_argument("--qlen-at", help="where to sweep q_len, LAYOUT:SIZES;... (default NxP:16M,64M)")
    g.add_argument("--pct", type=float, default=98, help="the fit minimizes this percentile of regret")


def parse_rows(text, nodes, ppn):
    """"1xP:16K,64K;NxP:1M" -> [((n, q), [sizes])]; N and P stand for the allocation."""
    out = []
    for part in filter(None, (x.strip() for x in text.split(";"))):
        lay, _, sizes = part.partition(":")
        n, _, q = lay.strip().upper().partition("X")
        n = nodes if n == "N" else int(n)
        q = ppn if q == "P" else int(q)
        out.append(((n, q), [parse_size(s) for s in sizes.split(",") if s.strip()]))
    return out


def default_rows(nodes, ppn):
    rows = "1xP:16K,64K,4M,64M"
    if nodes > 1:
        rows += ";2x1:16K,64K,4M,64M"
        if nodes > 2 or ppn > 1:
            rows += ";NxP:256K,512K,1M"
    return rows


def chunk_row(m, lo_small=1 * K, lo_large=4 * K, small=64 * K):
    """The chunks of a calibration row at message size m: every power of two from max(lo_small,
    m/16) (m <= small) or lo_large (larger m) up to m (one block)."""
    lo = max(lo_small, m // 16) if m <= small else lo_large
    out, c = [], 1
    while c <= m:
        if c >= lo:
            out.append(c)
        c *= 2
    if not out or out[-1] != m:
        out.append(m)
    return out


def calibrate(ctx, op, fixed, args):
    """Measure the calibration through ctx: ctx.nodes, ctx.ppn, ctx.device, ctx.cvar_values, and
    ctx.run(layout, m, [theta, ...]) (False if the point does not fit the allocation), whose
    measurements ctx.measurements collects. The rows first, then the q_len sweep at the chunk of a
    fit of the rows. Returns the settings to record in the job."""
    c0 = consts_for(op, **fixed)
    cell = (ctx.cvar_values or {}).get("MPIR_CVAR_CIRC_GRAPH_CHUNK_SIZE")
    if cell:
        print(f"pool cell {size_str(cell)}" + (" (packed chunks above it run as the cell)" if ctx.device else ""),
              file=sys.stderr)
    rows = args.rows or default_rows(ctx.nodes, ctx.ppn)
    for lay, sizes in parse_rows(rows, ctx.nodes, ctx.ppn):
        for m in sizes:
            chunks = chunk_row(m)
            if ctx.device and cell:
                # packed chunks above the pool cell run as the cell: measure each distinct schedule once
                chunks = sorted({min(c, cell) if c < m else c for c in chunks})
            ctx.run(lay, m, [{"chunk_size": c, "q_len": c0["Q"]} for c in chunks])
    qlens = [int(x) for x in args.qlen.split(",")] if args.qlen else []
    qat = args.qlen_at or ("NxP:16M,64M" if ctx.nodes > 1 else "1xP:16M,64M")
    if qlens:
        consts, _, _ = fit(op, ctx.measurements, {k: v for k, v in fixed.items() if k != "Qb"}, args)
        for lay, sizes in parse_rows(qat, ctx.nodes, ctx.ppn):
            for m in sizes:
                c = int(chunk(consts, lay[0] * lay[1], m))
                ctx.run(lay, m, [{"chunk_size": c, "q_len": q} for q in qlens])
    return {"rows": rows, "qlen": args.qlen, "qlen_at": qat if qlens else None, "pool_cell": cell}


def cells_of(measurements, Q):
    """Calibration cells from measurements: the chunk rows ({chunk: median time} at q_len Q, cells of
    at least two chunks) and the q_len cells ({q_len: time} at one chunk, at least two values)."""
    rows, qs = {}, {}
    for r in measurements:
        th = r.get("params") or {}
        if "chunk_size" not in th:
            continue
        t = float(np.median(r["t_us"]))
        key = (r["N"], r["ppn"], r["m"])
        if th.get("q_len") == Q:
            d = rows.setdefault(key, {})
            d[th["chunk_size"]] = min(d.get(th["chunk_size"], t), t)
        d = qs.setdefault(key + (th["chunk_size"],), {})
        d[th.get("q_len")] = min(d.get(th.get("q_len"), t), t)
    cells = [{"N": n, "ppn": q, "m": m, "t": t} for (n, q, m), t in sorted(rows.items()) if len(t) > 1]
    qcells = [{"N": n, "ppn": q, "m": m, "t": t} for (n, q, m, c), t in sorted(qs.items()) if len(t) > 1]
    return cells, qcells


class Cells:
    """Calibration cells as arrays: one cell per (layout, m), its times per measured chunk."""

    J = 41

    def __init__(self, cells):
        """cells: [{"N", "ppn", "m", "t": {chunk: time}}] (chunk 0 or >= m: one block)."""
        self.cells = [c for c in cells if c["t"]]
        self.p = np.array([c["N"] * c["ppn"] for c in self.cells], float)
        self.m = np.array([c["m"] for c in self.cells], float)
        self.ch = [np.array(sorted({min(int(k) if int(k) > 0 else c["m"], c["m"]) for k in c["t"]}), float)
                   for c in self.cells]
        self.T = []
        for c, chs in zip(self.cells, self.ch):
            by = {}
            for k, t in c["t"].items():
                e = min(int(k) if int(k) > 0 else c["m"], c["m"])
                by[e] = min(by.get(e, math.inf), float(t))
            self.T.append(np.array([by[int(x)] for x in chs]))
        self.best = np.array([t.min() for t in self.T])
        # R[i, j]: regret of chunk 2^j at cell i (the measured chunk nearest in log2 to min(2^j, m));
        # one[i]: of one block
        self.R = np.empty((len(self.cells), self.J))
        for i, (chs, T) in enumerate(zip(self.ch, self.T)):
            e = np.minimum(2.0 ** np.arange(self.J), self.m[i])
            near = np.argmin(np.abs(np.log2(chs)[None, :] - np.log2(e)[:, None]), axis=1)
            self.R[i] = T[near] / self.best[i]
        self.one = np.array([T[-1] / b for T, b in zip(self.T, self.best)])

    def __len__(self):
        return len(self.cells)

    def regret(self, chunks):
        """Regret per cell of a chunk per cell: the time at the measured chunk nearest in log2
        (after clamping to m; 0 is one block) over the cell's best."""
        c = np.asarray(chunks, float)
        whole = (c <= 0) | (c >= self.m)
        j = np.clip(np.round(np.log2(np.maximum(c, 1.0))).astype(int), 0, self.J - 1)
        return np.where(whole, self.one, self.R[np.arange(len(self)), j])


def fit_chunk(cells, op, fixed=None, pct=98):
    """(constants, report) of the chunk rule fitted to calibration cells by grid search: the
    fitted constants (those not in `fixed`) over their power-of-two grid, minimizing the pct
    percentile of regret, then the mean log regret. fixed: {constant: value}."""
    fixed = dict(fixed or {})
    base = consts_for(op, **fixed)
    t = cells if isinstance(cells, Cells) else Cells(cells)
    if not len(t):
        raise ValueError("no calibration cells")
    grid = {k: ([fixed[k]] if k in fixed else GRID[k]) for k in FITTED}
    best = None
    for F in grid["F"]:
        for A in grid["A"]:
            for Fb in grid["Fb"]:
                for C in grid["C"]:
                    if Fb > C:
                        continue
                    c = dict(base, F=F, A=A, Fb=Fb, C=C)
                    r = t.regret(chunk(c, t.p, t.m))
                    s = score(r, pct)
                    if best is None or s < best[0]:
                        best = (s, c, r)
    _, consts, r = best
    dflt = t.regret(np.full(len(t), float(DEFAULT_CHUNK)))
    report = {"cells": len(t), "pct": pct, "fixed": sorted(fixed), "regret": stats(r), "default_regret": stats(dflt),
              "worst": sorted(([int(t.p[i]), t.cells[i]["N"], t.cells[i]["ppn"], int(t.m[i]), round(float(r[i]), 3)]
                               for i in range(len(t))), key=lambda x: -x[4])[:5]}
    return consts, report


def fit_qlen(qcells, consts, tol=0.03):
    """The large-message q_len Qb from a q_len sweep at m >= Mq: the value with the least geomean
    regret; the small-message Q is kept unless Qb gains less than tol over it (then Qb = Q).
    qcells: [{"N", "ppn", "m", "t": {q_len: time}}]. Returns (Qb, report)."""
    cs = [c for c in qcells if c["m"] >= consts["Mq"] and c["t"]]
    if not cs:
        return consts["Qb"], {"cells": 0, "note": "no q_len sweep at m >= Mq: Qb kept"}
    qs = sorted(set.intersection(*[set(int(k) for k in c["t"]) for c in cs]))
    if not qs:
        return consts["Qb"], {"cells": len(cs), "note": "no q_len measured at every cell: Qb kept"}
    reg = {q: np.array([c["t"][q] / min(c["t"][k] for k in qs) for c in cs]) for q in qs}
    gm = {q: float(np.exp(np.mean(np.log(r)))) for q, r in reg.items()}
    qb = min(qs, key=lambda q: (round(gm[q], 3), q))
    if consts["Q"] in gm and gm[consts["Q"]] <= gm[qb] * (1 + tol):
        qb = consts["Q"]
    return qb, {"cells": len(cs), "geomean": {str(q): round(v, 3) for q, v in gm.items()}, "Qb": qb}


def fit(op, measurements, fixed, args):
    """(constants, fit report, report lines) from the calibration's measurements."""
    pct = getattr(args, "pct", 98)
    base = consts_for(op, **fixed)
    cells, qcells = cells_of(measurements, base["Q"])
    if not cells:
        raise ValueError("no chunk rows to fit (cells with at least two chunks at q_len %d)" % base["Q"])
    consts, rep = fit_chunk(cells, op, fixed, pct=pct)
    qb, qrep = fit_qlen(qcells, consts) if "Qb" not in fixed else (fixed["Qb"], {"note": "Qb fixed"})
    consts["Qb"] = qb
    report = {"chunk": rep, "q_len": qrep, "layouts": sorted({"%dx%d" % (c["N"], c["ppn"]) for c in cells})}
    t = Cells(cells)
    r = t.regret(chunk(consts, t.p, t.m))
    d = t.regret(np.full(len(t), float(DEFAULT_CHUNK)))
    fmt = lambda s: f"max {s['max']:.3f} p90 {s['p90']:.3f} median {s['median']:.3f} geomean {s['geomean']:.3f}"
    lines = [f"  fitted {', '.join(k for k in FITTED if k not in fixed)} on {len(t)} cells "
             f"(fixed: {', '.join(f'{k}={size_str(v)}' for k, v in sorted(fixed.items())) or '-'})",
             f"  regret on the rows: rule {fmt(stats(r))}; default {size_str(DEFAULT_CHUNK)}: {fmt(stats(d))}",
             "  %-8s %8s %8s %8s %7s %7s" % ("layout", "m", "best", "rule", "rule", size_str(DEFAULT_CHUNK))]
    for i, c in enumerate(t.cells):
        best = int(t.ch[i][int(np.argmin(t.T[i]))])
        ch = int(chunk(consts, t.p[i], t.m[i]))
        lines.append("  %-8s %8s %8s %8s %7.3f %7.3f" % ("%dx%d" % (c["N"], c["ppn"]), size_str(c["m"]),
                                                       size_str(best), size_str(ch) if ch else "whole", r[i], d[i]))
    if qrep.get("geomean"):
        lines.append(f"  q_len at m >= {size_str(consts['Mq'])} ({qrep['cells']} cells), geomean regret "
                     + ", ".join(f"{q}: {v}" for q, v in qrep["geomean"].items()) + f"; Qb = {qb}")
    else:
        lines.append(f"  q_len: {qrep.get('note')}")
    if stats(d)["max"] <= 1.03:
        lines.append("  note: the default is within 3% of the best everywhere on the rows: a map gains little")
    if score(d, pct) < score(r, pct):
        lines.append(f"  note: the default {size_str(DEFAULT_CHUNK)} scores better than the rule on the rows "
                     f"(p{pct:g} regret, then mean log regret): check the rows' noise, or fix constants with --set")
    return consts, report, lines
