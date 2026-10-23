"""Collective model builder v1: one algorithm, T(N, ppn, m), from a blank start.

    b = Builder(measure, layouts, sizes, n_max=32, ppn_max=96)
    b.run(budget_node_h=1.0)
    b.model.predict(n, ppn) -> log2 T at every size of `sizes`

measure(n, ppn, [m, ...]) -> [(t_us, wall_us), ...] sends one request per
size as one batch, so a live driver pays its round trip once per batch: the
replay (replay/replay_v1.py) or the live server. Sizes come from a fixed
discrete grid (4 per octave here), so repeated requests hit the same sizes
and a jump is bracketed to one grid step.

Three parts:

1. GridSampler, the m axis of one layout: a coarse pass every `step` grid
   sizes, then breadth-first over intervals, one level per batch. Each
   interval's grid midpoint is predicted by PCHIP (log-log) through the
   points measured so far, then the level's midpoints are measured.
   A miss of at most tau_acc ends the interval. A miss above tau_jump is
   re-measured with its end points (all such in one batch). If it still
   misses, it is a jump and is bisected to adjacent grid sizes, all jumps in
   lockstep (each midpoint goes to the side whose trend it matches). Any other miss splits the interval while it is wider than
   w_min grid steps. A size's value is the median of its log2 measurements.
   With a prior (log2 T at every grid size, e.g. the cross-layout model's
   prediction for this layout), PCHIP interpolates the residual from the
   prior instead, so structure the prior already has costs no refinement.

2. LayoutModel, across layouts. With s(p) = 2 - p / 2^floor(log2 p) for p not
   a power of two (1 just above 2^k, 0 at 2^(k+1)) and 0 for a power of two:
       T(N, ppn, m) = S(log2 N, log2 ppn, m) * (1 + a_class(m) * s(N * ppn))
   (a penalty that is largest just above a power of two and vanishes at the
   next one, in relative form). Classes: N = 1 (shared
   memory only) and N >= 2, never mixed. Every measured layout gives S = T /
   (1 + a s), and S is interpolated linearly in (log2 N, log2 ppn) over the
   measured layouts of the class (1-D in ppn for N = 1; nearest outside the
   hull), so a measured layout is reproduced exactly. a(m) is fitted by
   leave-one-out: for each value on a grid A_GRID, predict every interior
   layout (inside the hull of the others) from the others; per size, take
   the value with the smallest sum of |log2 error|, then a running median
   over m. With no interior non-pof2 layout, a = 0.

3. Builder, the layout loop. Initial layouts: N in {2, mid, N_max} x ppn in
   {1, mid, ppn_max}, plus N = 1 at ppn {smallest, mid, max}, where mid is
   the power of two nearest the log-midpoint. Then every added layout is
   first predicted by the current model and then measured (a check):
   - risk checks: the allowed layouts between two adjacent measured ones
     along N (fixed ppn) or ppn (fixed N), the one nearest the log-midpoint
     and the power of two nearest it, scored by the gap in octaves times the
     RMS log2 difference of the two neighbours' curves; the best is taken.
     They put layouts where the model is weakest, so their errors overstate
     the typical error.
   - random checks: every `check_every`-th layout is drawn uniformly from
     the unmeasured allowed layouts. Their errors (at the sizes measured
     there) are an unbiased sample of the error at unmeasured layouts; the
     estimate is the median and p90 pooled over the last `k_est` of them.
     The median estimate is the reliable one: the large errors tend to sit
     at a few layouts that a few random draws can miss, so the p90 estimate
     is noisy.
   Stop at the target (estimate median <= tau_med and p90 <= tau_p90; with
   `confirm`, also over the window one check earlier), at a
   plateau (neither improved by the factor `plateau` over the k_est random
   checks before), at the budget, or when every allowed layout is measured.
   With use_prior, the check's prediction is also the m sampler's prior
   there (coarse grid every `step_prior` sizes).
"""
from collections import deque

import numpy as np

from .interp import linear_nd, nearest, pchip


def is_pof2(p):
    return p & (p - 1) == 0


def sigma(p):
    """Non-pof2 coordinate: 0 at 2^k, 1 just above it, 0 again at 2^(k+1)."""
    if is_pof2(p):
        return 0.0
    k = 1 << (int(p).bit_length() - 1)
    return 2.0 - p / k


A_GRID = [-0.3, -0.15, 0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def pchip_log(xs, ys, xq):
    if len(xs) == 1:
        return np.full(len(xq), ys[0])
    return pchip(xs, ys, xq)


class GridSampler:
    def __init__(self, measure, sizes, step=8, tau_acc=0.10, tau_jump=0.15, w_min=2, budget=200,
                 prior=None, init=None):
        """measure(list of m) -> list of t, one request per size, sent as one
        batch; sizes: the allowed sizes, ascending; prior: log2 T at every
        size, or None; init: grid indices of the coarse pass (default every
        `step`, both ends included)."""
        self.measure, self.sizes = measure, np.asarray(sizes)
        self.lx = np.log2(self.sizes)
        self.prior = np.zeros(len(sizes)) if prior is None else np.asarray(prior)
        self.tau_acc, self.tau_jump, self.w_min, self.budget = tau_acc, tau_jump, w_min, budget
        self.vals = {}  # grid index -> list of log2 T
        self.n = 0
        self.jumps = []
        last = len(sizes) - 1
        self.meas(sorted(set(range(0, last, step)) | {last}) if init is None else sorted(set(init)))

    def meas(self, idx):
        """Measure grid indices idx (a list) in one batch."""
        if not idx:
            return
        for i, t in zip(idx, self.measure([int(self.sizes[i]) for i in idx])):
            self.vals.setdefault(i, []).append(np.log2(t))
        self.n += len(idx)

    def y(self, i):
        return float(np.median(self.vals[i]))

    def r(self, i):
        """residual from the prior"""
        return self.y(i) - self.prior[i]

    def predict(self, i):
        k = sorted(j for j in self.vals if j != i)
        return float(pchip_log(self.lx[k], [self.r(j) for j in k], [self.lx[i]])[0]) + self.prior[i]

    def slope(self, i, side):
        k = sorted(self.vals)
        j = k.index(i) + side
        if j < 0 or j >= len(k):
            return 0.0 if side < 0 or self.prior.any() else 1.0
        return (self.r(k[j]) - self.r(i)) / (self.lx[k[j]] - self.lx[i])

    def bisect(self, brackets):
        """Narrow confirmed jumps [(a, b, c), ...] in lockstep: each step
        measures one new midpoint per unfinished bracket, as one batch."""
        st = [[a, b, c, self.slope(a, -1), self.slope(b, +1)] for a, b, c in brackets]
        while True:
            todo = []
            for s in st:
                a, b, c, sl, sr = s
                if c is None:
                    continue
                yl = self.r(a) + sl * (self.lx[c] - self.lx[a])
                yr = self.r(b) - sr * (self.lx[b] - self.lx[c])
                if abs(self.r(c) - yl) <= abs(self.r(c) - yr):
                    a = c
                else:
                    b = c
                c = (a + b) // 2 if b - a > 1 and self.n < self.budget else None
                s[:3] = a, b, c
                if c is not None:
                    todo.append(c)
            if not todo:
                break
            self.meas(sorted(set(todo)))
        for a, b, _, _, _ in st:
            self.jumps.append((int(self.sizes[a]), int(self.sizes[b]), 2 ** (self.y(b) - self.y(a))))

    def run(self, active=None):
        """Breadth-first refinement, one level per batch: predict every
        interval's midpoint, measure them all, then re-measure the suspected
        jumps (midpoint and ends) together, then bisect the confirmed ones.
        active(a, b) -> bool limits refinement to some intervals (grid
        indices a < b); default all."""
        k = sorted(self.vals)
        level = list(zip(k[:-1], k[1:]))
        while level and self.n < self.budget:
            mids = [(a, b, (a + b) // 2) for a, b in level if b - a > 1 and (active is None or active(a, b))]
            preds = {c: self.predict(c) for _, _, c in mids}
            self.meas(sorted({c for _, _, c in mids}))
            nxt, confirm = [], []
            for a, b, c in mids:
                r = abs(self.y(c) - preds[c])
                if r <= self.tau_acc:
                    continue
                if r > self.tau_jump:
                    confirm.append((a, b, c))
                elif b - a > self.w_min:
                    nxt += [(a, c), (c, b)]
            if confirm:
                self.meas([i for a, b, c in confirm for i in (c, a, b)])
                jumps = []
                for a, b, c in confirm:
                    if abs(self.y(c) - self.predict(c)) > self.tau_jump:
                        jumps.append((a, b, c))
                    elif b - a > self.w_min:
                        nxt += [(a, c), (c, b)]
                if jumps:
                    self.bisect(jumps)
            level = nxt
        return self

    def curve(self):
        """log2 T at every grid size. With a prior, the residual is held
        constant beyond the measured sizes (no extrapolation); with no
        measurement at all, the curve is the prior."""
        k = sorted(self.vals)
        if not k:
            return self.prior.copy()
        if self.prior.any():
            xq = np.clip(self.lx, self.lx[k[0]], self.lx[k[-1]])
            return pchip_log(self.lx[k], [self.r(i) for i in k], xq) + self.prior
        return pchip_log(self.lx[k], [self.r(i) for i in k], self.lx)

    def points(self):
        return [(int(self.sizes[i]), 2 ** self.y(i)) for i in sorted(self.vals)]


class LayoutModel:
    def __init__(self, curves, penalty=True, smooth=3):
        """curves: {(n, ppn): log2 T at every grid size}."""
        self.curves = curves
        self.penalty = penalty
        self.a = {}
        for intra in (True, False):
            self.a[intra] = self.fit_a(intra, smooth) if penalty else None
        nsz = len(next(iter(curves.values())))
        self.S = {}
        for lay, y in curves.items():
            a = self.a_for(lay[0] == 1, nsz)
            self.S[lay] = self.depen(y, a, lay)

    @staticmethod
    def depen(y, a, lay):
        return y - np.log2(1 + a * sigma(lay[0] * lay[1]))

    def a_for(self, intra, nsz):
        a = self.a.get(intra)
        if a is None:
            a = self.a.get(not intra)
        return np.zeros(nsz) if a is None else a

    @staticmethod
    def interp(src, vals, q, intra):
        """Linear interpolation of vals (rows per src layout) at layout q in
        (log2 N, log2 ppn); 1-D in ppn for N = 1; nearest outside the hull."""
        pts = np.log2(np.array(src, float))
        qq = np.log2(np.array(q, float))
        vals = np.asarray(vals)
        if intra or len(set(pts[:, 0])) == 1:
            o = np.argsort(pts[:, 1])
            x = pts[o, 1]
            return np.array([np.interp(qq[1], x, vals[o, j]) for j in range(vals.shape[1])])
        if len(set(pts[:, 1])) == 1:
            o = np.argsort(pts[:, 0])
            x = pts[o, 0]
            return np.array([np.interp(qq[0], x, vals[o, j]) for j in range(vals.shape[1])])
        if len(src) >= 3 and np.linalg.matrix_rank(pts - pts[0]) == 2:
            lin = linear_nd(pts, vals, qq)
            if lin is not None:
                return lin
        return nearest(pts, vals, qq)

    @staticmethod
    def inside(src, q, intra):
        if len(src) < 2:
            return False
        pts = np.log2(np.array(src, float))
        qq = np.log2(np.array(q, float))
        if intra:
            return pts[:, 1].min() <= qq[1] <= pts[:, 1].max()
        if len(src) < 3 or np.linalg.matrix_rank(pts - pts[0]) < 2:
            return False
        return linear_nd(pts, np.zeros(len(src)), qq) is not None

    def fit_a(self, intra, smooth):
        cls = [l for l in self.curves if (l[0] == 1) == intra]
        inner = [l for l in cls if self.inside([x for x in cls if x != l], l, intra)]
        if not any(sigma(l[0] * l[1]) > 0 for l in inner):
            return None
        cost = []
        for av in A_GRID:
            S = {l: self.depen(self.curves[l], av, l) for l in cls}
            c = 0
            for l in inner:
                rest = [x for x in cls if x != l]
                pred = self.interp(rest, [S[x] for x in rest], l, intra) + np.log2(1 + av * sigma(l[0] * l[1]))
                c = c + np.abs(pred - self.curves[l])
            cost.append(c)
        a = np.array(A_GRID)[np.argmin(np.array(cost), axis=0)]
        if smooth > 1:
            pad = np.pad(a, smooth // 2, mode="edge")
            a = np.array([np.median(pad[i:i + smooth]) for i in range(len(a))])
        return np.clip(a, -0.5, 10.0)

    def to_json(self, sizes):
        """Model file content (format collmodel/1): every measured layout's
        curve on the size grid, and a(m) per class."""
        r = lambda v: [round(float(x), 6) for x in v]
        return {
            "format": "collmodel/1",
            "inputs": ["N", "ppn", "m"],
            "output": "t_us",
            "model": {
                "type": "layout-grid-pof2",
                "sizes": [int(m) for m in sizes],
                "layouts": [{"N": n, "ppn": p, "log2_t": r(self.curves[(n, p)])} for n, p in sorted(self.curves)],
                "penalty": {"form": "1 + a*s", "a_intra": None if self.a[True] is None else r(self.a[True]),
                            "a_inter": None if self.a[False] is None else r(self.a[False])},
            },
        }

    @classmethod
    def from_json(cls, d):
        m = d["model"]
        self = cls.__new__(cls)
        self.curves = {(l["N"], l["ppn"]): np.array(l["log2_t"]) for l in m["layouts"]}
        self.penalty = True
        self.a = {True: m["penalty"]["a_intra"], False: m["penalty"]["a_inter"]}
        self.a = {k: None if v is None else np.array(v) for k, v in self.a.items()}
        nsz = len(m["sizes"])
        self.S = {lay: self.depen(y, self.a_for(lay[0] == 1, nsz), lay) for lay, y in self.curves.items()}
        return self

    def predict(self, n, ppn):
        intra = n == 1
        src = [l for l in self.S if (l[0] == 1) == intra] or list(self.S)
        if (n, ppn) in self.curves:
            return self.curves[(n, ppn)]
        S = self.interp(src, [self.S[l] for l in src], (n, ppn), intra)
        a = self.a_for(intra, len(S))
        return S + np.log2(1 + a * sigma(n * ppn))


def nearest_allowed(vals, target, pof2_first=True):
    """The value in vals nearest target (log2), powers of two first."""
    vals = sorted(vals)
    if pof2_first and any(is_pof2(v) for v in vals):
        vals = [v for v in vals if is_pof2(v)]
    return min(vals, key=lambda v: (abs(np.log2(v) - target), v))


def initial_layouts(allowed, n_max, ppn_max):
    """N in {2, mid, N_max} x ppn in {1, mid, ppn_max}, plus N = 1 at ppn
    {smallest, mid, max}; mid = the power of two nearest the log-midpoint."""
    ns = {n for n, _ in allowed}
    ps = {p for _, p in allowed}
    mid_n = nearest_allowed([n for n in ns if 2 < n < n_max] or [2], np.log2(n_max * 2) / 2)
    mid_p = nearest_allowed([p for p in ps if 1 < p < ppn_max] or [1], np.log2(ppn_max) / 2)
    lay = [(n, p) for n in sorted({2, mid_n, n_max}) for p in sorted({1, mid_p, ppn_max})]
    p1 = min(p for n, p in allowed if n == 1 and p > 1)     # N = 1: from its smallest layout
    lay += [(1, p) for p in sorted({p1, mid_p, ppn_max})]
    return [l for l in lay if l in allowed and l[0] * l[1] > 1]


def candidate_layouts(allowed, measured, diff):
    """(score, layout) for allowed layouts between two adjacent measured ones
    along N (fixed ppn) or ppn (fixed N): the one nearest the log-midpoint and
    the power of two nearest it, scored by the gap in octaves times diff(a, b)
    of the two neighbours; best first."""
    meas = set(measured)
    out = {}
    for axis in (0, 1):
        lines = {}
        for l in meas:
            lines.setdefault(l[1 - axis], []).append(l)
        for fixed, ls in lines.items():
            ls = sorted(ls, key=lambda l: l[axis])
            for a, b in zip(ls[:-1], ls[1:]):
                between = [l for l in allowed if l[1 - axis] == fixed and a[axis] < l[axis] < b[axis]
                           and l not in meas and l[0] * l[1] > 1]
                if not between:
                    continue
                if (a[0] == 1) != (b[0] == 1):      # never interpolate across the shm/net split
                    continue
                mid = (np.log2(a[axis]) + np.log2(b[axis])) / 2
                score = np.log2(b[axis] / a[axis]) * diff(a, b)
                for first in (True, False):
                    v = nearest_allowed([l[axis] for l in between], mid, pof2_first=first)
                    c = (v, fixed) if axis == 0 else (fixed, v)
                    out[c] = max(out.get(c, 0), score)
    return sorted(((sc, c) for c, sc in out.items()), reverse=True)


class Builder:
    def __init__(self, measure, layouts, sizes, n_max, ppn_max, step=8, step_prior=12, tau_med=0.05,
                 tau_p90=0.20, check_every=3, k_est=4, confirm=False, plateau=0.9, seed=0, penalty=True,
                 use_prior=False, sampler_kw=None, clock=None):
        """layouts: allowed (n, ppn); measure(n, ppn, [m, ...]) -> [(t_us,
        wall_us), ...], one request per size, sent as one batch."""
        self.measure = measure
        self.allowed = sorted(set(layouts))
        self.sizes = np.asarray(sizes)
        self.n_max, self.ppn_max = n_max, ppn_max
        self.step, self.penalty = step, penalty
        self.tau_med, self.tau_p90, self.check_every, self.k_est, self.plateau = \
            tau_med, tau_p90, check_every, k_est, plateau
        self.confirm = confirm
        self.rng = np.random.RandomState(seed)
        self.step_prior, self.use_prior = step_prior, use_prior
        self.clock = clock      # seconds spent so far; default: sum of the requests' wall time
        self.sampler_kw = sampler_kw or {}
        self.curves, self.samplers = {}, {}
        self.wall_us = 0.0
        self.n_meas = 0
        self.checks = []        # (layout, kind 'risk'/'random', median err, p90 err, wall_us, errors)
        self.history = []       # (layout, cumulative wall_us, n_meas): model snapshots

    def add(self, lay, prior=None):
        n, ppn = lay

        def batch(ms):
            out = self.measure(n, ppn, ms)
            self.wall_us += sum(w for _, w in out)
            self.n_meas += len(ms)
            return [t for t, _ in out]
        g = GridSampler(batch, self.sizes, step=self.step if prior is None else self.step_prior, prior=prior,
                        **self.sampler_kw).run()
        self.samplers[lay] = g
        self.curves[lay] = g.curve()
        self.history.append((lay, self.wall_us, self.n_meas))

    def model(self, penalty=None):
        return LayoutModel(self.curves, self.penalty if penalty is None else penalty)

    def check(self, lay, kind):
        """Predict lay before measuring it; measure; record the relative error
        at the sizes measured there."""
        pred = self.model().predict(*lay)
        self.add(lay, pred if self.use_prior else None)
        g = self.samplers[lay]
        idx = sorted(g.vals)
        e = 2 ** np.abs(np.array([g.y(i) for i in idx]) - pred[idx]) - 1
        self.checks.append((lay, kind, float(np.median(e)), float(np.quantile(e, 0.9)), self.wall_us, e))

    def estimate(self, last=0):
        """(median, p90) relative error pooled over random checks
        [-(last + k_est), -last), or None if there are too few."""
        r = [c for c in self.checks if c[1] == "random"]
        r = r[len(r) - last - self.k_est:len(r) - last] if len(r) >= last + self.k_est else []
        if not r:
            return None
        e = np.concatenate([c[5] for c in r])
        return float(np.median(e)), float(np.quantile(e, 0.9))

    def stop_reason(self):
        est = self.estimate()
        if est is None:
            return None
        ok = lambda e: e is not None and e[0] <= self.tau_med and e[1] <= self.tau_p90
        if ok(est) and (not self.confirm or ok(self.estimate(1))):
            return "target"
        prev = self.estimate(self.k_est)
        if prev is not None and est[0] > self.plateau * prev[0] and est[1] > self.plateau * prev[1]:
            return "plateau"
        return None

    def run(self, budget_node_h=1.0, nodes_billed=None, max_layouts=None):
        """Initial layouts, then alternate: check_every - 1 risk candidates,
        then one layout drawn uniformly from the unmeasured allowed ones (the
        random checks are the error estimate). Stop at the target, at a
        plateau, at the budget, or when nothing is left."""
        billed = nodes_billed or self.n_max
        spent = lambda: (self.clock() if self.clock else self.wall_us * 1e-6) * billed / 3600
        for lay in initial_layouts(self.allowed, self.n_max, self.ppn_max):
            self.add(lay)
        self.stop = "budget"
        while spent() < budget_node_h and (max_layouts is None or len(self.curves) < max_layouts):
            why = self.stop_reason()
            if why:
                self.stop = why
                break
            free = [l for l in self.allowed if l not in self.curves and l[0] * l[1] > 1]
            if not free:
                self.stop = "all measured"
                break
            cand = candidate_layouts(self.allowed, self.curves,
                                     lambda a, b: np.sqrt(np.mean((self.curves[a] - self.curves[b]) ** 2)))
            if (len(self.checks) + 1) % self.check_every == 0 or not cand:
                self.check(free[self.rng.randint(len(free))], "random")
            else:
                self.check(cand[0][1], "risk")
        self.node_h = spent()
        return self
