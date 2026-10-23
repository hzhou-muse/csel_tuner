"""Collective model builder v2: all algorithms of one collective, sampled for selection.

    b = Builder2(measure, algs, layouts, sizes, n_max=32, ppn_max=96).run(budget_node_h=3)
    b.model().select(n, ppn) -> (index into algs of the predicted best at each size, log2 T of it)

measure(n, ppn, alg, [m, ...]) -> [(t_us, wall_us), ...]: one batch of
requests for one algorithm (live: the algorithm goes in each request as
cvar.*; replay: replay/replay_v2.py). The goal is the optimum model, the
fastest algorithm and its time at every (N, ppn, m), and the per-algorithm
models only need to be accurate where an algorithm can win.

1. MultiSampler, one layout, every algorithm (built on single.GridSampler):
   - coarse pass of every algorithm (every `step` grid sizes; algorithms in
     `sparse` every `sparse_step`);
   - an algorithm is a contender on a size interval if its curve comes
     within `margin` of the envelope (the best curve) somewhere on it; the
     m sampler refines an algorithm only on such intervals. A sparse
     algorithm that turns out to be a contender here gets the full coarse
     grid first;
   - crossovers: where the envelope's winner changes between two grid
     sizes, both algorithms are measured at the midpoints of the bracket
     until it is one grid step wide (the winner is decided by measured
     values).
2. OptimumModel: one single.LayoutModel per algorithm; selection = argmin
   of the predicted log2 T.
3. Builder2, the layout loop (shared by all algorithms), as v1:
   initial grid, then checks (predict, then measure): risk candidates
   between adjacent measured layouts, scored by the gap in octaves times the
   RMS difference of the two neighbours' envelopes plus the fraction of
   sizes whose winner differs; every `check_every`-th check is a random
   unmeasured layout, measured in full (every algorithm on the coarse grid),
   and the regret of the predicted selection there, T(selected) /
   T(fastest) at the coarse sizes, is the error estimate (geomean and p95
   pooled over the last `k_est` random checks).
   - An algorithm is sparse when, at every measured layout and size, it is
     more than `dom_margin` slower than the envelope; random checks still
     measure it in full, so it comes back if it starts to compete.
   - Stop at the target (geomean regret <= tau_gm and p95 <= tau_p95), at a
     plateau (neither excess regret improved by the factor `plateau` over
     the k_est random checks before), at the budget, or when every allowed
     layout is measured.
"""
import numpy as np

from .single import GridSampler, LayoutModel, candidate_layouts, initial_layouts


class MultiSampler:
    def __init__(self, measure, algs, sizes, step=8, sparse_step=16, margin=1.3, sparse=(), full=False,
                 prior=None, prune=2.0, sampler_kw=None):
        """measure(alg, [m, ...]) -> [t, ...]. full: every algorithm on the
        coarse grid (a random check), sparse algorithms included. prior: the
        model's prediction here (log2 T, one row per algorithm) or None; then
        an algorithm's coarse pass skips the sizes where it is predicted more
        than `prune` slower than the predicted best (unless full), and its
        curve is the prior corrected by the measured residuals."""
        self.measure, self.algs, self.sizes = measure, list(algs), np.asarray(sizes)
        self.step, self.sparse_step, self.margin = step, sparse_step, margin
        self.sparse = set() if full else set(sparse)
        self.prior, self.prune, self.full = prior, prune, full
        self.kw = sampler_kw or {}
        self.crossovers = []
        self.pruned = {}            # alg -> number of coarse sizes skipped

    def curve(self, a):
        return self.g[a].curve()

    def envelope(self):
        return np.min([self.curve(a) for a in self.algs], axis=0)

    def contends(self, a, i, j, env):
        """Is algorithm a within margin of the envelope somewhere in [i, j]?"""
        return bool(np.min(self.curve(a)[i:j + 1] - env[i:j + 1]) <= np.log2(self.margin))

    def run(self):
        n = len(self.sizes)
        self.g = {}
        if self.prior is not None:
            pbest = self.prior.min(axis=0)
        for k, a in enumerate(self.algs):
            st = self.sparse_step if a in self.sparse else self.step
            init = sorted(set(range(0, n - 1, st)) | {n - 1})
            prior = None
            if self.prior is not None:
                prior = self.prior[k]
                if not self.full:
                    keep = [i for i in init if prior[i] - pbest[i] <= np.log2(self.prune)]
                    self.pruned[a] = len(init) - len(keep)
                    init = keep
            self.g[a] = GridSampler(lambda ms, a=a: self.measure(a, ms), self.sizes, step=st, prior=prior,
                                    init=init, **self.kw)
        for a in self.algs:
            env = self.envelope()
            if a in self.sparse:
                if not self.contends(a, 0, n - 1, env):
                    continue
                last = n - 1
                self.g[a].meas([i for i in sorted(set(range(0, last, self.step)) | {last})
                                if i not in self.g[a].vals])
                env = self.envelope()
            if len(self.g[a].vals) < 2:
                continue            # pruned here: nothing to refine between
            self.g[a].run(active=lambda i, j, a=a, env=env: self.contends(a, i, j, env))
        self.locate_crossovers()
        return self

    def winner(self, i):
        """Index of the best algorithm measured at grid index i (None if any is missing)."""
        if any(i not in self.g[a].vals for a in self.algs):
            return None
        return int(np.argmin([self.g[a].y(i) for a in self.algs]))

    def locate_crossovers(self, max_steps=6):
        curves = np.array([self.curve(a) for a in self.algs])
        win = np.argmin(curves, axis=0)
        for i in np.nonzero(win[1:] != win[:-1])[0]:
            pair = [self.algs[win[i]], self.algs[win[i + 1]]]
            lo = max([k for k in range(i + 1) if all(k in self.g[a].vals for a in pair)], default=None)
            hi = min([k for k in range(i + 1, len(self.sizes)) if all(k in self.g[a].vals for a in pair)],
                     default=None)
            if lo is None or hi is None:
                continue
            first = lambda k: min(pair, key=lambda a: self.g[a].y(k))
            steps = 0
            while hi - lo > 1 and steps < max_steps and first(lo) != first(hi):
                mid = (lo + hi) // 2
                for a in pair:
                    if mid not in self.g[a].vals:
                        self.g[a].meas([mid])
                if first(mid) == first(lo):
                    lo = mid
                else:
                    hi = mid
                steps += 1
            self.crossovers.append((int(self.sizes[lo]), int(self.sizes[hi]), first(lo), first(hi)))

    @property
    def n(self):
        return sum(g.n for g in self.g.values())


class OptimumModel:
    def __init__(self, algs, curves, penalty=True):
        """curves: {alg: {(n, ppn): log2 T at every grid size}}."""
        self.algs = list(algs)
        self.models = {a: LayoutModel(curves[a], penalty) for a in self.algs}

    def predict(self, n, ppn):
        """log2 T, one row per algorithm."""
        return np.array([self.models[a].predict(n, ppn) for a in self.algs])

    def select(self, n, ppn):
        p = self.predict(n, ppn)
        w = np.argmin(p, axis=0)
        return w, p[w, np.arange(p.shape[1])]


class Builder2:
    def __init__(self, measure, algs, layouts, sizes, n_max, ppn_max, step=8, sparse_step=16, margin=1.3,
                 dom_margin=1.5, prune=2.0, check_every=3, k_est=4, tau_gm=1.02, tau_p95=1.10, plateau=0.9,
                 seed=0, penalty=True, sampler_kw=None, clock=None):
        """measure(n, ppn, alg, [m, ...]) -> [(t_us, wall_us), ...]."""
        self.measure, self.algs = measure, list(algs)
        self.allowed = sorted(set(layouts))
        self.sizes = np.asarray(sizes)
        self.n_max, self.ppn_max = n_max, ppn_max
        self.step, self.sparse_step, self.margin, self.dom_margin = step, sparse_step, margin, dom_margin
        self.prune = prune          # None: no model-based pruning
        self.check_every, self.k_est, self.plateau = check_every, k_est, plateau
        self.tau_gm, self.tau_p95, self.penalty = tau_gm, tau_p95, penalty
        self.sampler_kw = sampler_kw or {}
        self.clock = clock
        self.rng = np.random.RandomState(seed)
        self.samplers = {}          # layout -> MultiSampler
        self.curves = {a: {} for a in self.algs}
        self.wall_us = 0.0
        self.n_meas = 0
        self.n_meas_alg = {a: 0 for a in self.algs}
        self.sparse = set()
        self.checks = []            # (layout, kind, geomean regret, p95 regret, wall_us, regrets)
        self.history = []           # (layout, wall_us, n_meas)
        self.on_layout = None       # f(builder) after each layout: checkpoints
        self.stop_check = None      # f() -> reason or None, before each layout: deadline, signals
        self.stop = "running"
        self.billed = n_max         # nodes the allocation bills; run(nodes_billed=) sets it
        self.node_h = 0.0

    def add(self, lay, full=False, prior=None):
        n, ppn = lay

        def m1(alg, ms):
            out = self.measure(n, ppn, alg, ms)
            self.wall_us += sum(w for _, w in out)
            self.n_meas += len(ms)
            self.n_meas_alg[alg] += len(ms)
            return [t for t, _ in out]
        s = MultiSampler(m1, self.algs, self.sizes, step=self.step, sparse_step=self.sparse_step,
                         margin=self.margin, sparse=self.sparse, full=full, prior=prior, prune=self.prune,
                         sampler_kw=self.sampler_kw).run()
        self.samplers[lay] = s
        for a in self.algs:
            if s.g[a].vals:         # an algorithm pruned everywhere here adds nothing to its model
                self.curves[a][lay] = s.curve(a)
        self.history.append((lay, self.wall_us, self.n_meas))
        self.update_sparse()
        if self.on_layout:
            self.on_layout(self)

    def update_sparse(self):
        """Algorithms more than dom_margin slower than the envelope at every measured point."""
        sp = set()
        for a in self.algs:
            lays = [l for l in self.samplers if l in self.curves[a]]
            gap = min((float(np.min(self.curves[a][l] - self.envelope(l)[0])) for l in lays), default=np.inf)
            if gap > np.log2(self.dom_margin):
                sp.add(a)
        self.sparse = sp

    def model(self, penalty=None):
        return OptimumModel(self.algs, self.curves, self.penalty if penalty is None else penalty)

    def envelope(self, lay):
        """Best log2 T and its algorithm index at every size of a measured
        layout (pruned stretches filled from the model's prior)."""
        c = np.array([self.samplers[lay].curve(a) for a in self.algs])
        return c.min(axis=0), c.argmin(axis=0)

    def diff(self, a, b):
        ea, wa = self.envelope(a)
        eb, wb = self.envelope(b)
        return float(np.sqrt(np.mean((ea - eb) ** 2)) + np.mean(wa != wb))

    def check(self, lay, kind):
        m = self.model()
        pred = m.predict(*lay)
        sel = np.argmin(pred, axis=0)
        self.add(lay, full=kind == "random", prior=pred if self.prune else None)
        s = self.samplers[lay]
        r = []
        for i in sorted(set(range(0, len(self.sizes) - 1, self.step)) | {len(self.sizes) - 1}):
            if all(i in s.g[a].vals for a in self.algs):
                y = [s.g[a].y(i) for a in self.algs]
                r.append(2 ** (y[sel[i]] - min(y)))
        # a pruned risk check may have no size where every algorithm was measured: no regret then
        r = np.array(r)
        gm, p95 = (float(np.exp(np.log(r).mean())), float(np.quantile(r, 0.95))) if len(r) else (np.nan, np.nan)
        self.checks.append((lay, kind, gm, p95, self.wall_us, r))

    def estimate(self, last=0):
        """(geomean, p95) regret pooled over random checks [-(last + k_est), -last)."""
        rc = [c for c in self.checks if c[1] == "random"]
        rc = rc[len(rc) - last - self.k_est:len(rc) - last] if len(rc) >= last + self.k_est else []
        if not rc:
            return None
        r = np.concatenate([c[5] for c in rc])
        r = r[np.isfinite(r)]
        return float(np.exp(np.log(r).mean())), float(np.quantile(r, 0.95))

    def stop_reason(self):
        est = self.estimate()
        if est is None:
            return None
        if est[0] <= self.tau_gm and est[1] <= self.tau_p95:
            return "target"
        prev = self.estimate(self.k_est)
        if prev is not None and est[0] - 1 > self.plateau * (prev[0] - 1) and \
                est[1] - 1 > self.plateau * (prev[1] - 1):
            return "plateau"
        return None

    def run(self, budget_node_h=3.0, nodes_billed=None, max_layouts=None):
        self.billed = nodes_billed or self.n_max
        for lay in initial_layouts(self.allowed, self.n_max, self.ppn_max):
            if self.halt():
                return self
            self.add(lay)
        self.stop = "budget"
        while self.spent() < budget_node_h and (max_layouts is None or len(self.samplers) < max_layouts):
            if self.halt():
                break
            why = self.stop_reason()
            if why:
                self.stop = why
                break
            free = [l for l in self.allowed if l not in self.samplers and l[0] * l[1] > 1]
            if not free:
                self.stop = "all measured"
                break
            cand = candidate_layouts(self.allowed, self.samplers, self.diff)
            if (len(self.checks) + 1) % self.check_every == 0 or not cand:
                self.check(free[self.rng.randint(len(free))], "random")
            else:
                self.check(cand[0][1], "risk")
        self.node_h = self.spent()
        return self

    def spent(self):
        """Node-hours so far: elapsed time (clock) or summed request wall time, times the billed nodes."""
        return (self.clock() if self.clock else self.wall_us * 1e-6) * self.billed / 3600

    def halt(self):
        """Ask stop_check whether to stop before the next layout; record why."""
        why = self.stop_check() if self.stop_check else None
        if why:
            self.stop = why
            self.node_h = self.spent()
        return bool(why)
