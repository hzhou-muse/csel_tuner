"""The model file (collmodel/4): measurements with their provenance, and the fit derived from them.

    m = Model.load("bcast.json")
    m.predict("binomial", 8, 96)          # log2 T at every grid size
    m.sets()                              # {restriction mask: [layouts]}
    m.save("bcast.json")

`data` is the source of truth: every measurement (algorithm, N, ppn, m, the
repetitions, wall time, job, the inner selection it ran under, anchor
flag), the precheck results, the jobs and the inner selection files. `fit`
holds each algorithm's curve at each layout where it was sampled (log2 T
on the size grid, as the sampler left it: measured sizes exactly, the rest
interpolated or, where pruned, the cross-layout prediction corrected by the
measured residuals). An algorithm's model across layouts is a
single.LayoutModel over all its curves, fitted on load. See
docs/design.md, section 12.
"""
import datetime
import hashlib
import json
import os

import numpy as np

from .restrict import infer_mask, mem_known
from .single import LayoutModel

FORMAT = "collmodel/4"
FIT_VERSION = 1


def file_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def lay_key(lay):
    return "%dx%d" % lay


class Model:
    def __init__(self, op, sizes, layouts, domain, conditions=(), algorithms=None, mem="host"):
        """layouts: every allowed (N, ppn); algorithms: {name: {"leaf", "restrictions",
        "mask", "inner"}} (a catalog's view; leaf None: MPIR_<Coll>_intra_<name>);
        mem: the buffers' memory kind, "host" or "device" (a model has one)."""
        self.op = op
        self.mem = mem
        self.sizes = [int(x) for x in sizes]
        self.layouts = sorted(tuple(l) for l in layouts)
        self.domain = domain
        self.conditions = list(conditions)
        self.algorithms = dict(algorithms or {})
        self.masks = {}                 # layout -> restriction mask (conditions true there)
        self.left_out = []              # algorithms that fail without a restriction
        self.notes = []
        self.jobs = []                  # [{...}], a measurement's "job" is an index here
        self.measurements = []          # [{"alg", "N", "ppn", "m", "t_us", "wall_us", "job", "inner"?, "anchor"?}]
        self.prechecks = []             # [{"alg", "N", "ppn", "ok", "msg", "job"}]
        self.inner_files = {}           # hash -> content of a tuning file loaded for inner calls
        self.curves = {}                # alg -> {layout: log2 T on the grid}
        self.fit_inner = {}             # compositional alg -> hash of the inner selection its curves ran under
        self.runs = []                  # per collmodel run: settings, stop, estimate (results, not model)
        self._lm = {}

    # ---- file ----------------------------------------------------------

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        if d.get("format") != FORMAT:
            raise ValueError("%s: not a %s file (format %s; convert with bin/collmodel --import)"
                             % (path, FORMAT, d.get("format")))
        self = cls(d["op"], d["sizes"], d["layouts"], d["domain"], d["conditions"], d["algorithms"],
                   d.get("mem", "host"))
        self.masks = {(n, q): m for n, q, m in d["layout_masks"]}
        self.left_out = d.get("left_out", [])
        self.notes = d.get("notes", [])
        data = d["data"]
        self.jobs, self.measurements = data["jobs"], data["measurements"]
        self.prechecks, self.inner_files = data["prechecks"], data["inner_files"]
        fit = d["fit"]
        if fit.get("version") != FIT_VERSION:
            raise ValueError("%s: fit version %s, this tool has %d" % (path, fit.get("version"), FIT_VERSION))
        self.curves = {a: {(c["N"], c["ppn"]): np.array(c["log2_t"]) for c in cs} for a, cs in fit["curves"].items()}
        self.fit_inner = fit.get("inner", {})
        self.runs = d.get("runs", [])
        return self

    def to_dict(self):
        r = lambda v: [round(float(x), 6) for x in v]
        return {
            "format": FORMAT, "op": self.op, "what": "%s-latency" % self.op, "mem": self.mem,
            "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sizes": self.sizes, "domain": self.domain, "layouts": [list(l) for l in self.layouts],
            "conditions": self.conditions, "algorithms": self.algorithms,
            "layout_masks": [[l[0], l[1], self.masks.get(l, 0)] for l in self.layouts],
            "left_out": self.left_out, "notes": self.notes,
            "data": {"jobs": self.jobs, "measurements": self.measurements, "prechecks": self.prechecks,
                     "inner_files": self.inner_files},
            "fit": {"version": FIT_VERSION, "inner": self.fit_inner,
                    "curves": {a: [{"N": l[0], "ppn": l[1], "log2_t": r(c)} for l, c in sorted(cs.items())]
                               for a, cs in self.curves.items()}},
            "runs": self.runs,
        }

    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=None, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp, path)

    # ---- data ----------------------------------------------------------

    def add_job(self, job):
        self.jobs.append(job)
        return len(self.jobs) - 1

    def add_measurement(self, job, alg, n, ppn, m, t_us, wall_us, inner=None, anchor=False):
        rec = {"alg": alg, "N": n, "ppn": ppn, "m": int(m), "t_us": [round(float(t), 4) for t in t_us],
               "wall_us": round(float(wall_us), 1), "job": job}
        if inner:
            rec["inner"] = inner
        if anchor:
            rec["anchor"] = True
        self.measurements.append(rec)

    def add_precheck(self, job, alg, n, ppn, ok, msg=""):
        self.prechecks.append({"alg": alg, "N": n, "ppn": ppn, "ok": bool(ok), "msg": msg, "job": job})

    def add_inner_file(self, text):
        h = file_hash(text)
        self.inner_files[h] = text
        return h

    def feasible(self):
        """{layout: set of algorithms that ran there}, the latest precheck of each (alg, layout)."""
        last = {}
        for p in self.prechecks:
            last[(p["alg"], p["N"], p["ppn"])] = p["ok"]
        out = {l: set() for l in self.layouts}
        for (a, n, q), ok in last.items():
            if ok:
                out.setdefault((n, q), set()).add(a)
        return out

    def prechecked(self, alg, lay):
        return any(p["alg"] == alg and (p["N"], p["ppn"]) == tuple(lay) for p in self.prechecks)

    def values(self, alg, lay, inner=None):
        """{grid index: [log2 of each measurement's median]} of an algorithm at a
        layout, anchors excluded; for a compositional algorithm only those under
        the given inner selection."""
        idx = {m: i for i, m in enumerate(self.sizes)}
        out = {}
        for r in self.measurements:
            if r["alg"] != alg or (r["N"], r["ppn"]) != tuple(lay) or r.get("anchor") or r["m"] not in idx:
                continue
            if self.is_compositional(alg) and r.get("inner") != inner:
                continue
            out.setdefault(idx[r["m"]], []).append(float(np.log2(np.median(r["t_us"]))))
        return out

    # ---- algorithms and sets -------------------------------------------

    def mask_of(self, alg):
        return self.algorithms.get(alg, {}).get("mask", 0)

    def is_compositional(self, alg):
        return bool(self.algorithms.get(alg, {}).get("inner"))

    def allowed(self, alg, mask):
        return self.mask_of(alg) & ~mask == 0

    def leaf(self, alg):
        x = self.algorithms.get(alg, {}).get("leaf")
        return x or "MPIR_%s_intra_%s" % (self.op[0].upper() + self.op[1:], alg)

    def update_masks(self, algs):
        """Masks from the precheck: the conditions true at each layout, inferred
        from which of the algorithms (catalog Alg objects) ran. Returns notes."""
        feas, notes = self.feasible(), []
        on, off = mem_known(self.conditions, self.mem)
        for lay in self.layouts:
            m, undecided = infer_mask(feas.get(lay, set()), algs, self.conditions)
            undecided &= ~(on | off)
            self.masks[lay] = (m | on) & ~off
            if undecided:
                notes.append("N=%d ppn=%d: conditions %s not decided by the algorithms, taken as false" % (
                    lay[0], lay[1], ",".join(self.conditions[k] for k in range(len(self.conditions))
                                             if undecided >> k & 1)))
        return notes

    def sets(self):
        """{mask: [layouts]}: set 0 is every layout (the run-time fallback); a set S != 0
        the layouts whose mask is exactly S."""
        out = {0: list(self.layouts)}
        for lay in self.layouts:
            m = self.masks.get(lay, 0)
            if m:
                out.setdefault(m, []).append(lay)
        return out

    def set_algs(self, mask):
        """The modeled algorithms allowed in a set: their mask a subset of it."""
        return [a for a in self.algorithms if a in self.curves and self.curves[a] and self.allowed(a, mask)
                and a not in self.left_out]

    def measured_layouts(self):
        return sorted({l for cs in self.curves.values() for l in cs})

    # ---- fit -------------------------------------------------------------

    def set_curve(self, alg, lay, curve):
        self.curves.setdefault(alg, {})[tuple(lay)] = np.asarray(curve, float)
        self._lm.pop(alg, None)

    def drop_curves(self, alg):
        self.curves.pop(alg, None)
        self._lm.pop(alg, None)

    def layout_model(self, alg):
        if alg not in self._lm:
            self._lm[alg] = LayoutModel(self.curves[alg])
        return self._lm[alg]

    def predict(self, alg, n, ppn):
        """log2 T of an algorithm at every grid size of a layout."""
        return self.layout_model(alg).predict(n, ppn)
