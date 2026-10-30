"""The model file (collmodel/5): units of measurements with their provenance, and the fits derived
from them.

    f = ModelFile.load("models.json")     # every unit (a collmodel/4 file is one unit)
    m = f.unit({"coll": "bcast"})         # the one unit these tags select
    m = Model.load("bcast.json")          # the same, for a file of one unit (or Model.load(path, sel))
    m.predict("binomial", 8, 96)          # log2 T at every grid size
    m.sets()                              # {restriction mask: [layouts]}
    f.put(m); f.save("models.json")       # add or replace the unit with m's tags

A unit is one model: one collective measured one way, named by its tags
(tag_str): coll, variant (the input-condition assignment, kinds.py),
datatype, launch (the buffers' memory, today host or device) and, for the
v-operations, counts. A unit is self-contained: its jobs, inner selection
files and runs are its own, so units copy between files unchanged.

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

from .kinds import datatype, is_v, run_variant
from .restrict import infer_mask, mem_known
from .single import LayoutModel

FORMAT = "collmodel/5"
FORMAT_V4 = "collmodel/4"
FIT_VERSION = 1


def default_tags(coll, launch="host", inplace=False, counts="uniform"):
    """A unit's tags from how mpisrv measured it (MPI_SUM, its datatype for the collective)."""
    tags = {"coll": coll, "variant": run_variant(coll, inplace), "datatype": datatype(coll), "launch": launch}
    if is_v(coll):
        tags["counts"] = counts
    return tags


def v4_tags(d):
    """A collmodel/4 file's tags: its op and mem, and inplace/counts from its last job that has them
    (collmodel --inplace and --counts came shortly before collmodel/5)."""
    job = next((j for j in reversed(d["data"]["jobs"]) if "inplace" in j or "counts" in j), {})
    return default_tags(d["op"], d.get("mem", "host"), bool(job.get("inplace")), job.get("counts") or "uniform")


def tag_key(tags):
    return json.dumps(tags, sort_keys=True)


def tag_str(tags):
    """e.g. "allgatherv inplace=1 datatype=byte counts=linear launch=host"."""
    v = " ".join("%s=%s" % kv for kv in sorted(tags.get("variant", {}).items()))
    rest = " ".join("%s=%s" % (k, tags[k]) for k in sorted(tags) if k not in ("coll", "variant"))
    return " ".join(x for x in (tags.get("coll", "?"), v, rest) if x)


def tags_match(tags, sel):
    """sel: {key: value} over the tags and the variant's conditions (values compared as strings)."""
    for k, v in (sel or {}).items():
        have = tags.get(k, tags.get("variant", {}).get(k))
        if have is None or str(have) != str(v):
            return False
    return True


def parse_sel(text):
    """"coll=bcast,inplace=1" -> {"coll": "bcast", "inplace": "1"}."""
    out = {}
    for kv in filter(None, (text or "").split(",")):
        k, eq, v = kv.partition("=")
        if not eq:
            raise ValueError("%s: need KEY=VALUE" % kv)
        out[k.strip()] = v.strip()
    return out


def file_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def lay_key(lay):
    return "%dx%d" % lay


class Model:
    """One unit."""

    def __init__(self, op, sizes, layouts, domain, conditions=(), algorithms=None, mem="host", tags=None):
        """layouts: every allowed (N, ppn); algorithms: {name: {"leaf", "restrictions",
        "mask", "inner"}} (a catalog's view; leaf None: MPIR_<Coll>_intra_<name>);
        mem: the buffers' memory kind, "host" or "device" (the launch tag); tags: the unit's
        tags, default_tags(op, mem) for those left out."""
        self.tags = dict(default_tags(op, mem), **(tags or {}))
        if self.tags["coll"] != op or self.tags["launch"] != mem:
            raise ValueError("tags %s do not match op %s, mem %s" % (tag_str(self.tags), op, mem))
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

    @property
    def op(self):
        return self.tags["coll"]

    @property
    def mem(self):
        return self.tags["launch"]

    # ---- file ----------------------------------------------------------

    @classmethod
    def load(cls, path, sel=None):
        """The unit of the file that sel ({tag: value}) selects; without sel, the file's only unit."""
        return ModelFile.load(path).unit(sel)

    def save(self, path):
        """A file of just this unit."""
        ModelFile([self]).save(path)

    @classmethod
    def from_dict(cls, d, path="", tags=None):
        """A unit of a collmodel/5 file, or a whole collmodel/4 file (with tags given)."""
        tags = tags or d["tags"]
        self = cls(tags["coll"], d["sizes"], d["layouts"], d["domain"], d["conditions"], d["algorithms"],
                   tags["launch"], tags)
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
        """The unit's entry in a collmodel/5 file."""
        r = lambda v: [round(float(x), 6) for x in v]
        return {
            "tags": self.tags, "what": "%s-latency" % self.op,
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
        if x:
            return x
        Op = self.op[0].upper() + self.op[1:]
        if self.op.startswith("i") and alg.startswith("tsp_"):     # gen_coll.py's name for tsp algorithms
            return "MPIR_TSP_%s_sched_intra_%s" % (Op, alg[4:])
        return "MPIR_%s_intra_%s" % (Op, alg)

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


class ModelFile:
    """The units of a model file."""

    def __init__(self, units=()):
        self.units = list(units)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        fmt = d.get("format")
        if fmt == FORMAT_V4:
            return cls([Model.from_dict(d, path, v4_tags(d))])
        if fmt != FORMAT:
            raise ValueError("%s: not a %s file (format %s; convert collmodel/3 with bin/collmodel --import)"
                             % (path, FORMAT, fmt))
        return cls([Model.from_dict(u, path) for u in d["units"]])

    def save(self, path):
        d = {"format": FORMAT,
             "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "units": [u.to_dict() for u in self.units]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=None, separators=(",", ":"))
            f.write("\n")
        os.replace(tmp, path)

    def select(self, sel=None):
        return [u for u in self.units if tags_match(u.tags, sel)]

    def unit(self, sel=None):
        """The one unit sel selects (ValueError if none or several, listing the units)."""
        found = self.select(sel)
        if len(found) == 1:
            return found[0]
        units = "; ".join(tag_str(u.tags) for u in self.units) or "none"
        raise ValueError("%s units select %s (the file's units: %s)"
                         % ("no" if not found else "several", ",".join("%s=%s" % kv for kv in (sel or {}).items())
                            or "the only unit", units))

    def get(self, tags):
        k = tag_key(tags)
        return next((u for u in self.units if tag_key(u.tags) == k), None)

    def put(self, unit):
        """Add the unit, replacing one with the same tags; True if it replaced one."""
        k = tag_key(unit.tags)
        for i, u in enumerate(self.units):
            if tag_key(u.tags) == k:
                self.units[i] = unit
                return True
        self.units.append(unit)
        return False
