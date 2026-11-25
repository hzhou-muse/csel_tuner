"""Algorithm parameters: param maps and the generic part of their rules.

    pm = ParamMap.default("bcast", "circ_graph")      # the rule with its default constants
    pm.theta(p=64, m=1 << 20)                         # {"chunk_size": 131072, "q_len": 8}
    pm.req_cvars("bcast", 64, 1 << 20)                # {"cvar.MPIR_CVAR_BCAST_CIRC_GRAPH_CHUNK_SIZE": 131072, ...}
    maps = load_maps("cg.model", "bcast", "host")     # {alg: ParamMap} of a param-map file

A param map is the function theta*(comm_size, msg_size) -> the algorithm's leaf parameters. It is
a rule (a fixed expression, per algorithm family), the rule's version and its site constants;
bin/algoparam fits the constants from a few measured points, collmodel measures the algorithm at
theta* (per-request CVARs) and colltree writes the leaves with the theta of each leaf's region.
See docs/design.md, section 14.

Each rule is a module of its own, csel_tuner/param_<rule>.py (param_circ_graph.py), loaded by
rule() on first use. It provides
    VERSION, PARAMS, CVAR_PARAMS, DEFAULTS     its version, parameter names, their CVARs (as in
                                               coll_algorithms.txt's cvar_params), its constants
    consts_for(op, **consts)                   the default constants of op's family, consts over them
    theta(consts, p, m)                        {param: int array} over points
    describe(consts), check(consts, cvar_values, device)    a one-line form; a warning or None
and, for bin/algoparam, FAMILY ({op: family}), add_args(group), calibrate(ctx, op, fixed, args) and
fit(op, measurements, fixed, args) -> (consts, fit report, report lines).
"""
import importlib
import json
import re

import numpy as np

from .model import file_hash

K = 1 << 10
M = 1 << 20

_RULES = {}


def rule(name):
    """The rule module csel_tuner.param_<name>; ValueError if there is none."""
    if name not in _RULES:
        if not re.match(r"^[a-z][a-z0-9_]*$", str(name)):
            raise ValueError("param map rule %r: not a rule name" % (name,))
        try:
            _RULES[name] = importlib.import_module("csel_tuner.param_" + name)
        except ModuleNotFoundError as e:
            if e.name != "csel_tuner.param_" + name:
                raise
            raise ValueError("param map rule %r: no module csel_tuner/param_%s.py" % (name, name)) from None
    return _RULES[name]


def pow2(x):
    """The nearest power of two in log2, halves up (toward fewer, larger chunks); x >= 1."""
    return 2.0 ** np.floor(np.log2(np.maximum(x, 1.0)) + 0.5)


def parse_size(s):
    """"64K" -> 65536, "16M" -> 16777216, "8" -> 8."""
    s = str(s).strip()
    mult = {"K": K, "M": M, "G": 1 << 30}.get(s[-1:].upper(), 1)
    return int(float(s[:-1] if mult > 1 else s) * mult)


def size_str(v):
    v = int(v)
    for unit, f in (("G", 1 << 30), ("M", M), ("K", K)):
        if v >= f and v % f == 0:
            return "%d%s" % (v // f, unit)
    return str(v)


def score(r, pct=98):
    """What a calibration fit minimizes: the pct percentile of regret (rounded), then the mean log
    regret."""
    return (round(float(np.percentile(r, pct)), 3), float(np.mean(np.log(r))))


def stats(r):
    r = np.asarray(r, float)
    if not len(r):
        return {}
    return {"max": round(float(r.max()), 3), "p90": round(float(np.percentile(r, 90)), 3),
            "median": round(float(np.median(r)), 3), "geomean": round(float(np.exp(np.mean(np.log(r)))), 3),
            "n": int(len(r))}


class ParamMap:
    """theta*(p, m) of one algorithm: a rule, its version and its constants. `fit` is how the
    constants were obtained (algoparam's report); it is not part of the map's identity (hash). A map
    is always of the version of the rule in this code: from_dict refuses one of another version."""

    def __init__(self, alg, rule_name, consts, fit=None):
        self.mod = rule(rule_name)
        self.alg, self.rule, self.consts = alg, rule_name, {k: int(v) for k, v in consts.items()}
        self.version = self.mod.VERSION
        self.fit = fit or {}
        missing = set(self.mod.DEFAULTS) - set(self.consts)
        if missing:
            raise ValueError("param map %s: constants %s missing" % (rule_name, ", ".join(sorted(missing))))

    @classmethod
    def default(cls, op, alg, rule_name=None, **consts):
        """The rule's default constants for op's family, consts over them; the rule is the
        algorithm's own unless rule_name is given."""
        rule_name = rule_name or alg
        return cls(alg, rule_name, rule(rule_name).consts_for(op, **consts))

    @property
    def params(self):
        return self.mod.PARAMS

    @property
    def hash(self):
        return file_hash(json.dumps({"alg": self.alg, "rule": self.rule, "version": self.version,
                                     "consts": self.consts}, sort_keys=True))

    def to_dict(self):
        d = {"rule": self.rule, "version": self.version, "consts": self.consts, "hash": self.hash}
        if self.fit:
            d["fit"] = self.fit
        return d

    @classmethod
    def from_dict(cls, alg, d):
        """The map of a file; ValueError if it was made with another version of its rule (its
        parameters would differ from those it was measured with) or its hash does not match."""
        pm = cls(alg, d["rule"], d["consts"], d.get("fit"))
        if d.get("version") != pm.version:
            raise ValueError("param map %s of %s: made with %s rule version %s, this csel_tuner has version %d; "
                             "refit it (bin/algoparam --refit) and measure the algorithm again"
                             % (d.get("hash", "?"), alg, d["rule"], d.get("version", "(none)"), pm.version))
        if d.get("hash") and d["hash"] != pm.hash:
            raise ValueError("param map of %s: its hash %s does not match its constants (%s; edited?)"
                             % (alg, d["hash"], pm.hash))
        return pm

    def theta_arrays(self, p, m):
        """{param: int array} over points (p, m)."""
        return self.mod.theta(self.consts, p, m)

    def theta(self, p, m):
        return {k: int(v) for k, v in self.theta_arrays(p, m).items()}

    def cvars(self, op):
        """{param: its CVAR}, as MPICH's gen_coll.py names them (algos.Alg.params)."""
        return {name: "MPIR_CVAR_%s_%s" % (op.upper(), cv) for name, cv in zip(self.params, self.mod.CVAR_PARAMS)}

    def req_cvars(self, op, p, m):
        """The per-request CVAR tokens that run the algorithm at theta*(p, m)."""
        th = self.theta(p, m)
        return {"cvar." + cv: th[name] for name, cv in self.cvars(op).items()}

    def check(self, cvar_values, device):
        """A warning about the server's settings for this map, or None."""
        w = self.mod.check(self.consts, cvar_values, device)
        return f"{self.alg}: {w}" if w else None

    def describe(self):
        return f"{self.rule} v{self.version}: {self.mod.describe(self.consts)}"


def theta_of_req(req, op, pm):
    """The theta a request ran with (its cvar tokens), or None if it sets none of the map's CVARs."""
    out = {name: int(req["cvar." + cv]) for name, cv in pm.cvars(op).items() if "cvar." + cv in req}
    return out or None


def map_tags(op, launch):
    """The tags of a param-map unit (algoparam's output): a model unit's, plus kind=param_map, so
    that it never takes the place of the collective's model in a file."""
    from .model import default_tags
    return dict(default_tags(op, launch), kind="param_map")


def is_map_unit(u):
    return u.tags.get("kind") == "param_map"


def load_maps(path, op, launch):
    """{alg: ParamMap} of a param-map file (algoparam's output) for one collective and launch kind:
    the units of that launch kind, else of one with the same buffers (gpu and device)."""
    from .kinds import is_device
    from .model import ModelFile
    units = [u for u in ModelFile.load(path).units if u.tags.get("coll") == op and is_map_unit(u)]
    same = [u for u in units if u.tags.get("launch") == launch] or \
        [u for u in units if is_device(u.tags.get("launch")) == is_device(launch)]
    out = {}
    for u in same:
        for alg, info in u.algorithms.items():
            if info.get("param_map"):
                out[alg] = ParamMap.from_dict(alg, info["param_map"])
    return out


def model_maps(model):
    """{alg: ParamMap} recorded in a model's algorithm entries."""
    return {alg: ParamMap.from_dict(alg, info["param_map"]) for alg, info in model.algorithms.items()
            if info.get("param_map")}
