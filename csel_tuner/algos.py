"""The algorithms of a collective, from MPICH's src/mpi/coll/coll_algorithms.txt.

    cat = Catalog.load("coll_algorithms.txt")
    algs = cat.algorithms("bcast")        # [Alg], in file order, plus nb
    bits = cat.restriction_bits()         # restrictive conditions: bit k = bits[k], for every op
    algs[0].mask(bits), algs[0].leaf      # e.g. 0, "MPIR_Bcast_intra_binomial"

An algorithm's restrictions are conditions that must hold for it to run.
The restrictive conditions are the file's `restrictions:` section, each
listed with its index ("11 - MPIDI_CH4_release_gather: ..."), one numbering
for every collective; files before that section have them as the
non-threshold entries of `conditions:`, numbered in file order. An
algorithm's mask is the OR of its restrictions' bits (0: unrestricted), and
it runs in a restriction set S (the mask of the conditions that are true)
when its mask is a subset of S, so whatever runs in S also runs in any
larger set. MPICH files from before restrictions became positive-only use
`!c` (e.g. `!inplace`); each such restriction is read as a condition of its
own, numbered after the listed ones (a CSEL tree can test it as `!c`).

A compositional algorithm lists the collectives it calls through the
selection as `inner` (Inner: section, communicator, conditions). The general
algorithms nb (MPIR_Coll_nb, the blocking call through its nonblocking form)
and buffer_swap (MPIR_Coll_buffer_swap, device buffers swapped for host
buffers) run for every blocking collective; their inner calls are written
relative to the caller ($coll, comm, [k=v]) and are filled in per collective.
Catalogs without those inner keys get the same calls (GENERAL).
"""
import re

INNER_RE = re.compile(r"([\w$-]+)@(\w+)(?:\[(\w+)=([01])\])?$")
# the general algorithms of a blocking collective: model name -> (function, inner calls if the catalog has none)
GENERAL = {"nb": ("MPIR_Coll_nb", "i$coll-intra@comm"),
           "buffer_swap": ("MPIR_Coll_buffer_swap", "$coll-intra@comm[gpu=0]")}


class Inner:
    """An inner call: the section it selects from, the communicator (node_comm, node_roots_comm,
    or comm: the caller's own) and the restrictions it sets, e.g. {"gpu": 0}."""

    def __init__(self, section, comm, cond=None):
        self.section, self.comm, self.cond = section, comm, cond or {}

    @classmethod
    def parse(cls, text, coll):
        m = INNER_RE.match(text.replace(" ", ""))
        if not m:
            raise ValueError("inner %s is not <section>@<communicator>[k=v]" % text)
        cond = {m.group(3): int(m.group(4))} if m.group(3) else {}
        return cls(m.group(1).replace("$coll", coll), m.group(2), cond)

    def __eq__(self, other):
        return (self.section, self.comm, self.cond) == (other.section, other.comm, other.cond)

    def __str__(self):
        return "%s@%s%s" % (self.section, self.comm, "".join("[%s=%d]" % kv for kv in self.cond.items()))

    __repr__ = __str__


class Alg:
    def __init__(self, op, name, keys):
        self.op, self.name, self.keys = op, name, keys
        self.restrictions = [r.strip() for r in keys.get("restrictions", "").split(",") if r.strip()]
        self.extra_params = [p.strip() for p in keys.get("extra_params", "").split(",") if p.strip()]
        self.cvar_params = [p.strip() for p in keys.get("cvar_params", "").split(",") if p.strip()]
        # inner: the calls a compositional algorithm makes through the selection
        self.inner = [Inner.parse(x, op) for x in keys.get("inner", "").split(",") if x.strip()]
        coll = op[0].upper() + op[1:]
        self.leaf = keys.get("func_name") or "MPIR_%s_intra_%s" % (coll, name)

    def mask(self, bits):
        m = 0
        for r in self.restrictions:
            if r in bits:
                m |= 1 << bits.index(r)
        return m

    def allowed(self, bits, s):
        return self.mask(bits) & ~s == 0

    def __repr__(self):
        return "Alg(%s%s)" % (self.name, ", " + ",".join(self.restrictions) if self.restrictions else "")


class Catalog:
    def __init__(self, sections, conditions, indexed=None):
        self.sections = sections        # "bcast-intra" -> [(name, {key: value})]
        self.conditions = conditions    # name -> takes a threshold
        self.indexed = indexed          # {index: restriction} from a restrictions: section

    @classmethod
    def load(cls, path):
        sections, conditions, indexed, cur, alg = {}, {}, {}, None, None
        for raw in open(path):
            line = raw.rstrip("\n")
            s = line.strip()
            if not s or s.startswith(("#", "/*", "*")):
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent == 0:
                cur = s.rstrip(":")
                sections.setdefault(cur, [])
                alg = None
            elif cur == "restrictions":
                m = re.match(r"(\d+)\s+-\s+(\w+):", s)
                if not m:
                    raise ValueError("%s: restriction entry is not 'index - name: func': %s" % (path, s))
                indexed[int(m.group(1))] = m.group(2)
                conditions[m.group(2)] = False
            elif cur == "conditions":
                name = s.split(":", 1)[0].strip()
                m = re.match(r"(\w+)\((\w+)\)$", name)
                conditions[m.group(1) if m else name] = bool(m)
            elif indent == 4:
                alg = (s, {})
                sections[cur].append(alg)
            elif alg is not None and ":" in s:
                k, v = s.split(":", 1)
                alg[1][k.strip()] = v.strip()
        sections.pop("restrictions", None)
        if indexed and sorted(indexed) != list(range(len(indexed))):
            raise ValueError("%s: restriction indices are not 0..%d" % (path, len(indexed) - 1))
        return cls(sections, conditions, indexed or None)

    def algorithms(self, op, kind="intra", nb=True):
        """The op's algorithms in file order, then for a blocking op the general ones:
        nb (implied for every blocking op) and buffer_swap (if the file lists it). Both are
        compositional: nb on the op's nonblocking tree, buffer_swap on its host tree."""
        sec = "%s-%s" % (op, kind)
        if sec not in self.sections:
            raise KeyError("no section %s in coll_algorithms.txt" % sec)
        out = [Alg(op, n, k) for n, k in self.sections[sec]]
        if kind == "intra" and not op.startswith("i"):
            general = dict(self.sections.get("general", []))
            for name, (func, inner) in GENERAL.items():
                if (name == "nb" and nb) or func in general:
                    keys = dict(general.get(func, {}))
                    keys.setdefault("inner", inner)
                    out.append(Alg(op, name, dict(keys, func_name=func)))
        return out

    def restriction_bits(self):
        """The restrictive (non-threshold) conditions in file order, then the
        old-style negated restrictions in order of first use: bit k = [k]."""
        if self.indexed:
            bits = [self.indexed[k] for k in range(len(self.indexed))]
        else:
            bits = [c for c, thresh in self.conditions.items() if not thresh]
        for sec in self.sections.values():
            for _, keys in sec:
                for r in keys.get("restrictions", "").split(","):
                    r = r.strip()
                    if r.startswith("!") and r[1:] in bits and r not in bits:
                        bits.append(r)
        return bits

    def check(self, op, kind="intra"):
        """Every restriction of the op's algorithms is a restrictive condition."""
        bits = self.restriction_bits()
        for a in self.algorithms(op, kind):
            for r in a.restrictions:
                if r not in bits:
                    raise KeyError("%s: restriction %s is not a listed non-threshold condition" % (a.name, r))
