"""The algorithms of a collective, from MPICH's src/mpi/coll/coll_algorithms.txt.

    cat = Catalog.load("coll_algorithms.txt")
    algs = cat.algorithms("bcast")        # [Alg], in file order, plus nb
    bits = cat.restriction_bits()         # restrictive conditions: bit k = bits[k], for every op
    algs[0].requires(bits), algs[0].leaf  # e.g. (0, 0), "MPIR_Bcast_intra_binomial"

An algorithm's restrictions are conditions that must hold (`c`) or must not
hold (`!c`) for it to run. The restrictive conditions are the non-threshold
entries of the file's `conditions:` list, numbered in that order (one
numbering for every collective; appending a condition keeps the others'
numbers). A restriction set is the mask of the conditions that are true; an
algorithm runs in set S when its required bits are in S and its excluded
bits are not (allowed()). requires() == (0, 0): no restriction.
"""
import re


class Alg:
    def __init__(self, op, name, keys):
        self.op, self.name, self.keys = op, name, keys
        self.restrictions = [r.strip() for r in keys.get("restrictions", "").split(",") if r.strip()]
        self.extra_params = [p.strip() for p in keys.get("extra_params", "").split(",") if p.strip()]
        self.cvar_params = [p.strip() for p in keys.get("cvar_params", "").split(",") if p.strip()]
        coll = op[0].upper() + op[1:]
        self.leaf = keys.get("func_name") or "MPIR_%s_intra_%s" % (coll, name)

    def requires(self, bits):
        """(bits that must be set, bits that must be clear)"""
        pos = neg = 0
        for r in self.restrictions:
            name = r.lstrip("!")
            if name in bits:
                if r.startswith("!"):
                    neg |= 1 << bits.index(name)
                else:
                    pos |= 1 << bits.index(name)
        return pos, neg

    def mask(self, bits):
        """the bits the algorithm's restrictions involve; 0: unrestricted"""
        pos, neg = self.requires(bits)
        return pos | neg

    def allowed(self, bits, s):
        pos, neg = self.requires(bits)
        return pos & ~s == 0 and neg & s == 0

    def __repr__(self):
        return "Alg(%s%s)" % (self.name, ", " + ",".join(self.restrictions) if self.restrictions else "")


class Catalog:
    def __init__(self, sections, conditions):
        self.sections = sections        # "bcast-intra" -> [(name, {key: value})]
        self.conditions = conditions    # name -> takes a threshold

    @classmethod
    def load(cls, path):
        sections, conditions, cur, alg = {}, {}, None, None
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
        return cls(sections, conditions)

    def algorithms(self, op, kind="intra", nb=True):
        """The op's algorithms in file order; nb (MPIR_Coll_nb: the blocking call
        through its nonblocking form) is implied by the file for blocking ops."""
        sec = "%s-%s" % (op, kind)
        if sec not in self.sections:
            raise KeyError("no section %s in coll_algorithms.txt" % sec)
        out = [Alg(op, n, k) for n, k in self.sections[sec]]
        if nb and kind == "intra" and not op.startswith("i"):
            out.append(Alg(op, "nb", {"func_name": "MPIR_Coll_nb"}))
        return out

    def restriction_bits(self):
        """The restrictive (non-threshold) conditions in file order: bit k = [k]."""
        return [c for c, thresh in self.conditions.items() if not thresh]

    def check(self, op, kind="intra"):
        """Every restriction of the op's algorithms is a listed restrictive condition."""
        bits = self.restriction_bits()
        for a in self.algorithms(op, kind):
            for r in a.restrictions:
                if r.lstrip("!") not in bits:
                    raise KeyError("%s: restriction %s is not a listed non-threshold condition" % (a.name, r))
