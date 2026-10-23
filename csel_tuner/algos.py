"""The algorithms of a collective, from MPICH's src/mpi/coll/coll_algorithms.txt.

    cat = Catalog.load("coll_algorithms.txt")
    algs = cat.algorithms("bcast")        # [Alg], in file order, plus nb
    bits = cat.restriction_bits("bcast")  # restrictive conditions: bit k = bits[k]
    algs[0].mask(bits), algs[0].leaf      # e.g. 0, "MPIR_Bcast_intra_binomial"

An algorithm's restrictions are conditions that must hold for it to run. The
non-threshold ones (e.g. MPIDI_CH4_release_gather, hierarchical, !inplace;
a negated condition is its own entry) are numbered in file order over the
collective's algorithms; an algorithm's mask is the OR of its bits. Mask 0:
no restriction.
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

    def mask(self, bits):
        m = 0
        for r in self.restrictions:
            if r in bits:
                m |= 1 << bits.index(r)
        return m

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

    def restriction_bits(self, op, kind="intra"):
        bits = []
        for a in self.algorithms(op, kind):
            for r in a.restrictions:
                name = r.lstrip("!")
                if name not in self.conditions:
                    raise KeyError("%s: restriction %s is not a listed condition" % (a.name, r))
                if not self.conditions[name] and r not in bits:
                    bits.append(r)
        return bits
