"""The stages of a collection's algorithms, from their inner calls.

    st = Stages(Catalog.load("coll_algorithms.txt"))
    st.alg_stage("bcast", "host", alg)          # 1 plain; e.g. 2 for nb, 3 for smp
    st.unit_stages("bcast", "host")             # [(stage, Alg, [deps])] in catalog order
    st.tree_stage("ibcast", "host")             # the stage at which a tree is complete

A plain algorithm is stage 1. A compositional algorithm (one with inner calls)
is 1 + the largest stage of the trees its inner calls select from, a tree
being (collective, launch kind): an inner call selects from the collective's
own tree on the caller's launch, or with [gpu=0] from its host tree. A tree
is complete at the largest stage of its algorithms, with two kinds left out:
those whose inner calls lead back into the chain being computed (a cycle:
smp's inner bcast calls cannot run smp, so smp's bcast tree is bcast without
smp), and those the launch kind rules out (buffer_swap, restricted to gpu,
is in no host tree). So for host bcast with nb and smp: the plain algorithms
are stage 1, nb 2 (ibcast's tree is plain), smp 3 (its bcast tree has nb),
and on a gpu launch buffer_swap 4 (the host tree is complete at 3).

A stage-k algorithm is measured with an inner selection file
(--inner-selection) holding the trees it depends on, built from algorithms
up to stage k - 1: stage 1 needs none.
"""
from .kinds import LAUNCH


class Stages:
    def __init__(self, catalog, exclude=()):
        self.cat, self.exclude = catalog, set(exclude)
        self.memo = {}

    def algorithms(self, coll, launch):
        """The collective's algorithms that a launch kind allows (none if the catalog lacks it)."""
        try:
            algs = self.cat.algorithms(coll)
        except KeyError:
            return []
        off = {c for c, v in LAUNCH.get(launch, {}).items() if v == 0}
        return [x for x in algs if x.name not in self.exclude and not off & set(x.restrictions)]

    @staticmethod
    def deps(alg, launch):
        """The trees an algorithm's inner calls select from: [(coll, launch)], in order, unique."""
        out = []
        for i in alg.inner:
            d = (i.section.rsplit("-", 1)[0], "host" if i.cond.get("gpu") == 0 else launch)
            if d not in out:
                out.append(d)
        return out

    def alg_stage(self, coll, launch, alg, stack=frozenset()):
        if not alg.inner:
            return 1
        inner = stack | {(coll, launch)}
        return 1 + max(self.tree_stage(c, l, inner) for c, l in self.deps(alg, launch))

    def tree_stage(self, coll, launch, stack=frozenset()):
        """The stage at which the tree (coll, launch) is complete, without the algorithms that
        depend on a tree in stack (0: the catalog has no algorithm for it)."""
        key = (coll, launch, stack)
        if key not in self.memo:
            best = 0
            for x in self.algorithms(coll, launch):
                if any(d in stack for d in self.deps(x, launch)):
                    continue
                best = max(best, self.alg_stage(coll, launch, x, stack))
            self.memo[key] = best
        return self.memo[key]

    def unit_stages(self, coll, launch):
        """[(stage, Alg, deps)] of a unit's algorithms, in catalog order."""
        return [(self.alg_stage(coll, launch, x), x, self.deps(x, launch))
                for x in self.algorithms(coll, launch)]
