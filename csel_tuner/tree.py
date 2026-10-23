"""Builder v3: a decision tree (MPICH collective-selection JSON) from an optimum model.

    t = fit_tree(features, cost, names)       # cost: predicted log2 T, points x algorithms
    t = prune_to(t, tol=0.005)
    json_tree(t, ...)                         # a CSEL subtree, e.g. "name=bcast-intra-auto"

Input is the v2 model (multi.OptimumModel, or a collmodel/2 file): the
predicted time of every algorithm at every point of a grid, all allowed
layouts x all sizes. The tree is fitted to minimize regret, not
misclassification: with R[i, a] = predicted log2 T[i, a] - min_b log2 T[i, b],
a leaf picks the algorithm with the smallest sum of R over its points, and a
split is the threshold that most reduces that sum. Two algorithms within a
few percent of each other cost almost nothing to confuse, so the tree spends
its splits where a wrong choice is expensive.

1. Features. Today's CSEL conditions: comm_size(t) (p <= t),
   avg_msg_size(t) (m <= t; for bcast the message size) and pof2. With
   `node_aware`, also num_nodes(t) and ppn(t): the conditions this tool
   assumes CSEL will gain (proposed names; MPICH must add them before such a
   file can be loaded).
2. Growth: best-first, every candidate threshold of every feature (the
   distinct values on the grid, prefix sums per algorithm), choosing the
   split that most reduces the squared deviation of the regret vectors
   (a multi-output regression tree; a regret criterion is myopic, see
   best_split), until the reduction is below `min_gain` per point or there
   are `max_leaves` leaves. Leaves pick the least-regret algorithm.
3. Pruning: weakest link first (the split whose removal adds the least
   regret per leaf removed), down to the smallest tree whose predicted
   geomean regret is within `tol` of the full tree's, never merging a split
   that would give a point a predicted regret above `max_regret` (the
   geomean alone lets a few expensive mistakes through, e.g. an algorithm
   that wins at power-of-two p chosen for the non-power-of-two p next to
   them, where it is much slower).
4. JSON: a right-leaning chain of splits becomes one object of ordered
   sibling conditions ending in "any" (CSEL tries siblings in order); pof2
   is a 0/1 feature, so its split reads "!pof2" for the first sibling; a leaf
   is {"algorithm=MPIR_<Coll>_intra_<alg>": {extra params}}. A numeric
   split on value v sends v <= t to the first sibling, t being the largest
   grid value on that side.
"""
import heapq

import numpy as np

CSEL_NAME = {"p": "comm_size", "m": "avg_msg_size", "N": "num_nodes", "ppn": "ppn", "pof2": "pof2"}


class Node:
    def __init__(self, idx, R):
        self.idx = idx
        s = R[idx].sum(axis=0)
        self.alg = int(np.argmin(s))
        self.cost = float(s[self.alg])
        self.split = None       # (feature, threshold, left, right, gain)

    def leaves(self):
        if self.split is None:
            return [self]
        return self.split[2].leaves() + self.split[3].leaves()

    def subtree_cost(self):
        return sum(l.cost for l in self.leaves())


def best_split(node, X, R, names):
    """(gain, feature, threshold, left idx, right idx) of the split that most
    reduces the squared deviation of the regret vectors R (all algorithms at
    once), or None. A split on regret itself is myopic: when the best
    algorithm depends on two features jointly, no single split lowers the
    regret much until the second one is made, so growth would stop early.
    The leaves still pick by regret, and pruning works on regret."""
    idx = node.idx
    Ri = R[idx]
    tot, tot2 = Ri.sum(axis=0), (Ri ** 2).sum(axis=0)
    n = len(idx)
    sse0 = float((tot2 - tot ** 2 / n).sum())
    best = None
    for f in names:
        x = X[f][idx]
        o = np.argsort(x, kind="stable")
        xs = x[o]
        last = np.nonzero(xs[1:] != xs[:-1])[0]         # last position of each value but the largest
        if not len(last):
            continue
        c1 = np.cumsum(Ri[o], axis=0)[last]
        c2 = np.cumsum(Ri[o] ** 2, axis=0)[last]
        nl = (last + 1)[:, None]
        nr = n - nl
        sse = ((c2 - c1 ** 2 / nl) + ((tot2 - c2) - (tot - c1) ** 2 / nr)).sum(axis=1)
        k = int(np.argmin(sse))
        gain = sse0 - float(sse[k])
        if best is None or gain > best[0]:
            best = (gain, f, xs[last[k]], idx[o[:last[k] + 1]], idx[o[last[k] + 1:]])
    return best


def fit_tree(X, L, names, min_gain=1e-6, max_leaves=1024):
    """X: {feature: values per point}; L: predicted log2 T, points x algorithms;
    names: the features splits may use."""
    R = L - L.min(axis=1, keepdims=True)
    n = len(R)
    root = Node(np.arange(n), R)
    heap = []

    def push(node):
        s = best_split(node, X, R, names)
        if s and s[0] > min_gain * n:
            heapq.heappush(heap, (-s[0], id(node), node, s))
    push(root)
    nleaves = 1
    while heap and nleaves < max_leaves:
        _, _, node, (gain, f, t, li, ri) = heapq.heappop(heap)
        node.split = [f, t, Node(li, R), Node(ri, R), gain]
        nleaves += 1
        push(node.split[2])
        push(node.split[3])
    root.n = n
    root.R = R
    return root


def internal(node):
    if node.split is None:
        return []
    return [node] + internal(node.split[2]) + internal(node.split[3])


def prune_to(root, tol=0.005, max_regret=1.5):
    """Weakest-link pruning to the smallest tree whose predicted geomean
    regret is within tol of the full tree's. A split is never merged if the
    merged leaf would give some point a predicted regret above max_regret
    (unless that point already has it), so the few expensive mistakes that
    a geomean hides are kept out. Returns the pruned root (in place) and the
    path [(leaves, geomean regret)]."""
    n, R = root.n, root.R
    gm = lambda: 2 ** (root.subtree_cost() / n)
    full = gm()
    path = [(len(root.leaves()), full)]
    cap = np.log2(max_regret)

    def worst(node):
        """largest predicted log2 regret of the leaves under node, and of node as one leaf"""
        now = max(float(R[l.idx, l.alg].max()) for l in node.leaves())
        return now, float(R[node.idx, node.alg].max())
    while root.split is not None:
        cand = []
        for node in internal(root):
            now, merged = worst(node)
            if merged > cap and merged > now + 1e-12:
                continue
            nl = len(node.leaves())
            cand.append(((node.cost - node.subtree_cost()) / (nl - 1), id(node), node))
        if not cand:
            break
        _, _, weakest = min(cand)
        keep = weakest.split
        weakest.split = None
        if gm() > full * (1 + tol):
            weakest.split = keep           # one step too far: undo and stop
            break
        path.append((len(root.leaves()), gm()))
    return root, path


def select(root, x):
    """Algorithm index chosen for one point (dict of feature values)."""
    node = root
    while node.split is not None:
        f, t, left, right, _ = node.split
        node = left if x[f] <= t else right
    return node.alg


def json_tree(root, alg_names, leaf):
    """CSEL JSON object for the tree. alg_names: index -> algorithm name;
    leaf(name) -> (key, params) of an algorithm leaf, or a string such as
    "call=<subtree>" that stands for the leaf itself."""
    def obj(node):
        out = {}
        cur = node
        while cur.split is not None:
            f, t, left, right, _ = cur.split
            if f == "pof2":
                cond = "!pof2"          # pof2 is 0/1, so the split's left side (<= 0) is "not pof2"
            else:
                cond = "%s(%d)" % (CSEL_NAME[f], int(t))
            if cond in out:             # cannot be reached again on this chain
                cur = right
                continue
            out[cond] = obj(left)
            cur = right
        lf = leaf(alg_names[cur.alg])
        val = lf if isinstance(lf, str) else {lf[0]: lf[1]}
        if not out:
            return val
        out["any"] = val
        return out
    return obj(root)


def json_choice(obj, x, trees=None):
    """The algorithm a CSEL subtree (as json_tree writes it) chooses at one
    point, x = {"p": comm_size, "m": message size, "N": nodes, "ppn": ranks
    per node, "pof2": 0/1, "cond": {restrictive condition: bool}}: siblings
    are tried in order, the first true condition is followed, "any" always
    is; "call=NAME" continues in trees["NAME"] (a file's subtrees, keys
    without "name="). Returns the leaf's key after "algorithm=", e.g.
    "MPIR_Bcast_intra_binomial"."""
    feature = {"comm_size": "p", "avg_msg_size": "m", "num_nodes": "N", "ppn": "ppn"}
    while True:
        if isinstance(obj, str):
            if not obj.startswith("call="):
                raise ValueError("leaf %r" % obj)
            obj = trees[obj[len("call="):]]
            continue
        for key, sub in obj.items():
            if key.startswith("algorithm="):
                return key[len("algorithm="):]
            if key == "any":
                ok = True
            elif key in ("pof2", "!pof2"):
                ok = (x["pof2"] > 0.5) == (key == "pof2")
            elif "(" not in key:
                # a restrictive condition; an old-style "!c" restriction is a condition of its own
                ok = x["cond"][key] if key in x["cond"] else x["cond"][key.lstrip("!")] != key.startswith("!")
            else:
                name, t = key.rstrip(")").split("(")
                ok = x[feature[name]] <= int(t)
            if ok:
                obj = sub
                break
        else:
            raise ValueError("no sibling matched (an object must end with \"any\")")
