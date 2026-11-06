"""Builder v3: a decision tree (MPICH collective-selection JSON) from an optimum model.

    t = fit_tree(features, cost, names)       # cost: predicted log2 T, points x algorithms
    t = prune_to(t, tol=0.005)
    json_tree(t, ...)                         # a CSEL subtree, e.g. "name=bcast-intra-auto"

Input is the v2 model (multi.OptimumModel, or a unit of a collmodel/5 file): the
predicted time of every algorithm at every point of a grid, all allowed
layouts x all sizes. The tree is fitted to minimize regret, not
misclassification: with R[i, a] = predicted log2 T[i, a] - min_b log2 T[i, b],
a leaf picks the algorithm with the smallest sum of R over its points, and a
split is the threshold that most reduces that sum. Two algorithms within a
few percent of each other cost almost nothing to confuse, so the tree spends
its splits where a wrong choice is expensive.

1. Features. Today's CSEL conditions: comm_size(t) (p <= t),
   msg_size(t) (m <= t: the message size, or one rank's block; an MPICH
   before it has avg_msg_size, json_tree's `names`) and pof2. With
   `node_aware`, also num_nodes(t) and ppn(t) (ranks per node, averaged over
   the nodes), conditions MPICH has since pmodels/mpich#8048.
2. Growth: best-first, every candidate threshold of every feature (the
   distinct values on the grid, prefix sums per algorithm), choosing the
   split that most reduces the squared deviation of the regret vectors
   (a multi-output regression tree; a regret criterion is myopic, see
   best_split), until the reduction is below `min_gain` per point or there
   are `max_leaves` leaves. Leaves pick the least-regret algorithm.
3. Pruning: weakest link first (the split whose removal adds the least
   regret per leaf removed), down to the smallest tree whose predicted
   geomean regret is within `tol` of the full tree's (a depth limit,
   at most that many nested condition levels in the JSON, cond_depth, is
   applied while growing: fit_tree's max_depth), never merging a split
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

CSEL_NAME = {"p": "comm_size", "m": "msg_size", "N": "num_nodes", "ppn": "ppn", "pof2": "pof2"}


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


def fit_tree(X, L, names, min_gain=1e-6, max_leaves=1024, max_depth=None):
    """X: {feature: values per point}; L: predicted log2 T, points x algorithms;
    names: the features splits may use; max_depth: at most this many nested condition levels
    in the JSON (cond_depth): a split at a node with `lev` levels above it puts its condition at
    level lev + 1, its left child inside it (lev + 1), its right child in the same chain (lev)."""
    R = L - L.min(axis=1, keepdims=True)
    n = len(R)
    root = Node(np.arange(n), R)
    root.lev = 0
    heap = []

    def push(node):
        if max_depth is not None and node.lev >= max_depth:
            return
        s = best_split(node, X, R, names)
        if s and s[0] > min_gain * n:
            heapq.heappush(heap, (-s[0], id(node), node, s))
    push(root)
    nleaves = 1
    while heap and nleaves < max_leaves:
        _, _, node, (gain, f, t, li, ri) = heapq.heappop(heap)
        node.split = [f, t, Node(li, R), Node(ri, R), gain]
        node.split[2].lev, node.split[3].lev = node.lev + 1, node.lev
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


def prune_to(root, tol=0.005, max_regret=1.5, leaf_cost=None):
    """Weakest-link pruning to the smallest tree whose predicted geomean
    regret is within tol of the full tree's. A split is never merged if the
    merged leaf would give some point a predicted regret above max_regret
    (unless that point already has it), so the few expensive mistakes that
    a geomean hides are kept out. Returns the pruned root (in place) and the
    path [(leaves, geomean regret)]. With leaf_cost (the mean log2 regret over the tree's points
    that one leaf must buy) pruning stops at that price instead of at tol, so subtrees pruned at
    one price spend their leaves where they buy the most."""
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
        price, _, weakest = min(cand)
        if leaf_cost is not None and price / n > leaf_cost:
            break
        keep = weakest.split
        weakest.split = None
        if leaf_cost is None and gm() > full * (1 + tol):
            weakest.split = keep           # one step too far: undo and stop
            break
        path.append((len(root.leaves()), gm()))
    return root, path


def cond_depth(node):
    """Nesting of condition levels in the JSON that json_tree writes for node: a split's left
    branch nests one level deeper, its right branch continues the same level (a chain of
    splits becomes one object of sibling conditions); a leaf adds none."""
    if node.split is None:
        return 0
    left, right = node.split[2], node.split[3]
    return max(1 + cond_depth(left), cond_depth(right) if right.split is not None else 1)


def regret_stats(root):
    """(geomean, p90, max) of the predicted regret (ratios) over the fitted points."""
    r = np.concatenate([root.R[l.idx, l.alg] for l in root.leaves()])
    return float(2 ** r.mean()), float(2 ** np.quantile(r, 0.9)), float(2 ** r.max())


def select(root, x):
    """Algorithm index chosen for one point (dict of feature values)."""
    node = root
    while node.split is not None:
        f, t, left, right, _ = node.split
        node = left if x[f] <= t else right
    return node.alg


def json_tree(root, alg_names, leaf, names=CSEL_NAME):
    """CSEL JSON object for the tree. alg_names: index -> algorithm name;
    leaf(name) -> (key, params) of an algorithm leaf, or a string such as
    "call=<subtree>" that stands for the leaf itself; names: feature -> CSEL
    condition."""
    def obj(node):
        out = {}
        cur = node
        while cur.split is not None:
            f, t, left, right, _ = cur.split
            if f == "pof2":
                cond = "!pof2"          # pof2 is 0/1, so the split's left side (<= 0) is "not pof2"
            else:
                cond = "%s(%d)" % (names[f], int(t))
            if cond in out:             # cannot be reached again on this chain
                cur = right
                continue
            val = obj(left)
            # the previous sibling, a lower threshold of the same feature with the same value, is
            # redundant: whatever matches it matches this one too
            if out and f != "pof2":
                last = next(reversed(out))
                if last.startswith(names[f] + "(") and out[last] == val:
                    del out[last]
            out[cond] = val
            cur = right
        lf = leaf(alg_names[cur.alg])
        val = lf if isinstance(lf, str) else {lf[0]: lf[1]}
        if not out:
            return val
        out["any"] = val
        return out
    return obj(root)


def set_masks(trees, coll):
    """{(mask, gpu): name} of the restriction sets of `coll` (e.g. "bcast-intra") among a file's
    subtree names (keys without "name="): <coll>-<mask in hex>, or <coll>-<mask>-gpu."""
    out = {}
    pre = coll + "-"
    for name in trees:
        if not name.startswith(pre):
            continue
        rest = name[len(pre):]
        gpu = rest.endswith("-gpu")
        h = rest[:-len("-gpu")] if gpu else rest
        if h and all(c in "0123456789abcdef" for c in h) and (h == "0" or h[0] != "0"):
            out[(int(h, 16), gpu)] = name
    return out


def auto_lookup(trees, coll, m, v, bits):
    """The set MPICH's dispatch continues in (pmodels/mpich#8115, MPII_Csel_dispatch): m is the
    restrictions that hold, local ones already left out where a _coll one holds, and v the _coll
    restrictions the search's dispatch(<mask>)es left out, which select the variants (-gpu for
    gpu_coll). A local set (a mask with a <name>_local restriction) whose mask is within m comes
    first, then a set whose mask is within m and whose variants are within v. Within each, the most
    variants, then the most restrictions, then the larger variant mask, then the larger mask wins."""
    local = sum(1 << k for k, b in enumerate(bits) if b.endswith("_local"))
    gpu_bit = 1 << bits.index("gpu_coll") if "gpu_coll" in bits else 0
    sets = set_masks(trees, coll)

    def order(c):
        mask, gpu = c
        vm = gpu_bit if gpu else 0
        return (-bin(vm).count("1"), -bin(mask).count("1"), -vm, -mask)
    stages = [[c for c in sets if not c[1] and c[0] & local and c[0] & ~m == 0],
              [c for c in sets if not c[0] & local and c[0] & ~m == 0 and (gpu_bit if c[1] else 0) & ~v == 0]]
    for cands in stages:
        if cands:
            return sets[min(cands, key=order)]
    raise ValueError("%s: no set matches" % coll)


def file_choice(trees, coll, x, bits):
    """The algorithm a whole file (its subtrees by name, keys without "name=") selects for a call
    of `coll` (e.g. "bcast-intra"): from the legacy <coll>-auto dispatch if the file has one, else
    from MPICH's built-in dispatch over the reserved set names."""
    if coll + "-auto" in trees:
        return json_choice(trees[coll + "-auto"], x, trees, bits)
    return json_choice("dispatch", x, trees, bits, coll=coll)


def json_choice(obj, x, trees=None, bits=None, coll=None):
    """The algorithm a CSEL subtree (as json_tree writes it) chooses at one
    point, x = {"p": comm_size, "m": message size, "N": nodes, "ppn": ranks
    per node, "pof2": 0/1, "cond": {restrictive condition: bool}}: siblings
    are tried in order, the first true condition is followed, "any" always
    is; "call=NAME" continues in trees["NAME"] (a file's subtrees, keys
    without "name="); "dispatch(MASK)", and "dispatch" where a selection
    starts, continue in the restriction set of `coll` (e.g. "bcast-intra")
    that auto_lookup finds,
    the restrictions in MASK left out (cumulative within one search; a
    _coll one left out selects its variant, e.g. -gpu); "dispatch=PREFIX(BITS)" (the older format)
    continues in PREFIX-<the bits whose condition holds, in hex>, or its
    largest subset that exists (bits: the restriction names, bit k =
    bits[k]). Returns the leaf's key after "algorithm=", e.g.
    "MPIR_Bcast_intra_binomial"."""
    m = None
    feature = {"comm_size": "p", "msg_size": "m", "avg_msg_size": "m", "num_nodes": "N", "ppn": "ppn"}
    while True:
        if isinstance(obj, str) and obj.startswith("dispatch="):
            prefix, mask = obj[len("dispatch="):].rstrip(")").split("(")
            mask = int(mask, 0)
            m = sum(1 << k for k, b in enumerate(bits) if mask >> k & 1 and x["cond"].get(b, False))
            subs = [s for s in range(m + 1) if s & ~m == 0 and "%s-%x" % (prefix, s) in trees]
            obj = trees["%s-%x" % (prefix, max(subs, key=lambda s: (bin(s).count("1"), -s)))]
            continue
        if isinstance(obj, str) and coll is not None and (obj == "dispatch" or obj.startswith("dispatch(")):
            coll_bits = sum(1 << k for k, b in enumerate(bits) if b.endswith("_coll"))
            if m is None:
                # MPICH tests only the restrictions in the masks of the collective's sets
                aggregate = 0
                for mask, gpu in set_masks(trees, coll):
                    aggregate |= mask
                m = sum(1 << k for k, b in enumerate(bits) if x["cond"].get(b, False)) & aggregate
                if m & coll_bits:       # a _coll and a _local restriction exclude each other
                    m &= ~sum(1 << k for k, b in enumerate(bits) if b.endswith("_local"))
                v = 0
            if obj != "dispatch":
                drop = int(obj[len("dispatch("):-1], 16)
                v |= m & drop & coll_bits
                m &= ~drop
            obj = trees[auto_lookup(trees, coll, m, v, bits)]
            continue
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
