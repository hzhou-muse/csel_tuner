"""bin/colltree's options for a small default file: a joint fit of two models (--also), pruning
by price (--leaf-cost), thresholds on a grid (--msg-grid, --comm-grid) and the compact layout."""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from importlib.machinery import SourceFileLoader

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from csel_tuner.model import Model  # noqa: E402
from csel_tuner.tree import fit_tree, prune_to  # noqa: E402

ct = SourceFileLoader("colltree", os.path.join(REPO, "bin", "colltree")).load_module()
LAYOUTS = [(1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4)]
SIZES = [2 ** k for k in range(0, 21)]


def model(curves, layouts=LAYOUTS):
    """A bcast model; curves: {algorithm: (latency, bandwidth)}."""
    algs = {a: {"leaf": None, "restrictions": [], "mask": 0, "inner": []} for a in curves}
    m = Model("bcast", SIZES, layouts, {"N": [1, 2], "ppn": [1, 8], "m": [SIZES[0], SIZES[-1]]}, [], algs)
    m.masks = {lay: 0 for lay in layouts}
    for a, (lat, bw) in curves.items():
        for lay in layouts:
            m.set_curve(a, lay, np.log2(lat + np.array(SIZES, float) / bw))
    return m


def build(models, *argv):
    """The subtrees colltree writes for model files of these models, and its output."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, m in enumerate(models):
            paths.append(os.path.join(d, "m%d.json" % i))
            m.save(paths[-1])
        out = os.path.join(d, "out.json")
        args = [paths[0]] + [x for p in paths[1:] for x in ("--also", p)] + ["--out", out] + list(argv)
        with redirect_stdout(io.StringIO()):
            ct.main(args)
        with open(out) as f:
            return json.load(f)


def leaves(o):
    if isinstance(o, str):
        return []
    if any(k.startswith("algorithm=") for k in o):
        return [k[len("algorithm="):] for k in o]
    return [x for v in o.values() for x in leaves(v)]


def test_leaf_cost_prices_a_split():
    """Half the points gain 1 (log2) from a split: 0.5 of mean regret, for one leaf."""
    X = {"m": np.arange(100.0)}
    L = np.zeros((100, 2))
    L[:50, 0], L[50:, 1] = 1.0, 1.0
    assert len(prune_to(fit_tree(X, L, ["m"]), max_regret=4, leaf_cost=0.4)[0].leaves()) == 2
    assert len(prune_to(fit_tree(X, L, ["m"]), max_regret=4, leaf_cost=0.6)[0].leaves()) == 1


def test_leaf_cost_keeps_max_regret():
    """A split too cheap for the price stays if merging it passes --max-regret."""
    X = {"m": np.arange(100.0)}
    L = np.zeros((100, 2))
    L[0, 0], L[1:, 1] = 2.0, 0.05        # as one leaf: the first algorithm, 4x at point 0
    assert len(prune_to(fit_tree(X, L, ["m"]), leaf_cost=0.5, max_regret=1.5)[0].leaves()) == 2
    assert len(prune_to(fit_tree(X, L, ["m"]), leaf_cost=0.5, max_regret=8)[0].leaves()) == 1


def test_on_grid():
    v = np.array([0, 1, 2, 3, 4, 5, 16, 17, 2 ** 20])
    assert list(ct.on_grid(v, 2)) == [1, 1, 2, 4, 4, 8, 16, 32, 2 ** 20]
    assert list(ct.on_grid(v, 4)) == [1, 1, 4, 4, 4, 16, 16, 64, 4 ** 10]


def thresholds(o, name):
    if not isinstance(o, dict):
        return []
    here = [int(k[len(name) + 1:-1]) for k in o if k.startswith(name + "(")]
    return here + [x for v in o.values() for x in thresholds(v, name)]


def test_msg_grid():
    """A crossover between grid values: the thresholds are powers of 4, also for the true sizes."""
    m = model({"short": (1.0, 1e3), "long": (50.0, 1e6)})
    assert any(t & (t - 1) or t % 3 != 1 for t in thresholds(build([m])["name=bcast-intra-0"], "msg_size"))
    got = thresholds(build([m], "--msg-grid", "4")["name=bcast-intra-0"], "msg_size")
    assert got and all(4 ** round(np.log(t) / np.log(4)) == t for t in got)


def test_also_fits_both():
    """a is a little faster on the first model and much slower on the other: alone the first
    model picks a, jointly b. Layouts and sizes of the two may differ."""
    one = model({"a": (1.0, 1e9), "b": (1.1, 1e9)})
    two = model({"a": (4.0, 1e9), "b": (1.0, 1e9)}, LAYOUTS[:3])
    two.sizes = SIZES
    assert leaves(build([one])["name=bcast-intra-0"]) == ["MPIR_Bcast_intra_a"]
    assert leaves(build([one, two])["name=bcast-intra-0"]) == ["MPIR_Bcast_intra_b"]


def test_compact_layout():
    subtree = {"comm_size(8)": {"msg_size(4)": {"algorithm=X": {"k": 2, "t": "kary"}},
                                "any": {"algorithm=Y": {}}},
               "any": "call=auto"}
    ct.COMPACT = False
    standard = ct.format_main(subtree, 4)
    ct.COMPACT = True
    try:
        compact = ct.format_main(subtree, 4)
    finally:
        ct.COMPACT = False
    assert json.loads(compact) == json.loads(standard) == subtree
    assert len(compact.splitlines()) < len(standard.splitlines())
    assert '"msg_size(4)": {"algorithm=X": {"k": 2, "t": "kary"}},' in compact


if __name__ == "__main__":
    test_leaf_cost_prices_a_split()
    test_leaf_cost_keeps_max_regret()
    test_on_grid()
    test_msg_grid()
    test_also_fits_both()
    test_compact_layout()
    print("ok")
