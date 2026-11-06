"""bin/colltree's default output, MPICH's reserved set names (--format reserved): the sets only,
reached by MPICH's built-in dispatch; fallbacks and tie aliases as dispatch(<mask>); the -gpu
variant and the local set. Each file selects what the legacy layout selects, at every point."""
import io
import itertools
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from importlib.machinery import SourceFileLoader

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

from csel_tuner.model import Model  # noqa: E402
from csel_tuner.tree import file_choice, json_choice, set_masks  # noqa: E402
from test_gpu import BITS as GPU_BITS, LAYOUTS, SIZES, alg, curve, gpu_model, host_model  # noqa: E402


def run(paths, *argv):
    """{name: tree} (keys without "name=") of colltree's output in one format."""
    ct = SourceFileLoader("colltree", os.path.join(REPO, "bin", "colltree")).load_module()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.json")
        with redirect_stdout(io.StringIO()):
            ct.main(list(paths) + ["--out", out] + list(argv))
        with open(out) as f:
            return {k[len("name="):]: v for k, v in json.load(f).items()}


def legacy_choice(trees, coll, x, bits):
    """What a legacy file selects: from its name=main, else MPICH's built-in auto."""
    t = dict(trees)
    t.setdefault("auto", t[coll + "-auto"])
    start = t.get("main", "call=auto")
    if isinstance(start, dict):         # main's collective= dispatch, which json_choice does not take
        start = start.get("collective=" + coll, start["any"])
    return json_choice(start, x, t, bits)


def calls(obj):
    if isinstance(obj, str):
        return [obj] if obj.startswith(("call=", "dispatch")) else []
    return [c for v in obj.values() for c in calls(v)]


def check_mpich_rules(trees, coll, bits):
    """The load-time rules of MPICH's reserved names: no name=main, auto, -auto or dispatch; call=
    names free subtrees only; in a set, every dispatch(r) leaves out one of its restrictions, and in
    the local set only local ones, its own among them."""
    local = sum(1 << k for k, b in enumerate(bits) if b.endswith("_local"))
    sets = {name: mask for (mask, gpu), name in set_masks(trees, coll).items()}
    assert set(trees) == set(sets), sorted(set(trees) - set(sets))
    for name, mask in sets.items():
        for c in calls(trees[name]):
            assert c.startswith("dispatch("), (name, c)
            r = int(c[len("dispatch("):-1], 16)
            assert r & mask, (name, c)
            if mask & local:
                assert r & local and not r & ~local, (name, c)


def points(bits, layouts, sizes):
    for lay, m, conds in itertools.product(layouts, sizes, itertools.product((False, True), repeat=len(bits))):
        p = lay[0] * lay[1]
        yield {"p": p, "m": m, "N": lay[0], "ppn": lay[1], "pof2": float(p & (p - 1) == 0),
               "cond": dict(zip(bits, conds))}


def test_gpu_reserved():
    """Host and gpu models: bcast-intra-0, the -gpu variant and the local set (gpu_local, bit 1)."""
    with tempfile.TemporaryDirectory() as tmp:
        h, g = os.path.join(tmp, "h.json"), os.path.join(tmp, "g.json")
        host_model().save(h)
        gpu_model().save(g)
        new = run([h, g], "--node-aware")
        old = run([h, g], "--node-aware", "--format", "legacy")
    assert sorted(new) == ["bcast-intra-0", "bcast-intra-0-gpu", "bcast-intra-2", "bcast-intra-4"], sorted(new)
    check_mpich_rules(new, "bcast-intra", GPU_BITS)
    local = new["bcast-intra-2"]
    assert set(calls(local)) == {"dispatch(2)"}, local
    # the set of gpu_coll (bit 2) leaves it out, which selects the -gpu variant
    assert new["bcast-intra-4"] == "dispatch(4)", new["bcast-intra-4"]
    # a gpu_coll call never swaps, so the -gpu sets choose without buffer_swap
    assert "algorithm=MPIR_Coll_buffer_swap" not in json.dumps(new["bcast-intra-0-gpu"])
    for x in points(GPU_BITS, LAYOUTS, SIZES):
        b = file_choice(new, "bcast-intra", x, GPU_BITS)
        if x["cond"]["gpu_coll"]:
            # legacy lets gpu_coll calls swap; here they take the -gpu set
            assert b == json_choice(new["bcast-intra-0-gpu"], x, new, GPU_BITS), (x, b)
            continue
        a = legacy_choice(old, "bcast-intra", x, GPU_BITS)
        assert a == b, (x, a, b)


def restricted_model():
    """bcast with two restrictions that never hold together: smp (hierarchical) on several nodes
    with several ranks each, rg (release_gather) on one node. Sets 0, 1 and 2 tie at 3."""
    bits = ["hierarchical", "release_gather"]
    m = Model("bcast", SIZES, LAYOUTS, {"N": [1, 4], "ppn": [1, 8], "m": [SIZES[0], SIZES[-1]]}, bits,
              {"binomial": alg(0), "scatter": alg(0), "smp": alg(1), "rg": alg(2)}, mem="host")
    m.masks = {lay: (1 if lay[0] > 1 and lay[1] > 1 else 2 if lay[0] == 1 else 0) for lay in LAYOUTS}
    for name, (lat, bw, mask) in {"binomial": (1.0, 1e3, 0), "scatter": (3.0, 1e4, 0),
                                  "smp": (0.5, 2e3, 1), "rg": (0.2, 5e2, 2)}.items():
        for lay in LAYOUTS:
            if mask & ~m.masks[lay] == 0:
                m.set_curve(name, lay, curve(lay, lat, bw))
    return m, bits


def test_tie_alias():
    m, bits = restricted_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        m.save(path)
        new = run([path], "--node-aware", "--min-fallback-algs", "1")
        old = run([path], "--node-aware", "--min-fallback-algs", "1", "--format", "legacy")
    assert {"bcast-intra-0", "bcast-intra-1", "bcast-intra-2", "bcast-intra-3"} <= set(new), sorted(new)
    # the union of the tied sets goes where the legacy dispatch went: the lower mask, 1
    assert new["bcast-intra-3"] == "dispatch(2)", new["bcast-intra-3"]
    check_mpich_rules(new, "bcast-intra", bits)
    for x in points(bits, LAYOUTS, SIZES):
        assert legacy_choice(old, "bcast-intra", x, bits) == file_choice(new, "bcast-intra", x, bits), x


def test_fallback_mask():
    """A fallback leaves out every restriction the subset lacks, so the lookup lands on the subset
    whatever else holds: in a file with sets 0, 1, 2 and 3, set 3 falling back to 1 is auto(2)."""
    from csel_tuner.tree import auto_lookup
    trees = {"c-intra-0": {}, "c-intra-1": {}, "c-intra-2": {}, "c-intra-3": {}}
    bits = ["a", "b"]
    for m in range(4):
        assert auto_lookup(trees, "c-intra", m & ~2, 0, bits) == "c-intra-%x" % (m & 1)
    # of two sets of the same size, the larger mask
    assert auto_lookup({"c-intra-0": {}, "c-intra-1": {}, "c-intra-2": {}}, "c-intra", 3, 0, bits) == "c-intra-2"


def test_variant():
    """MPICH takes a -gpu set only after a dispatch that leaves gpu_coll out (here the set
    c-intra-4), and falls back to the host sets; a gpu_coll call never takes the local set."""
    bits = ["a", "gpu_local", "gpu_coll"]
    leaf = {k: {"algorithm=" + k: {}} for k in ("h0", "h1", "g0", "swap")}
    trees = {"c-intra-0": leaf["h0"], "c-intra-1": leaf["h1"], "c-intra-0-gpu": leaf["g0"],
             "c-intra-2": leaf["swap"], "c-intra-4": "dispatch(4)"}

    def pick(t, **cond):
        return file_choice(t, "c-intra", {"cond": dict({b: False for b in bits}, **cond)}, bits)
    assert pick(trees) == "h0"
    assert pick(trees, a=True) == "h1"
    assert pick(trees, gpu_local=True) == "swap"
    assert pick(trees, gpu_local=True, gpu_coll=True) == "g0"
    # without the set of gpu_coll the variant is never selected, and the local set is taken
    del trees["c-intra-4"]
    assert pick(trees, gpu_local=True, gpu_coll=True) == "swap"


if __name__ == "__main__":
    np.seterr(all="ignore")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
