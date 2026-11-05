"""A model grown over job sizes (docs/usage.md, section 6.2): each job adds only the layouts
its larger allocation newly reaches, with stages in order, growth anchors beyond the model, and
colltree --guard-above auto. On recorded data, with --replay-nodes standing for the allocation;
needs pandas and replay data (replay/README.md); skipped without data."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
COLLMODEL = os.path.join(REPO, "bin", "collmodel")
COLLTREE = os.path.join(REPO, "bin", "colltree")
ALGS = "binomial,scatter_ring_allgather,tree,circ_graph"
ALIAS = ["--replay-alias", "release_gather=auto@single", "--replay-alias", "smp=scatter_ring_allgather@hier"]

from csel_tuner.model import Model  # noqa: E402
from test_extend import have_data, true_regret  # noqa: E402
from test_growth_anchors import load_collmodel  # noqa: E402
from test_restrict import CATALOG  # noqa: E402


def run(*args, check=True):
    return subprocess.run([sys.executable, COLLMODEL, "--replay", *args], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def tree(*args):
    return subprocess.run([sys.executable, COLLTREE, *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def lays_of(ns, m):
    """The NxPPN list of an allocation's layouts with these N (their ppn from the model)."""
    return ",".join(f"{n}x{q}" for n in ns for q in sorted({q for _, q in m.layouts}))


def fitted(m):
    return [r for r in m.measurements if not r.get("anchor") and not r.get("growth")]


def test_grow_keeps_small_layouts_and_regret():
    """8, then 16, then 32 nodes: the layouts grow, nothing below is measured again, and the
    regret stays near a straight 32-node build at less cost."""
    if not have_data():
        print("skipped: no replay data")
        return
    with tempfile.TemporaryDirectory() as d:
        m8, m16, m32, once = (os.path.join(d, x + ".json") for x in ("m8", "m16", "m32", "once"))
        run("--algs", ALGS, "--replay-nodes", "8", "--out", m8)
        a = Model.load(m8)
        assert max(n for n, _ in a.layouts) == 8 and a.domain["N"][1] == 8
        r = run("--model", m8, "--layouts", lays_of([12, 16], a), "--replay-nodes", "16", "--out", m16)
        b = Model.load(m16)
        assert max(n for n, _ in b.layouts) == 16 and b.domain["N"][1] == 16
        assert b.runs[-1]["anchors"]["points"] > 0 and "--layouts" not in r.stderr
        old = {(x["alg"], x["N"], x["ppn"], x["m"]) for x in fitted(a)}
        new = [x for x in fitted(b) if (x["alg"], x["N"], x["ppn"], x["m"]) not in old]
        assert new and all(x["N"] in (12, 16) for x in new)
        assert len(fitted(b)) - len(fitted(a)) == len(new)
        for alg in a.curves:
            for lay in a.curves[alg]:
                assert np.array_equal(a.curves[alg][lay], b.curves[alg][lay])
        run("--model", m16, "--layouts", lays_of([24, 32], a), "--replay-nodes", "32", "--out", m32)
        c = Model.load(m32)
        assert max(n for n, _ in c.measured_layouts()) == 32
        run("--algs", ALGS, "--out", once)
        grown, straight = true_regret(m32), true_regret(once)
        cost = lambda m: sum(x.get("node_h") or 0 for x in m.runs)
        assert [x["mode"] for x in c.runs] == ["create", "add", "add"]
        print(f"grown 8/16/32: regret {grown:.4f}, {cost(c):.2f} node-h; "
              f"straight 32: regret {straight:.4f}, {cost(Model.load(once)):.2f} node-h")
        assert grown < straight * 1.05


def test_allocation_checks():
    """Layouts beyond the job's allocation, and a job smaller than the model, are refused."""
    if not have_data():
        print("skipped: no replay data")
        return
    with tempfile.TemporaryDirectory() as d:
        m8, m16 = (os.path.join(d, x + ".json") for x in ("m8", "m16"))
        run("--algs", "binomial,tree", "--replay-nodes", "8", "--budget-node-h", "0.1", "--out", m8)
        r = run("--model", m8, "--layouts", "32x8", "--replay-nodes", "16", "--out", m16, check=False)
        assert r.returncode and "not an allowed layout of this allocation" in r.stderr
        run("--model", m8, "--layouts", "16x8", "--replay-nodes", "16", "--out", m16)
        r = run("--model", m16, "--algs", "circ_graph", "--replay-nodes", "8",
                "--out", os.path.join(d, "x.json"), check=False)
        assert r.returncode and "not all in this allocation's" in r.stderr
        # cut short by the budget: a new layout where nothing was measured stays out of the model
        cut = os.path.join(d, "cut.json")
        run("--model", m8, "--layouts", "12x8,16x8,16x16", "--replay-nodes", "16", "--budget-node-h", "0",
            "--out", cut)
        assert Model.load(cut).layouts == Model.load(m8).layouts


def test_stages_at_new_layouts():
    """--layouts with --max-stage measures plain algorithms first; the compositional one comes in
    its own run under its inner selection, without touching its curves at the earlier layouts."""
    if not have_data():
        print("skipped: no replay data")
        return
    with tempfile.TemporaryDirectory() as d:
        cat = os.path.join(d, "coll_algorithms.txt")
        with open(cat, "w") as f:
            f.write(CATALOG)
        common = ["--catalog", cat, *ALIAS]
        p = lambda x: os.path.join(d, x)
        run("--algs", "all", *common, "--replay-nodes", "8", "--budget-node-h", "0.3", "--out", p("s1.json"))
        tree(p("s1.json"), "--out", p("t1.json"))
        run("--model", p("s1.json"), "--algs", "smp", *common, "--inner-selection", p("t1.json"),
            "--replay-nodes", "8", "--out", p("s2.json"))
        a = Model.load(p("s2.json"))
        lay = (16, 8)
        # stage 1 at the new layout: no compositional algorithm
        run("--model", p("s2.json"), *common, "--layouts", "16x8", "--max-stage", "1", "--inner-selection", p("t1.json"),
            "--replay-nodes", "16", "--out", p("b1.json"))
        b = Model.load(p("b1.json"))
        assert lay in b.curves["binomial"] and lay not in b.curves["smp"]
        # the compositional one, under a later tree: only there; its earlier curves stay, under their inner
        tree(p("b1.json"), "--out", p("t2.json"))
        r = run("--model", p("b1.json"), "--algs", "smp", *common, "--layouts", "16x8",
                "--inner-selection", p("t2.json"), "--replay-nodes", "16", "--out", p("b2.json"))
        assert "its curves are replaced" not in r.stderr
        c = Model.load(p("b2.json"))
        assert lay in c.curves["smp"]
        early = next(iter(a.curves["smp"]))
        assert np.array_equal(a.curves["smp"][early], c.curves["smp"][early])
        assert c.inner_at("smp", early) == a.fit_inner["smp"]
        assert c.inner_at("smp", lay) != a.fit_inner["smp"]
        assert c.inner_files[c.inner_at("smp", lay)] == open(p("t2.json")).read()
        # colltree reads the model, the stale check per inner selection included
        tree(p("b2.json"), "--out", p("t3.json"))
        assert "name=bcast-intra-0" in json.load(open(p("t3.json")))
        # without an inner selection the compositional one is left out, and says so
        r = run("--model", p("b2.json"), *common, "--layouts", "12x8", "--replay-nodes", "16", "--out", p("b3.json"))
        assert (12, 8) in Model.load(p("b3.json")).curves["binomial"] and (12, 8) not in Model.load(p("b3.json")).curves["smp"]
        assert "smp: compositional, not measured at --layouts without --inner-selection" in r.stderr


def test_growth_beyond_the_model():
    """Growth anchors at N the model has no layouts for: measured and precheck recorded, the
    model's layouts and curves unchanged; a top-up does not repeat them."""
    if not have_data():
        print("skipped: no replay data")
        return
    with tempfile.TemporaryDirectory() as d:
        cat = os.path.join(d, "coll_algorithms.txt")
        with open(cat, "w") as f:
            f.write(CATALOG)
        common = ["--catalog", cat, *ALIAS]
        m8, g, g2 = (os.path.join(d, x + ".json") for x in ("m8", "g", "g2"))
        run("--algs", "all", *common, "--replay-nodes", "8", "--budget-node-h", "0.3", "--growth-anchors", "--out", m8)
        a = Model.load(m8)
        run("--model", m8, *common, "--growth-anchors", "--growth-n", "32", "--growth-sizes", "1k,4k,16k",
            "--replay-nodes", "32", "--out", g)
        b = Model.load(g)
        assert b.layouts == a.layouts and b.measured_layouts() == a.measured_layouts()
        for alg in a.curves:
            for lay in a.curves[alg]:
                assert np.array_equal(a.curves[alg][lay], b.curves[alg][lay])
        far = {(x["alg"], x["N"]) for x in b.measurements if x.get("growth") and x["N"] == 32}
        assert ("binomial", 32) in far and ("release_gather", 32) not in far     # single-node only: its precheck fails
        assert any(p["N"] == 32 and p["alg"] == "binomial" for p in b.prechecks)
        run("--model", g, *common, "--growth-anchors", "--growth-n", "32", "--growth-sizes", "1k,4k,16k",
            "--replay-nodes", "32", "--out", g2)
        assert Model.load(g2).runs[-1]["growth_anchors"]["measured"] == 0
        # the guard's classes see them
        tree(g, "--out", os.path.join(d, "t.json"), "--guard-above", "auto")


def test_over_limit_points_left_out():
    """A p-block collective's anchors beyond the server's buffers are left out and counted."""
    cm = load_collmodel()

    class A:
        growth_n, growth_ppn, growth_sizes, catalog, replay = "2,4", "1", "1k,4k", None, False
        over_limit = staticmethod(lambda p, m: m * p > 8192)
    m = Model("allgather", [1024, 4096], [(1, 2), (2, 1), (4, 1)], {"N": [1, 4], "ppn": [1, 1], "m": [1024, 4096]})
    m.algorithms = {"x": {"leaf": None, "restrictions": [], "mask": 0, "inner": []}}
    sent = []
    rec = cm.growth_anchors(A, m, lambda n, q, alg, ms, tag=None: sent.append((n, ms)), None, lambda: None,
                            lambda: None, lambda: 0.0)
    assert sent == [(2, [1024, 4096]), (4, [1024])] and rec["over_limit"] == 1


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
