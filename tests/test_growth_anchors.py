"""bin/collmodel --growth-anchors: the fixed point set, its independence of
the fit, restrictions, top-up with --model, and --resume; on recorded data.
Needs pandas and replay data (replay/README.md); skipped without data."""
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
COLLMODEL = os.path.join(REPO, "bin", "collmodel")

from test_restrict import CATALOG  # noqa: E402

ALGS = "binomial,scatter_ring_allgather,tree"


def have_data():
    d = os.path.join(REPO, "replay", "data")
    return os.path.exists(os.path.join(d, "gt.csv.gz")) and os.path.exists(os.path.join(d, "sessions.csv.gz"))


def run(*args, check=True):
    return subprocess.run([sys.executable, COLLMODEL, "--replay", *args], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def unit(path):
    return json.load(open(path))["units"][0]


def growth(u):
    return [r for r in u["data"]["measurements"] if r.get("growth")]


def load_collmodel():
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("collmodel", COLLMODEL)
    spec = importlib.util.spec_from_loader("collmodel", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_point_set():
    """The default ladder, ppn classes and sizes, and the overrides; no data needed."""
    cm = load_collmodel()
    allowed = [(n, q) for n in cm.axis_values(32) for q in cm.axis_values(96) if n * q > 1]
    a = SimpleNamespace(growth_n=None, growth_ppn=None, growth_sizes=None)
    lays, ns, qs = cm.growth_layouts(a, allowed)
    assert ns == [2, 4, 8, 16, 32] and qs == [1, 8, 96]
    assert lays[:5] == [(2, 1), (4, 1), (8, 1), (16, 1), (32, 1)] and (1, 8) in lays and (1, 96) in lays
    assert len(lays) == 5 + 6 + 6
    sizes = cm.size_grid(1 << 26)
    assert cm.growth_sizes(a, sizes) == [1024, 4096, 16384, 1 << 22, 1 << 24]
    a = SimpleNamespace(growth_n="4,16", growth_ppn="2", growth_sizes="3k,1M")
    lays, ns, qs = cm.growth_layouts(a, allowed)
    assert lays == [(1, 2), (4, 2), (16, 2)]
    assert cm.growth_sizes(a, sizes) == [2896, 1 << 20]     # 3k snapped to the grid (4 per octave)
    assert cm.growth_sizes(a, [0]) == [0]


def test_independent_of_fit():
    """With anchors: the same sampling and curves as without, plus every algorithm at every point."""
    if not have_data():
        print("skipped: no replay data (replay/README.md)")
        return
    with tempfile.TemporaryDirectory() as d:
        g, n = os.path.join(d, "g.json"), os.path.join(d, "n.json")
        run("--algs", ALGS, "--budget-node-h", "0.3", "--growth-anchors", "--out", g)
        run("--algs", ALGS, "--budget-node-h", "0.3", "--out", n)
        a, b = unit(g), unit(n)
        assert a["fit"] == b["fit"]
        assert [r for r in a["data"]["measurements"] if not r.get("growth")] == b["data"]["measurements"]
        rec = a["runs"][-1]["growth_anchors"]
        assert rec["stop"] == "done" and rec["measured"] == rec["points"] == 17 * 5 * 3
        pts = {(r["alg"], r["N"], r["ppn"], r["m"]) for r in growth(a)}
        assert len(pts) == rec["points"] and {p[0] for p in pts} == set(ALGS.split(","))


def test_restrictions_and_top_up():
    """release_gather (single node only) is anchored only at N=1; a top-up adds the anchors to
    a model without them, and a second top-up measures nothing."""
    if not have_data():
        print("skipped: no replay data")
        return
    with tempfile.TemporaryDirectory() as d:
        cat = os.path.join(d, "coll_algorithms.txt")
        with open(cat, "w") as f:
            f.write(CATALOG)
        m0, m1, m2 = (os.path.join(d, x) for x in ("m0.json", "m1.json", "m2.json"))
        alias = ["--replay-alias", "release_gather=auto@single"]
        run("--algs", "all", "--catalog", cat, *alias, "--budget-node-h", "0.3", "--out", m0)
        assert not growth(unit(m0))
        run("--model", m0, "--growth-anchors", *alias, "--anchors", "0", "--out", m1)
        u = unit(m1)
        rg = {(r["N"], r["ppn"]) for r in growth(u) if r["alg"] == "release_gather"}
        assert rg == {(1, 8), (1, 96)}
        rec = u["runs"][-1]["growth_anchors"]
        assert rec["stop"] == "done" and u["runs"][-1]["complete"]
        assert {s["why"] for s in rec["skipped"] if s["alg"] == "release_gather"} == {"restriction"}
        assert unit(m0)["fit"] == u["fit"]
        run("--model", m1, "--growth-anchors", *alias, "--anchors", "0", "--out", m2)
        rec2 = unit(m2)["runs"][-1]["growth_anchors"]
        assert rec2["points"] == 0 and len(growth(unit(m2))) == len(growth(u))


def test_resume():
    """A journal cut in the anchor phase: --resume finishes it with the same records."""
    if not have_data():
        print("skipped: no replay data")
        return
    with tempfile.TemporaryDirectory() as d:
        full, part, log = (os.path.join(d, x) for x in ("full.json", "part.json", "j.log"))
        run("--algs", ALGS, "--budget-node-h", "0.3", "--growth-anchors", "--out", full, "--log", log)
        lines = open(log).read().splitlines(True)
        assert '"done": true' in lines[-1]
        cut = len(lines) - 1 - 100                     # the done record and the last 100 replies (anchors)
        assert all('"growth"' in x for x in lines[cut:-1])
        open(log, "w").writelines(lines[:cut])
        run("--algs", ALGS, "--budget-node-h", "0.3", "--growth-anchors", "--out", part, "--log", log, "--resume")
        a, b = unit(full), unit(part)
        key = lambda r: (r["alg"], r["N"], r["ppn"], r["m"])
        assert sorted(map(key, growth(a))) == sorted(map(key, growth(b)))
        assert a["fit"] == b["fit"]
        # the anchors' settings must not change on resume
        r = run("--algs", ALGS, "--budget-node-h", "0.3", "--growth-anchors", "--growth-ppn", "1",
                "--out", part, "--log", log, "--resume", check=False)
        assert r.returncode != 0 and "growth" in r.stderr


def test_growth_classes_synthetic():
    """growth.fit_exponent and classify on exact log p and p - 1 growth, with a single-node
    offset and per-size intercepts."""
    import numpy as np
    from csel_tuner.growth import classify, fit_exponent
    lays = [(1, 2), (1, 8), (1, 32)] + [(2, q) for q in (1, 2, 4, 8, 16, 32)]
    for f, want in ((lambda p: np.log2(p) + 1, "logarithmic"), (lambda p: p - 1 + 0.5, "linear")):
        rows = [(n, q, m, f(n * q) * m * (0.5 if n == 1 else 1.0)) for n, q in lays for m in (1024, 4096)]
        fit = fit_exponent(rows)
        assert classify(fit) == want, (want, fit)
    assert fit_exponent([(2, 1, 1024, 1.0), (2, 2, 1024, 2.0)]) is None     # too few layouts


def test_guard_on_replay():
    """colltree --guard-above: the classes from the anchors (ring linear in the latency band,
    binomial logarithmic), and with a lower threshold a guarded branch above P whose choice
    differs from the measured one."""
    if not have_data():
        print("skipped: no replay data")
        return
    from csel_tuner.growth import classes
    from csel_tuner.model import Model
    from csel_tuner.tree import json_choice
    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "m.json")
        run("--algs", "binomial,scatter_ring_allgather,circ_graph", "--budget-node-h", "0.3", "--growth-anchors",
            "--out", m)
        cls = classes(Model.load(m))
        assert cls["scatter_ring_allgather"]["lat"][0] == "linear" and cls["binomial"]["lat"][0] == "logarithmic"
        t = os.path.join(d, "t.json")
        r = subprocess.run([sys.executable, os.path.join(REPO, "bin", "colltree"), m, "--out", t, "--max-depth", "4",
                            "--guard-above", "768", "--linear-above", "0.45"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        tree = {k.split("=", 1)[1]: v for k, v in json.load(open(t)).items()}
        sub = tree["bcast-intra-0"]
        assert list(sub)[0] == "comm_size(768)", list(sub)
        assert "guard bcast-intra-0 above comm_size 768" in r.stdout + r.stderr
        x = lambda p, m: {"p": p, "m": m, "N": p, "ppn": 1, "pof2": 1.0, "cond": {}}
        assert json_choice(sub, x(4096, 64), tree) != json_choice(sub["comm_size(768)"], x(4096, 64), tree) or \
            json_choice(sub, x(4096, 1 << 20), tree) != json_choice(sub["comm_size(768)"], x(4096, 1 << 20), tree)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
