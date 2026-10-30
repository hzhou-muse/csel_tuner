"""bin/collmodel --model: adding an algorithm and layouts to a model
(docs/design.md, section 12), on recorded data. Needs pandas and replay data
(replay/README.md); skipped without data."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
COLLMODEL = os.path.join(REPO, "bin", "collmodel")

from csel_tuner.model import Model  # noqa: E402


def have_data():
    d = os.path.join(REPO, "replay", "data")
    return os.path.exists(os.path.join(d, "gt.csv.gz")) and os.path.exists(os.path.join(d, "sessions.csv.gz"))


def run(*args, check=True):
    return subprocess.run([sys.executable, COLLMODEL, "--replay", *args], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def true_regret(path):
    """Geomean ground-truth regret of the model's per-point selection over the whole recording."""
    sys.path.insert(0, os.path.join(REPO, "replay"))
    from common import FIXED, load_gt, table
    w = table(load_gt()).droplevel("p")
    m = Model.load(path)
    algs = [a for a in m.algorithms if a in m.curves]
    idx = {s: i for i, s in enumerate(m.sizes)}
    lr = []
    for (n, q), g in w.groupby(level=["N", "ppn"]):
        P = np.array([m.predict(a, n, q) for a in algs])
        for (_, _, size), row in g.iterrows():
            lr.append(np.log(row[algs[int(np.argmin(P[:, idx[int(size)]]))]] / row[FIXED].min()))
    return float(np.exp(np.mean(lr)))


def test_add_algorithm_matches_one_shot():
    """Four algorithms, then circ_graph added: about the regret of building all
    five at once, for the cost of the new measurements only."""
    if not have_data():
        print("skipped: no replay data (replay/README.md)")
        return
    four = "binomial,scatter_ring_allgather,tree,pipelined_tree"
    with tempfile.TemporaryDirectory() as d:
        m4, m5, once = (os.path.join(d, x) for x in ("m4.json", "m5.json", "once.json"))
        run("--algs", four, "--budget-node-h", "1", "--out", m4)
        run("--model", m4, "--algs", "circ_graph", "--out", m5)
        run("--algs", four + ",circ_graph", "--budget-node-h", "1", "--out", once)
        a, b = Model.load(m4), Model.load(m5)
        assert set(b.curves["circ_graph"]) == set(a.measured_layouts())
        added = b.runs[-1]
        assert added["mode"] == "add" and added["algs"] == ["circ_graph"] and added["anchors"]["points"] > 0
        assert abs(np.log(added["anchors"]["ratio_geomean"])) < 0.02     # replay noise is 1%
        # what the add measured: circ_graph, the anchors, and the old algorithms only at crossovers
        new = [r for r in b.measurements if r["job"] in added["jobs"]]
        assert sum(r["alg"] == "circ_graph" for r in new) > 0.6 * len(new)
        assert sum(1 for r in new if r.get("anchor")) == added["anchors"]["points"]
        # the earlier data is kept as it was
        assert b.measurements[:len(a.measurements)] == a.measurements
        r4, r5, r_once = true_regret(m4), true_regret(m5), true_regret(once)
        assert r5 < r4 and r5 < r_once * 1.02, (r4, r5, r_once)
        assert added["node_h"] < Model.load(once).runs[-1]["node_h"]


def test_add_layouts_and_resume():
    """--layouts measures every algorithm at new layouts; an add cut short
    resumes from its journal to the same model; --out must differ from --model."""
    if not have_data():
        print("skipped: no replay data (replay/README.md)")
        return
    with tempfile.TemporaryDirectory() as d:
        m1, m2, cut = (os.path.join(d, x) for x in ("m1.json", "m2.json", "cut.json"))
        run("--algs", "binomial,tree", "--budget-node-h", "0.3", "--out", m1)
        a = Model.load(m1)
        free = [l for l in a.layouts if l not in a.measured_layouts()]
        lays = ",".join("%dx%d" % l for l in (free[2], free[-3]))
        log = os.path.join(d, "add.log")
        run("--model", m1, "--algs", "circ_graph", "--layouts", lays, "--log", log, "--out", m2)
        b = Model.load(m2)
        for lay in (free[2], free[-3]):
            assert all(lay in b.curves[x] for x in ("binomial", "tree", "circ_graph"))
        lines = open(log).readlines()
        cut_log = os.path.join(d, "cut.log")
        with open(cut_log, "w") as f:
            f.writelines(lines[:len(lines) // 2])
        r = run("--model", m1, "--algs", "circ_graph", "--layouts", lays, "--log", cut_log, "--resume",
                "--out", cut)
        assert "from the journal" in r.stdout
        same = lambda p: (json.load(open(p))["units"][0]["fit"],
                          sorted(json.dumps({k: v for k, v in x.items() if k != "job"}, sort_keys=True)
                                 for x in json.load(open(p))["units"][0]["data"]["measurements"]))
        assert same(m2) == same(cut)
        r = run("--model", m1, "--algs", "circ_graph", "--out", m1, check=False)
        assert r.returncode != 0 and "must not be the --model file" in r.stderr


if __name__ == "__main__":
    test_add_algorithm_matches_one_shot()
    test_add_layouts_and_resume()
    print("ok")
