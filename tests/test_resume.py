"""bin/collmodel --resume: a run cut short and resumed from its journal ends
with the same model as an uninterrupted run; a resume may leave algorithms
out and keeps the other replies. The runs need pandas and replay data
(replay/README.md) and are skipped without data; the checks of resume_check
and the journal need neither."""
import json
import os
import re
import subprocess
import sys
import tempfile
from importlib.machinery import SourceFileLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLMODEL = os.path.join(REPO, "bin", "collmodel")
collmodel = SourceFileLoader("collmodel", COLLMODEL).load_module()


def job(algs, exclude="", **kw):
    d = dict(mode="live", op="allreduce", seed=0, batches=3, sync=1, prune=0, nodes=2, ppn_max=4, max_m=65536,
             compositional=None, probe_m=4, base=None, inner=None, layouts=None, anchors=None, mem="host",
             algs=algs, exclude_algs=exclude)
    d.update(kw)
    return d


def test_resume_check():
    check = collmodel.resume_check
    assert check([job(["all"])], job(["all"])) == (None, set())
    assert check([job(["all"])], job(["all"], "a")) == (None, {"a"})
    # against every earlier job: a third job that changes nothing still has the dropped algorithm
    assert check([job(["all"]), job(["all"], "a")], job(["all"], "a")) == (None, {"a"})
    assert check([job(["all"], "a")], job(["all"], "a,b")) == (None, {"b"})
    assert check([job(["a", "b", "c"])], job(["a", "c"])) == (None, {"b"})
    assert "only exclude more" in check([job(["all"], "a")], job(["all"]))[0]
    assert "adds c" in check([job(["a", "b"])], job(["a", "c"]))[0]
    assert "'all'" in check([job(["all"])], job(["a"]))[0]
    assert "differs in seed" in check([job(["all"])], job(["all"], seed=1))[0]
    # excluding an algorithm an explicit list does not have drops nothing
    assert check([job(["a", "b"])], job(["a", "b"], "z")) == (None, set())
    # journals written before exclude_algs existed
    old = job(["all"])
    del old["exclude_algs"]
    assert check([old], job(["all"], "a")) == (None, {"a"})


def test_journal_salvage():
    op, cvar = "allreduce", "cvar." + collmodel.OPS["allreduce"]
    req = lambda alg, m: {"op": op, "m": m, "p": 8, "ppn": 4, cvar: alg}
    rep = lambda t: {"status": "ok", "t_us": [t], "wall_us": t}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "j.log")
        with open(path, "w") as f:     # one job of 100 s, 4 replies: 25 s each
            f.write(json.dumps({"time": 0.0, "job": dict(job(["a", "b", "c"]), start=0.0)}) + "\n")
            for i, (alg, m) in enumerate([("a", 4), ("b", 4), ("c", 4), ("a", 8)]):
                f.write(json.dumps({"time": 25.0 * (i + 1), "req": req(alg, m), "rep": rep(i)}) + "\n")
        # in order: the first miss ends the replay
        j = collmodel.Journal(path, True)
        assert j.take(req("a", 4)) == rep(0) and j.take(req("a", 16)) is None
        assert j.take(req("c", 4)) is None and j.n_cached == 1
        # salvage: out of order, b left out; 3 of the 4 replies billed
        j = collmodel.Journal(path, True)
        j.salvage = True
        assert j.take(req("c", 4)) == rep(2) and j.take(req("a", 16)) is None
        assert j.take(req("a", 8)) == rep(3) and j.take(req("a", 4)) == rep(0)
        assert j.take(req("a", 4)) is None and j.n_cached == 3
        assert abs(j.reused - 75.0) < 1e-9 and j.earlier_seconds() == 100.0


def have_data():
    d = os.path.join(REPO, "replay", "data")
    return os.path.exists(os.path.join(d, "gt.csv.gz")) and os.path.exists(os.path.join(d, "sessions.csv.gz"))


def run(*args, check=True):
    return subprocess.run([sys.executable, COLLMODEL, "--replay", *args], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def model(path):
    """What must not depend on the cut: the fit, the measurements (without their
    job, a resumed run has two) and the prechecks."""
    d = json.load(open(path))
    strip = lambda rs: sorted(json.dumps({k: v for k, v in r.items() if k != "job"}, sort_keys=True) for r in rs)
    return d["fit"], strip(d["data"]["measurements"]), strip(d["data"]["prechecks"])


def test_cut_journal_resumes_to_the_same_model():
    if not have_data():
        print("skipped: no replay data (replay/README.md)")
        return
    sys.path.insert(0, os.path.join(REPO, "replay"))
    from common import FIXED
    algs = ",".join(FIXED[:2])
    with tempfile.TemporaryDirectory() as d:
        full_log, full = os.path.join(d, "full.log"), os.path.join(d, "full.json")
        run("--algs", algs, "--log", full_log, "--out", full)
        lines = open(full_log).readlines()
        cut_log, cut = os.path.join(d, "cut.log"), os.path.join(d, "cut.json")
        with open(cut_log, "w") as f:     # half the journal, the last line cut mid-way as by a kill
            f.writelines(lines[:len(lines) // 2])
            f.write(lines[len(lines) // 2][:40])
        r = run("--algs", algs, "--log", cut_log, "--out", cut, "--resume")
        assert "from the journal" in r.stdout
        assert model(full) == model(cut)
        for line in open(cut_log):        # the cut line was dropped, the journal is whole again
            json.loads(line)
        r = run("--algs", algs, "--log", cut_log, "--out", cut, "--resume", "--seed", "1", check=False)
        assert r.returncode != 0 and "differs in seed" in r.stderr


def test_resume_with_fewer_algorithms():
    if not have_data():
        print("skipped: no replay data (replay/README.md)")
        return
    sys.path.insert(0, os.path.join(REPO, "replay"))
    from common import FIXED
    algs3, algs2 = ",".join(FIXED[:3]), ",".join(FIXED[:2])
    with tempfile.TemporaryDirectory() as d:
        full_log = os.path.join(d, "full.log")
        run("--algs", algs3, "--log", full_log, "--out", os.path.join(d, "full.json"))
        lines = open(full_log).readlines()
        cut_log, cut = os.path.join(d, "cut.log"), os.path.join(d, "cut.json")
        with open(cut_log, "w") as f:
            f.writelines(lines[:len(lines) // 2])
        recs = [json.loads(x) for x in lines[:len(lines) // 2]]
        kept = sum(1 for x in recs if "req" in x and collmodel.req_alg(x["req"], "bcast") != FIXED[2])
        r = run("--algs", algs2, "--log", cut_log, "--out", cut, "--resume")
        assert "dropped: " + FIXED[2] in r.stderr
        reused = int(re.search(r"\((\d+) from the journal\)", r.stdout).group(1))
        assert 0 < reused <= kept
        assert {m["alg"] for m in json.load(open(cut))["data"]["measurements"]} <= set(FIXED[:2])
        r = run("--algs", algs3, "--log", cut_log, "--out", cut, "--resume", check=False)
        assert r.returncode != 0 and "adds " + FIXED[2] in r.stderr


if __name__ == "__main__":
    test_resume_check()
    test_journal_salvage()
    test_cut_journal_resumes_to_the_same_model()
    test_resume_with_fewer_algorithms()
    print("ok")
