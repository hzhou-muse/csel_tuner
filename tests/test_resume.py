"""bin/collmodel --resume: a run cut short and resumed from its journal ends
with the same model as an uninterrupted run. Needs pandas and replay data
(replay/README.md); skipped without data."""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLMODEL = os.path.join(REPO, "bin", "collmodel")


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


if __name__ == "__main__":
    test_cut_journal_resumes_to_the_same_model()
    print("ok")
