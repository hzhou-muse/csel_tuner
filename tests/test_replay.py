"""End to end on recorded data: bin/collmodel --replay (v2, a collmodel/4 model), then bin/colltree.
Needs pandas and replay data (replay/README.md); skipped without data."""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from csel_tuner.model import ModelFile  # noqa: E402


def have_data():
    d = os.path.join(REPO, "replay", "data")
    return os.path.exists(os.path.join(d, "gt.csv.gz")) and os.path.exists(os.path.join(d, "sessions.csv.gz"))


def test_v2_then_tree():
    if not have_data():
        print("skipped: no replay data (replay/README.md)")
        return
    with tempfile.TemporaryDirectory() as d:
        model, tuning = os.path.join(d, "m.model"), os.path.join(d, "t.json")
        sys.path.insert(0, os.path.join(REPO, "replay"))
        from common import FIXED
        subprocess.run([sys.executable, os.path.join(REPO, "bin", "collmodel"), "--replay", "--out", model,
                        "--algs", ",".join(FIXED[:3]), "--budget-node-h", "0.3"],
                       check=True, stdout=subprocess.DEVNULL)
        f = ModelFile.load(model)
        m = f.units[0].to_dict()
        assert ModelFile.format_of(model) == "collmodel/6" and len(f.units) == 1 and len(m["fit"]["curves"]) == 3 \
            and m["data"]["measurements"]
        subprocess.run([sys.executable, os.path.join(REPO, "bin", "colltree"), model, "--out", tuning],
                       check=True, stdout=subprocess.DEVNULL)
        t = json.load(open(tuning))
        assert list(t) == ["name=bcast-intra-0"]           # MPICH's call=auto reaches it


if __name__ == "__main__":
    test_v2_then_tree()
    print("ok")
