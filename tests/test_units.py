"""Model files of several units (collmodel/5): tags, selection, the round trip, collmodel/4 files
read as one unit, put() replacing by tags, and collmodel --merge."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from csel_tuner.model import (FORMAT, Model, ModelFile, default_tags, parse_sel, tag_str,  # noqa: E402
                              tags_match)


def unit(coll, inplace=False, counts="uniform", mem="host", t=1.0):
    tags = default_tags(coll, mem, inplace, counts)
    m = Model(coll, [1, 2], [(1, 2)], {"N": [1, 1], "ppn": [1, 2], "m": [1, 2]}, ["inplace"], mem=mem, tags=tags)
    j = m.add_job({"start": 0})
    m.add_measurement(j, "ring", 1, 2, 1, [t], 10.0)
    m.set_curve("ring", (1, 2), [t, t + 1])
    return m


def test_tags():
    assert default_tags("bcast") == {"coll": "bcast", "variant": {}, "datatype": "byte", "launch": "host"}
    t = default_tags("allreduce", inplace=True)
    assert t["variant"] == {"inplace": 1, "builtin_op": 1, "commutative": 1} and t["datatype"] == "int"
    assert default_tags("ialltoallv", counts="onebig")["counts"] == "onebig"
    assert "counts" not in default_tags("alltoall")
    assert tags_match(t, {"coll": "allreduce", "inplace": "1"}) and not tags_match(t, {"inplace": 0})
    assert not tags_match(t, {"counts": "uniform"})
    assert parse_sel("coll=bcast, launch=device") == {"coll": "bcast", "launch": "device"}
    assert tag_str(default_tags("allgatherv", counts="linear")) == \
        "allgatherv inplace=0 counts=linear datatype=byte launch=host"


def test_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        f = ModelFile([unit("allreduce"), unit("allreduce", True), unit("bcast", mem="device")])
        f.save(path)
        assert json.load(open(path))["format"] == FORMAT
        g = ModelFile.load(path)
        assert [u.tags for u in g.units] == [u.tags for u in f.units]
        u = g.unit({"inplace": 1})
        assert u.op == "allreduce" and u.mem == "host" and np.allclose(u.curves["ring"][(1, 2)], [1, 2])
        assert Model.load(path, {"coll": "bcast"}).mem == "device"
        for sel in (None, {"coll": "allreduce"}, {"coll": "scan"}):
            try:
                g.unit(sel)
                assert False, sel
            except ValueError:
                pass
        assert g.put(unit("allreduce", True, t=5.0)) and len(g.units) == 3
        assert g.get(default_tags("allreduce", inplace=True)).measurements[0]["t_us"] == [5.0]
        assert not g.put(unit("scan")) and len(g.units) == 4


def test_v4():
    """A collmodel/4 file is one unit; inplace and counts come from its jobs."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v4.json")
        d = unit("alltoallv", True, "linear").to_dict()
        del d["tags"]
        d.update(format="collmodel/4", op="alltoallv", mem="host")
        d["data"]["jobs"] = [{"start": 0, "inplace": True, "counts": "linear"}]
        json.dump(d, open(path, "w"))
        m = Model.load(path)
        assert m.tags == default_tags("alltoallv", "host", True, "linear")
        d["data"]["jobs"] = [{"start": 0}]
        json.dump(d, open(path, "w"))
        assert Model.load(path).tags == default_tags("alltoallv")


def test_merge():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c, out = (os.path.join(tmp, x) for x in ("a.json", "b.json", "c.json", "out.json"))
        ModelFile([unit("allreduce"), unit("bcast")]).save(a)
        ModelFile([unit("bcast"), unit("allreduce", True)]).save(b)
        ModelFile([unit("bcast", t=3.0)]).save(c)
        run = lambda *files: subprocess.run([sys.executable, os.path.join(REPO, "bin", "collmodel"),
                                             *sum((["--merge", x] for x in files), []), "--out", out],
                                            capture_output=True, text=True)
        r = run(a, b)
        assert r.returncode == 0, r.stderr
        assert sorted(tag_str(u.tags) for u in ModelFile.load(out).units) == sorted(
            tag_str(u.tags) for u in (unit("allreduce"), unit("allreduce", True), unit("bcast")))
        r = run(a, c)
        assert r.returncode != 0 and "also in" in r.stderr


if __name__ == "__main__":
    test_tags()
    test_file()
    test_v4()
    test_merge()
    print("ok")
