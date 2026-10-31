"""bin/colltree over a collection: several collectives' units into one tuning
file, --colls and --with, and the staleness check of inner calls into
another collective (nb's ibcast tree)."""
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

from csel_tuner.model import Model, ModelFile  # noqa: E402

LAYOUTS = [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 2)]
SIZES = [2 ** k for k in range(0, 17, 2)]
DOMAIN = {"N": [1, 4], "ppn": [1, 4], "m": [SIZES[0], SIZES[-1]]}
# nb was measured with this file: ibcast chose sched_slow
OLD_INNER = json.dumps({"name=ibcast-intra-0": {"algorithm=MPIR_Ibcast_intra_sched_slow": {}},
                        "name=ibcast-intra-auto": "call=ibcast-intra-0"})


def alg(inner=()):
    return {"leaf": None, "restrictions": [], "mask": 0, "inner": list(inner)}


def model(coll, curves, inner=None):
    algs = {name: alg(inner.get(name, ()) if inner else ()) for name in curves}
    m = Model(coll, SIZES, LAYOUTS, DOMAIN, [], algs)
    m.masks = {lay: 0 for lay in LAYOUTS}
    for name, (lat, bw) in curves.items():
        for lay in LAYOUTS:
            p = lay[0] * lay[1]
            m.set_curve(name, lay, np.log2(lat * (1 + np.log2(p)) + np.array(SIZES, float) / bw))
    return m


def bcast():
    """nb fastest everywhere, measured under OLD_INNER."""
    m = model("bcast", {"binomial": (10.0, 1e3), "nb": (1.0, 1e5)}, {"nb": ["ibcast-intra@comm"]})
    m.fit_inner["nb"] = m.add_inner_file(OLD_INNER)
    return m


def ibcast():
    """sched_fast beats sched_slow by 4x: the output's ibcast tree differs from nb's file."""
    return model("ibcast", {"sched_slow": (4.0, 1e3), "sched_fast": (1.0, 4e3)})


def colltree(argv):
    ct = SourceFileLoader("colltree", os.path.join(REPO, "bin", "colltree")).load_module()
    with redirect_stdout(io.StringIO()) as buf:
        ct.main(argv)
    return buf.getvalue()


def test_collection():
    with tempfile.TemporaryDirectory() as tmp:
        f, t, t1, t2 = (os.path.join(tmp, x) for x in ("m.json", "t.json", "t1.json", "t2.json"))
        ModelFile([bcast(), ibcast()]).save(f)
        log = colltree([f, "--out", t])
        out = json.load(open(t))
        assert sorted(out) == ["name=bcast-intra-0", "name=bcast-intra-auto", "name=ibcast-intra-0",
                               "name=ibcast-intra-auto"], sorted(out)
        assert out["name=ibcast-intra-0"] == {"algorithm=MPIR_Ibcast_intra_sched_fast": {}}
        # nb's inner ibcast now selects sched_fast, not the sched_slow it was measured with
        assert "stale: bcast nb's inner ibcast-intra@comm" in log, log
        assert "selects sched_fast, nb was measured with sched_slow" in log, log
        assert "remeasure: bin/collmodel --model %s --colls bcast --algs nb" % f in log, log
        # one collective: its inner tree is not in the output, so not checked; --with adds it back
        log = colltree([f, "--colls", "ibcast", "--out", t1])
        assert sorted(json.load(open(t1))) == ["name=ibcast-intra-0", "name=ibcast-intra-auto"]
        log = colltree([f, "--colls", "bcast", "--out", t2])
        assert "ibcast's tree is not in this output; not checked" in log, log
        colltree([f, "--colls", "bcast", "--with", t1, "--out", t2])
        assert json.load(open(t2)) == out


def test_collection_errors():
    with tempfile.TemporaryDirectory() as tmp:
        f, g, t = (os.path.join(tmp, x) for x in ("m.json", "g.json", "t.json"))
        ModelFile([bcast(), ibcast()]).save(f)
        ModelFile([ibcast()]).save(g)
        for argv, msg in (([f, "--colls", "allreduce", "--out", t], "no unit selected"),
                          ([f, g, "--colls", "bcast,allreduce", "--out", t], "no unit selected"),
                          ([f, g, "--out", t], "a second ibcast unit of host buffers")):
            try:
                colltree(argv)
                assert False, argv
            except SystemExit as e:
                assert msg in str(e), (argv, e)


if __name__ == "__main__":
    test_collection()
    test_collection_errors()
    print("ok")
