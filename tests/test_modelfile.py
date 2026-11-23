"""The text model file (collmodel/6, modelfile.py): a unit round-trips through it unchanged, the
reader takes hand-edited text (comments, blank lines, aligned columns, gzip), and it stops on an
edited inner file or a malformed line."""
import gzip
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from csel_tuner import modelfile  # noqa: E402
from csel_tuner.model import Model, default_tags  # noqa: E402

INNER = '{"name=main": {"any": {"algorithm=MPIR_Bcast_intra_binomial": {}}}}\n'


def unit():
    tags = dict(default_tags("allgatherv", inplace=True, counts="linear"), site="1.5")    # a numeric string tag
    sizes = [1, 2, 4]
    lays = [(1, 2), (1, 3), (1, 4), (2, 4)]
    m = Model("allgatherv", sizes, lays, {"N": [1, 2], "ppn": [2, 4], "m": [1, 4]}, ["inplace", "pof2"],
              {"ring": {"leaf": None, "restrictions": [], "mask": 0, "inner": []},
               "smp": {"leaf": None, "restrictions": ["pof2"], "mask": 2, "inner": ["allgather"],
                       "params": {"k": 2}}}, tags=tags)
    m.masks = {(1, 2): 2, (1, 4): 2, (1, 3): 0}
    m.left_out = ["nb"]
    m.notes = ["N=2 ppn=4: conditions pof2 not decided, taken as false", ""]
    j0 = m.add_job({"start": 1.5, "mode": "live", "hosts": ["a b"]})
    j1 = m.add_job({"start": 2.5})
    h = m.add_inner_file(INNER)
    m.add_precheck(j0, "nb", 1, 2, False, "  Algorithm selected\nis not supported ")
    m.add_precheck(j0, "ring", 1, 2, True)
    m.add_measurement(j0, "ring", 1, 2, 1, [1.25, 1.5, 270], 10.0)
    m.add_measurement(j0, "smp", 2, 4, 4, [3.0], 12.5, inner=h)
    m.add_measurement(j1, "ring", 1, 3, 2, [2.0], 11.0, anchor=True, growth=True)
    m.add_measurement(j1, "ring", 2, 4, 4, [4.0], 11.0, growth=True)
    for n, p in lays:
        m.set_curve("ring", (n, p), np.log2(np.array(sizes) * p * (1 + 0.5 * (p == 3))))
    m.set_curve("smp", (2, 4), [1.0, 2.0, 3.0])
    m.fit_inner = {"smp": h}
    m.runs = [{"mode": "create", "stop": "budget", "node_h": 0.1}]
    return m


def roundtrip(d):
    units, created = modelfile.loads(modelfile.dumps([d], "T"))
    assert created == "T" and len(units) == 1
    return Model.from_dict(units[0])


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_roundtrip():
    d = unit().to_dict()
    assert d["fit"]["penalty"]["ring"]["intra"] is not None and d["fit"]["penalty"]["ring"]["inter"] is None
    d["data"]["measurements"][0]["t_us"][2] = 270       # an int, as in some JSON models, stays one
    m = roundtrip(d)
    assert same(m.to_dict(), d)
    assert modelfile.dumps([m.to_dict()], "T") == modelfile.dumps([d], "T")
    assert m.tags["site"] == "1.5" and [p["msg"] for p in m.prechecks if p["alg"] == "nb"] == ["Algorithm selected is not supported"]
    assert m.measurements[0]["t_us"] == [1.25, 1.5, 270] and isinstance(m.measurements[0]["t_us"][2], int)
    assert m.algorithms["smp"]["params"] == {"k": 2} and "nb" not in m.algorithms


def test_hand_edited():
    """Comments, blank lines, aligned columns, gzip; an algorithm section without info."""
    text = modelfile.dumps([unit().to_dict()], "T")
    text = text.replace("\nsizes ", "\n\n  # a comment\nsizes\t ").replace("meas 1 2 1 ", "meas  1   2  1\t")
    m = roundtrip(unit().to_dict())
    units, _ = modelfile.loads(text)
    assert same(Model.from_dict(units[0]).to_dict(), m.to_dict())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.model")
        with gzip.open(path, "wt") as f:
            f.write(text)
        assert same(Model.from_dict(modelfile.read(path)[0][0]).to_dict(), m.to_dict())


def fails(text, what):
    try:
        modelfile.loads(text, "x.model")
    except ValueError as e:
        assert what in str(e), (what, str(e))
        return
    assert False, "no error for " + what


def test_errors():
    text = modelfile.dumps([unit().to_dict()], "T")
    fails(text.replace('{"algorithm=MPIR_Bcast_intra_binomial": {}}', '{"algorithm=MPIR_Bcast_intra_smp": {}}'),
          "hashes to")
    fails(text.replace("---- end\n", ""), "has no '---- end'")
    fails(text.replace("meas 1 2 1 ", "meas 1 2 1 x"), "meas")
    fails(text.replace("\nfit_version", "\nfitversion"), "unknown unit line")
    fails(text.replace("mask 1x2 2", "mask 8x8 2"), "not one of its layouts")
    fails("bcast", "not a collmodel/6 file")
    d = unit().to_dict()
    d["algorithms"]["two words"] = {}
    try:
        modelfile.dumps([d], "T")
        assert False
    except ValueError as e:
        assert "no blanks" in str(e)


if __name__ == "__main__":
    test_roundtrip()
    test_hand_edited()
    test_errors()
    print("ok")
