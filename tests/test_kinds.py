"""csel_tuner.kinds: the kinds of the catalog's restrictions, the variants of a
collective and their requests; and collmodel's check that a compositional
algorithm's inner calls have trees in the --inner-selection file."""
import argparse
import json
import os
import sys
import tempfile
from importlib.machinery import SourceFileLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csel_tuner.algos import Inner  # noqa: E402
from csel_tuner.kinds import classify, launch_known, measured, request, variants  # noqa: E402

# the restrictions of MPICH's catalog after pmodels/mpich#8065, in index order
BITS = ["inplace", "pof2", "commutative", "builtin_op", "hierarchical", "node_consecutive", "node_regular",
        "displs_ordered", "block_regular", "MPIDI_CH4_release_gather", "gpu", "gpu_coll"]


def test_classify():
    kinds, notes = classify(BITS)
    assert [r for r in BITS if kinds[r] == "input"] == ["inplace", "commutative", "builtin_op"]
    assert [r for r in BITS if kinds[r] == "launch"] == ["gpu", "gpu_coll"]
    assert kinds["displs_ordered"] == kinds["MPIDI_CH4_release_gather"] == "set" and notes == []
    kinds, notes = classify(BITS + ["datatype_contig"])
    assert kinds["datatype_contig"] == "set" and len(notes) == 1 and "datatype_contig" in notes[0]


def test_variants():
    assert variants("bcast", BITS) == [{}] and variants("barrier", BITS) == [{}]
    assert variants("alltoall", BITS) == [{"inplace": 0}, {"inplace": 1}]
    assert variants("ialltoall", BITS) == variants("alltoall", BITS)
    red = variants("reduce", BITS)
    # builtin_op implies commutative: 3 op cases, with and without in-place
    assert len(red) == 6 and {"inplace": 0, "commutative": 0, "builtin_op": 1} not in red
    meas = [v for v in red if measured(v)]
    assert meas == [{"inplace": 0, "commutative": 1, "builtin_op": 1},
                    {"inplace": 1, "commutative": 1, "builtin_op": 1}]
    assert request(meas[0]) == {"op": "sum"} and request(meas[1]) == {"inplace": 1, "op": "sum"}
    # only the conditions in the catalog count
    assert variants("reduce", ["commutative"]) == [{"commutative": 0}, {"commutative": 1}]


def test_launch_known():
    gpu, gpu_coll = 1 << BITS.index("gpu"), 1 << BITS.index("gpu_coll")
    assert launch_known(BITS, "device") == (gpu, 0)
    assert launch_known(BITS, "host") == (0, gpu | gpu_coll)
    assert launch_known(["hierarchical"], "device") == (0, 0)


def test_missing_inner():
    collmodel = SourceFileLoader("collmodel", os.path.join(REPO, "bin", "collmodel")).load_module()

    class A:
        inner = [Inner.parse("i$coll-intra@comm", "bcast")]
    bcast_only = json.dumps({"name=bcast-intra-0": {}, "name=bcast-intra-auto": "call=bcast-intra-0"})
    assert collmodel.missing_inner(A, bcast_only) == ["ibcast-intra"]
    both = json.dumps({"name=bcast-intra-auto": {}, "name=ibcast-intra-auto": {}})
    assert collmodel.missing_inner(A, both) == []
    assert collmodel.missing_inner(A, None) == ["ibcast-intra"]
    # adding nb with a file that has bcast's tree only: refused, naming ibcast-intra, before any probe
    from csel_tuner.model import Model
    from test_restrict import CATALOG
    with tempfile.TemporaryDirectory() as tmp:
        cat, sel = os.path.join(tmp, "coll_algorithms.txt"), os.path.join(tmp, "t.json")
        with open(cat, "w") as f:
            f.write(CATALOG)
        with open(sel, "w") as f:
            f.write(bcast_only)
        bits = ["inplace", "not_inplace", "pof2", "hierarchical", "MPIDI_CH4_release_gather"]
        m = Model("bcast", [1], [(1, 2)], {}, bits)
        a = argparse.Namespace(catalog=cat, op="bcast", exclude_algs="", algs=["nb"], inner_selection=sel,
                               replay=False, layouts=None, launch="host", max_stage=None,
                               cvar_values=None)
        fail = lambda *x: sys.exit("probed")
        try:
            collmodel.extend(a, m, fail, fail, None, None, 1, None, None, None)
            assert False, "added nb without the ibcast tree"
        except SystemExit as e:
            assert "nb" in str(e) and "ibcast-intra" in str(e), e
        a.algs = ["all"]
        try:
            collmodel.extend(a, m, fail, fail, None, None, 1, None, None, None)
        except SystemExit as e:     # nb is left out of "all"; the plain algorithms go on to the precheck
            assert str(e) == "probed", e


if __name__ == "__main__":
    test_classify()
    test_variants()
    test_launch_known()
    test_missing_inner()
    print("ok")
