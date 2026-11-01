"""bin/colltree over a collection: several collectives' units into one tuning
file, --colls and --with, the staleness check of inner calls into another
collective (nb's ibcast tree), and several variant units of one collective
(in-place and not) in one file."""
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

from csel_tuner.model import Model, ModelFile, default_tags  # noqa: E402
from csel_tuner.tree import json_choice  # noqa: E402

LAYOUTS = [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 2)]
SIZES = [2 ** k for k in range(0, 17, 2)]
DOMAIN = {"N": [1, 4], "ppn": [1, 4], "m": [SIZES[0], SIZES[-1]]}
# nb was measured with this file: ibcast chose sched_slow
OLD_INNER = json.dumps({"name=ibcast-intra-0": {"algorithm=MPIR_Ibcast_intra_sched_slow": {}},
                        "name=ibcast-intra-auto": "call=ibcast-intra-0"})


def alg(inner=()):
    return {"leaf": None, "restrictions": [], "mask": 0, "inner": list(inner)}


def model(coll, curves, inner=None, conditions=(), masks=None, tags=None):
    algs = {name: alg(inner.get(name, ()) if inner else ()) for name in curves}
    for name, mk in (masks or {}).items():
        algs[name]["mask"] = mk
    m = Model(coll, SIZES, LAYOUTS, DOMAIN, list(conditions), algs, tags=tags)
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


def alltoall(inplace, curves, masks=None, counts=None):
    """A variant unit of alltoall: conditions inplace (bit 0) and pof2 (bit 1); the precheck found
    inplace only where an algorithm restricts on it, so the masks start without it."""
    tags = default_tags("alltoall", inplace=inplace)
    if counts:
        tags["counts"] = counts
    return model("alltoall", curves, conditions=["inplace", "pof2"], masks=masks, tags=tags)


def test_variants():
    """In-place and not in one file: subtree 0 from the not-in-place unit, 0x1 from the in-place one,
    each chosen by its own curves (brucks wins not in place, pairwise in place), and the dispatch on
    the inplace bit; the in-place set is built even with no in-place-restricted algorithm."""
    def point(inplace):
        return {"p": 4, "m": 1024, "N": 2, "ppn": 2, "pof2": 1.0, "cond": {"inplace": inplace, "pof2": False}}
    with tempfile.TemporaryDirectory() as tmp:
        f, t = os.path.join(tmp, "m.json"), os.path.join(tmp, "t.json")
        ModelFile([alltoall(False, {"brucks": (1.0, 1e4), "pairwise": (4.0, 1e4)}),
                   alltoall(True, {"brucks": (4.0, 1e4), "pairwise": (1.0, 1e4)})]).save(f)
        colltree([f, "--out", t])
        out = json.load(open(t))
        assert sorted(out) == ["name=alltoall-intra-0", "name=alltoall-intra-1", "name=alltoall-intra-auto"], out
        assert out["name=alltoall-intra-auto"] == "dispatch=alltoall-intra(0x1)"
        trees = {k[len("name="):]: v for k, v in out.items()}
        bits = ["inplace", "pof2"]
        assert json_choice(trees["alltoall-intra-auto"], point(False), trees, bits) == "MPIR_Alltoall_intra_brucks"
        assert json_choice(trees["alltoall-intra-auto"], point(True), trees, bits) == "MPIR_Alltoall_intra_pairwise"
        # an in-place-only algorithm joins the in-place set; a subset subtree from the other variant is a
        # candidate only where this variant has curves of all its algorithms (no only_a here)
        ModelFile([alltoall(False, {"brucks": (1.0, 1e4), "pairwise": (4.0, 1e4), "only_a": (0.5, 1e4)}),
                   alltoall(True, {"brucks": (4.0, 1e4), "pairwise": (2.0, 1e4), "replace": (1.0, 1e4)},
                            {"replace": 1})]).save(f)
        colltree([f, "--out", t, "--min-fallback-algs", "1"])
        trees = {k[len("name="):]: v for k, v in json.load(open(t)).items()}
        assert json_choice(trees["alltoall-intra-auto"], point(True), trees, bits) == \
            "MPIR_Alltoall_intra_replace"
        assert json_choice(trees["alltoall-intra-auto"], point(False), trees, bits) == "MPIR_Alltoall_intra_only_a"
        assert "call=" not in json.dumps(trees["alltoall-intra-1"])
        # the in-place variant not measured yet: reported, its calls use the not-in-place subtree
        ModelFile([alltoall(False, {"brucks": (1.0, 1e4)})]).save(f)
        log = colltree([f, "--out", t])
        assert "alltoall-intra: missing (not measured yet): 1 variant inplace=1; its calls use " \
            "alltoall-intra-0" in log, log
        # two units with the same input conditions (here differing in a tag only) collide
        ModelFile([alltoall(False, {"brucks": (1.0, 1e4)}),
                   alltoall(False, {"brucks": (1.0, 1e4)}, counts="linear")]).save(f)
        try:
            colltree([f, "--out", t])
            assert False
        except SystemExit as e:
            assert "with the same input conditions" in str(e), e



def test_reduction_variants():
    """allreduce measures builtin commutative ops only: no variant has base 0, so the not-in-place one
    (the fewest input conditions) also builds subtree 0 from its unrestricted algorithms, the
    run-time fallback of user ops: correct for any op, scored on MPI_SUM."""
    bits = ["inplace", "commutative", "builtin_op"]

    def unit(inplace, curves, masks=None):
        return model("allreduce", curves, conditions=bits, masks=masks,
                     tags=default_tags("allreduce", inplace=inplace))
    with tempfile.TemporaryDirectory() as tmp:
        f, t = os.path.join(tmp, "m.json"), os.path.join(tmp, "t.json")
        ModelFile([unit(False, {"ring": (1.0, 1e4), "rsag": (0.5, 1e5)}, {"rsag": 4}),
                   unit(True, {"ring": (1.0, 1e4), "rsag": (2.0, 1e3)}, {"rsag": 4})]).save(f)
        log = colltree([f, "--out", t])
        assert "allreduce-intra: not measured (no recipe): 4 variants builtin_op=0; their calls use " \
            "allreduce-intra-0" in log, log
        out = json.load(open(t))
        # 0x6: builtin commutative, not in place; 0x7: also in place
        assert sorted(out) == ["name=allreduce-intra-0", "name=allreduce-intra-6", "name=allreduce-intra-7",
                               "name=allreduce-intra-auto"], sorted(out)
        assert out["name=allreduce-intra-0"] == {"algorithm=MPIR_Allreduce_intra_ring": {}}
        trees = {k[len("name="):]: v for k, v in out.items()}
        x = {"p": 4, "m": 65536, "N": 2, "ppn": 2, "pof2": 1.0,
             "cond": {"inplace": False, "commutative": True, "builtin_op": True}}
        assert json_choice(trees["allreduce-intra-auto"], x, trees, bits) == "MPIR_Allreduce_intra_rsag"
        x["cond"]["inplace"] = True
        assert json_choice(trees["allreduce-intra-auto"], x, trees, bits) == "MPIR_Allreduce_intra_ring"
        x["cond"]["builtin_op"] = False      # a user op: the run-time fallback
        assert json_choice(trees["allreduce-intra-auto"], x, trees, bits) == "MPIR_Allreduce_intra_ring"



def test_size_condition():
    """Size splits read msg_size (avg_msg_size with --size-condition); gatherv, without a message
    size the same on every rank, never splits on size even where the winner changes with it."""
    with tempfile.TemporaryDirectory() as tmp:
        f, t = os.path.join(tmp, "m.json"), os.path.join(tmp, "t.json")
        # linear wins small messages, binomial large ones
        curves = {"linear": (1.0, 1e3), "binomial": (8.0, 1e5)}
        ModelFile([model("gather", curves), model("gatherv", curves)]).save(f)
        colltree([f, "--out", t])
        out = json.load(open(t))
        assert "msg_size(" in json.dumps(out["name=gather-intra-0"])
        assert "msg_size" not in json.dumps(out["name=gatherv-intra-0"]), out["name=gatherv-intra-0"]
        colltree([f, "--out", t, "--size-condition", "avg_msg_size"])
        assert "avg_msg_size(" in json.dumps(json.load(open(t))["name=gather-intra-0"])


if __name__ == "__main__":
    test_collection()
    test_collection_errors()
    test_variants()
    test_reduction_variants()
    test_size_condition()
    print("ok")


def test_splice_into_main_file():
    """colltree --into: only the built collective's intra subtrees change; the rest stays byte
    for byte; old subtrees of that collective that the run lacks go; a second splice is a no-op."""
    import importlib.machinery
    import importlib.util
    path = os.path.join(REPO, "bin", "colltree")
    loader = importlib.machinery.SourceFileLoader("colltree_mod", path)
    spec = importlib.util.spec_from_loader("colltree_mod", loader)
    ct = importlib.util.module_from_spec(spec)
    loader.exec_module(ct)
    text = ('{\n    "name=main": "call=auto",\n    "name=bcast-intra-auto":\n    {\n'
            '            "comm_size<8":\n            {\n                "algorithm=MPIR_Bcast_intra_binomial":{}\n'
            '            },\n            "any":\n            {\n'
            '                "algorithm=MPIR_Bcast_intra_scatter_ring_allgather":{}\n            }\n    },\n'
            '    "name=bcast-intra-10": "call=bcast-intra-auto",\n'
            '    "name=bcast-inter-auto":\n    {\n        "algorithm=MPIR_Bcast_inter_remote_send_local_bcast":{}\n'
            '    }\n}\n')
    ours = {"name=bcast-intra-auto": "dispatch=bcast-intra(0x200)",
            "name=bcast-intra-0": {"msg_size(4096)": {"algorithm=MPIR_Bcast_intra_binomial": {}},
                                   "any": {"algorithm=MPIR_Bcast_intra_tree": {"k": 2}}},
            "name=bcast-intra-200": {"algorithm=MPIDI_POSIX_mpi_bcast_release_gather": {}}}
    new = ct.splice_into(text, ours, ["bcast"])
    d = json.loads(new)
    assert list(d) == ["name=main", "name=bcast-intra-auto", "name=bcast-intra-0", "name=bcast-intra-200",
                       "name=bcast-inter-auto"]
    assert d["name=bcast-intra-0"] == ours["name=bcast-intra-0"] and "name=bcast-intra-10" not in d
    assert new.startswith('{\n    "name=main": "call=auto",\n    "name=bcast-intra-auto": "dispatch=')
    assert new.endswith(text[text.index('    "name=bcast-inter-auto"'):])
    assert ct.splice_into(new, ours, ["bcast"]) == new
