"""Restriction sets: the coll_algorithms.txt catalog, the precheck, the set
plan, the dispatch subtree, and (with replay data) a two-set build through
bin/collmodel and bin/colltree."""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "bin"))

from csel_tuner.algos import Catalog  # noqa: E402
from csel_tuner.restrict import plan_sets, precheck  # noqa: E402
from csel_tuner.tree import json_choice  # noqa: E402

CATALOG = """/*
 * header comment
 */
# ----
conditions:
    inplace:          MPIR_Csel_sendbuf_inplace
    pof2:             MPIR_Csel_comm_size_is_pof2
    hierarchical:     MPIR_Csel_is_hierarchical
    comm_size(thresh):      MPIR_Csel_comm_size
    MPIDI_CH4_release_gather: MPIDI_POSIX_check_release_gather #if defined(MPIDI_CH4_SHM_POSIX)

# ----
bcast-intra:
    binomial
    smp
        restrictions: hierarchical
    tree
        extra_params: tree_type, k, is_non_blocking
        cvar_params: TREE_TYPE, TREE_KVAL, IS_NON_BLOCKING
    release_gather
        func_name: MPIDI_POSIX_mpi_bcast_release_gather
        inline: 1
        macro_guard: defined(MPIDI_CH4_SHM_POSIX)
        restrictions: MPIDI_CH4_release_gather
bcast-inter:
    remote_send_local_bcast
"""


def catalog():
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write(CATALOG)
    f.close()
    return Catalog.load(f.name)


def test_catalog():
    cat = catalog()
    algs = cat.algorithms("bcast")
    assert [a.name for a in algs] == ["binomial", "smp", "tree", "release_gather", "nb"]
    bits = cat.restriction_bits("bcast")
    assert bits == ["hierarchical", "MPIDI_CH4_release_gather"]
    assert [a.mask(bits) for a in algs] == [0, 1, 0, 2, 0]
    assert algs[3].leaf == "MPIDI_POSIX_mpi_bcast_release_gather" and algs[0].leaf == "MPIR_Bcast_intra_binomial"
    assert algs[4].leaf == "MPIR_Coll_nb"
    assert algs[2].extra_params == ["tree_type", "k", "is_non_blocking"]
    assert cat.conditions["comm_size"] and not cat.conditions["inplace"]


def test_precheck_and_sets():
    cat = catalog()
    algs = [a for a in cat.algorithms("bcast") if a.name != "smp"]
    bits = cat.restriction_bits("bcast")
    layouts = [(1, 2), (1, 4), (2, 1), (2, 4)]
    # release_gather runs on one node only; nb fails everywhere
    probe = lambda n, ppn, alg: (alg != "nb" and (alg != "release_gather" or n == 1), "no")
    feas, masks, notes = precheck(probe, layouts, algs, bits)
    assert masks == {(1, 2): 2, (1, 4): 2, (2, 1): 0, (2, 4): 0}
    assert any(n.startswith("nb fails") for n in notes)
    sets = plan_sets([a for a in algs if a.name != "nb"], bits, masks)
    assert sets[0] == (0, layouts, ["binomial", "tree"])
    assert sets[1] == (2, [(1, 2), (1, 4)], ["release_gather", "binomial", "tree"])


def test_dispatch_and_choice():
    from importlib.machinery import SourceFileLoader
    colltree = SourceFileLoader("colltree", os.path.join(REPO, "bin", "colltree")).load_module()
    bits = ["hierarchical", "MPIDI_CH4_release_gather"]
    d = colltree.dispatch("bcast", bits, [0, 2])
    assert d == {"MPIDI_CH4_release_gather": "call=bcast-intra-r2", "any": "call=bcast-intra-r0"}
    # a set with no subtree falls back to its largest present subset (a tie: the lower mask)
    d = colltree.dispatch("bcast", bits, [0, 1, 2])
    assert d == {"hierarchical": "call=bcast-intra-r1",
                 "any": {"MPIDI_CH4_release_gather": "call=bcast-intra-r2", "any": "call=bcast-intra-r0"}}
    trees = {"bcast-intra-auto": colltree.dispatch("bcast", bits, [0, 2]),
             "bcast-intra-r0": {"comm_size(4)": {"algorithm=MPIR_Bcast_intra_binomial": {}},
                                "any": {"algorithm=MPIR_Bcast_intra_tree": {}}},
             "bcast-intra-r2": {"avg_msg_size(1024)": {"algorithm=MPIDI_POSIX_mpi_bcast_release_gather": {}},
                                "any": "call=bcast-intra-r0"}}
    x = lambda p, m, rg: {"p": p, "m": m, "N": 1, "ppn": p, "pof2": 1.0,
                          "cond": {"hierarchical": False, "MPIDI_CH4_release_gather": rg}}
    assert json_choice(trees["bcast-intra-auto"], x(4, 64, True), trees) == "MPIDI_POSIX_mpi_bcast_release_gather"
    assert json_choice(trees["bcast-intra-auto"], x(4, 4096, True), trees) == "MPIR_Bcast_intra_binomial"
    assert json_choice(trees["bcast-intra-auto"], x(8, 64, False), trees) == "MPIR_Bcast_intra_tree"


def have_data():
    d = os.path.join(REPO, "replay", "data")
    return os.path.exists(os.path.join(d, "gt.csv.gz")) and os.path.exists(os.path.join(d, "sessions.csv.gz"))


def test_two_set_replay():
    """release_gather replayed from MPICH's default on one node (which is
    release_gather there): a second set on the single-node layouts, and a
    tuning file that dispatches to it."""
    if not have_data():
        print("skipped: no replay data")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cat = os.path.join(tmp, "coll_algorithms.txt")
        with open(cat, "w") as f:
            f.write(CATALOG)
        model, tuning = os.path.join(tmp, "m.json"), os.path.join(tmp, "t.json")
        subprocess.run([sys.executable, os.path.join(REPO, "bin", "collmodel"), "--replay", "--algs", "all",
                        "--coll-algorithms", cat, "--replay-alias", "release_gather=auto@single",
                        "--budget-node-h", "0.5", "--out", model], check=True, capture_output=True)
        d = json.load(open(model))
        assert d["format"] == "collmodel/3" and [s["mask"] for s in d["sets"]] == [0, 2]
        assert d["stage2"] == ["smp"] and d["left_out"] == ["nb"]
        assert all(l[0] == 1 for l in d["sets"][1]["layouts"])
        subprocess.run([sys.executable, os.path.join(REPO, "bin", "colltree"), model, "--out", tuning],
                       check=True, capture_output=True)
        t = json.load(open(tuning))
        assert set(t) == {"name=bcast-intra-r0", "name=bcast-intra-r2", "name=bcast-intra-auto"}
        assert t["name=bcast-intra-auto"] == {"MPIDI_CH4_release_gather": "call=bcast-intra-r2",
                                              "any": "call=bcast-intra-r0"}
        assert "MPIDI_POSIX_mpi_bcast_release_gather" in json.dumps(t["name=bcast-intra-r2"])


if __name__ == "__main__":
    test_catalog()
    test_precheck_and_sets()
    test_dispatch_and_choice()
    test_two_set_replay()
    print("ok")
