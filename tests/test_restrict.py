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
    not_inplace:      MPIR_Csel_sendbuf_not_inplace
    pof2:             MPIR_Csel_comm_size_is_pof2
    hierarchical:     MPIR_Csel_is_hierarchical
    comm_size(thresh):      MPIR_Csel_comm_size
    MPIDI_CH4_release_gather: MPIDI_POSIX_check_release_gather #if defined(MPIDI_CH4_SHM_POSIX)

# ----
bcast-intra:
    binomial
    smp
        restrictions: hierarchical
        inner: bcast-intra@node_comm, bcast-intra@node_roots_comm
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
alltoall-intra:
    brucks
        restrictions: not_inplace
    pairwise_sendrecv_replace
        restrictions: inplace
"""


def catalog(text=CATALOG):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write(text)
    f.close()
    return Catalog.load(f.name)


def test_catalog():
    cat = catalog()
    algs = cat.algorithms("bcast")
    assert [a.name for a in algs] == ["binomial", "smp", "tree", "release_gather", "nb"]
    bits = cat.restriction_bits()       # every non-threshold condition, file order, for every op
    assert bits == ["inplace", "not_inplace", "pof2", "hierarchical", "MPIDI_CH4_release_gather"]
    assert [a.mask(bits) for a in algs] == [0, 8, 0, 16, 0]
    a2a = cat.algorithms("alltoall")
    assert [a.mask(bits) for a in a2a] == [2, 1, 0]
    assert a2a[0].allowed(bits, 2) and a2a[0].allowed(bits, 3) and not a2a[0].allowed(bits, 1)
    assert algs[3].leaf == "MPIDI_POSIX_mpi_bcast_release_gather" and algs[0].leaf == "MPIR_Bcast_intra_binomial"
    assert algs[4].leaf == "MPIR_Coll_nb"
    assert algs[2].extra_params == ["tree_type", "k", "is_non_blocking"]
    assert algs[1].inner == [("bcast-intra", "node_comm"), ("bcast-intra", "node_roots_comm")]
    assert cat.conditions["comm_size"] and not cat.conditions["inplace"]


def test_old_style_negated_restriction():
    """A file from before positive-only restrictions: !inplace is a condition of its own."""
    cat = catalog(CATALOG.replace("    not_inplace:      MPIR_Csel_sendbuf_not_inplace\n", "")
                  .replace("restrictions: not_inplace", "restrictions: !inplace"))
    bits = cat.restriction_bits()
    assert bits == ["inplace", "pof2", "hierarchical", "MPIDI_CH4_release_gather", "!inplace"]
    cat.check("alltoall")
    assert [a.mask(bits) for a in cat.algorithms("alltoall")] == [16, 1, 0]


def test_precheck_and_sets():
    cat = catalog()
    algs = [a for a in cat.algorithms("bcast") if a.name != "smp"]
    bits = cat.restriction_bits()
    layouts = [(1, 2), (1, 4), (2, 1), (2, 4)]
    # release_gather runs on one node only; nb fails everywhere
    probe = lambda n, ppn, alg: (alg != "nb" and (alg != "release_gather" or n == 1), "no")
    feas, masks, notes = precheck(probe, layouts, algs, bits)
    assert masks == {(1, 2): 16, (1, 4): 16, (2, 1): 0, (2, 4): 0}
    assert any(n.startswith("nb fails") for n in notes)
    sets = plan_sets([a for a in algs if a.name != "nb"], bits, masks)
    assert sets[0] == (0, layouts, ["binomial", "tree"])
    assert sets[1] == (16, [(1, 2), (1, 4)], ["release_gather", "binomial", "tree"])


def test_infer_complementary():
    """inplace / not_inplace decided by which of the two alltoall algorithms ran."""
    from csel_tuner.restrict import infer_mask
    cat = catalog()
    algs, bits = cat.algorithms("alltoall", nb=False), cat.restriction_bits()
    assert infer_mask({"brucks"}, algs, bits) == (2, 0)
    assert infer_mask({"pairwise_sendrecv_replace"}, algs, bits) == (1, 0)
    assert infer_mask(set(), algs, bits) == (0, 0)      # both failed: both false


def test_dispatch_and_choice():
    from importlib.machinery import SourceFileLoader
    colltree = SourceFileLoader("colltree", os.path.join(REPO, "bin", "colltree")).load_module()
    bits = ["inplace", "not_inplace", "pof2", "hierarchical", "MPIDI_CH4_release_gather"]
    d = colltree.dispatch("bcast", bits, [0, 16])
    assert d == {"MPIDI_CH4_release_gather": "call=bcast-intra-10", "any": "call=bcast-intra-0"}
    # a set with no subtree falls back to its largest present subset (a tie: the lower mask)
    d = colltree.dispatch("bcast", bits, [0, 8, 16])
    assert d == {"hierarchical": "call=bcast-intra-8",
                 "any": {"MPIDI_CH4_release_gather": "call=bcast-intra-10", "any": "call=bcast-intra-0"}}
    trees = {"bcast-intra-auto": colltree.dispatch("bcast", bits, [0, 16]),
             "bcast-intra-0": {"comm_size(4)": {"algorithm=MPIR_Bcast_intra_binomial": {}},
                                "any": {"algorithm=MPIR_Bcast_intra_tree": {}}},
             "bcast-intra-10": {"avg_msg_size(1024)": {"algorithm=MPIDI_POSIX_mpi_bcast_release_gather": {}},
                                "any": "call=bcast-intra-0"}}
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
        assert d["format"] == "collmodel/3" and [s["mask"] for s in d["sets"]] == [0, 16]
        assert d["stage2"] == ["smp"] and d["left_out"] == ["nb"]
        assert all(l[0] == 1 for l in d["sets"][1]["layouts"])
        subprocess.run([sys.executable, os.path.join(REPO, "bin", "colltree"), model, "--out", tuning],
                       check=True, capture_output=True)
        t = json.load(open(tuning))
        assert set(t) == {"name=bcast-intra-0", "name=bcast-intra-10", "name=bcast-intra-auto"}
        assert t["name=bcast-intra-auto"] == {"MPIDI_CH4_release_gather": "call=bcast-intra-10",
                                              "any": "call=bcast-intra-0"}
        assert "MPIDI_POSIX_mpi_bcast_release_gather" in json.dumps(t["name=bcast-intra-10"])


if __name__ == "__main__":
    test_catalog()
    test_precheck_and_sets()
    test_old_style_negated_restriction()
    test_infer_complementary()
    test_dispatch_and_choice()
    test_two_set_replay()
    print("ok")
