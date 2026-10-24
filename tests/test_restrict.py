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
restrictions:
    0  - inplace:                  MPIR_Csel_sendbuf_inplace
    1  - not_inplace:              MPIR_Csel_sendbuf_not_inplace
    2  - pof2:                     MPIR_Csel_comm_size_is_pof2
    3  - hierarchical:             MPIR_Csel_is_hierarchical
    # conditional conditions
    4  - MPIDI_CH4_release_gather: MPIDI_POSIX_check_release_gather #if defined(MPIDI_CH4_SHM_POSIX)

conditions:
    comm_size(thresh):      MPIR_Csel_comm_size

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
    # an index that is not the next one is an error
    try:
        catalog(CATALOG.replace("    3  - hierarchical", "    5  - hierarchical"))
        assert False, "accepted a wrong index"
    except ValueError:
        pass


OLD_CONDITIONS = """conditions:
    inplace:          MPIR_Csel_sendbuf_inplace
    pof2:             MPIR_Csel_comm_size_is_pof2
    hierarchical:     MPIR_Csel_is_hierarchical
    comm_size(thresh):      MPIR_Csel_comm_size
    MPIDI_CH4_release_gather: MPIDI_POSIX_check_release_gather #if defined(MPIDI_CH4_SHM_POSIX)
"""


def test_old_style_negated_restriction():
    """A file from before the restrictions section and positive-only restrictions: the
    non-threshold conditions in file order, and !inplace a condition of its own."""
    a, b = CATALOG.index("restrictions:\n    0"), CATALOG.index("# ----\nbcast-intra")
    cat = catalog((CATALOG[:a] + OLD_CONDITIONS + "\n" + CATALOG[b:])
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
    # the restriction-mask node chooses the same; set 0x8 (hierarchical) has no subtree: its subset 0
    trees["bcast-intra-auto"] = "dispatch=bcast-intra(0x18)"
    for p, m, rg, want in [(4, 64, True, "MPIDI_POSIX_mpi_bcast_release_gather"),
                           (4, 4096, True, "MPIR_Bcast_intra_binomial"), (8, 64, False, "MPIR_Bcast_intra_tree")]:
        assert json_choice(trees["bcast-intra-auto"], x(p, m, rg), trees, bits) == want
    xh = dict(x(8, 64, False), cond={"hierarchical": True, "MPIDI_CH4_release_gather": False})
    assert json_choice(trees["bcast-intra-auto"], xh, trees, bits) == "MPIR_Bcast_intra_tree"


def test_masks_and_fixed_point():
    """A compositional algorithm prechecked into a model: its restriction moves
    the layouts where it runs to a new set; the two-stage check."""
    from csel_tuner.model import Model
    from csel_tuner.restrict import fixed_point
    cat = catalog()
    bits = cat.restriction_bits()
    algs = [a for a in cat.algorithms("bcast") if a.name in ("binomial", "smp", "release_gather")]
    smp = [a for a in algs if a.name == "smp"]
    layouts = [(1, 4), (2, 1), (2, 4), (4, 2)]
    m = Model("bcast", [1, 2], layouts, {}, bits)
    for lay in layouts:
        m.add_precheck(0, "binomial", lay[0], lay[1], True)
        m.add_precheck(0, "release_gather", lay[0], lay[1], lay[0] == 1)
    m.update_masks(algs)
    assert m.masks == {(1, 4): 16, (2, 1): 0, (2, 4): 0, (4, 2): 0}
    for lay in layouts:
        m.add_precheck(1, "smp", lay[0], lay[1], lay[0] > 1 and lay[1] > 1)
    m.update_masks(algs)
    assert m.masks == {(1, 4): 16, (2, 1): 0, (2, 4): 8, (4, 2): 8}
    assert m.sets() == {0: layouts, 8: [(2, 4), (4, 2)], 16: [(1, 4)]}
    feas = m.feasible()
    assert fixed_point(smp, feas, "bcast") == ([], [])
    # smp running on its own node_comm layout (1, 4) would need its own file to be measured with
    feas[(1, 4)] = feas[(1, 4)] | {"smp"}
    problems, _ = fixed_point(smp, feas, "bcast")
    assert problems and "node_comm" in problems[0]


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
        from csel_tuner.model import Model
        d = Model.load(model)
        assert sorted(d.sets()) == [0, 16] and d.left_out == ["nb"] and "smp" not in d.algorithms
        assert all(l[0] == 1 for l in d.sets()[16]) and set(d.curves["release_gather"]) <= set(d.sets()[16])
        subprocess.run([sys.executable, os.path.join(REPO, "bin", "colltree"), model, "--out", tuning],
                       check=True, capture_output=True)
        t = json.load(open(tuning))
        assert set(t) == {"name=bcast-intra-0", "name=bcast-intra-10", "name=bcast-intra-auto"}
        assert t["name=bcast-intra-auto"] == "dispatch=bcast-intra(0x10)"
        assert "MPIDI_POSIX_mpi_bcast_release_gather" in json.dumps(t["name=bcast-intra-10"])
        subprocess.run([sys.executable, os.path.join(REPO, "bin", "colltree"), model, "--out", tuning,
                        "--nested-dispatch"], check=True, capture_output=True)
        t = json.load(open(tuning))
        assert t["name=bcast-intra-auto"] == {"MPIDI_CH4_release_gather": "call=bcast-intra-10",
                                              "any": "call=bcast-intra-0"}


def test_compositional_add_replay():
    """smp (replayed from scatter_ring_allgather on the layouts with several
    nodes and ranks per node) added to a model with the first tuning file
    loaded: a set 0x8 (hierarchical in the fixture's numbering) over smp and
    the algorithms allowed there, and a file that dispatches to it."""
    if not have_data():
        print("skipped: no replay data")
        return
    from csel_tuner.model import Model
    with tempfile.TemporaryDirectory() as tmp:
        cat = os.path.join(tmp, "coll_algorithms.txt")
        with open(cat, "w") as f:
            f.write(CATALOG)
        m1, t1, m2, t2 = (os.path.join(tmp, x) for x in ("m1.json", "t1.json", "m2.json", "t2.json"))
        run = lambda *args: subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True)
        alias = ["--replay-alias", "release_gather=auto@single", "--replay-alias", "smp=scatter_ring_allgather@hier"]
        run(os.path.join(REPO, "bin", "collmodel"), "--replay", "--algs", "all", "--coll-algorithms", cat,
            *alias, "--budget-node-h", "0.3", "--out", m1)
        run(os.path.join(REPO, "bin", "colltree"), m1, "--out", t1)
        run(os.path.join(REPO, "bin", "collmodel"), "--replay", "--model", m1, "--algs", "smp", "--coll-algorithms",
            cat, *alias, "--inner-selection", t1, "--out", m2)
        d = Model.load(m2)
        assert sorted(d.sets()) == [0, 8, 16] and all(n > 1 and q > 1 for n, q in d.sets()[8])
        assert d.curves["smp"] and set(d.curves["smp"]) <= set(d.sets()[8])
        h = d.fit_inner["smp"]
        assert d.inner_files[h] == open(t1).read()
        assert all(r.get("inner") == h for r in d.measurements if r["alg"] == "smp")
        assert d.runs[-1]["algs"] == ["smp"] and d.runs[-1]["anchors"]["points"] > 0
        r = run(os.path.join(REPO, "bin", "colltree"), m2, "--out", t2)
        t = json.load(open(t2))
        assert t["name=bcast-intra-auto"] == "dispatch=bcast-intra(0x18)"
        assert "MPIR_Bcast_intra_smp" in json.dumps(t["name=bcast-intra-8"])
        # a compositional algorithm needs the file of its inner calls when live; a second run with
        # another file replaces its curves
        r = run(os.path.join(REPO, "bin", "collmodel"), "--replay", "--model", m2, "--algs", "smp",
                "--coll-algorithms", cat, *alias, "--inner-selection", t2, "--anchors", "0",
                "--out", os.path.join(tmp, "m3.json"))
        assert "its curves are replaced" in r.stderr
        d3 = Model.load(os.path.join(tmp, "m3.json"))
        assert d3.fit_inner["smp"] != h and d3.inner_files[d3.fit_inner["smp"]] == open(t2).read()


if __name__ == "__main__":
    test_catalog()
    test_precheck_and_sets()
    test_old_style_negated_restriction()
    test_infer_complementary()
    test_dispatch_and_choice()
    test_two_set_replay()
    test_masks_and_fixed_point()
    test_compositional_add_replay()
    print("ok")
