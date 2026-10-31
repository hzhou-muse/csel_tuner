"""csel_tuner.stages: the stage of each algorithm from its inner calls; and the
catalog details the stages rely on (disabled algorithms, nonblocking leaf
names, whether the catalog has inner keys at all)."""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from csel_tuner.algos import Catalog  # noqa: E402
from csel_tuner.stages import Stages  # noqa: E402

CATALOG = """
restrictions:
    0  - pof2:                     MPIR_Csel_comm_size_is_pof2
    1  - hierarchical:             MPIR_Csel_is_hierarchical
    2  - gpu:                      MPIR_Csel_gpu

conditions:
    comm_size(thresh):      MPIR_Csel_comm_size

general:
    MPIR_Coll_nb
        inner: i$coll-intra@comm
    MPIR_Coll_buffer_swap
        restrictions: gpu
        inner: $coll-intra@comm[gpu=0]

bcast-intra:
    binomial
    smp
        restrictions: hierarchical
        inner: bcast-intra@node_comm, bcast-intra@node_roots_comm
ibcast-intra:
    sched_binomial
    sched_smp
        restrictions: hierarchical
        disabled: 1
    tsp_tree
        extra_params: tree_type, k, chunk_size
        cvar_params: TREE_TYPE, TREE_KVAL, TREE_PIPELINE_CHUNK_SIZE
"""


def catalog(text=CATALOG):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "coll_algorithms.txt")
        with open(path, "w") as f:
            f.write(text)
        return Catalog.load(path)


def stages(st, coll, launch):
    return {x.name: s for s, x, _ in st.unit_stages(coll, launch)}


def test_stages():
    st = Stages(catalog())
    # nb waits for ibcast's tree, smp for bcast's tree with nb (its inner calls can't run smp)
    assert stages(st, "bcast", "host") == {"binomial": 1, "smp": 3, "nb": 2}
    assert st.tree_stage("bcast", "host") == 3
    # buffer_swap on a gpu launch selects from the complete host tree, and smp's gpu tree has it
    assert stages(st, "bcast", "gpu") == {"binomial": 1, "smp": 5, "nb": 2, "buffer_swap": 4}
    assert stages(st, "ibcast", "host") == {"sched_binomial": 1, "tsp_tree": 1}
    # an excluded algorithm leaves the chain: without nb, smp needs only the plain tree
    assert stages(Stages(catalog(), ["nb"]), "bcast", "host") == {"binomial": 1, "smp": 2}


def test_catalog_details():
    cat = catalog()
    # disabled: MPICH refuses it forced and in a tuning file
    assert [x.name for x in cat.algorithms("ibcast")] == ["sched_binomial", "tsp_tree"]
    # the leaf names of MPICH's maint/gen_coll.py
    leaf = {x.name: x.leaf for x in cat.algorithms("ibcast")}
    assert leaf == {"sched_binomial": "MPIR_Ibcast_intra_sched_binomial",
                    "tsp_tree": "MPIR_TSP_Ibcast_sched_intra_tree"}
    # leaf parameters at the server's CVAR values; "use_ring=0" is a constant, THROTTLE is alltoall's
    tsp = [x for x in cat.algorithms("ibcast") if x.name == "tsp_tree"][0]
    cv = {"MPIR_CVAR_IBCAST_TREE_TYPE": "kary", "MPIR_CVAR_IBCAST_TREE_KVAL": 4,
          "MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE": 0}
    assert tsp.params(cv) == ({"tree_type": "kary", "k": 4, "chunk_size": 0}, [])
    assert tsp.params({}) == ({}, ["MPIR_CVAR_IBCAST_TREE_TYPE", "MPIR_CVAR_IBCAST_TREE_KVAL",
                                   "MPIR_CVAR_IBCAST_TREE_PIPELINE_CHUNK_SIZE"])
    from csel_tuner.algos import Alg
    x = Alg("ialltoallv", "tsp_scattered", {"extra_params": "batch_size, use_ring=0, bblock",
                                             "cvar_params": "SCATTERED_BATCH_SIZE, -, THROTTLE"})
    assert x.params({"MPIR_CVAR_IALLTOALLV_SCATTERED_BATCH_SIZE": 4, "MPIR_CVAR_ALLTOALL_THROTTLE": 32}) == \
        ({"batch_size": 4, "bblock": 32}, [])
    assert cat.has_inner()
    assert not catalog(CATALOG.replace("        inner: bcast-intra@node_comm, bcast-intra@node_roots_comm\n",
                                       "")).has_inner()


if __name__ == "__main__":
    test_stages()
    test_catalog_details()
    print("ok")
