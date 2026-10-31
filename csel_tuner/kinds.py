"""The kinds of MPICH's restrictions, and the call variants of a collective.

    kinds, notes = classify(bits)          # {restriction: "input" | "launch" | "set"}, [warnings]
    variants("reduce", bits)               # [{"inplace": 0, "builtin_op": 1, "commutative": 1}, ...]
    run_variant("reduce", inplace=True)    # the assignment mpisrv measures
    datatype("allreduce")                  # "int": mpisrv's element type for the collective
    measured(v), launch_known(bits, mem)

The catalog lists the restrictions but not their kinds: how a tuning tool
treats a condition is its own policy (docs/design.md, section 11):

- input: a property of the call's arguments that a request can set (RECIPES).
  Each assignment of a collective's input conditions is a variant, a model
  of its own. An input condition applies to a collective where MPI allows it
  (in-place where the collective takes MPI_IN_PLACE, the op conditions for
  the reductions), whether or not an algorithm restricts on it, because it
  changes the algorithms' curves.
- launch: decided by how the ranks are started (LAUNCH): `gpu` and
  `gpu_coll` on device buffers. The launch kind is a tag of its own.
- set: everything else, the layout conditions (pof2, hierarchical, ...) and
  the black-box conditions (MPIDI_CH4_release_gather, ...). The precheck
  finds which hold at each layout, and they form restriction sets inside a
  model. Conditions that the requests always satisfy (displs_ordered,
  block_regular) come out true wherever an algorithm uses them.

A restriction in none of the tables is taken as a set condition, with a
warning: it may be a new input condition that needs a recipe.
"""

REDUCTIONS = ["reduce", "allreduce", "reduce_scatter", "reduce_scatter_block", "scan", "exscan"]
# the v-operations: their blocks follow mpisrv's counts= pattern (a tag of their models)
VCOUNTS = ["allgatherv", "alltoallv", "gatherv", "scatterv", "reduce_scatter"]
INPLACE = ["gather", "gatherv", "scatter", "scatterv", "allgather", "allgatherv", "alltoall", "alltoallv",
           "alltoallw"] + REDUCTIONS


def _with_nb(colls):
    return colls + ["i" + c for c in colls]


# input condition -> (the collectives it applies to, {value: request settings, or None: not measured})
RECIPES = {
    "inplace": (_with_nb(INPLACE), {0: {}, 1: {"inplace": 1}}),
    # user ops are not measured: colltree takes the default tree's subtree for them
    "commutative": (_with_nb(REDUCTIONS), {0: None, 1: {}}),
    "builtin_op": (_with_nb(REDUCTIONS), {0: None, 1: {"op": "sum"}}),
}
# (if, then): an assignment with the first true and the second false is impossible
IMPLIES = [("builtin_op", "commutative"), ("gpu_coll", "gpu")]
# launch kind (the model's memory kind) -> the conditions it decides
# gpu: device buffers with MPIR_CVAR_ENABLE_GPU_COLL=1 (Q4); device: device buffers without it (the
# launch of --mem device, before launch kinds)
LAUNCH = {"host": {"gpu": 0, "gpu_coll": 0}, "gpu": {"gpu": 1, "gpu_coll": 1}, "device": {"gpu": 1}}
LAUNCH_CONDITIONS = ["gpu", "gpu_coll"]
# the set conditions of the current catalog, so that a new restriction is noticed
KNOWN_SET = ["pof2", "hierarchical", "node_consecutive", "node_regular", "displs_ordered", "block_regular",
             "MPIDI_CH4_release_gather"]


def classify(bits):
    """({restriction: kind}, [warnings]) for the catalog's restrictions (algos.Catalog.restriction_bits())."""
    kinds, notes = {}, []
    for r in bits:
        if r in RECIPES:
            kinds[r] = "input"
        elif r in LAUNCH_CONDITIONS:
            kinds[r] = "launch"
        else:
            kinds[r] = "set"
            if r not in KNOWN_SET:
                notes.append("restriction %s has no kind; treated as a set condition (an input condition "
                             "needs a recipe in csel_tuner/kinds.py)" % r)
    return kinds, notes


def consistent(v):
    return all(not (v.get(a) == 1 and v.get(b) == 0) for a, b in IMPLIES)


def variants(coll, bits):
    """Every consistent assignment of the collective's input conditions, in a fixed order."""
    names = [r for r in bits if r in RECIPES and coll in RECIPES[r][0]]
    out = [{}]
    for r in names:
        out = [dict(v, **{r: x}) for v in out for x in (0, 1)]
    return [v for v in out if consistent(v)]


def measured(v):
    """A variant is measured when every one of its values has a recipe."""
    return all(RECIPES[r][1][x] is not None for r, x in v.items())


def request(v):
    """The request settings of a measured variant."""
    out = {}
    for r, x in v.items():
        out.update(RECIPES[r][1][x])
    return out


def launch_known(bits, mem):
    """(mask known true, mask known false) of the conditions the launch kind decides."""
    t = f = 0
    for r, x in LAUNCH[mem].items():
        if r in bits:
            if x:
                t |= 1 << bits.index(r)
            else:
                f |= 1 << bits.index(r)
    return t, f


def run_variant(coll, inplace=False):
    """The input-condition assignment a request of mpisrv has: the reductions sum MPI_INTs with
    MPI_SUM (builtin and commutative), and inplace=1 or not where MPI allows MPI_IN_PLACE."""
    return {r: int(inplace) if r == "inplace" else 1 for r, (colls, _) in RECIPES.items() if coll in colls}


# the collectives with a message size the same on every rank (MPICH's msg_size condition,
# MPIR_Csel_has_msg_size): the others (the v/w collectives whose ranks know only their own counts,
# barrier) cannot split on size, and MPICH refuses a tuning file in which they reach msg_size
SIZED = _with_nb(["bcast", "reduce", "allreduce", "scan", "exscan", "allgather", "allgatherv", "alltoall",
                  "gather", "scatter", "reduce_scatter", "reduce_scatter_block"])


def has_msg_size(coll):
    return coll in SIZED


def is_v(coll):
    return coll in _with_nb(VCOUNTS)


def datatype(coll):
    """mpisrv's element type: MPI_INT for the reductions, bytes for the rest."""
    return "int" if coll in _with_nb(REDUCTIONS) else "byte"


def is_device(launch):
    """Whether a launch kind measures device buffers (mpisrv --mem device)."""
    return LAUNCH.get(launch, {}).get("gpu") == 1
