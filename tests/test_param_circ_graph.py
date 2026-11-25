"""The circ_graph param rule (csel_tuner/param_circ_graph.py): its values, pinned for its version,
and its calibration fit on synthetic rows."""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from csel_tuner import param_circ_graph as cg  # noqa: E402
from csel_tuner.param_circ_graph import Cells, chunk_row, fit_chunk, fit_qlen  # noqa: E402
from csel_tuner.params import K, M, ParamMap, parse_size, pow2  # noqa: E402


def test_rule():
    b = ParamMap.default("bcast", "circ_graph")
    assert b.theta(2, 4 * M) == {"chunk_size": 0, "q_len": 8}                 # p = 2: one block
    assert b.theta(64, 16 * K)["chunk_size"] == 16 * K                         # max(F, m/8)
    assert b.theta(64, 1 * M)["chunk_size"] == 128 * K                         # sqrt(16K * 1M)
    assert b.theta(64, 256 * K)["chunk_size"] == 64 * K                        # floor Fb
    assert b.theta(64, 64 * M) == {"chunk_size": 512 * K, "q_len": 16}         # cap C; q_len above Mq
    r = ParamMap.default("allreduce", "circ_graph")
    assert r.theta(2, 4 * M)["chunk_size"] == 256 * K                          # no one-block rule
    assert pow2(2 ** 10.5) == 2 ** 11 and pow2(2 ** 10.49) == 2 ** 10 and pow2(0.3) == 1
    assert parse_size("64K") == 65536 and parse_size("16M") == 16 * M and parse_size("8") == 8
    th = b.theta_arrays(np.array([2.0, 64.0]), np.array([4.0 * M, 1.0 * M]))
    assert th["chunk_size"].tolist() == [0, 128 * K]


# The circ_graph rule's output, pinned for its version. A change to chunk() or qlen() in
# param_circ_graph.py that changes any of these values must bump param_circ_graph.VERSION (maps of
# the old version are then refused: they were measured with other parameters) and update the table.
PINNED_VERSION = 1
PINNED_SIZES = [1, 1024, 8192, 65536, 65537, 98304, 262144, 1048576, 3000000, 16777216, 67108864]
PINNED_BCAST = {        # defaults; bcast-like (one block at p <= 2)
    2: [(0, 8)] * 9 + [(0, 16)] * 2,
    3: [(16384, 8)] * 4 + [(65536, 8)] * 3 + [(131072, 8), (262144, 8), (524288, 16), (524288, 16)],
}
PINNED_ALLREDUCE = [(8192, 4)] * 3 + [(16384, 4)] * 2 + [(32768, 4)] * 2 + \
    [(65536, 4), (131072, 4), (262144, 32), (262144, 32)]       # every p
PINNED_CONSTS = dict(F=8 * K, A=4 * K, Fb=32 * K, C=256 * K, D=4, M0=128 * K, Q=4, Qb=32, Mq=4 * M)
PINNED_HASH = {"bcast": "aaf898766df59e10", "allreduce": "18578fa1ef339f32"}


def test_rule_pinned():
    assert cg.VERSION == PINNED_VERSION, "a new rule version: update the pinned table"
    b = ParamMap.default("bcast", "circ_graph")
    r = ParamMap.default("allreduce", "circ_graph", **PINNED_CONSTS)
    for p in (2, 3, 8, 96, 3072):
        got = [tuple(b.theta(p, m).values()) for m in PINNED_SIZES]
        assert got == PINNED_BCAST[min(p, 3)], (p, got)
        got = [tuple(r.theta(p, m).values()) for m in PINNED_SIZES]
        assert got == PINNED_ALLREDUCE, (p, got)
    # the hash covers the algorithm, the rule, its version and the constants: maps in files keep it
    assert (b.hash, r.hash) == (PINNED_HASH["bcast"], PINNED_HASH["allreduce"])


ROWS = {(1, 4): [16 * K, 64 * K, 4 * M], (2, 1): [16 * K, 64 * K, 4 * M], (2, 4): [256 * K, 512 * K, 1 * M]}


def synth_t(pm, n, q, m, c):
    """A time whose best chunk is the rule's (a V in log2 chunk around it)."""
    best = pm.theta(n * q, m)["chunk_size"]
    best = m if best == 0 or best >= m else best
    e = m if c == 0 or c >= m else c
    return 100.0 * (1 + 0.1 * abs(np.log2(e) - np.log2(best)))


def synth_cells(pm):
    return [{"N": n, "ppn": q, "m": m, "t": {c: synth_t(pm, n, q, m, c) for c in chunk_row(m)}}
            for (n, q), ms in ROWS.items() for m in ms]


def test_fit():
    truth = ParamMap.default("bcast", "circ_graph", F=8 * K, A=4 * K, Fb=32 * K, C=256 * K)
    cells = synth_cells(truth)
    consts, rep = fit_chunk(cells, "bcast")
    assert rep["regret"]["max"] == 1.0, rep
    assert rep["default_regret"]["max"] > 1.0
    # a fixed constant is kept
    consts, rep = fit_chunk(cells, "bcast", fixed={"A": 64 * K})
    assert consts["A"] == 64 * K
    # Cells: chunk 0 and chunks >= m are one block; a chunk between measured ones takes the nearest
    t = Cells([{"N": 2, "ppn": 4, "m": 64 * K, "t": {16 * K: 2.0, 32 * K: 1.0, 64 * K: 4.0}}])
    assert t.regret([0])[0] == 4.0 and t.regret([1 << 30])[0] == 4.0 and t.regret([8 * K])[0] == 2.0
    # q_len: the best large-message value, unless it gains less than tol over Q
    c = ParamMap.default("bcast", "circ_graph").consts
    qb, _ = fit_qlen([{"N": 2, "ppn": 4, "m": 16 * M, "t": {4: 1.2, 8: 1.1, 16: 1.0}}], c)
    assert qb == 16
    qb, _ = fit_qlen([{"N": 2, "ppn": 4, "m": 16 * M, "t": {4: 1.2, 8: 1.01, 16: 1.0}}], c)
    assert qb == 8
    qb, rep = fit_qlen([], c)
    assert qb == c["Qb"] and rep["cells"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
