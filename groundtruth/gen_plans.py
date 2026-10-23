#!/usr/bin/env python3
"""Generate mpisrv batch plans for dense bcast ground truth and debug checks
(see docs/usage.md, "Ground truth and checks").

    python3 gen_plans.py --set debug  --out plans/debug
    python3 gen_plans.py --set A      --out plans/setA
    python3 gen_plans.py --set B      --out plans/setB --per-octave 4 \\
        --N 1,2,3,4,6,8,12,16,24,32 --ppn 1,2,4,8,16,24,32,48,96 --sessions 3
    python3 gen_plans.py --set Rdebug --out plans/Rdebug
    python3 gen_plans.py --set R      --out plans/setR
    python3 gen_plans.py --set Ndebug --out plans/Ndebug
    python3 gen_plans.py --set Idebug --out plans/Idebug
    python3 gen_plans.py --set switch --out plans/switch --nodes 8
    python3 gen_plans.py --set validate --out plans/validate --nodes 8

Writes <out>/launches.txt, one launch per line in priority order:
    id  nodes  ppn  alg  idle  plan-file
(idle = MODE[:IDLE_NS[:SPINUP_US]], see run_set.sh) and <out>/<id>.plan with
one mpisrv request per line, in random order (set Idebug: grouped by layout).
Standard library only.
"""
import argparse
import os
import random

ALGS = ["auto", "binomial", "scatter_recursive_doubling_allgather", "scatter_ring_allgather",
        "circ_graph", "tree", "pipelined_tree"]
CHECK_SIZES = [12347, 4194311]  # odd sizes, data-checked once per layout


def sizes_per_octave(per_octave, max_m):
    """1 B .. max_m, `per_octave` log-spaced sizes per octave, unique ints."""
    out, k = set(), 0
    while True:
        m = round(2 ** (k / per_octave))
        if m > max_m:
            break
        out.add(m)
        k += 1
    return sorted(out | {max_m})


ALG_CVAR = "MPIR_CVAR_BCAST_INTRA_ALGORITHM"


def bcast(m, n, ppn, tag, batches, nodes=None, check=False, sync=None, place=None, alg=None):
    """One request line; alg: the algorithm for this request only (cvar.*,
    needs the runtime CVAR switch), else the launch's."""
    s = f"op=bcast m={m} p={n * ppn} ppn={ppn}"
    if alg:
        s += f" cvar.{ALG_CVAR}={alg}"
    if place:
        s += f" place={place}"
    if nodes:
        s += f" nodes={nodes}"
    if sync:
        s += f" sync={sync}"
    s += " check=1 batches=1 target_us=100" if check else f" batches={batches}"
    return s + f" tag={tag}"


def layout_requests(layouts, sizes, batches, tag, reps=1, nodes=None, check=True, sync=None,
                    place=None):
    """layouts: list of (N, ppn). Data checks first per layout, then the grid."""
    req = []
    for n, ppn in layouts:
        if check:
            req += [bcast(m, n, ppn, tag + "-check", 1, nodes, True, place=place) for m in CHECK_SIZES
                    if m <= max(sizes)]
        for _ in range(reps):
            req += [bcast(m, n, ppn, tag, batches, nodes, sync=sync, place=place) for m in sizes]
    return req


def write(out, launches, rng, shuffle=True):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "launches.txt"), "w") as f:
        f.write("# id nodes ppn alg idle plan\n")
        for lid, nodes, ppn, alg, idle, reqs in launches:
            checks = [r for r in reqs if "check=1" in r]
            grid = [r for r in reqs if "check=1" not in r]
            if shuffle:
                rng.shuffle(grid)
            plan = os.path.join(out, f"{lid}.plan")
            with open(plan, "w") as p:
                p.write(f"# launch {lid}: nodes={nodes} ppn={ppn} alg={alg} idle={idle}\n")
                p.write("\n".join(checks + grid) + "\n")
            f.write(f"{lid} {nodes} {ppn} {alg} {idle} {plan}\n")
    total = sum(len(r) for *_, r in launches)
    print(f"{out}: {len(launches)} launches, {total} requests")


def set_debug(a, rng):
    """2 nodes: every algorithm briefly, plus a low-ppn launch (binding check)."""
    sizes = sizes_per_octave(0.5, 1 << 20)
    lay = [(1, 8), (1, 96), (2, 1), (2, 8), (2, 96)]
    launches = [(f"debug-{alg}", 2, 96, alg, "spin", layout_requests(lay, sizes, 3, f"debug-{alg}"))
                for alg in ALGS]
    launches.append(("debug-ppn8", 2, 8, "binomial", "spin",
                     layout_requests([(2, 8)], sizes, 3, "debug-ppn8")))
    write(a.out, launches, rng)


def set_a(a, rng):
    """Set A: coarse cost grid per algorithm (A2) with a back-to-back vs.
    per-call-barrier comparison (A7), interference (A4), noise and placement
    (A3, A5). Sizes every 2 octaves up to --max-m."""
    sizes = sizes_per_octave(0.5, a.max_m)
    nodes = a.nodes
    ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= nodes]
    lay = [(n, ppn) for n in ns for ppn in (1, 8, 32, 96) if n * ppn > 1 and ppn <= a.ppn]
    launches = []
    seven = sorted({(2, a.ppn), (nodes, 1), (nodes, a.ppn)})             # A7
    for alg in ALGS:                                                    # A2
        req = layout_requests(lay, sizes, 5, f"A2-{alg}")
        req += layout_requests(seven, sizes, 5, f"A7-{alg}", check=False, sync="call")
        launches.append((f"A2-{alg}", nodes, a.ppn, alg, "spin", req))
    low = sorted({1, min(8, a.ppn)})                                     # A4
    four = [(nodes, ppn) for ppn in low]
    launches.append(("A4-sleep", nodes, a.ppn, "binomial", "sleep",
                     layout_requests(four, sizes, 5, "A4-sleep", reps=2, check=False)))
    for ppn in low:
        launches.append((f"A4-ppn{ppn}", nodes, ppn, "binomial", "spin",
                         layout_requests([(nodes, ppn)], sizes, 5, f"A4-ppn{ppn}", reps=2, check=False)))
    noise = [(n, ppn) for n in (2, nodes) for ppn in (1, a.ppn)]         # A3 (+A5)
    sub = max(1, nodes // 4)
    for alg in ("binomial", "auto"):
        for s in range(3):
            req = layout_requests(noise, sizes, 5, f"A3-{alg}", reps=2, check=False)
            if alg == "binomial":
                for off, stride in [(i * sub, 1) for i in range(4)] + [(0, 4)]:
                    req += layout_requests([(sub, 1), (sub, a.ppn)], sizes, 5,
                                           f"A5-{off}:{stride}", nodes=f"{off}:{stride}", check=False)
            launches.append((f"A3-{alg}-s{s}", nodes, a.ppn, alg, "spin", req))
    write(a.out, launches, rng)


def set_b(a, rng):
    """Set B: dense grid, one launch per algorithm per session."""
    sizes = sizes_per_octave(a.per_octave, a.max_m)
    ns = [int(v) for v in a.N.split(",")]
    ppns = [int(v) for v in a.ppn_list.split(",")]
    lay = [(n, ppn) for n in ns for ppn in ppns if n * ppn > 1]
    launches = []
    for s in range(a.sessions):  # session-major, so a cut-off keeps every algorithm
        for alg in ALGS:
            launches.append((f"B-s{s}-{alg}", a.nodes, a.ppn, alg, "spin",
                             layout_requests(lay, sizes, a.batches, f"B-{alg}", check=(s == 0),
                                             sync=a.sync)))
    write(a.out, launches, rng)


FIXED = [a for a in ALGS if a != "auto"]


def r_layouts():
    """r-sweep: extra processes r = p - 2^floor(log2 p) at a fixed power-of-two base."""
    lay = [(n, ppn) for ppn in (1, 8) for n in (16, 17, 18, 20, 24, 28, 32)]      # node sweep
    lay += [(16, ppn) for ppn in (16, 17, 18, 20, 24, 28, 32)]                    # process sweep
    lay += [(1, ppn) for ppn in (32, 33, 34, 36, 40, 48, 56, 64)]                 # single node
    return list(dict.fromkeys(lay))


def set_rdebug(a, rng):
    """2 nodes: every fixed algorithm on a few non-divisor ppn, with data checks."""
    sizes = sizes_per_octave(0.5, 1 << 20)
    lay = [(1, 17), (1, 33), (1, 64), (2, 17), (2, 24)]
    launches = [(f"Rdebug-{alg}", 2, 96, alg, "spin",
                 layout_requests(lay, sizes, 3, f"Rdebug-{alg}", sync="call")) for alg in FIXED]
    write(a.out, launches, rng)


def set_r(a, rng):
    """Set R (r-sweep): 6 fixed algorithms, --r-per-octave sizes, per-call timing,
    one launch per algorithm per session; data checks in session 0."""
    sizes = sizes_per_octave(a.r_per_octave, a.max_m)
    lay = r_layouts()
    launches = []
    for s in range(a.r_sessions):
        for alg in FIXED:
            launches.append((f"R-s{s}-{alg}", a.nodes, a.ppn, alg, "spin",
                             layout_requests(lay, sizes, 5, f"R-{alg}", check=(s == 0), sync="call")))
    write(a.out, launches, rng)


NIC_PPN = [8, 16, 24, 48]  # ppn whose place=spread NIC load differs from a real job


def set_ndebug(a, rng):
    """NIC placement check on 2 nodes: the same layouts as sub-communicators of
    a 96-rank launch with place=spread and place=nic, and as real ppn-rank
    launches (place=compact, the reference). Session-major, 2 sessions.
    circ_graph is left out."""
    sizes = sizes_per_octave(1, a.max_m)
    sizes = [m for m in sizes if m >= 1024]
    launches = []
    for s in range(2):
        for alg in [x for x in FIXED if x != "circ_graph"]:
            req = []
            for place in ("spread", "nic"):
                req += layout_requests([(2, q) for q in NIC_PPN], sizes, 5, f"N-{place}-{alg}",
                                       check=(s == 0), sync="call", place=place)
            launches.append((f"N-s{s}-full-{alg}", 2, 96, alg, "spin", req))
            for q in NIC_PPN:
                launches.append((f"N-s{s}-real{q}-{alg}", 2, q, alg, "spin",
                                 layout_requests([(2, q)], sizes, 5, f"N-real-{alg}", check=(s == 0),
                                                 sync="call", place="compact")))
    write(a.out, launches, rng)


IDLE_PPN = [8, 24, 48]
IDLE_ALGS = ["binomial", "scatter_recursive_doubling_allgather", "pipelined_tree"]
# full-launch idle variants: (name, run_set idle field)
IDLE_VARIANTS = [("spin", "spin"),                  # the default: MPI_Test every 50 us
                 ("spin1ms", "spin:1000000"),       # same, MPI_Test every 1 ms
                 ("split", "split:1000000:30000"),  # non-members sleep, cold members spin up 30 ms
                 ("busy", "busy")]                  # blocking MPI_Bcast


def grouped(layouts, sizes, tag, rng, check, place=None):
    """Requests grouped by layout (layouts in random order, sizes shuffled
    within each), data checks first, all sync=call with 5 batches."""
    req = []
    if check:
        req = [bcast(m, n, q, tag + "-check", 1, check=True, place=place)
               for n, q in layouts for m in CHECK_SIZES if m <= max(sizes)]
    lay = list(layouts)
    rng.shuffle(lay)
    for n, q in lay:
        ms = list(sizes)
        rng.shuffle(ms)
        req += [bcast(m, n, q, tag, 5, sync="call", place=place) for m in ms]
    return req


def set_idebug(a, rng):
    """Idle-rank check on 2 nodes: do sub-communicator layouts of a full
    launch match real launches, and does the idle mode matter? Same layouts,
    measured (1) in 2 x 96 launches under four idle modes,
    (2) in a 2 x 48 launch (fewer idle ranks), (3) as real launches (the
    reference: 2 x ppn, or 1 x ppn for the single-node layouts). 3
    algorithms, 1 B - max_m one size per octave, 2 sessions, session-major.
    Requests are grouped by layout so split mode keeps its members warm."""
    sizes = sizes_per_octave(1, a.max_m)
    full = [(2, q) for q in IDLE_PPN] + [(1, IDLE_PPN[0]), (1, IDLE_PPN[-1])]
    half = [(2, IDLE_PPN[0]), (2, IDLE_PPN[1]), (1, IDLE_PPN[0])]
    launches = []
    for s in range(2):
        for alg in IDLE_ALGS:
            for name, idle in IDLE_VARIANTS:
                launches.append((f"I-s{s}-full-{name}-{alg}", 2, a.ppn, alg, idle,
                                 grouped(full, sizes, f"I-full-{name}-{alg}", rng, s == 0)))
            launches.append((f"I-s{s}-half-spin-{alg}", 2, a.ppn // 2, alg, "spin",
                             grouped(half, sizes, f"I-half-spin-{alg}", rng, s == 0)))
            for n, q in full:
                launches.append((f"I-s{s}-real{n}x{q}-{alg}", n, q, alg, "spin",
                                 grouped([(n, q)], sizes, f"I-real-{alg}", rng, s == 0,
                                         place="compact")))
    write(a.out, launches, rng, shuffle=False)


def axis_values(top):
    """1, 2, 3, 4, 6, 8, 12, ... up to top, and top itself (csel_tuner.layouts)."""
    out, k = {top}, 1
    while k <= top:
        out.add(k)
        if 3 * k // 2 <= top and k >= 2:
            out.add(3 * k // 2)
        k *= 2
    return sorted(out)


def set_switch(a, rng):
    """Per-request algorithm switch check: the same requests with the
    algorithm chosen per request (cvar.*, one launch, "S-req") and for the
    whole launch (environment, one launch per algorithm, "S-env-<alg>"); the
    times must agree. Layouts include single-node non-power-of-two ppn,
    where circ_graph used to hang, and every algorithm's data is checked."""
    sizes = sizes_per_octave(0.5, a.max_m)
    lay = [l for l in [(2, 8), (a.nodes, a.ppn), (1, 24), (1, 48)] if l[0] <= a.nodes and l[1] <= a.ppn]
    req = []
    for alg in FIXED:
        for n, q in lay:
            req += [bcast(m, n, q, "S-req-check", 1, check=True, alg=alg) for m in CHECK_SIZES if m <= a.max_m]
            req += [bcast(m, n, q, "S-req", 5, sync="call", alg=alg) for m in sizes]
    launches = [("S-req", a.nodes, a.ppn, "auto", "spin", req)]
    for alg in FIXED:
        launches.append((f"S-env-{alg}", a.nodes, a.ppn, alg, "spin",
                         layout_requests(lay, sizes, 5, f"S-env-{alg}", sync="call")))
    write(a.out, launches, rng)


def set_validate(a, rng):
    """Validation of a tuning file, at points the model builder did not
    choose: --v-layouts layouts drawn at random from those it may use (N and
    ppn in 1, 2, 3, 4, 6, 8, 12, ...), --v-per-octave sizes per octave.
    Three plan directories, one launch each:
      truth/    every algorithm, per request (cvar.*): the fastest at each point
      default/  MPICH's own selection (auto)
      tuned/    auto again, to be run with MPIR_CVAR_COLL_SELECTION_JSON_FILE set
    groundtruth/score_live.py validate scores them. --v-algs replaces the
    algorithms of truth/; those in --v-single-node-algs (e.g. release_gather)
    are measured on the single-node layouts only."""
    allowed = [(n, q) for n in axis_values(a.nodes) for q in axis_values(a.ppn) if n * q > 1]
    lay = sorted(rng.sample(allowed, min(a.v_layouts, len(allowed))))
    sizes = sizes_per_octave(a.v_per_octave, a.max_m)
    algs = a.v_algs.split(",") if a.v_algs else FIXED
    single = set(a.v_single_node_algs.split(",")) if a.v_single_node_algs else set()
    truth = []
    for alg in algs + sorted(single - set(algs)):
        truth += [bcast(m, n, q, "V-truth", 5, sync="call", alg=alg) for n, q in lay for m in sizes
                  if alg not in single or n == 1]
    plain = lambda tag: [bcast(m, n, q, tag, 5, sync="call") for n, q in lay for m in sizes]
    write(os.path.join(a.out, "truth"), [("V-truth", a.nodes, a.ppn, "auto", "spin", truth)], rng)
    write(os.path.join(a.out, "default"), [("V-default", a.nodes, a.ppn, "auto", "spin", plain("V-default"))], rng)
    write(os.path.join(a.out, "tuned"), [("V-tuned", a.nodes, a.ppn, "auto", "spin", plain("V-tuned"))], rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["debug", "A", "B", "Rdebug", "R", "Ndebug", "Idebug", "switch",
                                                        "validate"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodes", type=int, default=32, help="nodes in the allocation")
    ap.add_argument("--ppn", type=int, default=96, help="ranks per node of the full launch")
    ap.add_argument("--max-m", type=int, default=1 << 26)
    ap.add_argument("--seed", type=int, default=1)
    g = ap.add_argument_group("set B")
    g.add_argument("--per-octave", type=float, default=4)
    g.add_argument("--N", default="1,2,3,4,6,8,12,16,24,32")
    g.add_argument("--ppn-list", default="1,2,4,8,16,24,32,48,96")
    g.add_argument("--sessions", type=int, default=3)
    g.add_argument("--batches", type=int, default=5)
    g.add_argument("--sync", choices=["batch", "call"], default=None,
                   help="timing mode for set B (default: mpisrv's back-to-back batches)")
    g = ap.add_argument_group("set validate")
    g.add_argument("--v-layouts", type=int, default=16)
    g.add_argument("--v-per-octave", type=float, default=1)
    g.add_argument("--v-algs", help="algorithms of truth/, comma-separated (default: the six unrestricted ones)")
    g.add_argument("--v-single-node-algs", help="algorithms measured in truth/ on single-node layouts only "
                   "(e.g. release_gather)")
    g = ap.add_argument_group("set R")
    g.add_argument("--r-per-octave", type=float, default=2)
    g.add_argument("--r-sessions", type=int, default=2)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    {"debug": set_debug, "A": set_a, "B": set_b, "Rdebug": set_rdebug, "R": set_r,
     "Ndebug": set_ndebug, "Idebug": set_idebug, "switch": set_switch, "validate": set_validate}[a.set](a, rng)


if __name__ == "__main__":
    main()
