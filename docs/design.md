# csel_tuner: design

csel_tuner measures a cluster's MPI collective algorithms within a
node-hour budget and produces a tuning file for MPICH's collective
selection (CSEL). This document describes how it works and why. For how to
run it, see [usage.md](usage.md).

Status (2026-10-24): the server, the three builders, the tree output and
the restricted / compositional algorithms of §11 are implemented, tested on
replays and in local functional runs, and validated live on a large system
(MPI_Bcast, the only collective wired through the builders so far). §12 is
the target design for incremental models and batch strategies, not yet
implemented.

## 1. Goal

A tuning file maps each call, keyed on communicator size, ranks per node
and message size, to an algorithm. Its quality is the **regret** of what it
selects: T(selected algorithm) / T(fastest algorithm), summarized over a
grid of layouts and sizes by the geometric mean, the p95 and the maximum.
The tool decides what to measure, how much, and when to stop, trading
measurement cost (node-hours) against regret.

§9 describes how the builders are evaluated.

## 2. Architecture

```
bin/collmodel (driver, Python)                server/mpisrv (one MPI job of P ranks)
  builder v1 / v2 (csel_tuner)                  rank 0 reads requests, broadcasts them,
  csel_tuner.server.Server   -- batch of  -->   the requested sub-communicator measures,
                             <-- replies ---    rank 0 writes one JSON reply per request
        |
        v  model file (collmodel/4: measurements + fit)
bin/colltree (builder v3)  -->  tuning JSON  -->  MPIR_CVAR_COLL_SELECTION_JSON_FILE
```

- One long-lived, warmed-up job serves every measurement. Launching a job
  per point is too slow, and processes run slow for a while after start.
- Any layout (N nodes x ppn ranks per node, p = N x ppn) up to the
  allocation is measured on a sub-communicator of the job.
- The driver talks to rank 0 through stdin/stdout (`mpiexec` forwards stdin
  to rank 0; checked with Hydra and with PALS).
- Batch mode (`--plan FILE --out FILE`) runs a fixed list of requests with
  no driver, for dense ground-truth sets (`groundtruth/`).

## 3. The benchmark server (`server/mpisrv.def`)

### 3.1 Requests and replies

A request is one line of `key=value` tokens; a reply is one JSON object per
line.

```
op=bcast m=65536 p=256 ppn=8 batches=5 sync=call cvar.MPIR_CVAR_BCAST_INTRA_ALGORITHM=binomial
{"status":"ok","seq":12,"op":"bcast","m":65536,"p":256,"ppn":8,"place":"nic",...,
 "cvars":{"MPIR_CVAR_BCAST_INTRA_ALGORITHM":"binomial"},"k":32,"t_us":[41.2,40.9,...],"wall_us":5210}
```

| Key | Meaning | Default |
|---|---|---|
| `op` | `bcast`, `allreduce`, `reduce`, `ireduce`, `iallreduce` (the reductions sum `MPI_INT`s; the nonblocking ones are waited on at once), `barrier` (`m=0`), `reduce_scatter_block`, `allgather`, `alltoall`, `gather`, `scatter`, `scan`, `exscan`, and the nonblocking `ibcast`, `ibarrier`, `ireduce_scatter_block`, `iallgather`, `ialltoall`, `igather`, `iscatter`, `iscan`, `iexscan` (waited on at once), the v-operations `allgatherv`, `alltoallv`, `gatherv`, `scatterv`, `reduce_scatter` and their nonblocking forms (blocks from `counts=`) (`m` is the per-member block: `MPI_INT`s for reduce_scatter_block, bytes otherwise; the p-block buffers must fit `max-m`), `pingpong`, `info`, `quit`; one file per operation in `server/colls/` | — |
| `m` | message size in bytes | — |
| `counts` | v-operations: `uniform` (every block `m`), `linear` (block i ∝ i+1), `onebig` (member 0 holds half); `m` is the mean block, so the blocks total p·m; in alltoallv members i and j exchange block (i+j) mod p both ways (symmetric, as in-place needs) | `uniform` |
| `inplace` | `1`: `MPI_IN_PLACE` where MPI allows it: the send buffer of allreduce, scan, exscan, reduce_scatter(_block), allgather(v), alltoall(v), the root's send buffer of reduce and gather(v), the root's receive buffer of scatter(v) (and the nonblocking forms); refused for the others | `0` |
| `p`, `ppn` | sub-communicator size and ranks per node | the whole job |
| `place` | which local ranks of a node: `nic` (§3.3), `spread`, `compact` | `nic` |
| `nodes` | `OFF[:STRIDE]`: which nodes | `0:1` |
| `batches`, `target_us` | timed batches; calibration target per batch | 3, 1000 |
| `sync` | `batch`: calls back to back; `call`: a barrier before each call, each timed alone | `batch` |
| `check` | `1`: run the operation once on known data before the measurement and verify the result on every rank; `check_err` is the number of wrong elements over all ranks. Every operation with data (not barrier, ibarrier, pingpong), through the measured call, with `counts=` and `inplace=1` (`server/macros/check.def`) | `0` |
| `cvar.<NAME>` | MPI_T control variable for this request only (§3.5), up to 4 | — |
| `root`, `cache`, `pair`, `tag` | root rank; hot/cold buffer; ping-pong ranks; free text echoed | |

`op=info` returns the job: nodes, ranks per node, hosts, the MPICH version,
the MPI-related environment, the topology checks, `cvar_switch`, `mem`
and `gpu` (the buffers' memory and the GPU runtime), and `colls`: every
operation with its properties (`layout` pair or comm, `elem` bytes per
element, `rooted`, `no_data`, `sbuf_p`/`rbuf_p` for buffers of p blocks,
`vcounts`), which collmodel reads instead of keeping its own table.

**Device buffers.** With `mpisrv --mem device`, `sbuf` and `rbuf` are
allocated once on the rank's GPU. The runtime is chosen at build time and
lives in `server/gpu.c`, behind init, alloc/free and fill; the measurement
code doesn't change. The memory kind is a property of the launch, not of a
request: a device run has its own ranks per node (one per GPU), its own
MPICH settings (`MPIR_CVAR_ENABLE_GPU=1`) and its own model.

- **Binding.** A wrapper (`launch/gpu_bind.sh`) makes one GPU visible to
  each rank, chosen by local rank. The server checks that it sees exactly
  one, and compares device UUIDs within each node. A rank the wrapper
  can't bind still starts, with the reason in `MPISRV_BIND_ERROR`, so every
  rank stops together instead of the rest waiting in `MPI_Init`.
- **Caches.** Fixed buffers keep MPICH's buffer-keyed caches (IPC handle
  mappings, NIC memory registrations) warm after the untimed first call.
  This is the target application's pattern. Cold or fresh buffers would
  measure something else and aren't offered.

### 3.2 Measurement

For each request, on the sub-communicator: one untimed call (first-call
effects such as connection setup), calibration of k calls per batch to
reach `target_us`/2, one untimed warm-up batch, one barrier, then `batches`
batches of k calls. A batch's value is the **synchronized latency**: from
the common barrier exit to the last rank finishing, per call. With
`sync=call` each call is timed on its own after a barrier, and the batch
value is the slowest rank's mean call time: latency without pipelining
between consecutive calls, which is what the builders use. `wall_us` is the
request's full cost.

`op=barrier` is itself the synchronization: no barrier is added before the
batches or, with `sync=call`, before each call. The measurement's other
communication (agreement on failures, spin-up, calibration and the check,
and the batch-time extremes) is point-to-point on binomial trees, and the
sub-communicator is created before the request's `cvar.*` are written, so a
forced algorithm reaches only the measured call.

### 3.3 Rank placement: `place=nic`

A sub-communicator with ppn ranks per node must load the NICs as a real
job with ppn ranks per node would. MPICH (ch4:ofi, one NIC per rank) gives
local rank i the close NIC `i mod (NICs per socket)`. Evenly spaced ranks
(`spread`) with a stride that is a multiple of that number put all of a
socket's ranks on one NIC: with 4 NICs per socket and 96 ranks per node,
ppn 24 puts 12 ranks on one NIC where a real 24-rank-per-node job puts 3,
which slows large messages.
`place=nic` moves each spread position within its group of M local ranks
(M = NICs per socket) so that the j-th rank has local rank = j mod M. The
NIC load then equals the real job's for every ppn.

The site config gives sockets and NICs per socket (`launch/sites/*.sh`).
mpisrv checks them at startup against MPICH's `MPI_INFO_ENV` keys
`num_close_nics` / `num_nics_available` and the ranks' core binding, and
reports mismatches in the info record (`topology_warnings`).

### 3.4 Idle ranks

Ranks outside the requested sub-communicator, and all ranks between
requests, wait for the next request:

- `spin` (default): check for it every 50 µs, with a pause-instruction loop
  in between (no MPI progress, no memory traffic). Cores that sleep drop to
  a low clock and need tens of ms of work to recover, so requests after an
  idle gap run slow.
- `sleep`, `busy` (blocking broadcast, drives MPI progress), `split` (the
  last request's members spin, the others sleep, with a spin-up when a
  member was asleep).

Measuring inside a job that runs more ranks per node than the layout uses
can still read slower than a real job, whatever the idle mode (§10);
`busy` is the slowest mode.

### 3.5 Per-request algorithms (`cvar.*`)

MPICH main applies writes to its algorithm CVARs at run time since commit
de0b8e6921 ("coll: read algorithm CVARs through a pointer table"); before,
it copied them at init. mpisrv writes each `cvar.<NAME>=<value>` on every
rank after the request's broadcast (the variables are all-equal scope) and
restores the old value after the measurement. Rank 0 validates the name
(MPI_T, no object binding, int or string type) and the value first; a bad
one gets an error reply.

Enum values go by name or number. Since f95b6ec93d MPICH registers an MPI_T
enum per enum CVAR, with the value names as items; for older builds mpisrv
reads the `name - text` lines of the CVAR's description, which is how
MPICH's generator numbers them.

A build's version string cannot tell whether it has the fix, so the site
config asserts it with `MPISRV_CVAR_SWITCH=1`; without it, `cvar.*`
requests are refused rather than silently measuring the default algorithm.

### 3.6 Interactive input

rank 0 reads stdin unbuffered, so `poll()` sees requests the driver sent
together; with stdio buffering a batch of requests hung the server.

## 4. The driver (`csel_tuner/server.py`, `bin/collmodel`)

- **Batches.** The driver sends a batch of requests, then reads the replies
  in order. A round trip through the launcher's stdin/stdout forwarding can
  cost as much as a request itself, so the builders send each refinement
  step as one batch: a handful of round trips per layout instead of one per
  request.
- **Own line reader** on the raw stdout descriptor, so replies that arrive
  together are never stuck in a buffer `select()` cannot see; a timeout
  kills the job.
- **Re-warm.** After the driver has been idle > 50 ms (model fitting), the
  next batch starts with an untimed 1 MB request: after a pause of a few
  hundred ms, mid-size messages were measured slow, while small and very
  large ones were not.
- **Cost.** Node-hours = elapsed time x nodes of the allocation, including
  launch, driver overhead and fitting (on replay: the recorded wall time).
- **Journal and resume.** Every reply is journaled (flushed per batch). The
  builders are deterministic given the measured values and their seed, so a
  new job replays the journal as a cache keyed by request (repeats in
  order) and continues live at the first miss; a line cut by a kill is
  dropped. Replay measurements are drawn from (seed, request, occurrence),
  not from call order, so a cut-and-resumed replay equals an uninterrupted
  one exactly (`tests/test_resume.py`).
- **Resume with fewer algorithms.** An algorithm found broken mid-run (it
  hangs, or MPICH rejects it) can be left out on resume: a longer
  `--exclude-algs` or a shorter explicit `--algs`. With fewer algorithms the
  builder's requests can change order (pruning and the sampler's choice of
  points depend on all algorithms' curves, and the restriction sets may
  change), so in-order replay would keep only the prefix before the first
  difference. Instead the journal answers every
  request it has, in any order ("salvage"); a reply is a real measurement
  of the same request on the same allocation shape, so the model stays
  valid, but it is no longer equal to any uninterrupted run. The dropped
  algorithms' replies are never requested. For the budget and the cost, an
  earlier job is charged its seconds per reply for each reply taken, so the
  discarded work does not use up the budget of the rest.
- **Stopping safely.** The model file is checkpointed after layouts; a
  deadline stops before a layout that might not fit; SIGTERM stops
  measuring and writes the model from the layouts whose recording finished
  (later SIGTERMs are ignored so the write completes).
- **Environment.** numpy only (`csel_tuner/interp.py` replaces scipy's PCHIP
  and Delaunay interpolation; checked against scipy) and Python ≥ 3.6, the
  system Python on some clusters' compute nodes.

## 5. Builder v1: one algorithm (`csel_tuner/single.py`)

**Sizes at one layout (GridSampler).** Sizes come from a fixed grid, 4 per
octave from 1 B to the largest size, so repeats hit the same sizes and a
jump is bracketed to one grid step. A coarse pass every 2 octaves, then
breadth-first over intervals: predict the grid midpoint with PCHIP
(log-log) through the points so far, measure it. A miss ≤ 0.10 (log2) ends
the interval; a miss > 0.15 that repeats is a jump and is bisected to
adjacent grid sizes; any other miss splits the interval. One level of the
breadth-first search is one batch.

**Across layouts (LayoutModel).**

```
T(N, ppn, m) = S(log2 N, log2 ppn, m) x (1 + a(m) s(N ppn))
s(p) = 2 - p / 2^floor(log2 p) for p not a power of two, else 0
```

S is interpolated linearly in (log2 N, log2 ppn) over the measured layouts
(a Delaunay triangulation; nearest outside the hull); single-node layouts
(shared memory only) are interpolated only among themselves. The penalty
(1 + a s) is largest just above a power of two and vanishes at the next
one, the form that fitted sweeps over non-power-of-two p best; a(m) is
fitted by leave-one-out.

**Layout loop.** Start from N in {2, mid, N_max} x ppn in {1, mid,
ppn_max} plus single-node layouts. Every further layout is first predicted,
then measured (a check). Risk checks take the allowed layout between two
adjacent measured ones whose neighbours' curves differ most; every third
check is a uniformly random unmeasured layout. The random checks' errors
are the estimate; stop when its median ≤ 5% and p90 ≤ 20%, when it stops
improving, or at the budget.

The median estimate is the reliable one; the large errors tend to sit at
a few layouts that a few random draws can miss, so the p90 estimate is
noisy. How accurate the model can get is bounded by the layout model
itself, which replay measures by leaving one layout out with all others
known.

## 6. Builder v2: all algorithms, for selection (`csel_tuner/multi.py`)

The aim is the **optimum model**, the fastest algorithm and its time at
every (N, ppn, m); each algorithm's model only needs to be good where it
can win.

- **One server**, the algorithm per request (§3.5).
- **At a layout (MultiSampler):** a coarse pass of every algorithm; an
  algorithm is refined only on size intervals where it comes within 1.3x of
  the best one (a contender); where the best algorithm changes between two
  sizes, both are measured at the midpoints until the crossover is one grid
  step wide.
- **Pruning:** at risk checks the model's prediction decides the coarse
  pass: sizes where an algorithm is predicted > 2x slower than the
  predicted best are skipped (its curve there is the prediction, corrected
  by the measured residuals). Random checks measure everything.
- **Model (OptimumModel):** one LayoutModel per algorithm; selection =
  argmin of the predicted times.
- **Layout loop** as v1, with the risk score = difference of the two
  neighbours' best times plus the fraction of sizes whose winner differs.
- **Estimate:** at each random check, the regret of the predicted selection
  at the coarse sizes; stop at geomean ≤ 1.02 and p95 ≤ 1.10, a plateau, or
  the budget. Replay checks this estimate against the true regret.

## 7. Builder v3: the tree (`csel_tuner/tree.py`, `bin/colltree`)

**Input:** the v2 model's predicted log2 T of every algorithm at every
allowed layout x size. With R[i, a] = log2 T[i, a] - min_b log2 T[i, b] (the
predicted regret of choosing a at point i):

- **Growth:** a multi-output regression tree on the regret vectors (the
  split with the largest reduction of their squared deviation), grown to
  1,024 leaves. Splitting on regret directly is myopic: when the best
  algorithm depends on size and comm_size jointly, no single split lowers
  the regret much, and growth stopped at 4 leaves.
- **Leaves** pick the algorithm with the least summed regret.
- **Pruning:** weakest link first, to the smallest tree whose predicted
  geomean regret is within `--tol` (0.5%) of the full tree's, never letting
  a point's predicted regret exceed `--max-regret` (1.5x). Without the
  second rule a geomean target lets a few expensive choices through, e.g.
  an algorithm that wins at power-of-two p chosen for the nearby
  non-power-of-two p, where it is much slower.
- **Output:** a CSEL subtree `name=<op>-intra-auto`. A right-leaning chain
  of splits becomes one object of ordered sibling conditions ending in
  `any`; a numeric split on value v uses the canonical threshold form
  `cond(t)` (v ≤ t); a leaf is `{"algorithm=MPIR_<Coll>_intra_<alg>":
  {extra params}}` with the algorithm's extra parameters at the CVAR
  defaults the measurements used.
- **Features:** today's CSEL conditions `comm_size`, `avg_msg_size` and
  `pof2`; with `--node-aware` also `num_nodes(t)` and `ppn(t)` (ranks per
  node, averaged over the nodes), conditions MPICH has since
  pmodels/mpich#8048 (an older one rejects such a file with "unknown
  condition").

## 8. File formats

**collmodel/1** (one algorithm):

```json
{"format": "collmodel/1", "inputs": ["N", "ppn", "m"], "output": "t_us", "what": "bcast-latency",
 "alg": "binomial",
 "model": {"type": "layout-grid-pof2", "sizes": [1, 2, 3, ...],
           "layouts": [{"N": 2, "ppn": 1, "log2_t": [...]}, ...],
           "penalty": {"form": "1 + a*s", "a_intra": [...], "a_inter": [...]}},
 "domain": {"N": [1, 32], "ppn": [1, 96], "m": [1, 67108864]},
 "jumps": [{"N": 2, "ppn": 1, "lo": ..., "hi": ..., "ratio": ...}, ...],
 "meta": {"created": "...", "tool": "collmodel 0.1", "launch": {...}, "library": "...",
          "n_measurements": ..., "n_layouts": ..., "node_h": ..., "stop": "plateau",
          "estimate": {"med": ..., "p90": ...}, "checks": [...]}}
```

`log2_t` is the measured layout's curve on the size grid; prediction at
other layouts follows §5 (`LayoutModel.from_json`).

**collmodel/4** (several algorithms, `csel_tuner/model.py`; §12.1):

```json
{"format": "collmodel/4", "op": "bcast", "sizes": [...], "domain": {...}, "layouts": [[1, 2], ...],
 "conditions": ["inplace", ...], "algorithms": {"binomial": {"leaf": "...", "restrictions": [], "mask": 0,
                                                              "inner": []}, ...},
 "layout_masks": [[1, 2, 2048], ...], "left_out": ["nb"],
 "data": {"jobs": [{...}], "measurements": [{"alg": "binomial", "N": 2, "ppn": 4, "m": 1024,
                                             "t_us": [...], "wall_us": ..., "job": 0}, ...],
          "prechecks": [{"alg": ..., "N": ..., "ppn": ..., "ok": true, "msg": "", "job": 0}, ...],
          "inner_files": {"<hash>": "<tuning file text>"}},
 "fit": {"version": 1, "inner": {"smp": "<hash>"},
         "curves": {"binomial": [{"N": 2, "ppn": 4, "log2_t": [...]}, ...], ...}},
 "runs": [{"mode": "create" | "add" | "import", "jobs": [...], "stop": ..., "node_h": ..., ...}]}
```

`"mem"` is the buffers' memory kind, `host` (also when the field is
absent) or `device`.

A measurement of a compositional algorithm also has `"inner": "<hash>"`,
an anchor `"anchor": true`. An algorithm's model across layouts is a
LayoutModel (§5) over all its curves, fitted on load. collmodel/2 and /3,
written by earlier versions, convert with `bin/collmodel --import`.

**Tuning JSON:** a CSEL subtree (`name=bcast-intra-auto`), loaded with
`MPIR_CVAR_COLL_SELECTION_JSON_FILE`; it replaces MPICH's built-in subtree
of that name.

## 9. Evaluation by replay

Replay (`replay/`) runs a builder against a recorded dense grid of one
collective (every algorithm at every layout and size, from the batch-mode
tools) instead of a cluster: a request returns a recorded value with a
little noise and costs its recorded wall time times the allocation's
nodes. Several seeds give the spread. The builders are scored against the
whole recording:

- v1: |T_pred / T - 1| at the layouts it did not measure (the real test)
  and at those it did (the m axis alone), against node-hours.
- v2 and v3: the regret of their selection at every recorded point
  (geomean, p95, max, total time / the oracle's), split into measured and
  unmeasured layouts, against node-hours; and the builder's own regret
  estimate against the true regret.
- References: MPICH's own selection (if recorded as `auto`), the best
  single algorithm, the best selection keyed on (comm_size, message size)
  and on (N, ppn, message size) with full knowledge, fixed sampling grids,
  and one v1 model per algorithm.

No recorded data is shipped; `replay/README.md` shows how to record and
convert it.

## 10. Lessons from measurements

Observations on a large system (MPI_Bcast, ch4:ofi over Slingshot) that
shaped the design:

- **NIC placement** (§3.3): evenly spaced ranks can overload some NICs;
  `place=nic` reproduces a real job's NIC load.
- **Idle cores must spin** (§3.4): sleeping cores make the next requests
  slow.
- **Launch with the ranks per node you measure.** Layouts with fewer ranks
  per node than the launch measured slower than the same layouts run as
  their own job, at all sizes, whatever the idle ranks did; a launch with
  fewer ranks per node matched. The cause was not found. The offset was
  nearly uniform across algorithms, so it moves absolute times more than
  the choice of algorithm; launching with no more ranks per node than the
  layouts need avoids it.
- **Driver pauses** slow the next mid-size requests (§4, re-warm).
- **The launcher's stdin forwarding** works but costs a round trip per
  batch (§4, batches).
- **Compute nodes may lack scipy** (§4): the tools need numpy only.
- **PCHIP, not random forests.** Along m, PCHIP reached a given median
  error with far fewer sizes than a random forest, and with a linear
  bandwidth tail it extrapolates past the largest measured size, which a
  forest cannot (it repeats its last leaf). Across layouts, the PCHIP layout
  model predicted unmeasured layouts better than a forest on (N, ppn, m),
  also with power-of-two features. The model file is orders of magnitude
  smaller, and its inference is a few dozen lines that port to C.
- **The selection path can change an algorithm's time.** On a single-node
  communicator, small bcasts with `release_gather` were consistently faster
  under MPICH's built-in selection with no file loaded than forced by its
  CVAR, and slower still when a tuning file selected it, well beyond the
  repetitions' spread; the launches differed only in the file. On a smaller
  communicator the three agreed. The cause was not found. Validation
  therefore times the tuned launch itself, not only the chosen algorithm.

## 11. Restricted algorithms and staged sampling

Status: implemented (`csel_tuner/algos.py`, `csel_tuner/restrict.py`,
`bin/collmodel --catalog`, mpisrv `probe=1` and error replies) for
restricted algorithms and for compositional ones whose inner calls are
their own op (bcast smp), tested on the replay (with `--replay-alias`), in
local functional runs and live. Compositional algorithms across ops
(allreduce smp needs reduce and bcast files) are not. Since §12, a
compositional algorithm is added to the model rather than built as a
second stage; the stages below describe that order of work.

Before, the builders chose among the algorithms that have no
`restrictions` in MPICH's `src/mpi/coll/coll_algorithms.txt`. On a
single-node communicator MPICH's default instead takes
`release_gather` (restriction `MPIDI_CH4_release_gather`), which a file of
unrestricted algorithms then overrides. The plan:

**Algorithms.** The candidates are all of the op's algorithms, from `--algs`
or (`--algs all`) from the build's `coll_algorithms.txt` (`--catalog
PATH`), checked against the items of the server's MPI_T enum for
`MPIR_CVAR_<OP>_INTRA_ALGORITHM`. The file also gives each algorithm's leaf
name (`func_name`, e.g. `MPIDI_POSIX_mpi_bcast_release_gather`), its extra
parameters, and its restrictions. Parameters stay at their CVAR defaults.

**Measurements fail instead of falling back.** The server runs with
`MPIR_CVAR_COLLECTIVE_FALLBACK=error`, so a forced algorithm whose
restriction fails returns an error instead of silently running `auto`.
Needs in mpisrv: `MPI_ERRORS_RETURN` on its communicators and an error reply
for the request (the restriction is a property of the communicator, so all
members fail together and nothing hangs).

**Precheck.** Before sampling a layout, one small request per algorithm;
an error reply marks the algorithm infeasible there and it is never
sampled at that layout. The result is the layout's feasible set. This
relies on condition checks without side effects (MPICH: the release_gather
condition used to count calls and create shared memory; now it only reads
state, and the creation happens before the selection).

**Restriction index.** Restrictions are positive: each names a condition
that must hold (MPICH's `coll_algorithms.txt` has no negated restrictions;
a restriction that needs a condition false names its own condition, e.g.
`not_inplace`). Bit k is the k-th restrictive (non-threshold) condition of
the file's `conditions:` list, one numbering for every collective, so a
set's name means the same in every op, and appending a condition keeps the
others' numbers (MPICH's list: `inplace` 0, `not_inplace` 1, ...,
`hierarchical` 5, ..., `MPIDI_CH4_release_gather` 11). An algorithm's mask
is the OR of its restrictions' bits; a set's mask is the conditions that
are true; an algorithm runs in set S when its mask is a subset of S, so
whatever runs in a set also runs in any larger one. A layout's mask comes
from the precheck: the bits of every algorithm that ran; an algorithm that
failed with all but one of its bits known true makes that one false; a bit
no algorithm decides is taken as false and reported. Complementary
conditions such as `inplace` / `not_inplace` give sets with exactly one of
the two; the others are empty and skipped. The model file records the bit
names. (Older MPICH files use `!inplace`; such a restriction is read as a
condition of its own, numbered after the listed ones, and tested as
`!inplace` in the tree.) The subtree `<op>-intra-r<mask>` serves calls whose
true conditions are exactly `mask`, written in hex without a prefix
(`<op>-intra-0` is set 0, written `r0` below), e.g. `bcast-intra-800` for
single-node communicators:

- `r0` holds the unrestricted algorithms and is fitted on **every** layout:
  it is the fallback whenever no other set applies at run time, also on
  layouts where one applied during tuning (e.g. `release_gather` is off
  under `MPI_THREAD_MULTIPLE`).
- `r<S>`, S ≠ 0, is fitted on the layouts whose mask is S, and chooses
  among the restricted algorithms allowed in S, plus the leaf
  `"call=<op>-intra-0"`, whose time at a point is that of r0's choice
  there. With one restricted algorithm it is a short split on size.
- A set with no layouts (mutually exclusive conditions) produces no subtree.
- `<op>-intra-auto` is the dispatch, `"dispatch=<op>-intra(<bits>)"`
  (pmodels/mpich#8048): MPICH evaluates the restrictions in `bits` with one
  call and continues in the matching subtree; a set with no subtree uses
  its largest present subset, at worst `r0` (with positive restrictions a
  subset's algorithms are always allowed), resolved when the file is
  loaded. It costs about 20 ns per call less than nested conditions for two
  restrictions. `--nested-dispatch` writes the nested conditions instead,
  each evaluated once on a path, for an MPICH without `dispatch=`.

**Stages for compositional algorithms.** An algorithm such as `smp` runs
other collectives on sub-communicators (the node comm, the comm of node
leaders), and those calls go through the selection itself, so its time
depends on the file being built.

1. Create the model with every non-compositional algorithm (with their
   restrictions), build the subtrees: file F1.
2. Add the compositional algorithms (`bin/collmodel --model M --algs smp
   --inner-selection F1`): the server runs with F1 loaded
   (`MPIR_CVAR_COLL_SELECTION_JSON_FILE`) and prechecks them; they decide
   new bits (smp: `hierarchical`), so some layouts move to a new mask S.
   They are sampled at the model's measured layouts where they run, against
   every algorithm the model has there (§12.2).
3. `bin/colltree M2` writes F2 from the one model: a subtree per set,
   S's candidates being the algorithms allowed there and calls of subsets'
   subtrees (§12.3), and a dispatch over all sets. Where F2 selects the
   inner calls differently from F1, colltree reports smp's measurements as
   stale (§12.3).

One file for the inner calls is enough when they do not reach a
compositional algorithm: `smp` needs `hierarchical`, which the node comm (one node) and
the leaders' comm (one rank per node) do not have, so F2 selects the same
algorithms as F1 for every inner call, up to the changes the staleness
check reports. The tool checks this on the precheck of the compositional
algorithms before measuring them (one that ran at an inner call's layout
fails the check); with pmodels/mpich#8048 a single-node communicator
is not `hierarchical`, so user layouts and inner sub-communicators of the
same shape have the same conditions.
Compositional algorithms carry an `inner` key in `coll_algorithms.txt`
(pmodels/mpich#8048): the collectives they call through the selection, as
`<section>@<communicator>`, e.g. bcast smp `bcast-intra@node_comm,
bcast-intra@node_roots_comm`, allreduce smp `reduce-intra@node_comm,
allreduce-intra@node_roots_comm, bcast-intra@node_comm`. `--compositional`
defaults to those algorithms (for files without the key: those restricted
to `hierarchical`). From the key the tool gets, per algorithm, the ops
whose tuning files must be loaded (other ops too: allreduce smp needs
reduce and bcast), and the layouts the inner calls see: `node_comm` gives
(1, ppn), `node_roots_comm` (N, 1), both in every stage-1 grid. The
two-stage check is then per inner (op, communicator): no compositional algorithm
of that op may be allowed under the precheck masks of those layouts (smp
needs `hierarchical`, false on both). An inner call can name its own op
(allreduce smp: `allreduce-intra@node_roots_comm`); the check covers it.
With #8048 a forced algorithm whose restriction fails on an internal
sub-communicator falls back to the selection even with
`MPIR_CVAR_COLLECTIVE_FALLBACK=error`, so forcing smp measures smp with its
inner calls selected by the loaded stage-1 file.
The general algorithms are compositional too. Their inner calls are
written relative to the caller (pmodels/mpich#8073: `$coll`, `comm`, and
`[k=v]` setting a restriction on the inner call), and an older catalog gets
the same calls: `nb` runs `i<op>-intra@comm`, the nonblocking op's tree, so
it is added only once the nonblocking op is tuned; `buffer_swap` runs
`<op>-intra@comm[gpu=0]`, the op's own tree on host buffers. collmodel
refuses a compositional algorithm whose inner sections have no tree
(`name=<section>-auto`) in the loaded file, rather than measure it under
MPICH's default selection for them.

**Kinds of restrictions.** The catalog lists restrictions without saying
how a tuner treats them; that is csel_tuner's policy, in
`csel_tuner/kinds.py`. An *input* condition is one a request can set (a
recipe: `inplace` by `MPI_IN_PLACE`, `builtin_op` and `commutative` by
`MPI_SUM`); each assignment of a collective's input conditions is a variant,
a model of its own, and an input condition applies wherever MPI allows it,
whether or not an algorithm restricts on it. Assignments that contradict an
implication (`builtin_op` without `commutative`) are skipped; those without
a recipe (user ops) are not measured. *Launch* conditions (`gpu`,
`gpu_coll`) are decided by how the ranks are started. Everything else is a
*set* condition: the layout conditions (`pof2`, `hierarchical`, ...) and the
black-box ones (`MPIDI_CH4_release_gather`), found by the precheck and
forming restriction sets inside a model; conditions the requests always
satisfy (`displs_ordered`, `block_regular`) come out true wherever an
algorithm uses them. A restriction that is in none of these tables is taken
as a set condition and reported, since it may be a new input condition that
needs a recipe.

**Validation** scores against the best of all feasible algorithms and
MPICH's default, per restriction set (e.g. single-node / multi-node).

**Known limits.**
- `release_gather` creates its shared memory after
  `MPIR_CVAR_POSIX_NUM_COLLS_THRESHOLD` calls (5) on a communicator, or at
  once when its CVAR forces it. A fresh communicator under the file runs r0
  for its first calls; validation warms up past that.
- mpisrv keeps every sub-communicator to the end, and with them their
  release_gather shared memory, under `MPIR_CVAR_COLL_SHM_LIMIT_PER_NODE`
  (64 MB). Small for bcast; reduce buffers (32 KB per rank per comm) can
  reach it for allreduce. A later creation failure turns the algorithm
  infeasible on that comm (an error reply mid-sampling), which the builder
  treats like a failed precheck and reports.
- Per-call conditions (`inplace`, `not_inplace`, `commutative`,
  `builtin_op`, `count_ge_pof2`) depend on the call's arguments, not the
  layout: the precheck runs per call variant, and mpisrv makes no in-place
  calls yet, so only the `not_inplace` sets are built. Bcast has none.
- Without a node-count condition, `r0` cannot tell 1 x 32 from 32 x 1; the
  restriction split removes that conflict only where a restricted
  algorithm wins.
- Budget: each later set keeps a share in proportion to its layouts, and
  reuses the measurements made before it at the same points.
- **Forced compositional algorithms need #8048.** The algorithm CVAR is
  global: forcing `smp` also forces it on smp's own inner calls, where
  `hierarchical` fails. Before #8048, with
  `MPIR_CVAR_COLLECTIVE_FALLBACK=error` those calls fail, so `smp` fails on
  every layout; with #8048 calls on internal sub-communicators fall back
  silently.

## 12. Incremental models and batch strategies

Status (2026-10-24): implemented (`csel_tuner/model.py`, `bin/collmodel
--model / --layouts / --import`, `bin/colltree` on collmodel/4) and tested
on the replay, except `--refit`, `--merge` and continuing the layout loop
when adding (§13). It replaces the stages of §11 as a property of the
tools; §11's restriction sets, dispatch and two-stage check stay.

**Principle.** Two tools around one model file, and the plan of what to
measure when outside both:

- `collmodel` measures algorithms into a model. It reads the existing
  model to decide where and how densely to sample, so an algorithm can be
  added, or an existing one refined on more layouts, at the cost of the new
  measurements only.
- `colltree` turns a model into a tuning file. It measures nothing and runs
  in seconds, so it can be
  rerun with other settings at will.
- A **batch strategy** (which algorithms, collectives and layouts, in which
  jobs, within which node-hours) is a sequence of calls to the two tools
  and a validation. Stage 1 + stage 2 of §11 is one such strategy.

### 12.1 The model file: data and fit

One file (collmodel/4) with two sections:

- **`data`**, the source of truth: every measurement as a record (op,
  algorithm, N, ppn, m, the repetitions `t_us`, `wall_us`, the job, the
  inner selection it ran under, an anchor flag), indexed by (algorithm,
  layout, size); a job table (hosts, start and end, MPICH version and
  commit, the MPI environment, tool version); the precheck results per
  (algorithm, layout) with the job and the error message; the catalog's
  restriction bit names by index; and each inner selection file by content
  hash, with its content (tens of kB).
- **`fit`**, derived: the per-algorithm layout models (§5) per restriction
  set, the selection and the regret estimate, with the fitting code's
  version. `collmodel --refit` recomputes it from `data`; any tool that
  finds an older fit version refits (not implemented yet: tools refuse a
  fit of another version).

*Alternative, not adopted for now:* store only `data` and fit on load. The
fit is a function of the data and the fitter version (per-layout PCHIP and
the penalty a(m): seconds, less than building
the tree), and fitting from measured points would free
it from the 4-per-octave grid, which now fixes the size resolution of
crossovers and thresholds at one grid step (about 19%); off-grid sizes
could bracket a jump to a few bytes. The costs: rules the stored curve now
hides (the value at sizes pruned at a layout, merging repeated measurements
of a point across jobs) and a refit in every tool that reads the model.

The **journal** stays a per-job crash log for `--resume` (§4) and audit.
At the end of a job its measurements are merged into `data`
(`collmodel --merge JOURNAL`, not implemented yet, would do the same for a
job killed before that; checkpoints and `--resume` cover it meanwhile);
nothing reads a journal for sampling decisions. A journal is unsuited for
that, not because of size (ten thousand measurements are a few MB and
parse in under a second) but because of its semantics: it is one job's requests in
order, keyed by the exact request, with probes and error replies mixed in,
and replays only in order up to the first miss. A merged, deduplicated
table with provenance is what sampling and fitting need, and one file is
also what moves between clusters and kits. Existing collmodel/3 models
import from their journals.

### 12.2 collmodel: measure into a model

`bin/collmodel --model M.json [--algs A,B] [--layouts ...]
[--inner-selection F.json] [budget, stop]` creates M or adds to it.

- **Sampling from the whole model.** Pruning (§6) compares a candidate's
  predicted time with the predicted best of every algorithm already in the
  model and allowed at the layout; refinement covers crossovers with old
  algorithms too. Adding an algorithm cannot make old pruning wrong: the
  best only gets faster, so a size skipped as > 2x slower stays so, and an
  old winner is already sampled densely where the new algorithm would have
  to beat it. A new algorithm is therefore cheap where it clearly loses.
- **Same layouts.** A new algorithm is sampled on the model's measured
  layouts (a coarse pass, then refinement where it contends), so every
  comparison is between measured curves; it may stop early on layouts
  where it is nowhere near the best.
- **Refining existing algorithms.** New layouts (`--layouts`, or the
  layout loop of §5 continuing) get every allowed algorithm, pruned against
  the prediction; their measurements merge into `data` and the fit is
  redone.
- **Incremental precheck.** Only (algorithm, layout) pairs without a
  precheck record are probed. Restriction bits are matched by name: a
  catalog that appends conditions is compatible (old masks keep their
  meaning, §11), a renamed or removed one is an error. An algorithm with a
  new restriction moves the layouts where it runs to a new mask; that set
  holds the old algorithms' data for those layouts plus the new one.
- **Anchors.** Each job remeasures a few points of the predicted best at
  layouts measured by earlier jobs and stores them as anchors. Their ratio
  to the stored values is the drift between jobs (other nodes, other days),
  reported and kept, not corrected; a large one asks for remeasuring rather
  than mixing.
- **Compositional algorithms** are measured with `--inner-selection`
  loaded (§11), and their records carry that file's hash. No `auto`
  candidate is needed: smp's curves compete directly with the other
  algorithms'. The two-stage check of §11 still runs on the precheck.
- **Stop** by the regret estimate of the whole model, the budget, a
  deadline or SIGTERM, as now. `--resume` within a job is unchanged.

### 12.3 colltree: models to one file

- **Sets** from the model's layout masks, as §11; subtrees built in order
  of increasing mask, so every subset exists before its supersets; r0 is
  always built (the run-time fallback of the dispatch).
- **Candidates of set S:** each algorithm allowed in S as a leaf, and
  `call=<op>-intra-<T>` for each built subtree T with mask a proper subset
  of S whose leaves use at least `--min-fallback-algs` distinct algorithms.
  A subtree with fewer is not worth the call; its algorithms enter S
  directly. Every candidate is scored at S's own layouts: a `call=` leaf at
  a point costs T's tree's choice evaluated there, not T's fit on its own
  layouts. (Today every set falls back to r0, which is fitted on all
  layouts and without node-aware conditions chooses by comm_size alone: the
  single-node set can hand large messages to the algorithm that wins at
  the same p across nodes.)
- **Staleness check for compositional algorithms.** For every
  compositional algorithm the output uses, colltree compares the output's
  choices at its inner layouts (node_comm: (1, ppn), node_roots_comm:
  (N, 1); other ops from their files) with those of the inner selection
  its data were measured under, and lists the layouts where they differ as
  "remeasure". With the two-stage check this converges in one remeasure.
- The stage-specific paths of §11 (`MODEL1 MODEL2`, `auto` leaves) are gone.
- **Several memory kinds, one file.** Host and device buffers have their
  own model each, because an algorithm's cost on device buffers says
  little about its cost on host buffers. A device call must therefore
  never fall back to a host subtree. colltree builds each model's sets on
  their own.
  - The device model's subtrees have the prefix `<op>-intra-gpu`. Its
    `gpu` bit holds everywhere, so it is dropped from the masks. An
    algorithm restricted to `gpu` alone (`buffer_swap`) is unrestricted
    there, and `<op>-intra-gpu-0` is that model's fallback.
  - `<op>-intra-auto` tests `gpu` first:
    `{"gpu": "dispatch=<op>-intra-gpu(<bits>)", "any": <host dispatch>}`.
  - MPICH's `dispatch=` only looks up `<prefix>-<hex>` names, so the
    prefixes keep the two kinds' subtrees apart, including their subset
    fallbacks. For example, without `MPIR_CVAR_ENABLE_GPU_COLL`, a set
    that needs `gpu_coll` falls back to its device subset.

### 12.4 Batch strategies

What a strategy weighs: node-hours and queue limits (job length, nodes per
job), the collectives to cover and their inner dependencies (allreduce smp
needs reduce and bcast files), how complete the algorithm set must be, the
allocation sizes, and when to validate. Examples:

- **All at once** (§11's stages): every non-compositional algorithm, tree,
  compositional algorithms with that file loaded, tree, validation.
- **Start small:** MPICH's default algorithms first, a file early, the
  rest added in later jobs as node-hours allow.
- **Several collectives:** ops in the order of their inner dependencies,
  one model per op, compositional algorithms after their inner ops' files.
- **Remeasure:** after a staleness report or a large anchor drift.

For now strategies are a usage guide written for an AI to turn into a
handoff kit (job scripts, plans, checkpoints, the expected outputs) or an
execution plan; an orchestration tool may follow once they settle.
Validation stays part of every strategy: the same algorithm can cost
differently by how it is selected (§10), which no model of forced
measurements predicts.

## 13. Limitations and next steps

- **Incremental models** (§12): no `--refit` or `--merge` yet; adding to a
  model samples the given algorithms at the measured layouts and the given
  `--layouts`, but does not continue the layout loop (risk and random
  checks, the regret estimate) on its own.
- **Compositional algorithms across ops** (allreduce smp, nb): neither the
  two-stage check nor the staleness check covers inner calls of another op;
  collmodel only checks that the loaded file has their trees.
- **Collectives:** only bcast is wired through the builders (`OPS` in
  `bin/collmodel`, leaf parameters in `bin/colltree`); mpisrv also measures
  allreduce and ping-pong.
- **GPU buffers**: one rank per GPU (or tile) only, warm buffers only, no
  mixed host/device buffers. `check=1` and a check of reductions on device
  buffers are not there yet. Placement keeps the host `place` modes, and
  whether tile/GPU placement needs one of its own is open. The inner call
  of `buffer_swap` (the host selection, on the same communicator) isn't
  covered by the staleness check.
- **Extrapolation beyond the allocation** (N > N_max) is not modeled: the
  tree's last branch covers it.
- The v1 p90 error estimate is unreliable (few random checks); v2's regret
  estimate is reliable on the replay.
