# csel_tuner: design

csel_tuner measures a cluster's MPI collective algorithms within a
node-hour budget and produces a tuning file for MPICH's collective
selection (CSEL). This document describes how it works and why. For how to
run it, see [usage.md](usage.md).

Status (2026-10-31): the server measures every intra-communicator
collective, blocking and nonblocking, in place where MPI allows it, on host
or device buffers. collmodel measures many units (a collective in one
variant and launch kind, §12) in one run, in stages for the compositional
algorithms (§11), into one model file, and colltree writes one tuning file
over the whole collection. Validated live on a large system for MPI_Bcast
(host and device buffers) and MPI_Allreduce (host); the other collectives
are tested in local functional runs and on synthetic models.

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
        v  model file (collmodel/5: units of measurements + fit)
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
- **Output:** a CSEL subtree, e.g. `name=<op>-intra-0`. A right-leaning chain
  of splits becomes one object of ordered sibling conditions ending in
  `any`; a numeric split on value v uses the canonical threshold form
  `cond(t)` (v ≤ t); a leaf is `{"algorithm=MPIR_<Coll>_intra_<alg>":
  {extra params}}` with the algorithm's extra parameters at the CVAR
  defaults the measurements used.
- **Features:** the CSEL conditions `comm_size`, `msg_size` and `pof2`.
  `msg_size` is the message size, or one rank's block (the mean block for
  allgatherv and reduce_scatter), the same on every rank, which is mpisrv's
  `m`; the collectives without one (gatherv, scatterv, alltoallv,
  alltoallw, barrier) do not split on size, and MPICH refuses a file in
  which they reach `msg_size`. An MPICH before `msg_size` (pmodels/mpich#8083) has
  `avg_msg_size` (`colltree --size-condition avg_msg_size`). With `--node-aware` also `num_nodes(t)` and `ppn(t)` (ranks per
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

**collmodel/5** (several algorithms, `csel_tuner/model.py`; §12.1): a list
of units, each one model of one collective measured one way, named by its
**tags**:

| Tag | Meaning |
|---|---|
| `coll` | the collective (`--op`) |
| `variant` | the input-condition assignment the requests had (`kinds.run_variant`): `inplace` where MPI allows `MPI_IN_PLACE`, `builtin_op`/`commutative` (always 1: mpisrv sums with `MPI_SUM`) for the reductions |
| `datatype` | mpisrv's element type: `int` for the reductions, `byte` otherwise |
| `counts` | v-operations only: the blocks' pattern (`--counts`) |
| `launch` | the buffers' memory, `host` or `device` (`--mem`) |

```json
{"format": "collmodel/5", "created": "...",
 "units": [{"tags": {"coll": "allreduce", "variant": {"builtin_op": 1, "commutative": 1, "inplace": 0},
                     "datatype": "int", "launch": "host"},
            "sizes": [...], ...the rest as in a collmodel/4 file, below, without format, op and mem...}]}
```

A unit is self-contained (its jobs, inner files and runs are its own), so
units copy between files unchanged: a collmodel run puts its unit into
`--out` and keeps the file's other units (adding those of `--model` it
lacks, so adding to a model keeps its other units), `collmodel --merge` combines
files, `collmodel --model` and `colltree --unit` select units by their tags.
A collmodel/4 file reads as one unit; new tags get defaults (a unit written
before a tag existed has its default value).

**collmodel/4**, one unit per file:

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

An algorithm with `extra_params` in the catalog also has `"params"`, the
leaf parameters at the values the server ran with: mpisrv's info record
lists every global int and enum CVAR (`cvar_values`, enums by item name),
and each non-constant parameter takes `MPIR_CVAR_<COLL>_<cvar_param>`, as
MPICH's `maint/gen_coll.py` maps them. colltree writes them into the
leaves, so a tuning file runs an algorithm the way it was measured, also
under `--env` overrides. A boolean comes out as 0 or 1 (MPI_T reports it as
an int), which MPICH's JSON parser accepts since csel-json-bool-int. A
model without `params` (recorded earlier, or replay) gets colltree's
frozen table of the 2026-10 defaults.

A measurement of a compositional algorithm also has `"inner": "<hash>"`,
an anchor `"anchor": true`. An algorithm's model across layouts is a
LayoutModel (§5) over all its curves, fitted on load. collmodel/2 and /3,
written by earlier versions, convert with `bin/collmodel --import`.

**Tuning JSON:** CSEL subtrees, the restriction sets (`name=bcast-intra-0`,
...), loaded with `MPIR_CVAR_COLL_SELECTION_JSON_FILE`; each replaces MPICH's
built-in subtree of that name.

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

Status: implemented (`csel_tuner/algos.py`, `restrict.py`, `kinds.py`,
`stages.py`, `bin/collmodel --catalog / --plan / --max-stage`, mpisrv
`probe=1` and error replies) for restricted algorithms and for
compositional ones within and across collectives, tested on the replay
(with `--replay-alias`), in local functional runs and, for bcast, live. A
compositional algorithm is added to the model (§12) in a later stage of
the same model file.

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
that must hold (a restriction that needs a condition false names its own
condition). Bit k is the restriction with index k in the catalog's
`restrictions:` section (pmodels/mpich#8065: explicit, checked indices, new
ones appended), one numbering for every collective, so a set's name means
the same in every collective. (A catalog without the section numbers its
restrictive conditions in order, then old-style `!c` restrictions, each
read as a condition of its own and tested as `!c` in the tree.) An
algorithm's mask is the OR of its restrictions' bits; a set's mask is the
conditions that are true; an algorithm runs in set S when its mask is a
subset of S, so whatever runs in a set also runs in any larger one. A
layout's mask comes from the precheck: the bits of every algorithm that
ran; an algorithm that failed with all but one of its bits known true
makes that one false; a bit no algorithm decides is taken as false and
reported. The conditions the launch kind decides (`gpu_local`, `gpu_coll`) are
known before the precheck, and colltree sets a unit's input conditions
(`inplace`, ...) in all its masks from its variant (§12.3). The model file
records the bit names. The subtree `<op>-intra-<mask>` serves calls whose
true conditions are exactly `mask`, written in hex without a prefix
(`<op>-intra-0` is set 0, written `r0` below), e.g. with the current
catalog `bcast-intra-200` for single-node communicators
(`MPIDI_CH4_release_gather`, bit 9):

- `r0` holds the unrestricted algorithms and is fitted on **every** layout:
  it is the fallback whenever no other set applies at run time, also on
  layouts where one applied during tuning (e.g. `release_gather` is off
  under `MPI_THREAD_MULTIPLE`).
- `r<S>`, S ≠ 0, is fitted on the layouts whose mask is S, and chooses
  among the restricted algorithms allowed in S, plus the leaf
  `"call=<op>-intra-0"`, whose time at a point is that of r0's choice
  there. With one restricted algorithm it is a short split on size.
- A set with no layouts (mutually exclusive conditions) produces no subtree.
- The sets carry MPICH's reserved names, `<op>-intra-<mask>`, and MPICH's
  built-in dispatch routes to them: it evaluates the restrictions its sets
  use with one call and continues in the matching set; a combination with no
  set uses its largest present subset, at worst `r0` (with positive
  restrictions a subset's algorithms are always allowed), resolved when the
  file is loaded. So the file holds only the sets. A set reaches a subset with
  `dispatch(<mask>)`, which leaves out the restrictions the subset lacks;
  MPICH allows `call=` only to free names, so a set cannot reach one whose
  restrictions do not hold. MPICH refuses two sets of the same size that could
  both match one call, since it cannot know which combinations happen;
  colltree adds their union, a one-line `dispatch(<mask>)` to the one the
  older dispatch picked. `--format legacy` writes the older form for an MPICH
  without the reserved names: `<op>-intra-auto` as the dispatch,
  `"dispatch=<op>-intra(<bits>)"` (pmodels/mpich#8048; about 20 ns per call
  less than nested conditions for two restrictions), or with
  `--nested-dispatch` the nested conditions, and `call=` between sets.

**Stages for compositional algorithms.** A compositional algorithm runs
collectives through the selection: its `inner` calls in the catalog, as
`<section>@<communicator>` (pmodels/mpich#8048), e.g. bcast `smp`
`bcast-intra@node_comm, bcast-intra@node_roots_comm`, allreduce `smp`
`reduce-intra@node_comm, allreduce-intra@node_roots_comm,
bcast-intra@node_comm`. The general algorithms are compositional too, their
calls written relative to the caller (pmodels/mpich#8073: `$coll`, `comm`,
`[k=v]` setting a condition on the inner call; an older catalog gets the
same calls): `nb` runs `i<op>-intra@comm`, the nonblocking collective's
tree, and `buffer_swap` runs `<op>-intra@comm[gpu_local=0]`, the collective's own
tree on host buffers. Its time depends on the trees those calls select
from, so it is measured with a file of them loaded (`--inner-selection`,
via `MPIR_CVAR_COLL_SELECTION_JSON_FILE`), and collmodel refuses one whose
inner sections have no tree (`name=<section>-auto`) in that file rather
than measure it under MPICH's default selection.

That fixes the order of work (`csel_tuner/stages.py`). A plain algorithm is
stage 1. A compositional one is 1 + the stage at which the trees its inner
calls select from are complete; a tree, (collective, launch kind), is
complete at the largest stage of its algorithms, leaving out those whose
inner calls lead back into the chain (smp's bcast tree has no smp: smp
needs `hierarchical`, which neither inner communicator has) and those the
launch kind rules out (`buffer_swap`, restricted to `gpu_local`, is in no host
tree). With MPICH's catalog: host bcast `nb` 2 (ibcast's tree is plain),
`smp` 3 (its bcast tree has `nb`); on a gpu launch `buffer_swap` 4, `smp` 5;
host allreduce `smp` 4. `collmodel --plan` prints each unit's algorithms by
stage and the trees each needs. A stage-k run (`--model ... --max-stage k`)
adds the algorithms up to k, with an inner selection file that colltree
builds from the units' stages below k; it prechecks them (they decide new
bits, e.g. `hierarchical`, so some layouts move to a new set) and samples
them at the model's measured layouts against every algorithm there
(§12.2). colltree then writes the file from the one model (§12.3), and
where it selects an inner call differently from the file a compositional
algorithm was measured under, it reports that algorithm as stale (§12.3).

The **two-stage check** runs on the precheck of a compositional algorithm,
per inner (collective, communicator) of its own collective: no
compositional algorithm of that collective may be allowed under the
precheck masks of the inner layouts (`node_comm` gives (1, ppn),
`node_roots_comm` (N, 1), both in every grid), so one inner file is enough.
With pmodels/mpich#8048 a single-node communicator is not `hierarchical`,
so user layouts and inner communicators of the same shape have the same
conditions. Algorithm CVARs apply to the user's call only, so forcing `smp`
measures it with its inner calls selected by the loaded file.

colltree places `nb` and `buffer_swap` like any algorithm: as leaves of
the `<op>-intra-<mask>` subtrees under `<op>-intra-auto`, each scored per
layout and size, not as fixed wrappers ahead of the selection. They cannot
loop there, because their own code changes the call before it selects
again: `nb` turns it into the nonblocking collective, `buffer_swap` clears
the gpu_local condition (and is restricted to it). This fits MPICH's planned
split into a `main` tree for user calls and an `auto` tree for every
internal call. `main` chooses whether a device call swaps (§12.3); inner calls
select from `auto` and don't swap, and `buffer_swap` as a leaf of a `gpu_coll`
subtree still applies to a user's call.

Whether an inner call on device buffers keeps `gpu_coll` is the algorithm's own
decision, made in its code, and the catalog does not say. Today reduce and scan
`smp` clear it because they use host temporaries; with device temporaries they
could keep it. So the tuner assumes neither: such an inner call needs both the
device tree and the host tree (the stages), and the stale check compares its
choices with and without `gpu_coll`, which can report a change in the tree the
call doesn't use. An inner selection file holds both trees, so the measurement
uses whichever the algorithm reaches.

**Kinds of restrictions.** The catalog lists restrictions without saying
how a tuner treats them; that is csel_tuner's policy, in
`csel_tuner/kinds.py`. An *input* condition is one a request can set (a
recipe: `inplace` by `MPI_IN_PLACE`, `builtin_op` and `commutative` by
`MPI_SUM`); each assignment of a collective's input conditions is a variant,
a model of its own, and an input condition applies wherever MPI allows it
and every rank of a call sees the same value, whether or not an algorithm
restricts on it. The rooted collectives (gather, gatherv, scatter, scatterv,
reduce and their nonblocking forms) have no in-place variant: MPI_IN_PLACE
is given at the root only, so a dispatch on it would send the root and the
other ranks into different subtrees, possibly different algorithms. Their
one tree serves both cases (every algorithm handles MPI_IN_PLACE at the
root, and no catalog restriction separates them); colltree refuses an
in-place unit of a rooted collective measured before this rule. Assignments that contradict an
implication (`builtin_op` without `commutative`) are skipped; those without
a recipe (user ops) are not measured. Their calls reach subtree 0, whose
algorithms are the unrestricted ones: restrictions are positive, so these
take any op, non-commutative included. That is correct, if not optimal (it
is scored on builtin-op timings). User ops and derived datatypes are too
varied to model as a whole, since their cost depends on the actual op and
type; the way to tune a common case among them is a narrow restrictive
condition in MPICH (a vector datatype, a specific op), which joins as a
black-box set condition, measured in a later enhancement run. *Launch* conditions (`gpu_local`,
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
- Without `--node-aware` (the `num_nodes` and `ppn` conditions), `r0`
  cannot tell 1 x 32 from 32 x 1; the restriction split removes that
  conflict only where a restricted algorithm wins.
- Budget: each later set keeps a share in proportion to its layouts, and
  reuses the measurements made before it at the same points.
- **Forced compositional algorithms need #8048.** The algorithm CVAR is
  global: forcing `smp` also forces it on smp's own inner calls, where
  `hierarchical` fails. Before #8048, with
  `MPIR_CVAR_COLLECTIVE_FALLBACK=error` those calls fail, so `smp` fails on
  every layout; with #8048 calls on internal sub-communicators fall back
  silently.

## 12. Incremental models and batch strategies

Status (2026-10-31): implemented (`csel_tuner/model.py`, `bin/collmodel
--model / --layouts / --colls / --launch / --merge / --import`, `bin/colltree`
on collmodel/5) and tested on the replay and in local functional runs,
except `--refit` and continuing the layout loop when adding (§13).

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
  and a validation (§12.4).

### 12.1 The model file: data and fit

One unit of a file (collmodel/5, §12) has two sections:

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

**Growth anchors** (`collmodel --growth-anchors`) are records of a fixed
point set: every modeled algorithm at an N ladder (powers of two) × a few
fixed ppn × a latency and a bandwidth size band. The sampler measures an
algorithm only where it can win, so its data says little about how a losing
algorithm, or one past its crossover, grows with the communicator; fits of
that growth (to select beyond the measured layouts) need the same points
for every algorithm. The records carry `growth: true` and the fit skips
them, as it skips drift anchors: the curves and the tree do not depend on
them. Fixed ppn keeps pof2 constant along each ladder.

Because the anchors measure losing algorithms too, they reach points no
sampler would: a ring alltoall at the largest N and size moves p times the
data of a linear one in p - 1 dependent rounds, and one such call can take
minutes. The anchors therefore carry their own guard: a point whose request
is predicted (from the smaller N of the same ladder, which come first) to
take longer than a limit is left out, so is every point that dominates
(more N, ppn and m) one left out or timed out, and a timeout is a result —
journaled and recorded in the run, not retried — rather than a fault. The
sampler keeps retrying a timed-out batch: there a point is needed for the
curve, and a hang is more likely a fault than a cost.

*Alternative, not adopted for now:* store only `data` and fit on load. The
fit is a function of the data and the fitter version (per-layout PCHIP and
the penalty a(m): seconds, less than building
the tree), and fitting from measured points would free
it from the 4-per-octave grid, which now fixes the size resolution of
crossovers and thresholds at one grid step (about 19%); off-grid sizes
could bracket a jump to a few bytes. The costs: rules the stored curve now
hides (the value at sizes pruned at a layout, merging repeated measurements
of a point across jobs) and a refit in every tool that reads the model.

The **journal** stays a per-job crash log for `--resume` (§4) and audit,
one per unit when a run has several. At the end of a job its measurements
are in `data`; a job killed before that continues with `--resume`, and
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
  redone. The layouts may lie beyond the model's, up to the job's allocation,
  so a model grows over job sizes and each size pays only for its new N. The
  size grid must stay one (`--grid-nodes`). A compositional algorithm's inner
  selection is then per layout (its latest measurement there), since the
  later layouts run under later trees.
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
- **Units.** A run measures one or more units, each a collective in one
  variant (its input conditions, §11) and launch kind, named by its tags:
  `coll`, `variant`, `datatype`, `launch` and, for the v-collectives,
  `counts`. `--colls COLL[:all|:inplace=..+counts=..],...` and `--launch
  host,gpu` list them; they run one after another, one server launch per
  launch kind, each unit with its own journal, budget (`--budget-h`) and a
  `done` record, so `--resume` skips finished units and a killed job loses
  at most the unit in progress. All go into one `--out` file, which keeps
  its other units and gets those of `--model` it lacks; `--merge` combines
  files from split jobs (a unit in two files must be the same).
- **Leaf parameters.** mpisrv's info record lists the global int and enum
  CVARs; each algorithm with `extra_params` records the values of its
  parameters' CVARs (§8), which colltree writes into the leaves.

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
  layouts.
- **Staleness check for compositional algorithms.** For every
  compositional algorithm the output uses, colltree compares the output's
  choices at its inner layouts (node_comm: (1, ppn), node_roots_comm:
  (N, 1), comm: the caller's own) with those of the inner selection its data
  were measured under, for any inner collective whose tree is in the same
  output, read with that collective's unit of the caller's variant, and
  lists where they differ as "remeasure" with the collmodel command. With
  the two-stage check this converges in one remeasure.
- **Several collectives, one file.** Every collective among the selected
  units (`--colls`, `--unit`) gets its subtrees in the output; `--with`
  copies in another file's subtrees of the collectives not built here.
- **Several variants of a collective, one file.** Each unit's input
  conditions that hold are set in all its masks, so a variant's sets have
  names of their own and the dispatch tests those bits too. Its base set
  (exactly its input conditions) covers every layout and is built from its
  own curves even without an algorithm restricted to it, since the variant
  changes every algorithm's cost. The variants share one table of subtrees,
  built in order of increasing mask: a set may call a subset's subtree from
  another variant, scored with this variant's own curves of the algorithms
  that subtree uses (only where it has them all). Where no variant is free
  of input conditions (the reductions measure builtin commutative ops), the
  one with the fewest builds r0 from its unrestricted algorithms: the
  run-time fallback of user ops (§11). Two units with the same input
  conditions (e.g. differing in `counts` only) are refused. colltree lists
  the variants without a unit, those not measured by design and those
  missing, with the subtree their calls reach.
- **Sizes.** The size splits use `msg_size` (§7); collectives without a
  message size the same on every rank build trees without size splits.
- **Several memory kinds, one file.** Host and device buffers have their
  own model each, because an algorithm's cost on device buffers says
  little about its cost on host buffers. colltree builds each model's sets
  on their own. All ranks of a call must reach the same algorithm, which
  decides how each device condition can shape the file:
  - `gpu_coll` is the same on all ranks, so it selects the device model's own
    sets, MPICH's `-gpu` variant, `<op>-intra-<mask>-gpu`. The dispatch falls back
    within the variant, and to a host set only where no `-gpu` set matches.
    `gpu_local` and `gpu_coll` hold everywhere in the model (`--launch gpu`), so
    they are dropped from the masks. An algorithm restricted to them alone
    (`buffer_swap`) is unrestricted there, and `<op>-intra-0-gpu` is that model's
    fallback.
  - `gpu_local` can differ between ranks. MPICH takes it through a local set
    first, `<op>-intra-<gpu_local bit>`, whose leaves are an algorithm restricted
    on it or `dispatch(<that bit>)`, where the host ranks go; a `gpu_coll` call
    never takes it. So a device call without `gpu_coll` runs the host sets'
    algorithm A, directly or after a swap, and ranks converge either way, because
    the swap clears the condition and selects again, reaching A. colltree fits
    this two-way choice per collective, on the device model's curves for A and
    `buffer_swap`, as the local set. Host calls skip it in the same table lookup.
    The device collection already holds both curves, so this needs no collection
    of its own. `--launch gpu` has `gpu_coll` set, so where A is compositional and
    keeps `gpu_coll` on its inner calls (allreduce `smp`), its curve used the device
    sets for them, while a call without `gpu_coll` uses the host sets: there the
    choice is an approximation. Inner calls select from the dispatch with the flags
    they inherit, so on device buffers they reach the local set too, unless their
    algorithm cleared `gpu_local`. (`--format legacy` writes the choice into
    `<op>-intra-main`, `{"gpu_coll": "call=auto", "gpu_local": <the choice>,
    "any": "call=auto"}`, which `main` sends the collective to, and the device
    subtrees under the prefix `<op>-intra-gpu`, with `<op>-intra-auto` testing
    `gpu_coll` first; there inner calls select from `auto` and never swap.)
  - A more restricted tree is written only if it is unique. No algorithm requires
    `gpu_coll` today, so the `gpu_coll` subtrees differ only where their free choice
    among all algorithms beats the host tree's algorithm with the swap choice; if it
    never chooses differently they are left out. Likewise a restricted subtree
    that chooses like its fallback.
  - MPICH does not define `gpu_coll` for gatherv, scatterv and alltoallv (a rank
    may have no buffer to query), and refuses a file that splits on it there. They
    have no device subtrees: their device model only serves the swap choice.

### 12.4 Batch strategies

What a strategy weighs: node-hours and queue limits (job length, nodes per
job), the collectives, variants and launch kinds to cover, how complete
the algorithm set must be, the allocation sizes, and when to validate.

The tools support one run over many units up to a stage (`--max-stage`) or
a launch kind, and any split of that run: the unit of splitting is (coll,
variant, launch, algorithm set). The only ordering that matters is the
stage order of §11: a stage-k algorithm needs the trees of the units'
stages below k, built by colltree from the model so far. Everything else is
stage 1 and independent. Examples:

- **The whole collection:** `--plan`; stage 1 of every unit (split across
  jobs as the queue allows, `--merge` the files); colltree into an inner
  file; each later stage with `--model` and that file; one colltree over
  the collection; validation.
- **Start small:** MPICH's default algorithms first, a file early, the
  rest added in later jobs as node-hours allow.
- **Remeasure:** after a staleness report or a large anchor drift.
- **Enhancement runs:** when MPICH gains a narrow restrictive condition for
  a common case the variants leave out (a vector datatype, a specific op,
  §11), a later run adds the algorithms that need it to the units it
  concerns.

For now strategies are a usage guide written for an AI to turn into an
execution plan (the repo's batch skill); an orchestration tool may follow
once they settle. Validation stays part of every strategy: the same
algorithm can cost differently by how it is selected (§10), which no model
of forced measurements predicts.

## 13. Limitations and next steps

- **Incremental models** (§12): no `--refit` yet; adding to a
  model samples the given algorithms at the measured layouts and the given
  `--layouts`, but does not continue the layout loop (risk and random
  checks, the regret estimate) on its own.
- **Compositional algorithms across collectives** (allreduce smp, nb): the
  staleness check covers them when colltree builds the inner collective's
  tree in the same output; collmodel's two-stage check does not, it only
  checks that the loaded file has their trees.
- **GPU buffers**: one rank per GPU (or tile) only, warm buffers only, no
  mixed host/device buffers. `check=1` and a check of reductions on device
  buffers are not there yet. Placement keeps the host `place` modes, and
  whether tile/GPU placement needs one of its own is open.
- **Count patterns of the v-collectives** (`counts=uniform|linear|onebig`)
  are units of their own, but no condition tells them apart at run time:
  colltree takes one of them per collective (`--unit`).
- **Extrapolation beyond the allocation** (N > N_max) is not modeled: the
  tree's last branch covers it. Growth anchors (§12.1) record the data
  for growth fits; `csel_tuner/growth.py` turns them into coarse classes
  (the exponent of t in p per size band, linear or logarithmic), and
  `colltree --guard-above` uses the classes to replace linear-growth leaves
  above the sampled comm size. Finer growth (k log_k p, effects that show
  only at large p) needs cost models per algorithm.
- The v1 p90 error estimate is unreliable (few random checks); v2's regret
  estimate is reliable on the replay.
