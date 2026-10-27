# csel_tuner: usage

How to build the benchmark server, describe a site, measure a cluster,
and load the resulting tuning file into MPICH. For how it works, see
[design.md](design.md).

## 1. Requirements

- **MPICH** with the ch4 device, built with an `mpicc`. To measure several
  algorithms in one run (builder v2, the normal case) it must apply
  algorithm CVAR writes at run time: MPICH main since commit `de0b8e6921`
  (for enum names through MPI_T also `f95b6ec93d`). To load the tuning file
  it must have the JSON collective selection (CSEL) of current main.
- **Python ≥ 3.6 with numpy** where the driver runs, i.e. on the node that
  calls `mpiexec` (for a batch job, its first compute node). Nothing else.
- **pandas**, only for the replay scripts and tests (`replay/`, `tests/`).

## 2. Repository layout

| Path | Contents |
|---|---|
| `bin/collmodel` | builds models: one algorithm (v1) or several (v2) |
| `bin/colltree` | turns a v2 model into an MPICH tuning file (v3) |
| `csel_tuner/` | the Python package (interpolation, server driver, builders) |
| `server/` | the benchmark server `mpisrv.c` and its Makefile |
| `launch/` | `config.sh` (launch function) and `sites/<site>.sh` (node topology, MPICH path) |
| `replay/` | replay of recorded measurements, for developing the builders (no data shipped) |
| `groundtruth/` | batch-mode tools: dense ground-truth plans, a runner, checks |
| `tests/` | quick checks |

## 3. Build the server

```sh
cd server
make MPICC=/path/to/mpich/bin/mpicc
```

This builds `server/mpisrv`, the default `--server` of `bin/collmodel`.

To measure on GPU buffers (`collmodel --mem device`, §6.6), build with a
device backend, `GPU=`:

| `GPU=` | For | Needs |
|---|---|---|
| `none` (default) | host buffers only | — |
| `ze` | Intel GPUs (Level Zero) | `level_zero/ze_api.h`, `libze_loader`; `ZE_PREFIX=` if not in the system paths |
| `cuda` | NVIDIA GPUs | `cuda_runtime.h`, `libcudart`; `CUDA_HOME=` (default `/usr/local/cuda`) |
| `fake` | functional runs without a GPU | — (host memory standing in for a device) |

```sh
make -B -C server MPICC=$MPICH_PREFIX/bin/mpicc GPU=ze
```

`-B` rebuilds when only `GPU` changed. The MPICH under test needs the same
GPU support (e.g. `--with-ze`); a `GPU=ze` server still measures host
buffers without `--mem device`. All device code is in `server/gpu.c`.

## 4. Describe the site

A site file, `launch/sites/<site>.sh`, holds everything specific to a
system; `launch/config.sh` sources it (`SITE=<site>`, default `aurora`).

| Variable | Meaning |
|---|---|
| `NODE_CORES` | cores a launch binds ranks to, socket 0 first, e.g. `1-48,53-100` |
| `SOCKETS` | sockets per node |
| `NICS_PER_SOCKET` | NICs close to each socket (MPICH gives local rank i the NIC `i mod` this) |
| `MPICH_PREFIX` | the MPICH install |
| `MAX_M` | largest message size to measure, bytes |
| `MPISRV_CVAR_SWITCH` | `1` only if this MPICH applies algorithm CVAR writes at run time (§1) |
| `launch()` | optional: how to start `NODES x PPN` ranks; the default uses `mpiexec -n --ppn --cpu-bind list:` |
| `GPU_RUNTIME` | `ze`, `cuda` or `fake`: how `launch/gpu_bind.sh` binds a rank to its GPU |
| `GPU_DEVICES` | GPU devices (tiles, under Level Zero's FLAT hierarchy) per node: at most one rank each |
| `GPU_TILES_PER_GPU` | for `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`: `ZE_AFFINITY_MASK=GPU.TILE` |

With `MEM=device` (set by `collmodel --mem device --site ...`),
`config.sh` exports `MPIR_CVAR_ENABLE_GPU=1` and runs mpisrv under
`launch/gpu_bind.sh`, which makes only the local rank's GPU visible
(`ZE_AFFINITY_MASK`, `CUDA_VISIBLE_DEVICES`); core binding stays with
`launch()`. The site file can set other values for `MEM=device` (Aurora:
`NICS_PER_SOCKET=2` for 12 ranks per node). A site file that defines its
own `launch()` puts `$GPU_WRAPPER` in front of the command.

`launch/sites/aurora.sh` is a complete example (ALCF Aurora, PALS
`mpiexec`). `launch/sites/local.sh` runs on one machine with MPICH's fake
nodes (`MPIR_CVAR_NUM_CLIQUES=2`, tcp): good for trying the tools, its
timings mean nothing.

The server checks `SOCKETS` and `NICS_PER_SOCKET` at startup against the
ranks' binding and against what MPICH reports, and lists mismatches as
`topology_warnings` in its info record and the model file. Fix the site
file rather than ignoring them.

## 5. Check the setup

`groundtruth/drive_check.py` starts the server in interactive mode, sends
requests through `mpiexec`'s stdin, and reports the startup time, the
driver overhead per request, and whether the launcher forwards stdin:

```sh
SITE=aurora python3 groundtruth/drive_check.py /tmp/drive --nodes 2 --ppn 96
```

It must end with `DRIVE CHECK PASSED`.

## 6. Build models (`bin/collmodel`)

### 6.1 All algorithms of a collective (v2)

```sh
bin/collmodel --op bcast \
    --algs binomial,scatter_recursive_doubling_allgather,scatter_ring_allgather,circ_graph,tree,pipelined_tree \
    --site aurora --nodes 8 --ppn 96 \
    --budget-node-h 2 --out bcast.json --log bcast.jsonl
```

- `--site/--nodes/--ppn` launch the server through `launch/config.sh`;
  `--launcher "CMD"` gives any other command prefix instead (the server and
  its max-m argument are appended).
- The allocation (nodes, ranks per node) and the largest size come from the
  server. The tool measures layouts N in {1, 2, 3, 4, 6, 8, 12, ...} up to
  the allocation and ppn likewise up to its ranks per node, at sizes 4 per
  octave from 1 B.
- It stops when its regret estimate meets the target (`--tau-gm 1.02`,
  `--tau-p95 1.10`), when the estimate stops improving, or when the budget
  (node-hours = elapsed time x nodes) is spent.
- Needs the per-request algorithm switch; the server must report
  `cvar_switch=1` (`MPISRV_CVAR_SWITCH=1` in the site file).
- `--log` keeps a journal of every measurement, which lets a later job
  continue the run (§6.5). Always give one on a cluster.

**Every algorithm, restricted ones included** (docs/design.md §11):

```sh
bin/collmodel --op bcast --algs all \
    --catalog $MPICH_SRC/src/mpi/coll/coll_algorithms.txt \
    --site aurora --nodes 8 --ppn 96 --budget-node-h 2 --out bcast.json --log bcast.jsonl
```

- `--catalog` is the `coll_algorithms.txt` of the MPICH build under
  test. `--algs all` takes every algorithm of the op from it (plus `nb`);
  `--algs a,b,...` with `--catalog` takes those.
- The run starts with a precheck: one probe call (`--probe-m`, 1 KB) of each
  algorithm at each allowed layout. The server runs with
  `MPIR_CVAR_COLLECTIVE_FALLBACK=error`, so an algorithm whose restriction
  fails returns an error. An unrestricted algorithm that fails anywhere is
  left out (e.g. `nb` without a nonblocking selection that works here).
- Then one builder per restriction set: `r0` (unrestricted algorithms) on
  every layout, then each other set on the layouts where exactly its
  conditions hold (e.g. `800`, `release_gather` on single-node layouts). Each
  later set keeps a share of the budget in proportion to its layouts, and
  reuses the measurements made before it.
- Compositional algorithms (`--compositional`, default those with an
  `inner` key in `coll_algorithms.txt`, e.g. `smp`; for files without the
  key, those restricted to `hierarchical`) are left out: they are added to
  the model afterwards with a tuning file for their inner calls (§6.2).
- Needs an MPICH whose `release_gather` condition has no side effects
  (pmodels/mpich#8048); with an older one the precheck of `release_gather`
  depends on how often it was evaluated.
- The model file is `collmodel/4` (design.md §12): every measurement with
  its job, the precheck results, the catalog's conditions and each layout's
  mask (`data`), and each algorithm's curve at each layout it was sampled
  at (`fit`).

A PBS job on Aurora that stops cleanly before its walltime and continues a
run an earlier job did not finish:

```sh
#!/bin/bash
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -q debug-scaling
#PBS -A <project>
cd "$PBS_O_WORKDIR"
resume=""; [ -s bcast.log ] && resume=--resume
bin/collmodel --algs binomial,scatter_recursive_doubling_allgather,scatter_ring_allgather,circ_graph,tree,pipelined_tree \
    --site aurora --nodes 8 --ppn 96 --budget-node-h 6 \
    --log bcast.log $resume --deadline 3300 --out bcast.json
```

The budget counts every job of the run; the deadline is this job's
(walltime minus a margin for writing the model).

### 6.2 Add to a model

`--model` adds algorithms, or layouts, to a model, measuring only what is
new (design.md §12). The allocation must have the model's size; write to a
new file (`--out` must not be the `--model` file, so that `--resume` finds
the model the run started from).

```sh
CAT=$MPICH_SRC/src/mpi/coll/coll_algorithms.txt
# an algorithm the model lacks (all: every one of the catalog's it lacks)
bin/collmodel --model bcast.json --algs pipelined_tree --catalog $CAT \
    --site aurora --nodes 8 --ppn 96 --budget-node-h 1 --out bcast2.json --log add.jsonl
# more layouts for the algorithms it has
bin/collmodel --model bcast.json --layouts 4x24,16x8 --catalog $CAT ... --out bcast2.json
```

- A new algorithm is prechecked at every layout; a new restriction moves
  the layouts where it runs to a new set. Then it is sampled at the
  layouts the model has measured where it may run: its coarse pass skips
  the sizes where it is predicted (from its first layouts) more than
  `--prune` slower than the model's best, it is refined where it comes near
  the best, and at its crossovers with the model's algorithms both are
  measured. The model's earlier measurements are kept as they are.
- A new layout gets every algorithm allowed there.
- Each run first remeasures a few points of the model's best algorithms at
  `--anchors` layouts (3) and prints the ratio to the earlier values, the
  drift between jobs; above `--drift-tol` (1.10) it warns. Anchors are kept
  in the data, flagged, and not used for the fit.

**Compositional algorithms** (e.g. smp; design.md §11) are added with the
tuning file of their inner calls loaded in the launch:

```sh
bin/colltree bcast.json --out bcast_tuning.json
bin/collmodel --model bcast.json --algs smp --inner-selection bcast_tuning.json --catalog $CAT \
    --site aurora --nodes 8 --ppn 96 --budget-node-h 1 --out bcast2.json --log add.jsonl
bin/colltree bcast2.json --out bcast_tuning2.json
```

- It stops before measuring if an inner call could itself run a
  compositional algorithm (the two-stage check); needs pmodels/mpich#8048.
- The model keeps the file (by hash, with its content); smp's measurements
  record it. Adding smp again with another file replaces its curves.
- `colltree` reports where its output selects smp's inner calls
  differently from that file (§7); then add smp again with the output.

**Convert older model files** (collmodel/3, with the journal of each run;
a stage-2 file needs the stage-1 tuning file it ran with):

```sh
bin/collmodel --import bcast.json:bcast.jsonl --import bcast2.json:bcast2.jsonl \
    --inner-selection bcast_tuning.json --out bcast4.json
```

### 6.3 One algorithm (v1)

```sh
bin/collmodel --alg binomial --site aurora --nodes 8 --ppn 96 --out bcast_binomial.json
```

The algorithm is selected per request when the server supports it, else
for the whole launch (`--switch auto|request|launch`). It stops when the
estimated median error ≤ `--tau-med` (5%) and p90 ≤ `--tau-p90` (20%), at a
plateau, or at the budget. The output (collmodel/1) is a performance model
of that algorithm (design.md §8).

### 6.4 Options

Also `--log`, `--resume`, `--deadline`, `--checkpoint-every` (§6.5) and
`--seed` (random checks; a resumed run must keep it).


| Option | Default | Meaning |
|---|---|---|
| `--op` | `bcast` | the collective |
| `--budget-node-h` | 2 | stop after this many node-hours |
| `--prune` | 2.0 | v2: skip sizes predicted this much slower than the best; 0 = off |
| `--batches` | 5 | timed batches per request |
| `--sync` | `call` | `call`: a barrier before each call; `batch`: back to back |
| `--max-m` | 64 MB | largest size; the server allocates buffers of this size per rank |
| `--timeout` | 300 | seconds to wait for a reply before killing the job |
| `--rewarm-gap` | 0.05 | seconds of driver idle time after which a batch starts with a warm-up request |
| `--env KEY=VALUE` | | extra environment for the launch |

### 6.5 Long runs, deadlines and interruptions

A run can span several jobs and survive a kill:

- **Journal (`--log FILE`).** Every measurement reply is appended and
  flushed after each batch, plus one record per job (start, allocation,
  settings). A kill loses at most the batch in flight.
- **Resume (`--resume`, same `--log`).** The builder makes the same
  decisions from the same values, so requests the journal already answers
  are taken from it and measuring continues where the earlier job stopped.
  Earlier jobs' time counts against `--budget-node-h`. A resume is refused
  if the algorithms, seed, batches, sync, pruning, the allocation's shape
  (nodes, ranks per node, largest size), the input model, the inner
  selection file or the added layouts differ; other nodes of the same shape
  are allowed and noted.
- **Resume with fewer algorithms.** The one change to the algorithms a
  resume allows is leaving some out: add them to `--exclude-algs`, or drop
  them from an explicit `--algs` list (not from `all`; exclude those
  instead). Their replies in the journal are ignored, every other request
  the journal answers is taken from it in any order, and an earlier job
  counts against the budget only for the replies taken (the job's time per
  reply). The job's stderr names the dropped algorithms and how many
  replies are left out, and at the end the earlier jobs' node-hours and the
  part billed. Put an algorithm back by starting a new journal.

  ```sh
  bin/collmodel ... --exclude-algs A --log a.log --out a.json
  # algorithm B turns out to be broken: resume without it, keeping the rest
  bin/collmodel ... --exclude-algs A,B --log a.log --resume --out a.json
  ```
- **Checkpoints.** The model file is rewritten after a layout at most every
  `--checkpoint-every` seconds (60) and at the end; the run's record in
  the file's `runs` has the stop reason.
- **Deadline (`--deadline SEC`).** A layout is not started if it might not
  finish SEC seconds after the job's start (judged by the slowest layout so
  far); the model is written and the job ends.
- **SIGTERM** (what a batch system sends at the walltime): measuring stops,
  the model is written from the finished layouts, and the server is killed.

To chain jobs under PBS, submit the job above again with a dependency, e.g.
`qsub -W depend=afterany:<jobid> job.pbs`; a job that finds the run already
finished replays the whole journal and writes the same model.

### 6.6 GPU buffers

A model is of one memory kind. Device buffers get a model of their own,
measured on a launch with one GPU per rank:

```sh
make -B -C server MPICC=$MPICH_PREFIX/bin/mpicc GPU=ze        # §3
bin/collmodel --mem device --algs all --catalog $MPICH_SRC/src/mpi/coll/coll_algorithms.txt \
    --out bcast_gpu.json --site aurora --nodes 8 --ppn 12 --log bcast_gpu.log
```

- mpisrv runs with `--mem device`. It allocates its buffers on the rank's
  GPU once, so caches keyed by buffer (IPC handles, NIC registrations) are
  warm after the first call. This matches applications that reuse their
  buffers.
- Each rank must see exactly one GPU, and no two ranks of a node may
  share one. `--ppn` is at most the site's `GPU_DEVICES`. mpisrv stops on
  every rank if the binding failed, and lists shared GPUs in
  `topology_warnings`.
- The memory kind decides the `gpu` and `gpu_coll` restrictions, if the
  catalog has them. `gpu` holds on device buffers, and neither holds on
  host buffers. The precheck reports algorithms that ran or failed against
  that.
- The `gpu_coll` restriction also needs `MPIR_CVAR_ENABLE_GPU_COLL=1` in
  the launch (`--env`).
- Not with device buffers: `--cold` and `check=1`. Mixed host and device
  buffers aren't measured.

Then build one tuning file from both models (§7).

## 7. Make and load the tuning file (`bin/colltree`)

```sh
bin/colltree bcast.json --out bcast_tuning.json
bin/colltree bcast.json bcast_gpu.json --out bcast_tuning.json   # host and device buffers (§6.6)
```

It prints the tree size, the predicted regret and the algorithms used.

| Option | Default | Meaning |
|---|---|---|
| `--tol` | 0.005 | allowed predicted geomean regret over the full tree; larger = smaller tree |
| `--max-regret` | 1.5 | pruning never lets a point's predicted regret exceed this |
| `--node-aware` | off | also split on `num_nodes` and `ppn` (MPICH with pmodels/mpich#8048) |
| `--min-fallback-algs` | 3 | a set may call a subset's subtree that uses at least this many algorithms; a smaller one's algorithms are candidates directly |
| `--stale-tol` | 1.05 | report a compositional algorithm's changed inner choice when the two are predicted more than this apart |
| `--nested-dispatch` | off | dispatch on restrictions with nested conditions instead of `dispatch=` (MPICH without pmodels/mpich#8048) |
| `--param ALG.KEY=VALUE` | | extra parameter of an algorithm leaf, e.g. `tree.k=4` |

Use it with MPICH:

```sh
export MPIR_CVAR_COLL_SELECTION_JSON_FILE=$PWD/bcast_tuning.json
# leave MPIR_CVAR_BCAST_INTRA_ALGORITHM unset (auto)
mpiexec ... ./app
```

The file defines `name=bcast-intra-auto`, which replaces MPICH's built-in
subtree of that name; everything else keeps its defaults. A malformed file
stops `MPI_Init` with the line and column of the error.

The output has a subtree per restriction set (the suffix is the mask of
true conditions, bit k = the condition with index k in
`coll_algorithms.txt`): `name=bcast-intra-0`, `name=bcast-intra-800`, ...,
and `name=bcast-intra-auto` dispatches on the restrictions, e.g.
`"dispatch=bcast-intra(0x820)"`: MPICH continues in `bcast-intra-800` on a
single-node communicator, `bcast-intra-20` on a hierarchical one, else
`bcast-intra-0`. An MPICH without `dispatch=` (pmodels/mpich#8048) needs
`--nested-dispatch`, which writes nested conditions ending in `call=`. A
restricted set's leaves are the algorithms allowed there or `call=` a
subset's subtree that uses at least `--min-fallback-algs` algorithms, each
scored at the set's own layouts. To see which algorithms ran, set
`MPIR_CVAR_DUMP_COLL_ALGO_COUNTERS=0` (rank 0 prints its counts at
finalize).

With a model of device buffers too, the file also has that model's
subtrees, named `bcast-intra-gpu-<mask>`. In them the `gpu` restriction is
left out of the mask, because it holds everywhere in that model. The file
also has a dispatch of its own, and `name=bcast-intra-auto` becomes
`{"gpu": "dispatch=bcast-intra-gpu(...)", "any": <the host dispatch>}`. A
call on device buffers falls back to `bcast-intra-gpu-0`, never to a host
subtree. The device model's catalog must have the `gpu` restriction.

With a compositional algorithm in the model, `colltree` compares its
output's choices for that algorithm's inner calls with those of the file
it was measured under, and prints `stale:` lines where they differ by more
than `--stale-tol`, followed by the `collmodel` command that remeasures it
with the output loaded.

## 8. Replay (development)

The builders can run against recorded measurements instead of a cluster.
No data is shipped: record a dense grid of your system with the batch-mode
tools (§9) and convert it with `replay/make_data.py`; `replay/README.md`
has the steps. Then:

```sh
bin/collmodel --replay --algs binomial,scatter_ring_allgather,circ_graph --out m.json
python3 replay/replay_v1.py --algs binomial --seeds 3 --tag t && python3 replay/replay_v1.py --report t
python3 replay/replay_v2.py --seeds 3 --v1 --tag t && python3 replay/replay_v2.py --report t
python3 replay/replay_v3.py --seeds 3
```

Reports and CSVs go to `replay/results/`. Scores are against the recorded
ground truth: model error for v1, selection regret for v2 and v3.

## 9. Ground truth and checks (batch mode)

`groundtruth/` runs fixed lists of requests without a driver, for dense
reference data or for checking a system:

```sh
python3 groundtruth/gen_plans.py --set B --out plans/setB --nodes 32 --ppn 96
SITE=aurora groundtruth/run_set.sh plans/setB results/setB 10000     # budget in seconds; resumable
python3 groundtruth/check_r.py plans/setB results/setB               # completeness, errors, data checks, placement
```

`gen_plans.py --help` lists the sets: a quick debug set, set A (coarse
grid, noise, interference), set B (dense grid), set R (non-power-of-two
sweeps), and the 2-node checks of rank placement (Ndebug) and idle modes
(Idebug). Each launch line in `launches.txt` names its algorithm and idle
mode (`MODE[:IDLE_NS[:SPINUP_US]]`).

Two sets check a live build and a tuning file, scored by
`groundtruth/score_live.py`:

```sh
# the per-request algorithm switch: per request vs. per launch, every algorithm
python3 groundtruth/gen_plans.py --set switch --out plans/switch --nodes 8 --ppn 96
groundtruth/run_set.sh plans/switch results/switch 900
python3 groundtruth/score_live.py switch results/switch            # ends with SWITCH CHECK PASSED

# a tuning file at random points the builder did not choose
python3 groundtruth/gen_plans.py --set validate --out plans/validate --nodes 8 --ppn 96 \
    [--v-algs binomial,...,nb --v-single-node-algs release_gather --v-hier-algs smp]
groundtruth/run_set.sh plans/validate/truth results/validate/truth 1800      # every algorithm, per request
groundtruth/run_set.sh plans/validate/default results/validate/default 600   # MPICH's own selection
MPIR_CVAR_COLL_SELECTION_JSON_FILE=$PWD/bcast_tuning.json \
    groundtruth/run_set.sh plans/validate/tuned results/validate/tuned 600   # with the tuning file
python3 groundtruth/score_live.py validate --truth results/validate/truth --default results/validate/default \
    --tuned results/validate/tuned --model bcast.json --tuning bcast_tuning.json
```

The validation reports the regret of MPICH's default, of MPICH with the
tuning file, of the tree's and the model's own choices, whether MPICH
applied the file, and whether the default beats every candidate algorithm
somewhere (then the candidates miss one, e.g. `release_gather`). Regret is
against the fastest of every algorithm measured and MPICH's default, split
into single-node and multi-node layouts. `--model` takes a collmodel/4
file and `--tuning` a file with one or several subtrees. With a
compositional algorithm in the model, run `truth/` with
`MPIR_CVAR_COLL_SELECTION_JSON_FILE` set to the tuning file too, so that
smp's inner calls follow it.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no reply in ...s to: op=info` | the launcher does not forward stdin to rank 0; check with `drive_check.py` |
| `cvar.* needs an MPICH that applies CVAR writes at run time` | set `MPISRV_CVAR_SWITCH=1`, only for a build with the fix (§1) |
| `topology_warnings` in the info record | `SOCKETS` / `NICS_PER_SOCKET` / `NODE_CORES` do not match the binding or MPICH |
| launches fail right after a killed one | stale `/dev/shm/mpich_shm_*` files filling shared memory; remove your own |
| `tree nests N levels` | MPICH's parser allows 64; use a larger `--tol` |
| `unknown condition` at `MPI_Init` | a `--node-aware` file on an MPICH without `num_nodes`/`ppn` conditions (before pmodels/mpich#8048) |
| `precheck: X fails at ... without restrictions` | an unrestricted algorithm errors on a layout; it is left out; the message is MPICH's |
| `Algorithm selected is not supported` in an error reply | a forced algorithm whose restriction fails, with `MPIR_CVAR_COLLECTIVE_FALLBACK=error` |

## 11. Tests

```sh
python3 tests/test_interp.py     # numpy only
python3 tests/test_tree.py       # numpy only
python3 tests/test_replay.py     # replay end to end; needs pandas and replay data, else skipped
python3 tests/test_resume.py     # a cut journal resumes to the same model; same needs
python3 tests/test_restrict.py   # catalog, precheck, sets, dispatch; the builds with sets need replay data
python3 tests/test_extend.py     # adding an algorithm and layouts to a model, resume; needs replay data
python3 tests/test_gpu.py        # memory-kind restrictions; colltree merging host and device models
```

They also run under pytest.
