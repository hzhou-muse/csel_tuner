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
- `--log` keeps every request and reply (JSONL).

A PBS job on Aurora, for example:

```sh
#!/bin/bash
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -q debug-scaling
#PBS -A <project>
cd "$PBS_O_WORKDIR"
bin/collmodel --algs binomial,scatter_recursive_doubling_allgather,scatter_ring_allgather,circ_graph,tree,pipelined_tree \
    --site aurora --nodes 8 --ppn 96 --budget-node-h 4 --out bcast.json --log bcast.jsonl
```

Leave a margin between `--budget-node-h` and the walltime; the budget is
checked between layouts.

### 6.2 One algorithm (v1)

```sh
bin/collmodel --alg binomial --site aurora --nodes 8 --ppn 96 --out bcast_binomial.json
```

The algorithm is selected per request when the server supports it, else
for the whole launch (`--switch auto|request|launch`). It stops when the
estimated median error ≤ `--tau-med` (5%) and p90 ≤ `--tau-p90` (20%), at a
plateau, or at the budget. The output (collmodel/1) is a performance model
of that algorithm (design.md §8).

### 6.3 Options

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

## 7. Make and load the tuning file (`bin/colltree`)

```sh
bin/colltree bcast.json --out bcast_tuning.json
```

It prints the tree size, the predicted regret and the algorithms used.

| Option | Default | Meaning |
|---|---|---|
| `--tol` | 0.005 | allowed predicted geomean regret over the full tree; larger = smaller tree |
| `--max-regret` | 1.5 | pruning never lets a point's predicted regret exceed this |
| `--node-aware` | off | also split on `num_nodes` and `ppn` (needs CSEL support MPICH does not have yet) |
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

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no reply in ...s to: op=info` | the launcher does not forward stdin to rank 0; check with `drive_check.py` |
| `cvar.* needs an MPICH that applies CVAR writes at run time` | set `MPISRV_CVAR_SWITCH=1`, only for a build with the fix (§1) |
| `topology_warnings` in the info record | `SOCKETS` / `NICS_PER_SOCKET` / `NODE_CORES` do not match the binding or MPICH |
| launches fail right after a killed one | stale `/dev/shm/mpich_shm_*` files filling shared memory; remove your own |
| `tree nests N levels` | MPICH's parser allows 64; use a larger `--tol` |
| `unknown condition` at `MPI_Init` | a `--node-aware` file on an MPICH without `num_nodes`/`ppn` conditions |

## 11. Tests

```sh
python3 tests/test_interp.py     # numpy only
python3 tests/test_tree.py       # numpy only
python3 tests/test_replay.py     # replay end to end; needs pandas and replay data, else skipped
```

They also run under pytest.
