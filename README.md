# csel_tuner

Tune MPICH's collective algorithm selection for a cluster from a small
measurement budget.

csel_tuner runs one long-lived MPI job as a benchmark server, measures the
collective algorithms adaptively over communicator layouts (nodes x ranks
per node) and message sizes, models where each algorithm wins, and writes a
decision tree in MPICH's collective-selection JSON format (CSEL).

```
bin/collmodel --algs a,b,...   measure, model all algorithms  ->  model.json
bin/colltree model.json        regret-minimizing decision tree ->  tuning.json
MPIR_CVAR_COLL_SELECTION_JSON_FILE=tuning.json mpiexec ... ./app
```

Status: tested on replays of recorded measurements and in local
functional runs; live validation is next. Only MPI_Bcast is wired through
the builders so far.

## Quick start

```sh
# 1. build the server against the MPICH to tune
make -C server MPICC=/path/to/mpich/bin/mpicc

# 2. describe the site: copy and edit launch/sites/aurora.sh (cores, sockets,
#    NICs per socket, MPICH path; MPISRV_CVAR_SWITCH=1 for MPICH main >= de0b8e6921)

# 3. measure and model (inside an allocation)
bin/collmodel --algs binomial,scatter_recursive_doubling_allgather,scatter_ring_allgather,circ_graph,tree,pipelined_tree \
    --site aurora --nodes 8 --ppn 96 --budget-node-h 2 --out bcast.json

# 4. make the tuning file and use it
bin/colltree bcast.json --out bcast_tuning.json
export MPIR_CVAR_COLL_SELECTION_JSON_FILE=$PWD/bcast_tuning.json
```

To try it without a cluster, run on fake nodes on one machine
(`launch/sites/local.sh`; the timings mean nothing), or replay recorded
measurements of your own system (`replay/README.md`).

Requirements: MPICH (ch4; main for per-request algorithm switching and the
JSON selection), Python ≥ 3.6 with numpy; pandas only for replay and tests.

## Documentation

- [docs/usage.md](docs/usage.md): building, site configuration, running,
  options, loading the tuning file, replay, troubleshooting.
- [docs/design.md](docs/design.md): the server and its measurement method,
  the three builders, file formats, evaluation by replay, and the lessons
  from measurements behind the design.

## Layout

```
bin/          collmodel, colltree
csel_tuner/   Python package: interp, server driver, builders single (v1), multi (v2), tree (v3)
server/       mpisrv.c (benchmark server), Makefile
launch/       config.sh, sites/*.sh
replay/       replay of recorded measurements: converter, scripts
groundtruth/  batch-mode plans, runner, checks
docs/         design.md, usage.md
tests/        quick checks
```
