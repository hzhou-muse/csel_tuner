# csel_tuner

Tune MPICH's collective algorithm selection for a cluster from a small
measurement budget.

csel_tuner runs one long-lived MPI job as a benchmark server, measures the
collective algorithms adaptively over communicator layouts (nodes x ranks
per node) and message sizes, models where each algorithm wins, and writes a
decision tree in MPICH's collective-selection JSON format (CSEL).

```
bin/collmodel --plan --colls ... --catalog CAT      the units and their stages
bin/collmodel --colls bcast,allreduce:all --algs all --max-stage 1 ...   measure  ->  model.json
bin/collmodel --model model.json --max-stage 2 --inner-selection t1.json ...  add  ->  model2.json
bin/colltree model2.json                            decision trees, one file  ->  tuning.json
MPIR_CVAR_COLL_SELECTION_JSON_FILE=tuning.json mpiexec ... ./app
```

The model file keeps every measurement with its provenance, so a model
grows over several runs: an algorithm added later costs only its own
measurements (docs/design.md, section 12).

Status: every intra-communicator collective (blocking and nonblocking, in
place where MPI allows, host and device buffers) is measured into one model
file of units and written as one tuning file. Tested on replays of recorded
measurements and in local functional runs; validated live on a large system
for MPI_Bcast and MPI_Allreduce.

## Quick start

```sh
# 1. build the server against the MPICH to tune
make -C server MPICC=/path/to/mpich/bin/mpicc

# 2. describe the site: copy and edit launch/sites/aurora.sh (cores, sockets,
#    NICs per socket, MPICH path; MPISRV_CVAR_SWITCH=1 for MPICH main >= de0b8e6921)

# 3. measure and model (inside an allocation): every plain bcast algorithm
bin/collmodel --colls bcast --algs all --max-stage 1 --catalog $MPICH_SRC/src/mpi/coll/coll_algorithms.txt \
    --site aurora --nodes 8 --ppn 96 --out bcast.json --log bcast.log

# 4. make the tuning file and use it
bin/colltree bcast.json --out bcast_tuning.json
export MPIR_CVAR_COLL_SELECTION_JSON_FILE=$PWD/bcast_tuning.json
```

To try it without a cluster, run on fake nodes on one machine
(`launch/sites/local.sh`; the timings mean nothing), or replay recorded
measurements of your own system (`replay/README.md`).

Requirements: MPICH (ch4; main for per-request algorithm switching, the
JSON selection and `msg_size`), Python ≥ 3.6 with numpy; pandas only for
replay and tests. docs/usage.md §6.7 walks through a whole collection in
stages.

## Documentation

- [docs/usage.md](docs/usage.md): building, site configuration, running,
  options, loading the tuning file, replay, troubleshooting.
- [skills/csel-tuner-batch](skills/csel-tuner-batch/SKILL.md): batch
  strategies, a guide written as a Claude Code skill: which algorithms,
  layouts and collectives to measure in which jobs, and how to package that
  as a handoff kit (job script and operator plan templates). Copy or link
  the directory into `~/.claude/skills/` to use it as a skill.
- [docs/design.md](docs/design.md): the server and its measurement method,
  the three builders, file formats, evaluation by replay, and the lessons
  from measurements behind the design.

## Layout

```
bin/          collmodel, colltree
csel_tuner/   Python package: interp, server driver, builders single (v1), multi (v2), tree (v3),
              catalog (algos), restriction kinds and variants (kinds), stages, model files (model)
server/       mpisrv.def, macros/*.def, colls/*.def (benchmark server, MyDef; generates mpisrv.c), gpu.c, Makefile
launch/       config.sh, sites/*.sh
replay/       replay of recorded measurements: converter, scripts
groundtruth/  batch-mode plans, runner, checks
docs/         design.md, usage.md
skills/       csel-tuner-batch: batch strategies and kit templates, as a Claude Code skill
tests/        quick checks
```
