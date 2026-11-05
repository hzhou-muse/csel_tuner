---
name: csel-tuner-batch
description: Plan and package csel_tuner measurement campaigns (batch strategies) — which collectives, variants, launch kinds, algorithms and layouts to measure in which jobs within a node-hour budget, in what order (the stages of the compositional algorithms), and how to turn that into a handoff kit (PBS job scripts, an operator plan with checkpoints and a Log, a result tarball) or an execution plan run directly. Use whenever the user asks to tune MPICH collectives with csel_tuner on a cluster, to build or update a measurement kit, to add algorithms or layouts to an existing model, to remeasure, to validate a tuning file live, or to estimate what a tuning campaign costs.
---

# csel_tuner batch strategies

csel_tuner (https://github.com/hzhou-muse/csel_tuner; this skill lives in its `skills/` directory)
measures MPICH collective algorithms and writes a tuning file for MPICH's
collective selection (CSEL). Its two tools do one thing each; a **batch
strategy** is the plan of which calls to make in which jobs. This skill is
that layer. Read `docs/design.md` §11–12 and `docs/usage.md` §6–9 in the
repo for the tools themselves; this file assumes them.

## The building blocks

| Block | Command | In → out | Cost |
|---|---|---|---|
| **plan** | `bin/collmodel --plan --colls ... --launch ... --catalog CAT` | → the units, their algorithms by stage, the trees each needs | none |
| **create** | `bin/collmodel --colls C1,C2:all,... --launch host,gpu --algs all --max-stage 1 --catalog CAT ...` | → model `M0.json` (one unit per collective, variant and launch kind) | per unit its budget (`--budget-h`) or less; precheck: one probe per algorithm × layout |
| **add** | `bin/collmodel --model Mi.json --colls C --algs X[,Y] ...` or `--layouts NxP,...` | `Mi` → `Mi+1` (keeps `Mi`'s other units) | only the new measurements, plus a few anchors |
| **next stage** | `... --model Mi.json --colls ... --algs all --max-stage k --inner-selection Ti.json` | `Mi` → `Mi+1` | as add; the launch loads `Ti`, the trees of the stages below k |
| **merge** | `bin/collmodel --merge A.json --merge B.json --out C.json` | split jobs' files → one | none |
| **tree** | `bin/colltree Mi.json --out Ti.json [--colls ...] [--node-aware]` | `Mi` → tuning file `Ti` for every collective and variant in it | seconds to a minute on the login or compute node; no measurement |
| **validate** | `groundtruth/gen_plans.py --set validate`, `run_set.sh` ×3, `score_live.py validate` | `Mi`, `Ti` → report | truth = every algorithm at the plan's random points; default and tuned one request per point (gen_plans prints the counts) |
| **import** | `bin/collmodel --import old.json:old.log ...` | collmodel/3 → /4 | none |

Rules the tools enforce or assume:

- **Same allocation shape.** An add runs on an allocation with the model's
  nodes and ranks per node (layouts and size grid must match). Other nodes
  of the same shape are fine; the anchors measure the drift.
- **`--out` ≠ `--model`.** Each step writes a new file (`M0`, `M1`, ...);
  the next step's `--model` is the previous `--out`. A resumed add needs
  its unchanged input model and its journal.
- **Journal and resume.** Always `--log`. A job that hit its `--deadline`
  or SIGTERM is re-submitted with `--resume` and the same arguments; the
  journal replays what was measured. Settings must not change between
  resumes (it refuses).
- **Stages.** A compositional algorithm (an `inner` key in
  `coll_algorithms.txt`: bcast `smp`, and the general `nb` and
  `buffer_swap`) is measured with a tuning file for its inner calls loaded,
  so it comes in a later stage: `--plan` gives each algorithm's stage (with
  MPICH's catalog host bcast `nb` 2, `smp` 3, gpu `buffer_swap` 4, host
  allreduce `smp` 4). Run stage k with `--max-stage k` and an
  `--inner-selection` file that colltree built from the stages below k;
  collmodel refuses an algorithm whose inner trees the file lacks. If
  earlier stages change later, `colltree` prints `stale:` lines with the
  command that re-adds the compositional algorithm (it replaces its
  curves). The two-stage check stops a run whose inner calls could reach a
  compositional algorithm. Everything else is stage 1 and independent.
- **One model file, many units.** A unit is (collective, variant, launch
  kind); a run measures several one after another, each with its own
  journal, budget and `done` record. Split jobs by unit (collective,
  variant, launch kind) or by stage, any way the queue suggests, and
  `--merge` the files. User ops and derived datatypes are not measured by
  design: their calls reach subtree 0, correct for any op; a narrow MPICH
  condition for a common case (a vector type, a specific op) is covered
  later by an enhancement run.
- **MPICH requirements.** The build under test needs the per-request
  algorithm switch (`MPISRV_CVAR_SWITCH=1`; MPICH main ≥ de0b8e6921), the
  `restrictions:` section and side-effect-free release_gather condition
  (pmodels/mpich#8048, merged) for `--catalog`, and the reserved set
  names with the built-in `call=auto` for the default tree output (else
  `--format legacy`, which needs `dispatch=` or `--nested-dispatch`). The tree's size
  splits use `msg_size` (pmodels/mpich#8083; else `colltree
  --size-condition avg_msg_size`), and leaf parameters write booleans as
  0/1 (pmodels/mpich#8080). `mpisrv` must be rebuilt against the MPICH
  under test.
- **Validate live.** The same algorithm can run at different speeds
  depending on how it is selected (built-in selection, CVAR, file), so a
  model built from forced measurements cannot replace timing MPICH with the
  file loaded. Every strategy ends with a validation of the file it ships.
  With a compositional algorithm in the model, run `truth/` with the tuning
  file loaded too.

## Choosing a strategy

Ask (or infer) four things: the node-hours available and the queue limits
(job length, nodes per job), the collectives to cover, how complete the
algorithm set must be, and the allocation sizes the file must serve (one
model per allocation shape; the tree extrapolates beyond it only crudely).

| Strategy | When | Steps |
|---|---|---|
| **All at once** | one op, enough node-hours, a fresh MPICH build | create (`--algs all`) → tree → add compositional → tree → validate |
| **Start small** | little time now, more later; or a quick first file | create with MPICH's default algorithms (those its built-in selection uses) → tree → validate; later adds of the rest as node-hours allow, each followed by tree (+ compositional re-add if stale) |
| **Grow** | an existing model; a new MPICH algorithm or a new restriction | add the new algorithm(s) → tree → (re-add compositional if stale) → validate |
| **More layouts** | validation shows a poor layout region; the builder stopped early | add `--layouts` for the region → tree → validate |
| **Whole collection** | several collectives, variants, launch kinds | plan → stage 1 of every unit (split across jobs, merge) → tree → each later stage with the tree of the ones before → one tree over the collection → validate per collective |
| **Enhancement** | MPICH gains a narrow condition for a common case the variants leave out (a vector datatype, a specific op) | add the algorithms that need it to the units it concerns → tree → validate |
| **Remeasure** | `stale:` lines from colltree, or an anchor drift warning | stale: re-add the compositional algorithm with the latest tree. Drift: compare the anchor ratio with the validation; if the layouts moved, plan a fresh create rather than mixing jobs |
| **Validate only** | a tuning file to check on a new build or other nodes | gen_plans + run_set + score_live |

Sizing a job: a create step costs at most its units' `--budget-h` (each
stops earlier at its regret target or when every layout is measured); an
add costs a fraction of a create in proportion to the algorithms added,
since the existing algorithms are not measured again; a tree costs no
node-hours; a validation costs what its plans hold (the truth plan is the
largest: every algorithm at every point). Costs grow with the number of
algorithms and with log(N)·log(ppn) of the allocation. The best guide is
an earlier model of the same system; read its history:

```sh
python3 -c "import json,sys; [print(r['mode'], r.get('algs'), r.get('stop'), r.get('node_h')) for r in json.load(open(sys.argv[1]))['runs']]" M.json
```

Give each unit a `--budget-h` cap (wall-clock hours, all jobs) if the
plan needs one, and each job a `--deadline` that leaves time for the tree
and for writing (5 min is plenty). A job may hold several steps; a step
that cannot finish resumes in the next job (`--resume` skips finished
units).

## Two ways to run a strategy

**Execution plan** (you have the cluster yourself, or the user runs
commands for you): a numbered list of commands with, for each, what to
check in its output (see "What to check") and what to do if it differs.
Record outcomes in the experiment's plan or session log as you go.

**Handoff kit** (another operator, human or AI, runs it): a directory with

- `plan.md`, the operator plan — template `templates/plan.md`: what the
  kit tests, ground rules (which jobs may be submitted without asking and
  how often re-submitted; what may and may not be changed; never delete
  `results/`; stop-and-ask conditions), Step 0 environment checks (MPICH
  commit and install, launcher, Python/numpy on the compute node, repo
  commit floor, rebuilding `mpisrv`), one step per job with the exact
  lines to copy into the Log and their expected values, the tarball step,
  and a Log section the operator appends to. The plan goes back in the
  tarball, so the Log is the report.
- job scripts — templates `templates/job_steps.pbs` (create/add/tree
  steps, each skipped when done, resumed when not) and
  `templates/job_validate.pbs`. Site settings (account, queue, paths)
  are variables the operator may adjust; measurement settings are not.
- nothing else: the tool comes from the public repo at a stated commit
  floor; plans that `gen_plans.py` makes deterministically are generated
  by the operator (state the expected request counts).

Package it as `<kit>.tgz` in the experiment directory and give its sha256.

**Before handing off, dry-run the kit locally** on fake nodes (site
`local`, 2 nodes × 4, small `--max-m` and budgets) with only the site
substitutions: both jobs end to end, a re-submission (kill one mid-step and
run it again), the plans and `check_r.py`. Fix the kit, not the plan's
expectations, if the dry run fails. Local timings mean nothing; do not
derive anything else from them (anchor drift warnings are normal there).
The templates read their site settings from the environment, so the dry
run needs no edits (a copy of the kit in the scratchpad; an MPICH install
with the per-request switch and #8048, its `bin` first in `PATH`):

```sh
export PATH=$MPICH/bin:$PATH MPICH_PREFIX=$MPICH NICS_PER_SOCKET=1 \
       CSEL_TUNER=<the repository clone> MPICH_SRC=<its source tree> \
       SITE=local NODES=2 PPN=4 MAX_M=65536 EXTRA="--max-m 65536"
bash job_steps.pbs
bash job_steps.pbs & sleep 7; pkill -TERM -f "[b]in/collmodel"; wait; bash job_steps.pbs   # resume
python3 $CSEL_TUNER/groundtruth/gen_plans.py --set validate --out plans/validate --nodes 2 --ppn 4 \
    --max-m 65536 --v-algs ... ; bash job_validate.pbs
```

(`[b]in/...` keeps `pkill -f` from matching its own shell.)

## What to check (expected outputs to put in a plan)

- **plan:** per unit, `tree complete at stage k` and the algorithms of each
  stage with the trees they need; the number of stages is the number of
  rounds of add + tree.
- **create:** the `set <op>-intra-<mask>: N layouts, algorithms ...`
  lines (number of layouts per set: set 0 has every allowed layout, the
  release_gather set the single-node ones, the smp set those with several
  nodes and several ranks per node; compute them for the allocation with
  `python3 -c "import sys; sys.path.insert(0, '<repo>'); from csel_tuner.layouts import axis_values as a; N, P = <nodes>, <ppn>; L = [(n, q) for n in a(N) for q in a(P) if n * q > 1]; print(len(L), sum(n == 1 for n, _ in L), sum(n > 1 and q > 1 for n, q in L))"`), any
  `precheck:` lines (an unrestricted algorithm failing is unexpected), any
  `catalog:` lines (a restriction without a kind), the
  `compositional (add them later ...)` line, the per-set
  lines and the total line (layouts, measurements, node-h, stop).
- **add:** the `set` lines (a compositional or restricted algorithm adds a
  set, e.g. `bcast-intra-10` with smp), the `anchors:` line (ratio geomean
  near 1; a `warning: an anchor moved` is a stop-and-report), `two-stage
  check` lines (failed = stop), the total line with `stop: done` (`budget`
  = the cap was too small; `deadline`/`terminated` = re-submit).
- **tree:** per subtree, leaves, predicted regret and `used:`; the
  variant lines (`not measured (no recipe)` for the user ops is expected;
  `missing (not measured yet)` names a unit the plan should still have);
  any `stale:` lines (then the plan's remeasure step runs).
- **validate:** `check_r.py` ends with `ALL CHECKS PASSED` for every
  result directory; copy the `score_live` report verbatim. The operator
  does not interpret it; you do, afterwards.
- A step is complete when the last record in the model's `runs` says so:
  `mode` create with `complete: true`, or `mode` add with `stop` `done`
  (the templates' `status` function).

## After the results come back

Load the tarball into the experiment's `data/` directory, read the Log
for deviations first, then the reports; re-score offline with
`score_live.py` if the tree changed; and note what the next strategy step
is (more layouts, a remeasure, the next op).
