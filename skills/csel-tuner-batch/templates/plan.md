# <KIT NAME> on <system> (<nodes> nodes x <ppn>): operator plan

This kit runs csel_tuner, a tool that measures MPICH's collective
algorithms and writes a tuning file for MPICH's collective selection.
<One paragraph: what this kit does in this campaign, e.g. "builds a model of
every bcast algorithm, adds smp with the first tuning file loaded, and
validates the final file live".> Read this whole file before doing
anything. Keep it updated: append to the **Log** at the end after every
step, with what you ran, job IDs and outcomes. This file goes back to the
user in the result tarball, so the Log is your report.

The kit holds this plan and the job scripts <list>. The tool is the public
repository https://github.com/hzhou-muse/csel_tuner (Step 1).

## What the test checks

| Job | Step | Question |
|---|---|---|
| 1 | <step> | <question; the expected outcome> |
| 2 | validation | At random points the builder did not choose: every algorithm per request, MPICH's default, MPICH with the tuning file; the regret of each |

<N> jobs of <walltime> on <nodes> nodes in <queue>; expected run times.

Do not draw performance conclusions or change the method on your own. Your
job is to run the plan, check the data, report, and ask.

## Ground rules

- <job scripts> (<nodes> nodes, <queue>, <walltime>) may be submitted
  without asking, and each may be re-submitted up to <k> more times to
  finish (they continue where they stopped). Ask before any other job,
  queue or size.
- Do not change the repository's code, the plans, or the job scripts'
  measurement settings. Site settings may be adjusted (PBS directives; the
  MPICH install, its source tree, Python or repository path through
  `MPICH_PREFIX`, `MPICH_SRC`, `PYTHON`, `CSEL_TUNER`); record each change
  in the Log. Do not edit files inside the repository clone; if you think a
  change is needed there, stop and ask.
- Do not change `NICS_PER_SOCKET` or `SOCKETS` to silence a topology
  warning: report it.
- Do not install Python packages or load modules to make numpy import. If
  it does not import, stop and ask the user.
- Never delete anything under `results/`.
- Login nodes are shared: `git`, `gen_plans.py`, `colltree` and the scoring
  scripts are fine there; MPI jobs are not.
- If launches fail within seconds (rc 15 or 255, PMI or "BAD TERMINATION")
  right after a killed launch, look for stale `mpich_shm_*` files in
  `/dev/shm` on the compute nodes; report it before re-submitting.
- Stop and report (do not continue to the next job) on: <the stop
  conditions of this kit: a missing set, a two-stage check failure, an
  anchor drift warning, a check_r failure, ...>.

## Step 0: environment (verify)

1. **MPICH.** The install is `<MPICH_PREFIX>`, built from <branch/commit
   floor>. In its source tree:

   ```sh
   git log -1 --format='%h %ci %s'
   git merge-base --is-ancestor <commit floor> HEAD && echo ok || echo "TOO OLD"
   grep -c '^restrictions:' src/mpi/coll/coll_algorithms.txt      # 1
   ```

   The install's `lib/libmpi.so` must be newer than the tree's last commit.
   If not, stop and ask (the user rebuilds it).
2. **Launcher and fabric.** <the site's launcher and network checks: which
   `mpiexec` the site config uses, and that the expected libfabric
   provider is listed>.
3. **Python.** `python3 --version` and numpy on the login node; the job
   checks the compute node.

## Step 1: the repository

```sh
REPO=$HOME/csel_tuner
git -C $REPO status --short          # must print nothing; if not, stop and ask
git -C $REPO pull --ff-only
git -C $REPO merge-base --is-ancestor <repo commit floor> HEAD && echo "repo new enough" || echo "repo TOO OLD"
(cd $REPO && export SITE=<site> && source launch/config.sh && make -B -C server MPICC=$MPICH_PREFIX/bin/mpicc)
```

(If there is no clone: `git clone https://github.com/hzhou-muse/csel_tuner $REPO`.)

## Step 2: validation plans (in the kit directory)

```sh
python3 $REPO/groundtruth/gen_plans.py --set validate --out plans/validate --nodes <nodes> --ppn <ppn> \
    --v-algs <plain algorithms> [--v-single-node-algs release_gather] [--v-hier-algs smp]
```

Expected: `plans/validate/truth: 1 launches, <n> requests`, `default: <n>`,
`tuned: <n>` (seeded: exact). Record them.

## Step 3: job 1

```sh
qsub job_steps.pbs
```

It ends with `job end; all steps done`. On `re-submit`, re-submit it. Copy
into the Log, per step:

1. <create step>: the `set ...` lines (expected: <sets and layout counts>),
   any `precheck:` lines, the per-set and total lines.
2. <add step>: the `set ...` lines, the `anchors:` line (a `warning:` is a
   stop condition), any `two-stage check` lines, the total line (`stop:
   done` expected).
3. Each tree: its subtree lines, the dispatch line
   (`python3 -c "import json; print(json.load(open('results/<t>.json'))['name=<op>-intra-auto'])"`,
   expected `<dispatch>`), and any `stale:` lines.

## Step 4: job 2

```sh
qsub job_validate.pbs      # or: qsub -W depend=afterok:<job 1 id> job_validate.pbs
```

1. `python3 $REPO/groundtruth/check_r.py plans/validate/<k> results/validate/<k>` for `truth`,
   `default`, `tuned`: each must end with `ALL CHECKS PASSED`.
2. Copy `results/validate_report.md` into the Log. Do not interpret it.

## Step 5: hand back

```sh
tar czf <kit>_<system>_$(date +%Y%m%d).tgz results plans plan.md <job output files>
```

Report in the Log and to the user: job IDs, hosts, run times, MPICH and
repository commits, the check results, any problems, and the tarball's full
path and size.

## Log

- (start here) Kit received. Record environment checks, job IDs, and
  outcomes below.
