# Replay

Replay runs the builders against recorded measurements instead of a live
cluster: a request returns a recorded value and costs its recorded wall
time. It is how the builders are developed and compared, and the replay
scripts score them against the full recording (model error for v1,
selection regret for v2 and v3).

No data is shipped. Record a dense grid on your system and convert it.

## 1. Record a dense grid

With the batch-mode tools in `groundtruth/` (docs/usage.md, §9), on the
largest allocation you want the replay to cover, one launch per algorithm
and session:

```sh
python3 groundtruth/gen_plans.py --set B --out plans/setB \
    --nodes 32 --ppn 96 --per-octave 4 --sessions 2 \
    --N 1,2,3,4,6,8,12,16,24,32 --ppn-list 1,2,4,8,16,24,32,48,96
SITE=<site> groundtruth/run_set.sh plans/setB results/setB <budget seconds>
python3 groundtruth/check_r.py plans/setB results/setB
```

- Every algorithm must be measured at every (layout, size): the replay
  needs all of them at each point. Set B includes `auto` (MPICH's own
  selection), which the reports use as a reference.
- Plan for the cost: set B measures every layout at 4 sizes per octave for
  every algorithm; fewer sizes per octave, layouts or sessions shrink it
  (`--per-octave`, `--N`, `--ppn-list`, `--sessions`). A cut-off job
  resumes when `run_set.sh` is run again.
- Keep one allocation: measurements on different nodes differ by more than
  the noise. `make_data.py` keeps one (below).

## 2. Convert

```sh
python3 replay/make_data.py results/setB [--alloc-host <text in a host name>]
```

This writes `replay/data/sessions.csv.gz` (one row per algorithm, layout,
size and session: the median of the request's batches and its wall time)
and `replay/data/gt.csv.gz` (the ground truth: the median over all
batches of all sessions). The format is in `replay/common.py`.

## 3. Run

```sh
bin/collmodel --replay --algs binomial,scatter_ring_allgather,circ_graph --out m.json
python3 replay/replay_v1.py --algs binomial --seeds 3 --tag t && python3 replay/replay_v1.py --report t
python3 replay/replay_v2.py --seeds 3 --v1 --tag t && python3 replay/replay_v2.py --report t
python3 replay/replay_v3.py --seeds 3
```

The allowed layouts and sizes are those in the data, and a measurement's
cost is its wall time times the largest N in the data (the allocation).
Reports go to `replay/results/`.
