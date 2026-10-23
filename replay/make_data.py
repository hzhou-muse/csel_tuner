#!/usr/bin/env python3
"""Convert batch-mode results (groundtruth/run_set.sh output) into replay data.

    python3 replay/make_data.py RESULT_DIR [RESULT_DIR ...] [--out replay/data]

Reads every <launch>.jsonl in the result directories: the info record
(hosts, environment) and the replies. Keeps the ok bcast replies that are
not data checks. The algorithm is the request's cvar.*_ALGORITHM if it has
one, else the launch's MPIR_CVAR_BCAST_INTRA_ALGORITHM; the session is the
`-s<k>-` part of the file name (0 if absent). Writes:

    sessions.csv.gz  alg, N, ppn, m, session, t, wall_us
                     (t = median of the request's batches, us)
    gt.csv.gz        alg, N, ppn, p, m, t, n_sess, noise, wall_us
                     (t = median over all batches of all sessions; noise =
                     spread of the session medians / their median)

Measurements on different node allocations differ by more than the noise,
so only one allocation is kept: the one with the most replies (or
--alloc-host TEXT, the allocation with a host name containing TEXT).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ALG_CVAR = "MPIR_CVAR_BCAST_INTRA_ALGORITHM"


def read(dirs):
    rows = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            m = re.search(r"-s(\d+)-", os.path.basename(path))
            session = int(m.group(1)) if m else 0
            hosts, launch_alg = None, None
            for line in open(path):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                if r.get("op") == "info":
                    hosts = tuple(r.get("hosts", []))
                    launch_alg = r.get("env", {}).get(ALG_CVAR, "auto")
                    continue
                if r.get("op") != "bcast" or r.get("status") != "ok" or "check" in r:
                    continue
                alg = r.get("cvars", {}).get(ALG_CVAR, launch_alg)
                for b, t in enumerate(r["t_us"]):
                    rows.append((hosts, alg, session, r["p"] // r["ppn"], r["ppn"], r["p"], r["m"], b, t,
                                 r["wall_us"]))
    return pd.DataFrame(rows, columns=["hosts", "alg", "session", "N", "ppn", "p", "m", "batch", "t", "wall_us"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="result directories of groundtruth/run_set.sh")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--alloc-host", help="keep the allocation with a host name containing this text")
    a = ap.parse_args()
    raw = read(a.dirs)
    if raw.empty:
        sys.exit("no bcast replies found")
    allocs = raw.groupby("hosts").size().sort_values(ascending=False)
    if a.alloc_host:
        pick = [h for h in allocs.index if any(a.alloc_host in x for x in h)]
        if not pick:
            sys.exit(f"no allocation includes {a.alloc_host}")
        hosts = pick[0]
    else:
        hosts = allocs.index[0]
    if len(allocs) > 1:
        print(f"{len(allocs)} allocations; keeping the one with {allocs[hosts]} batches "
              f"({len(hosts)} nodes, first {hosts[0] if hosts else '?'})")
    raw = raw[raw.hosts == hosts].drop(columns="hosts")
    key = ["alg", "N", "ppn", "m"]
    sess = raw.groupby(key + ["session"]).agg(t=("t", "median"), wall_us=("wall_us", "first")).reset_index()
    g = raw.groupby(key + ["p"])
    gt = g.t.median().rename("t").to_frame()
    per = raw.groupby(key + ["p", "session"]).t.median().unstack()
    gt["n_sess"] = per.count(axis=1)
    gt["noise"] = ((per.max(axis=1) - per.min(axis=1)) / per.median(axis=1)).fillna(0)
    gt["wall_us"] = g.wall_us.mean()
    gt = gt.reset_index()[["alg", "N", "ppn", "p", "m", "t", "n_sess", "noise", "wall_us"]]
    full = gt.groupby(["N", "ppn", "m"]).alg.nunique()
    if (full < gt.alg.nunique()).any():
        print(f"warning: {int((full < gt.alg.nunique()).sum())} (N, ppn, m) points lack some algorithm; "
              "the replay needs every algorithm at every point")
    os.makedirs(a.out, exist_ok=True)
    sess.to_csv(os.path.join(a.out, "sessions.csv.gz"), index=False, compression="gzip")
    gt.to_csv(os.path.join(a.out, "gt.csv.gz"), index=False, compression="gzip")
    print(f"{a.out}: {len(gt)} points ({gt.alg.nunique()} algorithms, {gt[['N', 'ppn']].drop_duplicates().shape[0]} "
          f"layouts, {gt.m.nunique()} sizes), {len(sess)} session medians")


if __name__ == "__main__":
    main()
