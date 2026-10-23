#!/usr/bin/env python3
"""Check a batch-mode result directory (any plan set). Standard library only.

    python3 check_r.py PLAN_DIR RESULT_DIR

Per launch: done marker, exit code, coverage of the plan's requests (every
seq answered with status ok), error replies, data checks, the placement
record (node0_locals must match the reply's place: compact j, spread
floor(j * n / ppn), nic that base moved to local rank = j mod nic_mod within
its group of nic_mod; n = node size, j < ppn), the idle settings
(MPISRV_IDLE / _IDLE_NS / _SPINUP_US against the launch's idle field), the
node allocation (hosts and cabinets), the MPICH library and algorithm, and
the topology warnings mpisrv reports (site config vs. binding and MPICH).
Prints a report and exits 1 if anything is wrong.
"""
import json
import os
import sys


def plan_requests(path):
    n = 0
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            n += 1
    return n


def placement(place, ppn, n, nic_mod):
    if place == "compact":
        return list(range(ppn))
    base = [j * n // ppn for j in range(ppn)]
    if place == "spread":
        return base
    return sorted(b - b % nic_mod + j % nic_mod for j, b in enumerate(base))


def cabinets(hosts):
    return sorted({h[:5] for h in hosts})


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    plans, results = sys.argv[1], sys.argv[2]
    bad = []
    log = {}
    if os.path.exists(os.path.join(results, "launches.log")):
        for line in open(os.path.join(results, "launches.log")):
            if line.strip():
                d = json.loads(line)
                log[d["id"]] = d
    allocs = {}
    print("| launch | done | rc | requests | answered | errors | checks ok/run | placement | cabinets |")
    print("|---|---|---|---|---|---|---|---|---|")
    for line in open(os.path.join(plans, "launches.txt")):
        if line.startswith("#") or not line.strip():
            continue
        lid, nodes, ppn, alg, idle, plan = line.split()
        n_req = plan_requests(plan)
        out = os.path.join(results, lid + ".jsonl")
        seqs, errors, checks, checks_ok, place_bad, infos = set(), 0, 0, 0, 0, []
        if os.path.exists(out):
            for l in open(out):
                d = json.loads(l)
                if d.get("op") == "info":
                    infos.append(d)
                    continue
                if d["status"] == "ok":
                    seqs.add(d["seq"])
                    if "check" in d:
                        checks += 1
                        checks_ok += d["check"] == "ok"
                        info = infos[-1] if infos else {}
                        want = placement(d.get("place", "spread"), d["ppn"], info.get("min_ppn", 96),
                                         info.get("nic_mod", 1))
                        if d.get("node0_locals") != want:
                            place_bad += 1
                elif d["status"] == "error":
                    errors += 1
        done = os.path.exists(os.path.join(results, lid + ".done"))
        rc = log.get(lid, {}).get("rc", "-")
        cabs = sorted({c for i in infos for c in cabinets(i["hosts"])})
        for i in infos:
            allocs.setdefault(tuple(i["hosts"]), []).append(lid)
            if i.get("topology_warnings"):
                bad.append(f"{lid}: topology warnings: {i['topology_warnings']}")
            env = i.get("env", {})
            want = dict(zip(["MPISRV_IDLE", "MPISRV_IDLE_NS", "MPISRV_SPINUP_US"], idle.split(":")))
            for k, v in want.items():
                if env.get(k) != v:
                    bad.append(f"{lid}: info record {k}={env.get(k)} != {v}")
            if env.get("MPIR_CVAR_BCAST_INTRA_ALGORITHM") != alg:
                bad.append(f"{lid}: info record algorithm {env.get('MPIR_CVAR_BCAST_INTRA_ALGORITHM')} != {alg}")
        print(f"| {lid} | {'yes' if done else 'NO'} | {rc} | {n_req} | {len(seqs)} | {errors} | "
              f"{checks_ok}/{checks} | {'ok' if not place_bad else f'{place_bad} BAD'} | {','.join(cabs)} |")
        if not done or len(seqs) != n_req:
            bad.append(f"{lid}: incomplete ({len(seqs)}/{n_req} answered, done={done})")
        if errors:
            bad.append(f"{lid}: {errors} error replies")
        if checks_ok != checks:
            bad.append(f"{lid}: {checks - checks_ok} failed data checks")
        if place_bad:
            bad.append(f"{lid}: {place_bad} placement records differ from the expected local ranks")
    print()
    libs, topo = set(), set()
    for hosts, lids in allocs.items():
        print(f"allocation {len(hosts)} nodes, cabinets {','.join(cabinets(hosts))}, first host {hosts[0]}: "
              f"{len(lids)} launch records")
    for fn in sorted(os.listdir(results)):
        if fn.endswith(".jsonl"):
            for l in open(os.path.join(results, fn)):
                d = json.loads(l)
                if d.get("op") == "info":
                    libs.add(d.get("library", "?"))
                    topo.add((d.get("nic_mod"), d.get("num_close_nics"), d.get("num_nics_available")))
                break
    print(f"libraries: {sorted(libs)}")
    print("topology (nic_mod, MPICH num_close_nics, num_nics_available):", sorted(topo, key=str))
    if len(allocs) > 1:
        print("NOTE: more than one node allocation (expected if the set needed several jobs)")
    print()
    if bad:
        print("PROBLEMS:")
        for b in bad:
            print(" -", b)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
