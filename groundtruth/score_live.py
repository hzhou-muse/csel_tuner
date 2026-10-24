#!/usr/bin/env python3
"""Score the live checks (numpy only).

    python3 score_live.py switch RESULT_DIR
    python3 score_live.py validate --truth DIR --default DIR --tuned DIR --model MODEL.json \\
        --tuning TUNING.json [--tuning-node-aware T.json] [--json SUMMARY.json]

switch: the plan set "switch" (gen_plans.py) measures the same requests with
the algorithm chosen per request (S-req, cvar.*) and for the launch
(S-env-<alg>). Per algorithm: the ratio T(per request) / T(per launch) at
every (layout, size), the data checks, and errors. An algorithm is "ok"
when its median ratio is within 10% and 90% of its points within 25%,
"noisy" when only the median is (the launches run at different times),
and FAIL otherwise or on a failed data check. An ignored switch moves the
medians far from 1. The verdict: SWITCH CHECK PASSED (all ok), PASSED
WITH NOISE (no FAIL), or FAILED.

validate: the plan set "validate" measures, at points the model builder
did not choose, every algorithm per request (truth), MPICH's own selection
(default) and auto with the tuning file loaded (tuned). Regret =
T(selection) / T(fastest algorithm), over all points and split into layouts
the builder measured and those it did not, for:
  default            MPICH's own selection, measured
  tuned              MPICH with the tuning file, measured
  tree (offline)     the algorithm the tuning file picks, timed by truth
  model, per point   the model's own choice, timed by truth
plus tuned / T(tree's algorithm) (about 1 if MPICH applied the file), and
the model's own regret estimate for comparison.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ALG_CVAR = "MPIR_CVAR_BCAST_INTRA_ALGORITHM"


def records(path):
    """(info env, [ok non-check bcast replies], [check replies], [errors]) of one JSONL file."""
    env, ok, checks, errors = {}, [], [], []
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("op") == "info":
            env = r.get("env", {})
        elif r.get("status") == "error":
            errors.append(r.get("msg", ""))
        elif r.get("op") == "bcast" and r.get("status") == "ok":
            (checks if "check" in r else ok).append(r)
    return env, ok, checks, errors


def alg_of(r, env):
    return r.get("cvars", {}).get(ALG_CVAR, env.get(ALG_CVAR, "auto"))


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def switch(a):
    req, env_t, checks, errors = {}, {}, {}, []
    for path in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        env, ok, chk, err = records(path)
        errors += [f"{name}: {e}" for e in err]
        mode = "req" if name.startswith("S-req") else "env"
        for r in ok:
            key = (alg_of(r, env), r["p"] // r["ppn"], r["ppn"], r["m"])
            (req if mode == "req" else env_t)[key] = float(np.median(r["t_us"]))
        for r in chk:
            c = checks.setdefault((alg_of(r, env), mode), [0, 0])
            c[0] += 1
            c[1] += r["check"] == "ok"
    algs = sorted({k[0] for k in req} | {k[0] for k in env_t})
    out = ["# Per-request algorithm switch check", "",
           "Ratio = T(algorithm per request, cvar.*) / T(algorithm for the launch, environment), same "
           "layout and size.", "",
           "| algorithm | points | median ratio | p10 | p90 | within 25% | data checks per request / launch | verdict |",
           "|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for alg in algs:
        keys = sorted(k for k in req if k[0] == alg and k in env_t)
        r = np.array([req[k] / env_t[k] for k in keys])
        cr = checks.get((alg, "req"), [0, 0])
        ce = checks.get((alg, "env"), [0, 0])
        if not len(r):
            verdicts.append("FAIL")
            out.append(f"| {alg} | 0 | | | | | | FAIL |")
            continue
        median_ok = abs(np.median(r) - 1) <= 0.10 and cr[0] == cr[1] and ce[0] == ce[1]
        tail_ok = np.mean(np.abs(np.log(r)) <= np.log(1.25)) >= 0.9
        v = "ok" if median_ok and tail_ok else ("noisy" if median_ok else "FAIL")
        verdicts.append(v)
        out.append(f"| {alg} | {len(r)} | {np.median(r):.3f} | {np.quantile(r, .1):.3f} | {np.quantile(r, .9):.3f} | "
                   f"{np.mean(np.abs(np.log(r)) <= np.log(1.25)):.0%} | {cr[1]}/{cr[0]} , {ce[1]}/{ce[0]} | {v} |")
    if errors:
        verdicts.append("FAIL")
        out += ["", f"Error replies ({len(errors)}):", ""] + [f"- {e}" for e in errors[:20]]
    if not verdicts or "FAIL" in verdicts:
        final = "SWITCH CHECK FAILED"
    elif "noisy" in verdicts:
        final = "SWITCH CHECK PASSED WITH NOISE"
    else:
        final = "SWITCH CHECK PASSED"
    out += ["", final]
    print("\n".join(out))
    return 1 if final.endswith("FAILED") else 0


def load_model(path):
    """The model file (collmodel/4; older ones convert with bin/collmodel --import)."""
    from csel_tuner.model import Model
    try:
        return Model.load(path)
    except ValueError as e:
        sys.exit(str(e))


def layout_mask(masks, n, q):
    """The restriction mask of a layout: recorded by the precheck, or else that
    of the recorded layouts of the same kind (single-node, one rank per node, or
    neither), or 0."""
    if (n, q) in masks:
        return masks[(n, q)]
    same = {m for (n2, q2), m in masks.items() if (n2 == 1) == (n == 1) and (q2 == 1) == (q == 1) and n2 * q2 > 1}
    return same.pop() if len(same) == 1 else 0


def times(directory):
    """{(alg or None, N, ppn, m): t} of the result directory, and the launch env."""
    out, env_all = {}, {}
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        env, ok, _, _ = records(path)
        env_all.update(env)
        for r in ok:
            out[(r.get("cvars", {}).get(ALG_CVAR), r["p"] // r["ppn"], r["ppn"], r["m"])] = \
                float(np.median(r["t_us"]))
    return out, env_all


def summary(r):
    r = np.asarray(r, float)
    return {"n": int(len(r)), "geomean": gm(r), "p95": float(np.quantile(r, .95)), "max": float(r.max()),
            "frac_gt_1.1": float(np.mean(r > 1.1))}


def validate(a):
    from csel_tuner.restrict import GPU
    from csel_tuner.tree import json_choice
    model = load_model(a.model)
    measured = set(model.measured_layouts())
    masks = model.masks

    def set_algs(mask):
        return [x for x in model.set_algs(mask)]

    def model_choice(mask, n, q, i):
        """The model's algorithm at (n, q) and size index i: the fastest predicted of those allowed there."""
        algs = set_algs(mask)
        if (mask, n, q) not in pred:
            pred[(mask, n, q)] = np.array([model.predict(alg, n, q) for alg in algs])
        return algs[int(np.argmin(pred[(mask, n, q)][:, i]))]
    sizes = np.array(model.sizes)
    op = model.op
    bits = model.conditions
    leaf_alg = {model.leaf(x): x for x in model.algorithms}
    truth, _ = times(a.truth)
    default, env_d = times(a.default)
    tuned, env_t = times(a.tuned)
    algs = sorted({k[0] for k in truth if k[0] is not None})
    trees = {"tree (offline)": json.load(open(a.tuning))}
    if a.tuning_node_aware:
        trees["tree, node-aware (offline)"] = json.load(open(a.tuning_node_aware))
    # every subtree of a file by name; the selection starts at <op>-intra-auto
    trees = {k: {n.split("=", 1)[1]: t for n, t in v.items()} for k, v in trees.items()}
    points = sorted({k[1:] for k in truth if k[0] is not None})
    pred = {}
    rows = {name: [] for name in ["default", "tuned", *trees, "model, per point"]}
    split = {name: ([], []) for name in rows}
    by_node = {name: ([], []) for name in rows}
    consistency, skipped, default_wins = [], 0, []
    for n, q, m in points:
        mask = layout_mask(masks, n, q)
        t = {alg: truth[(alg, n, q, m)] for alg in algs if (alg, n, q, m) in truth}
        if any(alg not in t for alg in set_algs(mask)):
            skipped += 1
            continue
        # the fastest of every algorithm measured here and MPICH's default
        best = min(list(t.values()) + ([default[(None, n, q, m)]] if (None, n, q, m) in default else []))
        x = {"p": n * q, "m": m, "N": n, "ppn": q, "pof2": float((n * q) & (n * q - 1) == 0),
             "cond": {b: bool(mask >> k & 1) for k, b in enumerate(bits)}}
        x["cond"][GPU] = model.mem == "device"     # a file of host and device models starts with gpu
        i = int(np.argmin(np.abs(np.log2(sizes) - np.log2(max(m, 1)))))
        alg = model_choice(mask, n, q, i)
        r = {"model, per point": t[alg] / best if alg in t else float("nan")}
        for name, tree in trees.items():
            key = json_choice(tree[f"{op}-intra-auto"], x, tree, bits)
            alg = leaf_alg.get(key, key.split("_intra_", 1)[-1])
            r[name] = t[alg] / best if alg in t else float("nan")
            if name == "tree (offline)" and (None, n, q, m) in tuned and alg in t:
                consistency.append(tuned[(None, n, q, m)] / t[alg])
        if (None, n, q, m) in default:
            r["default"] = default[(None, n, q, m)] / best
            default_wins.append(default[(None, n, q, m)] / min(t.values()))
        if (None, n, q, m) in tuned:
            r["tuned"] = tuned[(None, n, q, m)] / best
        for name, v in r.items():
            if v == v:
                rows[name].append(v)
                split[name][0 if (n, q) in measured else 1].append(v)
                by_node[name][0 if n == 1 else 1].append(v)
    first = next((r for r in model.runs if r.get("mode") == "create"), {})
    imported = next((r.get("meta") or {} for r in model.runs if r.get("mode") == "import"), {})
    est = next((x.get("estimate", {}) for x in first.get("sets", []) if x["mask"] == 0), None) or \
        imported.get("estimate") or next((x["meta"].get("estimate", {}) for r in model.runs
                                          for x in (r.get("sets") or []) if "meta" in x), {})
    info = {"stop": first.get("stop", imported.get("stop")), "n_layouts": len(measured),
            "node_h": sum(r.get("node_h") or 0 for r in model.runs) or imported.get("node_h")}
    out = ["# Validation of the tuning file", "",
           f"{len(points) - skipped} points ({len({p[:2] for p in points})} layouts); regret = T(selection) / "
           f"T(fastest at that point), the fastest of every algorithm measured per request and MPICH's default.", "",
           "| selection | points | geomean | p95 | max | > 1.1 | geomean, layouts the builder measured / not "
           "| geomean, single-node / multi-node |",
           "|---|---|---|---|---|---|---|---|"]
    res = {}
    for name, v in rows.items():
        if not v:
            continue
        s = summary(v)
        res[name] = dict(s, measured=gm(split[name][0]), unmeasured=gm(split[name][1]),
                         single_node=gm(by_node[name][0]), multi_node=gm(by_node[name][1]))
        out.append(f"| {name} | {s['n']} | {s['geomean']:.3f} | {s['p95']:.2f} | {s['max']:.2f} | "
                   f"{s['frac_gt_1.1']:.0%} | {gm(split[name][0]):.3f} / {gm(split[name][1]):.3f} | "
                   f"{gm(by_node[name][0]):.3f} / {gm(by_node[name][1]):.3f} |")
    d_r = np.array(default_wins)
    if len(d_r) and np.mean(d_r < 0.95) > 0:
        res["default faster than every candidate"] = {"fraction": float(np.mean(d_r < 0.95)),
                                                      "median_factor": float(np.median(d_r[d_r < 0.95]))}
        out += ["", f"MPICH's default is more than 5% faster than every candidate algorithm at "
                f"{np.mean(d_r < 0.95):.0%} of the points (median {np.median(d_r[d_r < 0.95]):.2f}x): it uses an "
                "algorithm the builder did not consider (e.g. release_gather, smp, nb); add it to --algs."]
    if consistency:
        c = np.array(consistency)
        res["tuned / truth(tree algorithm)"] = {"median": float(np.median(c)), "p10": float(np.quantile(c, .1)),
                                                "p90": float(np.quantile(c, .9))}
        out += ["", f"MPICH with the tuning file vs. the tree's algorithm timed per request: median "
                f"{np.median(c):.3f} (p10 {np.quantile(c, .1):.3f}, p90 {np.quantile(c, .9):.3f}); about 1 if MPICH "
                "applied the file."]
    out += ["", f"The model's own estimate: regret geomean {est.get('regret_geomean')}, p95 {est.get('regret_p95')} "
            f"(stop: {info['stop']}, {info['n_layouts']} layouts, {info['node_h']} node-h)."]
    if "MPIR_CVAR_COLL_SELECTION_JSON_FILE" not in env_t:
        out.append("\nWARNING: the tuned launch's environment has no MPIR_CVAR_COLL_SELECTION_JSON_FILE.")
    if "MPIR_CVAR_COLL_SELECTION_JSON_FILE" in env_d:
        out.append("\nWARNING: the default launch's environment has MPIR_CVAR_COLL_SELECTION_JSON_FILE set.")
    print("\n".join(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"points": len(points) - skipped, "results": res, "model_estimate": est}, f, indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("switch")
    s.add_argument("dir")
    v = sub.add_parser("validate")
    for k in ("truth", "default", "tuned", "tuning"):
        v.add_argument("--" + k, required=True)
    v.add_argument("--model", required=True, help="model file (collmodel/4)")
    v.add_argument("--tuning-node-aware")
    v.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "switch":
        sys.exit(switch(a))
    if a.cmd == "validate":
        sys.exit(validate(a))
    ap.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
