"""Param maps (csel_tuner/params.py): rule modules, the map's identity and version check,
measurements with their parameters in the .model file, bin/algoparam --refit, collmodel's map
checks, and colltree's leaves with the theta of their region. The circ_graph rule itself:
test_param_circ_graph.py."""
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stderr, redirect_stdout
from importlib.machinery import SourceFileLoader

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from csel_tuner import modelfile  # noqa: E402
from csel_tuner.model import Model, ModelFile  # noqa: E402
from csel_tuner.params import K, M, ParamMap, load_maps, map_tags, rule, theta_of_req  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_param_circ_graph import synth_cells  # noqa: E402


def tool(name):
    return SourceFileLoader(name, os.path.join(REPO, "bin", name)).load_module()


def test_version_check():
    d = ParamMap.default("bcast", "circ_graph").to_dict()
    assert d["version"] == rule("circ_graph").VERSION
    for bad in (dict(d, version=rule("circ_graph").VERSION + 1), {k: v for k, v in d.items() if k != "version"}):
        try:
            ParamMap.from_dict("circ_graph", bad)
            assert False, "a map of another rule version must be refused"
        except ValueError as e:
            assert "rule version" in str(e) and "--refit" in str(e), e
    # collmodel and colltree refuse such a map in a model, before measuring or writing anything
    ct = tool("colltree")
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "m.model")
        m = Model("bcast", [1], LAYS, {"N": [1, 2], "ppn": [1, 4], "m": [1, 1]},
                  algorithms={"circ_graph": {"leaf": None, "restrictions": [], "mask": 0, "inner": [],
                                             "param_map": dict(d, version=rule("circ_graph").VERSION + 1)}})
        ModelFile([m]).save(f)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                ct.main([f, "--out", os.path.join(tmp, "t.json")])
            assert False, "colltree must refuse the map"
        except SystemExit as e:
            assert "rule version" in str(e), e
        cm = tool("collmodel")
        try:
            cm.param_maps(types.SimpleNamespace(param_map=None, op="bcast", launch="host", model=f), m)
            assert False, "collmodel must refuse the map"
        except SystemExit as e:
            assert "rule version" in str(e), e


def test_identity():
    assert rule("circ_graph").PARAMS == ["chunk_size", "q_len"]
    for bad in ("no_such_rule", "../x", ""):
        try:
            rule(bad)
            assert False, "no rule %r" % bad
        except ValueError as e:
            assert "rule" in str(e), e
    a = ParamMap.default("bcast", "circ_graph")
    b = ParamMap.default("bcast", "circ_graph", A=8 * K)
    assert a.hash != b.hash
    assert ParamMap("circ_graph", "circ_graph", a.consts, {"note": "x"}).hash == a.hash    # fit is not identity
    d = a.to_dict()
    assert ParamMap.from_dict("circ_graph", d).consts == a.consts
    d["consts"] = dict(d["consts"], C=1 << 20)
    try:
        ParamMap.from_dict("circ_graph", d)
        assert False, "an edited map must be refused"
    except ValueError as e:
        assert "hash" in str(e)
    cv = a.req_cvars("ibcast", 64, 1 * M)
    assert cv == {"cvar.MPIR_CVAR_IBCAST_CIRC_GRAPH_CHUNK_SIZE": 128 * K, "cvar.MPIR_CVAR_IBCAST_CIRC_GRAPH_Q_LEN": 8}
    assert theta_of_req(dict(cv, op="ibcast"), "ibcast", a) == {"chunk_size": 128 * K, "q_len": 8}
    assert theta_of_req({"op": "ibcast"}, "ibcast", a) is None


LAYS = [(1, 4), (2, 1), (2, 4)]
def param_unit(op="bcast", pm=None, meas=True):
    pm = pm or ParamMap.default(op, "circ_graph")
    u = Model(op, [1], LAYS, {"N": [1, 2], "ppn": [1, 4], "m": [1, 4 * M]},
              tags=map_tags(op, "host"),
              algorithms={"circ_graph": {"leaf": None, "restrictions": [], "mask": 0, "inner": [],
                                         "param_map": pm.to_dict()}})
    if meas:
        truth = ParamMap.default(op, "circ_graph", F=8 * K, A=4 * K, Fb=32 * K, C=256 * K)
        u.jobs = [{"start": 0}]
        for c in synth_cells(truth):
            for ch, t in c["t"].items():
                u.add_measurement(0, "circ_graph", c["N"], c["ppn"], c["m"], [t, t], 10.0,
                                  params={"chunk_size": ch, "q_len": 8})
    return u


def test_modelfile_params():
    u = param_unit()
    u.add_measurement(0, "circ_graph", 2, 4, 1 * M, [5.0], 1.0, anchor=True, params={"q_len": 4, "chunk_size": 4096})
    u.add_measurement(0, "circ_graph", 2, 4, 1 * M, [5.0], 1.0, growth=True)
    text = modelfile.dumps([u.to_dict()], "2026-11-25T00:00:00Z")
    assert "\nparams 0 {\"chunk_size\":1024,\"q_len\":8}\n" in text
    assert " ap%d " % (text.count("\nparams ") - 1) in text                    # anchor flag and params
    assert "param_map" in text
    units, _ = modelfile.loads(text)
    back = Model.from_dict(units[0])
    assert back.measurements == u.measurements
    assert back.algorithms == u.algorithms
    # a flag naming a missing params line is refused
    try:
        modelfile.loads(text.replace("\nparams 0 ", "\n# params 0 ", 1))
        assert False, "p0 without params 0 must be refused"
    except modelfile.ParseError as e:
        assert "params" in str(e)


def test_algoparam_refit():
    ap = tool("algoparam")
    with tempfile.TemporaryDirectory() as tmp:
        f, g = os.path.join(tmp, "cal.model"), os.path.join(tmp, "maps.model")
        ModelFile([param_unit()]).save(f)
        with redirect_stdout(io.StringIO()) as out:
            ap.main(["--op", "bcast", "--also", "ibcast,allgather", "--refit", f, "--out", g])
        assert "regret on the rows: rule max 1.000" in out.getvalue(), out.getvalue()
        maps = {op: load_maps(g, op, "host")["circ_graph"] for op in ("bcast", "ibcast", "allgather")}
        assert len({pm.hash for pm in maps.values()}) == 1
        assert maps["ibcast"].fit == {"from": "bcast", "op": "ibcast"}
        assert load_maps(g, "bcast", "gpu") == {}           # host maps do not serve device buffers
        # --set keeps a constant; the refit replaces the units of the same tags, keeps others
        with redirect_stdout(io.StringIO()):
            ap.main(["--op", "bcast", "--refit", f, "--out", g, "--set", "A=64K"])
        assert load_maps(g, "bcast", "host")["circ_graph"].consts["A"] == 64 * K
        assert load_maps(g, "ibcast", "host")["circ_graph"].consts["A"] != 64 * K
        assert len(ModelFile.load(g).units) == 3
        assert load_maps(g, "bcast", "host")["circ_graph"].fit["set"] == {"A": 64 * K}
        try:
            with redirect_stderr(io.StringIO()):
                ap.main(["--op", "bcast", "--also", "reduce", "--refit", f, "--out", g])
            assert False, "another family must be refused"
        except SystemExit:
            pass
        # a refit of a refit keeps the constants fixed before; --set replaces one, --unset drops one
        h = os.path.join(tmp, "maps2.model")

        def refit(src, *extra):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                ap.main(["--op", "bcast", "--refit", src, "--out", h] + list(extra))
            return load_maps(h, "bcast", "host")["circ_graph"], err.getvalue()
        pm, err = refit(g)
        assert pm.consts["A"] == 64 * K and pm.fit["set"] == {"A": 64 * K} and "fixed in" in err, err
        pm, _ = refit(g, "--set", "A=32K", "--set", "F=4K")
        assert (pm.consts["A"], pm.consts["F"]) == (32 * K, 4 * K) and pm.fit["set"] == {"A": 32 * K, "F": 4 * K}
        pm, _ = refit(g, "--unset", "A")
        assert pm.consts["A"] == 4 * K and pm.fit["set"] == {}        # fitted again: the rows' own A
        for bad in (["--refit", g, "--unset", "Z"], ["--unset", "A", "--launcher", "x"]):
            try:
                with redirect_stderr(io.StringIO()):
                    ap.main(["--op", "bcast", "--out", h] + bad)
                assert False, bad
            except SystemExit:
                pass


def test_collmodel_maps():
    cm = tool("collmodel")
    with tempfile.TemporaryDirectory() as tmp:
        g = os.path.join(tmp, "maps.model")
        pm = ParamMap.default("bcast", "circ_graph", A=4 * K)
        ModelFile([param_unit(pm=pm, meas=False)]).save(g)
        a = types.SimpleNamespace(param_map=g, op="bcast", launch="host")
        with redirect_stderr(io.StringIO()):
            assert cm.param_maps(a, None)["circ_graph"].hash == pm.hash
            # the model's map is used without --param-map
            base = Model("bcast", [1], LAYS, {"N": [1, 2], "ppn": [1, 4], "m": [1, 1]},
                         algorithms={"circ_graph": {"leaf": None, "param_map": pm.to_dict()}})
            assert cm.param_maps(types.SimpleNamespace(param_map=None, op="bcast", launch="host"), base) \
                ["circ_graph"].hash == pm.hash
            # another map for a model's algorithm, or a map for one measured at its defaults, is refused
            for algs, meas in (({"circ_graph": {"leaf": None,
                                                "param_map": ParamMap.default("bcast", "circ_graph").to_dict()}}, []),
                               ({"circ_graph": {"leaf": None}}, [("circ_graph", 2, 4, 1)])):
                base = Model("bcast", [1], LAYS, {"N": [1, 2], "ppn": [1, 4], "m": [1, 1]}, algorithms=algs)
                for alg, n, q, m in meas:
                    base.add_measurement(0, alg, n, q, m, [1.0], 1.0)
                try:
                    cm.param_maps(a, base)
                    assert False, "a conflicting map must be refused"
                except SystemExit:
                    pass


SIZES = [2 ** k for k in range(0, 25, 2)]
TLAYS = [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 4)]


def walk(obj, x):
    """The leaf {"algorithm=...": params} a CSEL subtree reaches at x (comm_size, msg_size, any)."""
    while True:
        for key, sub in obj.items():
            if key.startswith("algorithm="):
                return key, sub
            if key == "any":
                ok = True
            elif key in ("pof2", "!pof2"):
                ok = (x["pof2"] > 0.5) == (key == "pof2")
            else:
                name, t = key.rstrip(")").split("(")
                ok = x[{"comm_size": "p", "msg_size": "m"}[name]] <= int(t)
            if ok:
                obj = sub
                break


def test_colltree_leaves():
    ct = tool("colltree")
    pm = ParamMap.default("bcast", "circ_graph")
    algs = {"binomial": {"leaf": None, "restrictions": [], "mask": 0, "inner": []},
            "circ_graph": {"leaf": None, "restrictions": [], "mask": 0, "inner": [],
                           "params": {"chunk_size": 131072, "q_len": 8}, "param_map": pm.to_dict()}}
    m = Model("bcast", SIZES, TLAYS, {"N": [1, 4], "ppn": [1, 4], "m": [1, SIZES[-1]]}, [], algs)
    m.masks = {lay: 0 for lay in TLAYS}
    s = np.array(SIZES, float)
    for lay in TLAYS:
        p = lay[0] * lay[1]
        m.set_curve("binomial", lay, np.log2(1 + np.log2(p) * (1 + s / 1e3)))
        m.set_curve("circ_graph", lay, np.log2(20 + s / 1e3 * 1.5))      # wins at large m
    with tempfile.TemporaryDirectory() as tmp:
        f, t = os.path.join(tmp, "m.model"), os.path.join(tmp, "t.json")
        ModelFile([m]).save(f)
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            ct.main([f, "--out", t])
        assert "leaf parameters from its param map" in out.getvalue(), out.getvalue()
        tree = json.load(open(t))["name=bcast-intra-0"]
    seen = set()
    for n, q in TLAYS:
        for size in SIZES:
            x = {"p": n * q, "m": size, "pof2": float(n * q & (n * q - 1) == 0)}
            key, params = walk(tree, x)
            if key.endswith("circ_graph"):
                assert params == pm.theta(n * q, size), (n, q, size, params)
                seen.add(params["chunk_size"])
    assert len(seen) >= 3, seen          # several theta regions, split inside circ_graph's leaves


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
