"""The growth anchors' guard and timeouts: points predicted too costly are left out, a timeout
is a result (journaled, not retried, skipped by later runs), and the server keeps the replies
of a batch that came before a timeout. Synthetic: no replay data, no MPI."""
import json
import os
import sys
import tempfile
import textwrap
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_growth_anchors import load_collmodel  # noqa: E402
from csel_tuner.model import Model  # noqa: E402
from csel_tuner.server import Server, ServerError, Timeout  # noqa: E402

SIZES = [1024, 4096, 16384, 21845]
ALG = {"leaf": None, "restrictions": [], "mask": 0, "inner": []}


def call_us(alg, n, m):
    """ring: p^2 growth (the whole buffer in each of p - 1 rounds), 19.2 s per call at 16 nodes
    and the largest size; lin: cheap."""
    if alg == "ring":
        return 0.3e6 * (n / 2) ** 2 * m / 21845 if n > 1 else 50.0 * m / 1024
    return 10.0 * n * m / 1024


def make(cm, timeout=600, max_s=None, runs=()):
    lays = [(n, 96) for n in (1, 2, 4, 8, 16, 32)]
    model = Model("alltoall", SIZES, lays, {}, algorithms={"ring": dict(ALG), "lin": dict(ALG)})
    model.runs = list(runs)
    a = SimpleNamespace(growth_n=None, growth_ppn="96", growth_sizes="1k,4k,16k,4M", growth_max_s=max_s,
                        timeout=timeout, batches=5, catalog=None, over_limit=None)
    asked = []

    def measure(n, ppn, alg, ms, tag=None, timeout_ok=False):
        assert tag == "growth" and timeout_ok and len(ms) == 1
        asked.append((alg, n, ppn, ms[0]))
        t = call_us(alg, n, ms[0])
        if (cm.GROWTH_CALLS + a.batches) * t * 1e-6 > timeout:      # the server would not answer in time
            return [(None, timeout * 1e6)]
        model.add_measurement(0, alg, n, ppn, ms[0], [t] * 5, 8 * t)
        return [(t, 8 * t)]
    rec = cm.growth_anchors(a, model, measure, None, lambda: None, lambda: None, lambda: 0.0, allocation=lays)
    return rec, asked, model


def test_guard_leaves_out_predicted_points():
    """Default limit (timeout / 2 = 300 s): ring at 32x96 is predicted from 8 and 16 nodes
    (exponent 2): 16 KB over the limit, 21845 beyond it; the smaller sizes and lin are measured."""
    cm = load_collmodel()
    rec, asked, _ = make(cm)
    g = {(x["alg"], x["N"], x["m"]): x for x in rec["guarded"]}
    assert set(g) == {("ring", 32, 16384), ("ring", 32, 21845)}, g
    assert g[("ring", 32, 16384)]["why"] == "predicted over the limit"
    assert abs(g[("ring", 32, 16384)]["predicted_s"] - 8 * 57.6) < 1, g
    assert "exponent 2.00" in g[("ring", 32, 16384)]["source"]
    assert g[("ring", 32, 21845)]["why"] == "beyond 32x96 m=16384"
    assert ("ring", 16, 96, 21845) in asked and ("ring", 32, 96, 4096) in asked
    assert ("ring", 32, 96, 16384) not in asked and not rec["timeouts"]
    assert rec["max_s"] == 300 and rec["measured"] == rec["points"] - 2 and rec["stop"] == "done"


def test_timeout_is_a_result():
    """No prediction limit: ring 32x96 at 21845 times out (614 s > 600 s) once; it is recorded, not
    asked again, and a later run on the model skips it unless its --timeout is longer."""
    cm = load_collmodel()
    rec, asked, model = make(cm, max_s=1e9)
    assert [(x["alg"], x["N"], x["m"], x["after_s"]) for x in rec["timeouts"]] == [("ring", 32, 21845, 600)]
    assert asked.count(("ring", 32, 96, 21845)) == 1 and not rec["guarded"]
    runs = [{"growth_anchors": rec}]
    rec2, asked2, _ = make(cm, max_s=1e9, runs=runs)
    assert ("ring", 32, 96, 21845) not in asked2
    assert any(x["why"].startswith("timeout after 600 s in an earlier run") for x in rec2["guarded"])
    rec3, asked3, _ = make(cm, timeout=900, max_s=1e9, runs=runs)
    assert ("ring", 32, 96, 21845) in asked3 and not rec3["timeouts"]


def test_guard_dominance_and_sources():
    cm = load_collmodel()
    m = Model("alltoall", SIZES, [(n, q) for n in (1, 2, 4) for q in (8, 96)], {}, algorithms={"x": dict(ALG)})
    gd = cm.GrowthGuard(m, 300, 600, 5)
    gd.block("x", 2, 96, 4096)
    assert gd.check("x", 4, 96, 16384)["why"] == "beyond 2x96 m=4096"
    assert gd.check("x", 4, 8, 16384) is None          # fewer ranks per node: not dominated
    assert gd.check("x", 4, 96, 1024) is None          # smaller size
    assert gd.predict_us("x", 4, 8, 1024) == (None, None)       # no records, no curves
    m.add_measurement(0, "x", 2, 8, 1024, [100.0], 800.0)
    t, src = gd.predict_us("x", 4, 8, 1024)
    assert t == 200.0 and src.startswith("growth from N=2 (exponent 1.00")     # one point: linear in N
    assert gd.predict_us("x", 2, 8, 1024) == (100.0, "measurement")


def test_journal_keeps_timeouts():
    """make_measure: a timeout reply comes back as None and is journaled; a resumed run takes it from
    the journal without sending; without timeout_ok a timeout is an error."""
    cm = load_collmodel()

    def make_req(n, ppn, alg, m):
        return {"op": "alltoall", "m": m, "p": n * ppn, "ppn": ppn, "cvar.X": alg}
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "j.log")
        j = cm.Journal(log, False)
        j.write({"job": {"start": 0}})
        sent = []

        def send(reqs, occ, timeout_ok=False):
            sent.extend(reqs)
            return [{"status": "timeout", "msg": "no reply", "wall_us": 6e8} if r["m"] > 4096 else
                    {"status": "ok", "t_us": [1.0], "wall_us": 10.0} for r in reqs]
        meas = cm.make_measure(j, send, make_req)
        assert meas(32, 96, "ring", [1024, 21845], tag="growth", timeout_ok=True) == [(1.0, 10.0), (None, 6e8)]
        j.f.close()
        assert any('"timeout"' in x for x in open(log))
        j2 = cm.Journal(log, True)

        def no_send(reqs, occ, timeout_ok=False):
            raise AssertionError("sent again: %s" % reqs)
        meas2 = cm.make_measure(j2, no_send, make_req)
        assert meas2(32, 96, "ring", [21845], tag="growth", timeout_ok=True) == [(None, 6e8)]
        j2.f.close()
        meas3 = cm.make_measure(cm.Journal(None, False), send, make_req)
        try:
            meas3(32, 96, "ring", [21845])
            assert False, "a timeout without timeout_ok must raise"
        except ServerError:
            pass


FAKE_SERVER = textwrap.dedent("""\
    import json, sys, time
    for line in sys.stdin:
        req = dict(kv.split("=", 1) for kv in line.split())
        if req.get("op") == "quit":
            break
        if req.get("op") == "info":
            print(json.dumps({"status": "ok", "nodes": 1, "min_ppn": 1, "max_m": 1 << 20}), flush=True)
            continue
        if req.get("m") == "999":
            time.sleep(30)
        print(json.dumps({"status": "ok", "t_us": [1.0], "wall_us": 5.0, "m": int(req["m"])}), flush=True)
""")


def test_server_timeout_keeps_earlier_replies():
    with tempfile.TemporaryDirectory() as d:
        srv_py = os.path.join(d, "fake_srv.py")
        with open(srv_py, "w") as f:
            f.write(FAKE_SERVER)
        s = Server(launcher=[sys.executable], server=srv_py, timeout=1, warmup=())
        try:
            s.measure_many([{"op": "x", "m": 1}, {"op": "x", "m": 2}, {"op": "x", "m": 999}, {"op": "x", "m": 3}])
            assert False, "expected a timeout"
        except Timeout as e:
            assert [r["m"] for r in e.replies] == [1, 2] and "m=999" in str(e)
        s.restart()
        assert s.measure(op="x", m=4)["m"] == 4
        s.close()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
