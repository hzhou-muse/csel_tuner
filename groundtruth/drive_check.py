#!/usr/bin/env python3
"""Interactive-mode check: drive mpisrv through rank 0's stdin, as the model
builder will. Standard library only (Python 3.6+).

    python3 drive_check.py OUT_DIR [--nodes 2 --ppn 96 --alg binomial --idle spin]
                                   [--layouts 2x8,2x48]

Starts mpisrv without --plan through config.sh's launch(), so PALS mpiexec
must forward stdin to rank 0 and rank 0's stdout back. Then:
  1. op=info; the time to its reply is the startup time;
  2. layouts (default 2 x 8 and 2 x 48), one size per octave from 1 B to MAX_M, sizes
     shuffled, sync=call, 5 batches; this pass is repeated with the driver
     waiting 0, 10 and 200 ms before every request (driver think time);
  3. a request that does not fit (error reply expected; the server must go on);
  4. op=quit; the launch must exit with 0.
Writes OUT_DIR/I-drive-<alg>.jsonl (info record, then every reply with
"gap_ms" and the driver's round trip "rtt_ms"), .stderr, and
OUT_DIR/drive_report.txt; exits 1 if any step failed.
"""
import argparse
import json
import os
import random
import selectors
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GAPS_MS = [0, 10, 200]


class Srv:
    def __init__(self, cmd, env, stderr):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                  env=env, universal_newlines=True, bufsize=1)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.p.stdout, selectors.EVENT_READ)

    def ask(self, line, timeout):
        """Send one request; return (reply dict, round trip in ms) or raise."""
        t0 = time.time()
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        while True:
            left = timeout - (time.time() - t0)
            if left <= 0 or not self.sel.select(left):
                raise RuntimeError("no reply in %ds to: %s" % (timeout, line))
            out = self.p.stdout.readline()
            if not out:
                raise RuntimeError("server output closed (exit %s) after: %s" % (self.p.poll(), line))
            out = out.strip()
            if out.startswith("{"):          # anything else would be launcher noise
                return json.loads(out), (time.time() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--ppn", type=int, default=96)
    ap.add_argument("--alg", default="binomial")
    ap.add_argument("--idle", default="spin", help="MODE[:IDLE_NS[:SPINUP_US]] as in run_set.sh")
    ap.add_argument("--max-m", type=int, default=None, help="default: MAX_M of the site config")
    ap.add_argument("--layouts", default="2x8,2x48", help="NxPPN,... measured in the launch")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    layouts = [tuple(int(v) for v in l.split("x")) for l in a.layouts.split(",")]
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "I-drive-%s" % a.alg)
    report = []

    def say(s):
        print(s, flush=True)
        report.append(s)

    env = dict(os.environ, MPIR_CVAR_BCAST_INTRA_ALGORITHM=a.alg)
    parts = a.idle.split(":")
    for k, v in zip(["MPISRV_IDLE", "MPISRV_IDLE_NS", "MPISRV_SPINUP_US"], parts):
        env[k] = v
    max_m = "${MAX_M}" if a.max_m is None else str(a.max_m)
    script = 'source "%s/../launch/config.sh" && launch %d %d "$MPISRV" %s' % (HERE, a.nodes, a.ppn, max_m)
    rng = random.Random(a.seed)
    ok = True
    t_start = time.time()
    with open(base + ".stderr", "w") as err, open(base + ".jsonl", "w") as out:
        srv = Srv(["bash", "-c", script], env, err)
        try:
            info, _ = srv.ask("op=info", 600)
            info["startup_s"] = round(time.time() - t_start, 2)
            out.write(json.dumps(info) + "\n")
            say("startup: info reply after %.1f s, world_size=%s, hosts=%s, topology_warnings=%r"
                % (info["startup_s"], info.get("world_size"), ",".join(info.get("hosts", [])),
                   info.get("topology_warnings")))
            mm = info.get("max_m", 1 << 26)
            sizes = [1 << k for k in range(64) if 1 << k <= mm]
            seq = 0
            for gap in GAPS_MS:
                rtt_over = []
                for n, q in layouts:
                    ms = list(sizes)
                    rng.shuffle(ms)
                    for m in ms:
                        time.sleep(gap / 1e3)
                        line = "op=bcast m=%d p=%d ppn=%d sync=call batches=5 tag=I-drive-gap%d" % (
                            m, n * q, q, gap)
                        rep, rtt = srv.ask(line, 600)
                        rep["gap_ms"], rep["rtt_ms"], rep["seq"] = gap, round(rtt, 3), seq
                        seq += 1
                        out.write(json.dumps(rep) + "\n")
                        out.flush()
                        if rep.get("status") != "ok":
                            ok = False
                            say("FAIL: error reply to %s: %s" % (line, rep.get("msg")))
                            continue
                        rtt_over.append(rtt - rep["wall_us"] / 1e3)
                rtt_over.sort()
                say("gap %3d ms: %d requests ok; driver round trip minus server wall time: "
                    "median %.2f ms, max %.2f ms" % (gap, len(rtt_over), rtt_over[len(rtt_over) // 2],
                                                     rtt_over[-1]))
            rep, _ = srv.ask("op=bcast m=1 p=%d ppn=%d tag=I-drive-bad" % (a.nodes * a.ppn * 2, a.ppn), 120)
            if rep.get("status") == "error":
                say("bad request: error reply as expected (%s)" % rep.get("msg"))
            else:
                ok = False
                say("FAIL: bad request got %r" % rep)
            info2, _ = srv.ask("op=info", 120)
            say("server still answers after the bad request: %s" % (info2.get("status") == "ok"))
            ok &= info2.get("status") == "ok"
            srv.p.stdin.write("op=quit\n")
            srv.p.stdin.close()
            rc = srv.p.wait(300)
            say("quit: launch exited with %d after %.1f s in total" % (rc, time.time() - t_start))
            ok &= rc == 0
        except Exception as e:      # report and kill; the hand-back still has the partial data
            ok = False
            say("FAIL: %s" % e)
            if srv.p.poll() is None:
                srv.p.kill()
                srv.p.wait()
    say("DRIVE CHECK PASSED" if ok else "DRIVE CHECK FAILED")
    with open(os.path.join(a.out, "drive_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
