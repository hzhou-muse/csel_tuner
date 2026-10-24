"""Driver for the benchmark server (server/mpisrv): start it, send requests, read replies.

    with Server(nprocs=8, env={"MPIR_CVAR_NUM_CLIQUES": "2"}) as s:
        s.info()
        r = s.measure(op="allreduce", m=65536, p=8, ppn=4)
        r["t_us"], r["wall_us"]

Every request and reply is appended to a JSONL log (`log=`), so a session
can be replayed in the simulator. A request that gets no reply within
`timeout` seconds kills the server and raises ServerError; `restart()`
starts a fresh one.
"""
import json
import os
import selectors
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MPIEXEC = "mpiexec"
SERVER = os.path.join(ROOT, "server", "mpisrv")


class ServerError(RuntimeError):
    pass


def warmup_sizes(max_m, step=4):
    sizes, m = [0], 1
    while m <= max_m:
        sizes.append(m)
        m *= step
    return sizes


class Server:
    def __init__(self, nprocs=2, env=None, bind=("-bind-to", "core"), launcher=None,
                 max_m=1 << 24, timeout=120, log=None, warmup=("pingpong",), server=SERVER, args=()):
        """launcher: full command prefix (e.g. ["srun", "--input=0", ...]);
        default is the local mpiexec with -n nprocs and `bind`. server: the
        mpisrv binary; args: its options after max-m (e.g. ["--mem", "device"]).
        warmup: ops to sweep over a coarse size grid right after start."""
        self.cmd = (list(launcher) if launcher else [MPIEXEC, "-n", str(nprocs), *bind]) + \
            [server, str(max_m), *args]
        self.env = dict(os.environ, **(env or {}))
        self.max_m = max_m
        self.timeout = timeout
        self.log = open(log, "a") if log else None
        self.warmup_ops = warmup
        self.count = 0
        self.wall_us = 0.0  # sum of reported wall_us: server-side benchmark cost
        self.start()

    def start(self):
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  universal_newlines=True, env=self.env, bufsize=1)
        self.buf = b""
        self.last = time.time()     # end of the last reply: driver idle time starts here
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.p.stdout, selectors.EVENT_READ)
        self.world = self.info()
        for op in self.warmup_ops:
            self.warmup(op)

    def restart(self):
        self.kill()
        self.start()

    def request(self, **req):
        return self.request_many([req])[0]

    def readline(self, timeout, what):
        """One reply line from the server's stdout. Reads the raw descriptor
        into our own buffer, so replies that arrive together are never stuck
        in a buffer select() cannot see. Lines that are not JSON (launcher
        noise) are skipped."""
        fd = self.p.stdout.fileno()
        deadline = time.time() + timeout
        while True:
            nl = self.buf.find(b"\n")
            if nl >= 0:
                line, self.buf = self.buf[:nl], self.buf[nl + 1:]
                line = line.decode(errors="replace").strip()
                if line.startswith("{"):
                    return line
                continue
            left = deadline - time.time()
            if left <= 0 or not self.sel.select(left):
                self.kill()
                raise ServerError(f"no reply in {timeout}s to: {what}")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise ServerError(f"server closed its output; exit code {self.p.wait()}")
            self.buf += chunk

    def request_many(self, reqs):
        """Send several requests at once, then read their replies in order:
        one driver round trip for the batch."""
        lines = [" ".join(f"{k}={v}" for k, v in r.items() if v is not None) for r in reqs]
        if self.p.poll() is not None:
            raise ServerError(f"server exited with {self.p.returncode}")
        self.p.stdin.write("".join(l + "\n" for l in lines))
        self.p.stdin.flush()
        t0 = time.time()
        reps = []
        for line, req in zip(lines, reqs):
            rep = json.loads(self.readline(self.timeout, line))
            if self.log:
                self.log.write(json.dumps({"time": t0, "req": req, "rep": rep}) + "\n")
            reps.append(rep)
        if self.log:
            self.log.flush()
        self.last = time.time()
        return reps

    def measure_many(self, reqs):
        reps = self.request_many(reqs)
        for rep in reps:
            if rep["status"] != "ok":
                raise ServerError(rep.get("msg", "error"))
            self.count += 1
            self.wall_us += rep["wall_us"]
        return reps

    def info(self):
        return self.request(op="info")

    def measure(self, **req):
        rep = self.request(**req)
        if rep["status"] != "ok":
            raise ServerError(rep.get("msg", "error"))
        self.count += 1
        self.wall_us += rep["wall_us"]
        return rep

    def warmup(self, op, **layout):
        """One pass of `op` over a coarse size grid; not counted as cost."""
        for m in warmup_sizes(self.max_m):
            if op == "allreduce" and m % 4:
                continue
            self.request(op=op, m=m, batches=1, target_us=200, tag="warmup", **layout)

    def kill(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait()

    def close(self, wait=30):
        """Ask the server to quit (it first answers the requests already
        sent) and wait up to `wait` seconds; kill it after that. A clean exit
        lets MPI remove its shared-memory files; a kill leaves them."""
        if self.p.poll() is None:
            try:
                self.p.stdin.write("op=quit\n")
                self.p.stdin.close()
                self.p.wait(wait)
            except (subprocess.TimeoutExpired, OSError):
                self.kill()
        if self.log:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
