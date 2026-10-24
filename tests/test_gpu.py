"""Device buffers: the restrictions the memory kind decides, and bin/colltree
merging a host and a device model into one tuning file whose device calls
never reach a host subtree."""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from csel_tuner.model import Model  # noqa: E402
from csel_tuner.restrict import infer_mask, mem_known  # noqa: E402
from csel_tuner.tree import json_choice  # noqa: E402

BITS = ["hierarchical", "gpu", "gpu_coll"]
HIER, GPU, GPU_COLL = 1, 2, 4
LAYOUTS = [(1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4)]
SIZES = [2 ** k for k in range(0, 21, 2)]


def alg(mask, leaf=None):
    return {"leaf": leaf, "restrictions": [], "mask": mask, "inner": []}


def curve(lay, lat, bw):
    """log2 T: a latency growing with log2 p, plus m / bandwidth."""
    p = lay[0] * lay[1]
    return np.log2(lat * (1 + np.log2(p)) + np.array(SIZES, float) / bw)


def model(mem, algs, masks, curves, conditions=BITS):
    m = Model("bcast", SIZES, LAYOUTS, {"N": [1, 4], "ppn": [1, 8], "m": [SIZES[0], SIZES[-1]]},
              conditions, algs, mem=mem)
    m.masks = dict(masks)
    for name, (lat, bw, where) in curves.items():
        for lay in LAYOUTS:
            if where(lay):
                m.set_curve(name, lay, curve(lay, lat, bw))
    return m


def host_model():
    return model("host", {"binomial": alg(0), "scatter": alg(0)}, {l: 0 for l in LAYOUTS},
                 {"binomial": (1.0, 1e3, lambda l: True), "scatter": (3.0, 1e4, lambda l: True)})


def device_model(conditions=BITS):
    """binomial/scatter as on host but slower; buffer_swap (gpu) fastest for
    large messages; ipc_read (gpu, gpu_coll) fastest on one node."""
    masks = {l: GPU | (GPU_COLL if l[0] == 1 else 0) for l in LAYOUTS}
    algs = {"binomial": alg(0), "scatter": alg(0), "buffer_swap": alg(GPU, "MPIR_Coll_buffer_swap"),
            "ipc_read": alg(GPU | GPU_COLL, "MPIDI_IPC_bcast_ipc_read")}
    return model("device", algs, masks, {
        "binomial": (10.0, 1e3, lambda l: True), "scatter": (30.0, 1e4, lambda l: True),
        "buffer_swap": (40.0, 1e6, lambda l: True), "ipc_read": (0.5, 1e7, lambda l: l[0] == 1)},
        conditions)


def colltree(argv):
    from importlib.machinery import SourceFileLoader
    ct = SourceFileLoader("colltree", os.path.join(REPO, "bin", "colltree")).load_module()
    with redirect_stdout(io.StringIO()) as buf:
        ct.main(argv)
    return buf.getvalue()


def test_mem_known():
    assert mem_known(BITS, "device") == (GPU, 0)
    assert mem_known(BITS, "host") == (0, GPU | GPU_COLL)
    assert mem_known(["hierarchical"], "device") == (0, 0)


def test_model_mem_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "d.json")
        device_model().save(path)
        assert Model.load(path).mem == "device"
        d = json.load(open(path))
        del d["mem"]                    # a file from before the field: host
        json.dump(d, open(path, "w"))
        assert Model.load(path).mem == "host"


def test_merge():
    with tempfile.TemporaryDirectory() as tmp:
        h, d, t = (os.path.join(tmp, x) for x in ("h.json", "d.json", "t.json"))
        host_model().save(h)
        device_model().save(d)
        colltree([h, d, "--out", t, "--node-aware"])
        out = json.load(open(t))
        names = sorted(out)
        assert names == ["name=bcast-intra-0", "name=bcast-intra-auto", "name=bcast-intra-gpu-0",
                         "name=bcast-intra-gpu-4"], names
        assert out["name=bcast-intra-auto"] == {"gpu": "dispatch=bcast-intra-gpu(0x4)", "any": "call=bcast-intra-0"}
        trees = {k[len("name="):]: v for k, v in out.items()}
        host_leaves = {"MPIR_Bcast_intra_binomial", "MPIR_Bcast_intra_scatter"}
        device_only = {"MPIR_Coll_buffer_swap", "MPIDI_IPC_bcast_ipc_read"}

        def choose(lay, m, gpu, gpu_coll):
            x = {"p": lay[0] * lay[1], "m": m, "N": lay[0], "ppn": lay[1],
                 "pof2": float(lay[0] * lay[1] & (lay[0] * lay[1] - 1) == 0),
                 "cond": {"hierarchical": False, "gpu": gpu, "gpu_coll": gpu_coll}}
            return json_choice(trees["bcast-intra-auto"], x, trees, BITS)
        for lay in LAYOUTS:
            for m in SIZES:
                assert choose(lay, m, False, False) in host_leaves
                # device buffers, without MPIR_CVAR_ENABLE_GPU_COLL, and with it (where the
                # model has gpu_coll: one node)
                for coll in {False, lay[0] == 1}:
                    leaf = choose(lay, m, True, coll)
                    if not coll:
                        assert leaf != "MPIDI_IPC_bcast_ipc_read"
                    if m == SIZES[-1]:
                        assert leaf in device_only, (lay, m, coll, leaf)
            assert choose(lay, 1, True, lay[0] == 1) == ("MPIDI_IPC_bcast_ipc_read" if lay[0] == 1
                                                         else "MPIR_Bcast_intra_binomial")
        # --nested-dispatch: the device dispatch as nested conditions
        colltree([h, d, "--out", t, "--nested-dispatch"])
        auto = json.load(open(t))["name=bcast-intra-auto"]
        assert auto == {"gpu": {"gpu_coll": "call=bcast-intra-gpu-4", "any": "call=bcast-intra-gpu-0"},
                        "any": "call=bcast-intra-0"}, auto
        # one model alone: today's output
        colltree([h, "--out", t])
        assert sorted(json.load(open(t))) == ["name=bcast-intra-0", "name=bcast-intra-auto"]


def test_merge_errors():
    with tempfile.TemporaryDirectory() as tmp:
        h, d, d2, t = (os.path.join(tmp, x) for x in ("h.json", "d.json", "d2.json", "t.json"))
        host_model().save(h)
        device_model().save(d)
        device_model(["hierarchical", "other", "gpu_coll"]).save(d2)
        for argv, msg in [([d, "--out", t], "no model of host buffers"),
                          ([h, h, "--out", t], "a second model of host buffers"),
                          ([h, d2, "--out", t], "has no gpu restriction")]:
            try:
                colltree(argv)
            except SystemExit as e:
                assert msg in str(e), (msg, e)
            else:
                raise AssertionError("no error: " + msg)


def test_precheck_known():
    """The memory kind decides gpu over what the precheck infers, and the
    precheck reports the contradiction."""
    from csel_tuner.restrict import precheck

    class A:
        def __init__(self, name, mask):
            self.name, self._m = name, mask

        def mask(self, bits):
            return self._m

        def allowed(self, bits, m):
            return self._m & ~m == 0
    algs = [A("binomial", 0), A("buffer_swap", GPU), A("ipc_read", GPU | GPU_COLL)]
    assert infer_mask({"binomial"}, algs, BITS) == (0, GPU_COLL)    # buffer_swap failed: gpu false
    probe = lambda n, q, name: (name == "binomial", "")
    _, masks, notes = precheck(probe, [(1, 2)], algs, BITS, mem_known(BITS, "device"))
    assert masks[(1, 2)] == GPU
    assert any("buffer_swap" in x and "failed although" in x for x in notes), notes
    assert not any("not decided" in x and "gpu," in x for x in notes), notes
    probe = lambda n, q, name: (name != "ipc_read", "")
    _, masks, notes = precheck(probe, [(1, 2)], algs, BITS, mem_known(BITS, "host"))
    assert masks[(1, 2)] == 0 and any("buffer_swap" in x and "ran although" in x for x in notes)
    assert not any("not decided" in x for x in notes), notes


if __name__ == "__main__":
    test_mem_known()
    test_model_mem_roundtrip()
    test_merge()
    test_merge_errors()
    test_precheck_known()
    print("ok")
