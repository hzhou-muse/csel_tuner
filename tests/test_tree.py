"""csel_tuner.tree: regret trees and their CSEL JSON."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csel_tuner.tree import fit_tree, json_tree, prune_to, select  # noqa: E402


def joint_case(n=2000):
    """Two algorithms; which is faster depends on m and p jointly, so no
    single split lowers the regret much on its own."""
    rng = np.random.RandomState(0)
    X = {"p": rng.randint(1, 100, n).astype(float), "m": rng.randint(1, 20, n).astype(float),
         "pof2": np.zeros(n)}
    L = np.zeros((n, 2))
    L[:, 0] = np.where(X["m"] <= 10, 0, 1.0)
    L[:, 1] = np.where(X["m"] <= 10, 1.0, 0) + np.where(X["p"] > 50, 2.0, 0)
    return X, L


def test_joint_structure_gives_zero_regret():
    X, L = joint_case()
    t, _ = prune_to(fit_tree(X, L, ["p", "m"]), tol=0.0)
    choice = np.array([select(t, {f: X[f][i] for f in X}) for i in range(len(L))])
    assert np.all(L[np.arange(len(L)), choice] == L.min(axis=1))
    assert len(t.leaves()) <= 4


def check_csel(obj, depth=1):
    """Every object but an algorithm leaf ends with "any"; values are objects."""
    assert depth <= 64
    keys = list(obj)
    if len(keys) == 1 and keys[0].startswith("algorithm="):
        return
    assert keys[-1] == "any", keys
    for v in obj.values():
        assert isinstance(v, dict)
        check_csel(v, depth + 1)


def test_json_shape():
    X, L = joint_case()
    t, _ = prune_to(fit_tree(X, L, ["p", "m"]), tol=0.0)
    j = json_tree(t, ["a", "b"], lambda alg: ("algorithm=MPIR_Bcast_intra_" + alg, {}))
    check_csel(j)
    text = str(j)
    assert "msg_size(10)" in text and "avg_msg_size" not in text and "comm_size(50)" in text   # m <= 10 and p <= 50 are the boundaries


if __name__ == "__main__":
    test_joint_structure_gives_zero_regret()
    test_json_shape()
    print("ok")
