"""csel_tuner.interp: PCHIP and 2-D linear interpolation (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csel_tuner.interp import Delaunay, linear_nd, nearest, pchip  # noqa: E402


def test_pchip_through_points_and_monotone():
    rng = np.random.RandomState(0)
    for _ in range(200):
        n = rng.randint(2, 20)
        x = np.sort(rng.choice(np.linspace(0, 26, 105), n, replace=False))
        y = np.sort(rng.normal(0, 3, n))
        assert np.allclose(pchip(x, y, x), y)
        q = np.linspace(x[0], x[-1], 400)
        assert np.all(np.diff(pchip(x, y, q)) >= -1e-9), "monotone data must give a monotone curve"


def test_pchip_two_points_is_a_line():
    assert np.allclose(pchip([0.0, 2.0], [1.0, 5.0], [-1.0, 1.0, 3.0]), [-1.0, 3.0, 7.0])


def test_delaunay_tiles_the_hull_and_is_exact_for_planes():
    rng = np.random.RandomState(1)
    grid = np.log2(np.array([(n, p) for n in (2, 3, 4, 6, 8, 12, 16, 24, 32) for p in (1, 2, 4, 8, 16, 24, 32, 48, 96)],
                            float))
    for _ in range(100):
        pts = grid[rng.choice(len(grid), rng.randint(4, len(grid)), replace=False)]
        if np.linalg.matrix_rank(pts - pts[0]) < 2:
            continue
        d = Delaunay(pts)
        cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
        area = sum(abs(cross(pts[b] - pts[a], pts[c] - pts[a])) / 2 for a, b, c in d.tri)
        # hull area by the monotone chain
        p = sorted(map(tuple, pts))
        def half(ps):
            h = []
            for q in ps:
                while len(h) >= 2 and cross(np.subtract(h[-1], h[-2]), np.subtract(q, h[-2])) <= 0:
                    h.pop()
                h.append(q)
            return h
        hull = half(p)[:-1] + half(p[::-1])[:-1]
        hull_area = abs(sum(cross(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull)))) / 2
        assert abs(area - hull_area) < 1e-6
        vals = (2 * pts[:, 0] - 3 * pts[:, 1] + 1)[:, None]
        q = pts.mean(axis=0)
        assert np.allclose(linear_nd(pts, vals, q), 2 * q[0] - 3 * q[1] + 1)


def test_outside_hull_is_none_and_nearest_works():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vals = np.array([[1.0], [2.0], [3.0]])
    assert linear_nd(pts, vals, [2.0, 2.0]) is None
    assert nearest(pts, vals, [0.9, 0.1])[0] == 2.0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("ok")
