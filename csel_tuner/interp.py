"""Interpolation without scipy (numpy only, Python 3.6+), for the model builder.

The builder runs on the cluster's compute node, where Python may have numpy
but not scipy (some clusters' system Python 3.6). These match scipy where it matters:

  pchip(x, y, xq)         PchipInterpolator(x, y, extrapolate=True)(xq):
                          Fritsch-Carlson slopes, the same end rule, and the end
                          cubics extended outside [x0, xn].
  Delaunay(pts)           2-D Delaunay triangulation (Bowyer-Watson). Points on
                          a grid are co-circular, so every point gets a tiny
                          deterministic jitter; any valid triangulation of a
                          co-circular set is accepted (scipy may pick the other
                          diagonal of a square).
  linear_nd(pts, v, q)    LinearNDInterpolator: barycentric weights in the
                          triangle that holds q; None outside the hull.
  nearest(pts, v, q)      NearestNDInterpolator.
"""
import numpy as np


def pchip_slopes(x, y):
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    if n == 2:
        return np.array([delta[0], delta[0]])
    d = np.zeros(n)
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    same = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    d[1:-1] = np.where(same, hm, 0.0)

    def edge(h0, h1, m0, m1):
        e = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        if np.sign(e) != np.sign(m0) or e == 0 or m0 == 0:
            return 0.0
        if np.sign(m0) != np.sign(m1) and abs(e) > abs(3 * m0):
            return 3 * m0
        return e
    d[0] = edge(h[0], h[1], delta[0], delta[1])
    d[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


def pchip(x, y, xq):
    """Monotone cubic Hermite interpolation of (x, y) at xq; x ascending.
    y may be 1-D (one curve) or 2-D (one curve per column of x's length rows)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xq = np.asarray(xq, float)
    if len(x) == 1:
        return np.full(xq.shape + y.shape[1:], y[0])
    if y.ndim == 2:
        return np.stack([pchip(x, y[:, j], xq) for j in range(y.shape[1])], axis=-1)
    d = pchip_slopes(x, y)
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = (xq - x[i]) / h
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)
    return h00 * y[i] + h10 * h * d[i] + h01 * y[i + 1] + h11 * h * d[i + 1]


class Delaunay:
    def __init__(self, pts):
        p = np.asarray(pts, float)
        self.pts = p
        n = len(p)
        self.tri = np.zeros((0, 3), int)
        if n < 3:
            return
        rng = np.random.RandomState(12345)
        q = p + (rng.rand(n, 2) - 0.5) * 1e-7
        lo, hi = q.min(0), q.max(0)
        span = max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
        c = (lo + hi) / 2
        sup = np.array([c + [-1e3 * span, -1e3 * span], c + [1e3 * span, -1e3 * span], c + [0, 1e3 * span]])
        allp = np.vstack([q, sup])
        tris = [(n, n + 1, n + 2)]
        for k in range(n):
            px, py = allp[k]
            bad = []
            for t in tris:
                if self._incircle(allp[t[0]], allp[t[1]], allp[t[2]], px, py):
                    bad.append(t)
            edges = {}
            for t in bad:
                for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    key = (min(e), max(e))
                    edges[key] = edges.get(key, 0) + 1
            badset = set(bad)
            tris = [t for t in tris if t not in badset]
            for (a, b), cnt in edges.items():
                if cnt == 1:
                    tris.append((a, b, k))
        tris = [t for t in tris if max(t) < n]
        tri = []
        for a, b, c3 in tris:
            area = (p[b, 0] - p[a, 0]) * (p[c3, 1] - p[a, 1]) - (p[c3, 0] - p[a, 0]) * (p[b, 1] - p[a, 1])
            if abs(area) > 1e-12:                    # drop slivers of collinear points
                tri.append((a, b, c3))
        self.tri = np.array(tri, int).reshape(-1, 3)

    @staticmethod
    def _incircle(a, b, c, px, py):
        ax, ay = a[0] - px, a[1] - py
        bx, by = b[0] - px, b[1] - py
        cx, cy = c[0] - px, c[1] - py
        det = ((ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) +
               (cx * cx + cy * cy) * (ax * by - bx * ay))
        orient = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        return det * np.sign(orient) > 0

    def weights(self, q, eps=1e-9):
        """(vertex indices, barycentric weights) of the triangle holding q, or None."""
        if not len(self.tri):
            return None
        p = self.pts
        a, b, c = p[self.tri[:, 0]], p[self.tri[:, 1]], p[self.tri[:, 2]]
        det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        l1 = ((b[:, 1] - c[:, 1]) * (q[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (q[1] - c[:, 1])) / det
        l2 = ((c[:, 1] - a[:, 1]) * (q[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (q[1] - c[:, 1])) / det
        l3 = 1 - l1 - l2
        ok = np.nonzero((l1 >= -eps) & (l2 >= -eps) & (l3 >= -eps))[0]
        if not len(ok):
            return None
        j = ok[0]
        return self.tri[j], np.array([l1[j], l2[j], l3[j]])


_cache = {}


def delaunay(pts):
    key = tuple(map(tuple, np.round(np.asarray(pts, float), 12)))
    d = _cache.get(key)
    if d is None:
        if len(_cache) > 4096:
            _cache.clear()
        d = _cache[key] = Delaunay(pts)
    return d


def linear_nd(pts, vals, q):
    """Linear interpolation of rows `vals` (one per point) at q; None outside."""
    w = delaunay(pts).weights(np.asarray(q, float))
    if w is None:
        return None
    idx, lam = w
    return lam @ np.asarray(vals)[idx]


def nearest(pts, vals, q):
    d = ((np.asarray(pts, float) - np.asarray(q, float)) ** 2).sum(1)
    return np.asarray(vals)[int(np.argmin(d))]
