"""The allowed layouts and sizes of a model build, from the allocation."""


def axis_values(top):
    """1, 2, 3, 4, 6, 8, 12, ... up to top, and top itself."""
    out = {top}
    k = 1
    while k <= top:
        out.add(k)
        if 3 * k // 2 <= top and k >= 2:
            out.add(3 * k // 2)
        k *= 2
    return sorted(out)


def size_grid(max_m, per_octave=4):
    out, k = set(), 0
    while round(2 ** (k / per_octave)) <= max_m:
        out.add(round(2 ** (k / per_octave)))
        k += 1
    return sorted(out | {max_m})
