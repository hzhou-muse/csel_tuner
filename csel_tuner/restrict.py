"""Restriction sets: which algorithms run on which layouts, and the model sets built from that.

    feas, masks, notes = precheck(probe, layouts, algs, bits)
    sets = plan_sets(algs, bits, masks)    # [(mask, layouts, [alg names])], mask 0 first

probe(n, ppn, alg_name) -> (ok, msg): one call of the algorithm forced on the
layout (live: a probe=1 request with MPIR_CVAR_COLLECTIVE_FALLBACK=error, so a
failed restriction is an error, not a silent fallback). A layout's mask is
the set of restrictive conditions that hold there; bit k holds where an
algorithm needing only bit k ran (with no such algorithm: where any
algorithm needing bit k ran). Set 0 (the unrestricted algorithms) is built
on every layout: it is the fallback whenever no restriction holds at run
time, also where one held during tuning. Set S != 0 is built on the layouts
whose mask is exactly S, over the restricted algorithms whose mask is a
subset of S plus the unrestricted ones (for the "call=r0" choice). See
docs/design.md, section 11.
"""


def precheck(probe, layouts, algs, bits):
    feas, notes = {}, []
    for lay in layouts:
        ok = set()
        for a in algs:
            good, msg = probe(lay[0], lay[1], a.name)
            if good:
                ok.add(a.name)
            elif a.mask(bits) == 0:
                notes.append("%s fails at N=%d ppn=%d without restrictions: %s" % (a.name, lay[0], lay[1], msg))
        feas[lay] = ok
    masks = {}
    for lay, ok in feas.items():
        m = 0
        for k in range(len(bits)):
            need = [a for a in algs if a.mask(bits) == 1 << k] or [a for a in algs if a.mask(bits) >> k & 1]
            if any(a.name in ok for a in need):
                m |= 1 << k
        masks[lay] = m
        for a in algs:
            am = a.mask(bits)
            if am and (a.name in ok) != (am & ~m == 0):
                notes.append("%s at N=%d ppn=%d: %s although its conditions %s" % (
                    a.name, lay[0], lay[1], "ran" if a.name in ok else "failed",
                    "hold" if am & ~m == 0 else "do not hold"))
    return feas, masks, notes


def plan_sets(algs, bits, masks):
    """[(mask, [layouts], [algorithm names])], set 0 first; sets without layouts or
    without a restricted algorithm are left out."""
    base = [a.name for a in algs if a.mask(bits) == 0]
    out = [(0, sorted(masks), base)]
    for s in sorted(set(masks.values()) - {0}):
        lays = sorted(l for l, m in masks.items() if m == s)
        extra = [a.name for a in algs if a.mask(bits) and a.mask(bits) & ~s == 0]
        if lays and extra:
            out.append((s, lays, extra + base))
    return out


def set_name(op, mask):
    return "%s-intra-r%d" % (op, mask)
