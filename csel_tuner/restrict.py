"""Restriction sets: which algorithms run on which layouts, and the model sets built from that.

    feas, masks, notes = precheck(probe, layouts, algs, bits)
    sets = plan_sets(algs, bits, masks)    # [(mask, layouts, [alg names])], mask 0 first

bits: the catalog's restrictive conditions (algos.Catalog.restriction_bits(),
one numbering for every collective). probe(n, ppn, alg_name) -> (ok, msg):
one call of the algorithm forced on the layout (live: a probe=1 request with
MPIR_CVAR_COLLECTIVE_FALLBACK=error, so a failed restriction is an error,
not a silent fallback). A layout's mask is the set of conditions true there,
inferred from which algorithms ran: an algorithm that ran has its required
bits true and its excluded bits false; one that failed has some requirement
violated, which pins a bit when all its others are known. Bits the
algorithms do not decide are taken as false and reported.

Set 0 holds the unrestricted algorithms and is built on every layout: it is
the fallback whenever no other set applies at run time, also where one
applied during tuning (e.g. release_gather is off under
MPI_THREAD_MULTIPLE). Set S != 0 is built on the layouts whose mask is
exactly S, over the restricted algorithms allowed in S plus the unrestricted
ones (for the "call=r0" choice). See docs/design.md, section 11.
"""


def infer_mask(ok, algs, bits):
    """(mask of true conditions, mask of bits left undecided) from the algorithms that ran (ok)."""
    used = 0
    for a in algs:
        used |= a.mask(bits)
    true = false = 0
    for a in algs:
        if a.name in ok:
            pos, neg = a.requires(bits)
            true |= pos
            false |= neg
    changed = True
    while changed:
        changed = False
        for a in algs:
            if a.name in ok or not a.mask(bits):
                continue
            pos, neg = a.requires(bits)
            # the requirements not known to hold; a single one must be the violated one
            open_pos, open_neg = pos & ~true, neg & ~false
            if bin(open_pos | open_neg).count("1") == 1:
                if open_pos & ~false:
                    false |= open_pos
                    changed = True
                elif open_neg & ~true:
                    true |= open_neg
                    changed = True
    return true, used & ~(true | false)


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
        m, undecided = infer_mask(ok, algs, bits)
        masks[lay] = m
        if undecided:
            notes.append("N=%d ppn=%d: conditions %s not decided by the algorithms, taken as false" % (
                lay[0], lay[1], ",".join(bits[k] for k in range(len(bits)) if undecided >> k & 1)))
        for a in algs:
            if a.mask(bits) and (a.name in ok) != a.allowed(bits, m):
                notes.append("%s at N=%d ppn=%d: %s although its restrictions %s" % (
                    a.name, lay[0], lay[1], "ran" if a.name in ok else "failed",
                    "hold" if a.allowed(bits, m) else "do not hold"))
    return feas, masks, notes


def plan_sets(algs, bits, masks):
    """[(mask, [layouts], [algorithm names])], set 0 first; sets without layouts or
    without a restricted algorithm are left out."""
    base = [a.name for a in algs if a.mask(bits) == 0]
    out = [(0, sorted(masks), base)]
    for s in sorted(set(masks.values()) - {0}):
        lays = sorted(l for l, m in masks.items() if m == s)
        extra = [a.name for a in algs if a.mask(bits) and a.allowed(bits, s)]
        if lays and extra:
            out.append((s, lays, extra + base))
    return out


def set_name(op, mask):
    return "%s-intra-r%d" % (op, mask)
