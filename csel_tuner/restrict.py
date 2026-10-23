"""Restriction sets: which algorithms run on which layouts, and the model sets built from that.

    feas, masks, notes = precheck(probe, layouts, algs, bits)
    sets = plan_sets(algs, bits, masks)    # [(mask, layouts, [alg names])], mask 0 first

bits: the catalog's restrictive conditions (algos.Catalog.restriction_bits(),
one numbering for every collective; restrictions are positive conditions).
probe(n, ppn, alg_name) -> (ok, msg): one call of the algorithm forced on the
layout (live: a probe=1 request with MPIR_CVAR_COLLECTIVE_FALLBACK=error, so
a failed restriction is an error, not a silent fallback). A layout's mask is
the set of conditions true there: the bits of every algorithm that ran; an
algorithm that failed with all but one of its bits known true makes that
one false. Bits the algorithms do not decide are taken as false and
reported.

Set 0 holds the unrestricted algorithms and is built on every layout: it is
the fallback whenever no other set applies at run time, also where one
applied during tuning (e.g. release_gather is off under
MPI_THREAD_MULTIPLE). Set S != 0 is built on the layouts whose mask is
exactly S, over the restricted algorithms allowed in S (mask a subset of S)
plus the unrestricted ones (for the "call=r0" choice). See docs/design.md,
section 11.
"""


def infer_mask(ok, algs, bits):
    """(mask of true conditions, mask of bits left undecided) from the algorithms that ran (ok)."""
    used = true = false = 0
    for a in algs:
        used |= a.mask(bits)
        if a.name in ok:
            true |= a.mask(bits)
    for a in algs:
        rest = a.mask(bits) & ~true
        if a.name not in ok and bin(rest).count("1") == 1:
            false |= rest           # it failed, and every other condition it needs holds
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
    """The set's CSEL subtree name: the mask of true conditions in hex, e.g. bcast-intra-800."""
    return "%s-intra-%x" % (op, mask)


def stage2_sets(algs2, bits, masks1, feas2):
    """Stage 2: the compositional algorithms (algs2, those with an inner key),
    prechecked (feas2) in a launch with the stage-1 tuning file loaded.
    Returns (sets, masks2, notes): each layout's mask is its stage-1 mask
    plus the bits the stage-2 algorithms decide; a set is
    (mask, layouts, [algorithm names], fallback): its layouts' masks changed
    from the stage-1 mask `fallback` to `mask`, its candidates are the stage-2
    algorithms allowed there and "auto" (MPICH with the stage-1 file: the
    stage-1 choice, to be replaced by call=<fallback set> in the tree)."""
    masks2, notes, groups = {}, [], {}
    for lay, m1 in masks1.items():
        m2 = m1 | infer_mask(feas2.get(lay, set()), algs2, bits)[0]
        masks2[lay] = m2
        if m2 != m1:
            groups.setdefault((m2, m1), []).append(lay)
    sets = []
    for (m2, m1), lays in sorted(groups.items()):
        if any(k[0] == m2 and k[1] != m1 for k in groups):
            notes.append("mask %x comes from different stage-1 masks; set built per stage-1 mask %x" % (m2, m1))
        names = [a.name for a in algs2 if a.allowed(bits, m2)]
        if names:
            sets.append((m2, sorted(lays), names + ["auto"], m1))
    return sets, masks2, notes


def role_layout(lay, comm):
    """The layout of an inner call on `comm` from a call at `lay`."""
    n, ppn = lay
    return {"node_comm": (1, ppn), "node_roots_comm": (n, 1)}[comm]


def fixed_point(algs2, feas2, op):
    """Two stages suffice when no inner call of a stage-2 algorithm can run a
    stage-2 algorithm of its own op: (problems, notes). Checked on the
    precheck: a stage-2 algorithm that ran at an inner call's layout."""
    names = {a.name for a in algs2}
    problems, notes = [], []
    for a in algs2:
        for section, comm in a.inner:
            if section != op + "-intra":
                notes.append("%s: inner %s@%s is another op; not checked here" % (a.name, section, comm))
                continue
            for lay, ok in sorted(feas2.items()):
                if a.name not in ok:
                    continue
                role = role_layout(lay, comm)
                hit = sorted(names & feas2.get(role, set()))
                if hit:
                    problems.append("%s at N=%d ppn=%d: its %s call (N=%d ppn=%d) can run %s" % (
                        a.name, lay[0], lay[1], comm, role[0], role[1], ",".join(hit)))
    return problems, notes
