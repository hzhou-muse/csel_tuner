"""csel_tuner: measure a cluster's collective algorithms and tune MPICH's
collective selection (see docs/design.md).

    interp   numpy-only interpolation (PCHIP, 2-D linear)
    server   driver for the benchmark server (server/mpisrv)
    layouts  allowed layouts and sizes of a build
    single   builder v1: model of one algorithm, T(N, ppn, m)
    multi    builder v2: all algorithms, sampled for selection
    tree     builder v3: regret decision tree -> MPICH selection JSON
"""
