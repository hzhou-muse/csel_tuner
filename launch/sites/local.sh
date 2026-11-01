# One machine with MPICH's fake nodes (cliques): each launch of NODES x PPN
# ranks sets MPIR_CVAR_NUM_CLIQUES=NODES, over loopback tcp, one rank per
# core of NODE_CORES (at most as many ranks as cores: an oversubscribed core
# distorts timings). Timings describe this machine, not a real network.
#   SITE=local MPICH_PREFIX=/path/to/mpich bin/collmodel --site local --nodes 4 --ppn 2 ...
NODE_CORES=${NODE_CORES:-0-7}; SOCKETS=${SOCKETS:-1}; NICS_PER_SOCKET=${NICS_PER_SOCKET:-1}
MPICH_PREFIX=${MPICH_PREFIX:?set MPICH_PREFIX to the MPICH install}; MAX_M=${MAX_M:-2097152}
export FI_PROVIDER=${FI_PROVIDER:-tcp}
# MEM=device: GPU_RUNTIME=fake needs mpisrv built with make GPU=fake (no GPU); a real GPU
# has a site file of its own (local-cuda.sh).
GPU_RUNTIME=${GPU_RUNTIME:-fake}; GPU_DEVICES=${GPU_DEVICES:-8}
launch() {
    local nodes=$1 ppn=$2; shift 2
    MPIR_CVAR_NUM_CLIQUES=$nodes mpiexec -n $((nodes * ppn)) -bind-to core $GPU_WRAPPER "$@"
}
