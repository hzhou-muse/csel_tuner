# Functional tests on one machine (timings are not representative): MPICH
# splits the ranks into 2 fake nodes (cliques), over loopback tcp. Cores 0-7.
#   SITE=local NICS_PER_SOCKET=1 MPICH_PREFIX=/path/to/mpich bin/collmodel --site local ...
NODE_CORES=0-7; SOCKETS=${SOCKETS:-1}; NICS_PER_SOCKET=${NICS_PER_SOCKET:-4}
MPICH_PREFIX=${MPICH_PREFIX:?set MPICH_PREFIX to the MPICH install}; MAX_M=2097152
export MPIR_CVAR_NUM_CLIQUES=2 FI_PROVIDER=tcp
launch() { local nodes=$1 ppn=$2; shift 2; mpiexec -n $((nodes * ppn)) -bind-to core "$@"; }
