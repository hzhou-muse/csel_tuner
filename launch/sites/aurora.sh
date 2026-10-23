# Aurora (ALCF) site settings, sourced by config.sh. Everything specific to
# the system lives here; the tool's code has no site constants.

# Node topology. A node has 2 sockets x 52 cores and 8 Slingshot NICs, 4
# close to each socket. MPICH (ch4:ofi, MAX_NICS=1) gives a rank the close
# NIC local_rank % NICS_PER_SOCKET of its socket. mpisrv checks these values
# against the core binding and MPICH's num_close_nics at startup.
NODE_CORES=${NODE_CORES:-1-48,53-100}   # cores a launch binds to, socket 0 first; cores 0 and 52 left to the OS
SOCKETS=${SOCKETS:-2}
NICS_PER_SOCKET=${NICS_PER_SOCKET:-4}

# MPICH under test: ch4:ofi, --with-pm=no (PALS mpiexec launches it via
# PMIx). Per-request algorithm switching needs MPICH main with commit
# de0b8e6921 (and, for enum names via MPI_T, f95b6ec93d); then set
# MPISRV_CVAR_SWITCH=1 below or in the environment.
MPICH_PREFIX=${MPICH_PREFIX:-$HOME/MPI}
MAX_M=${MAX_M:-67108864}                # 64 MB
# 1 only for an MPICH that applies algorithm CVAR writes at run time (main
# since commit de0b8e6921, "coll: read algorithm CVARs through a pointer
# table"); mpisrv then accepts per-request cvar.* tokens.
MPISRV_CVAR_SWITCH=${MPISRV_CVAR_SWITCH:-0}

# Environment for every launch (recorded by mpisrv in its info record).
export MPIR_CVAR_ENABLE_GPU=${MPIR_CVAR_ENABLE_GPU:-0}   # host buffers only

# The PBS project and queue go in the job script (docs/usage.md, §6.1).
