# Launch settings for the tools (bin/collmodel --site, groundtruth/run_set.sh,
# groundtruth/drive_check.py). Site facts (node topology, MPICH install,
# message size limit, launcher) come from sites/$SITE.sh (default aurora);
# this file only holds what is the same on every system.
here_cfg=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SITE=${SITE:-aurora}
source "$here_cfg/sites/$SITE.sh"

export MPICH_PREFIX
export PATH=$MPICH_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$MPICH_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MPISRV_NIC_MOD=$NICS_PER_SOCKET MPISRV_SOCKETS=$SOCKETS MPISRV_CVAR_SWITCH=${MPISRV_CVAR_SWITCH:-0}

MPISRV=${MPISRV:-$here_cfg/../server/mpisrv}

# Seconds kept free at the end of each launch / of the job.
LAUNCH_MARGIN=${LAUNCH_MARGIN:-120}
MIN_LAUNCH=${MIN_LAUNCH:-240}

# launch NODES PPN CMD...: start CMD with NODES*PPN ranks, PPN per node; rank
# i of a node is bound to core floor(i * n / PPN) of NODE_CORES (n cores), the
# same positions mpisrv's place=spread selects in a full launch. A site file
# may define its own launch().
if ! declare -F launch > /dev/null; then
launch() {
    local nodes=$1 ppn=$2; shift 2
    local all=() part lo hi i
    for part in ${NODE_CORES//,/ }; do
        lo=${part%-*}; hi=${part#*-}
        for ((i = lo; i <= hi; i++)); do all+=("$i"); done
    done
    local list=""
    for ((i = 0; i < ppn; i++)); do list+="${list:+:}${all[i * ${#all[@]} / ppn]}"; done
    mpiexec -n $((nodes * ppn)) --ppn "$ppn" --cpu-bind "list:$list" "$@"
}
fi
