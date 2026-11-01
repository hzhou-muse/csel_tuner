# Two GPU nodes of a CI cluster, launched with Hydra over ssh from the first host
# (JGPU_HOSTS, comma-separated, launch host first; /home shared). Per node 2 sockets
# x 16 cores (cores 0-15 and 16-31; hardware threads 32-63 unused), one 10 GbE NIC
# (on socket 0), 2 CUDA GPUs (GPU0 on socket 0, GPU1 on socket 1), no GPUDirect RDMA.
#   SITE=jenkins-gpu MPICH_PREFIX=... bin/collmodel --site jenkins-gpu --nodes 2 --ppn 32 ...
NODE_CORES=${NODE_CORES:-0-31}
SOCKETS=${SOCKETS:-2}
NICS_PER_SOCKET=${NICS_PER_SOCKET:-1}
JGPU_HOSTS=${JGPU_HOSTS:?set JGPU_HOSTS to the nodes, comma-separated, launch host first}
MPICH_PREFIX=${MPICH_PREFIX:?set MPICH_PREFIX to the MPICH install}
MAX_M=${MAX_M:-16777216}                # 16 MB
MPISRV_CVAR_SWITCH=${MPISRV_CVAR_SWITCH:-1}
# libfabric's default here: psm3 over TCP on eno1; fixed for every run
export FI_PROVIDER=${FI_PROVIDER:-psm3}
export MPIR_CVAR_ENABLE_GPU=${MPIR_CVAR_ENABLE_GPU:-0}   # host buffers (config.sh sets 1 for MEM=device)
# Device buffers: each rank uses the GPU of its socket (gpu_bind.sh GPU_BIND=socket),
# so the ranks of a socket share one GPU.
GPU_RUNTIME=${GPU_RUNTIME:-cuda}
GPU_DEVICES=${GPU_DEVICES:-2}
GPU_BIND=${GPU_BIND:-socket}
# launch() also runs in run_set.sh's subshell: export what it reads
export GPU_BIND GPU_RUNTIME GPU_DEVICES JGPU_HOSTS NODE_CORES
# launch NODES PPN CMD...: rank i of a node on physical core floor(i * 32 / PPN), so
# the ranks spread over both sockets in order (the first half on socket 0).
launch() {
    local nodes=$1 ppn=$2; shift 2
    local n i list="" hosts
    n=$(( ${NODE_CORES#*-} - ${NODE_CORES%-*} + 1 ))
    for ((i = 0; i < ppn; i++)); do list+="${list:+,}$(( ${NODE_CORES%-*} + i * n / ppn ))"; done
    hosts=$(echo "$JGPU_HOSTS" | cut -d, -f1-"$nodes")
    # -genvall: every rank, on the remote nodes too, gets the whole environment (the
    # collmodel/config.sh settings such as MPIR_CVAR_COLLECTIVE_FALLBACK, MPIR_CVAR_ENABLE_GPU,
    # FI_PROVIDER, MPIR_CVAR_COLL_SELECTION_JSON_FILE). -genvlist would pass only the listed
    # variables to the ranks Hydra starts over ssh, so the nodes would run differently.
    GPU_PPN=$ppn mpiexec -hosts "$hosts" -n $((nodes * ppn)) -ppn "$ppn" -bind-to "user:$list" \
        -genvall $GPU_WRAPPER "$@"
}
