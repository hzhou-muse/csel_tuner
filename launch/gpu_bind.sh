#!/bin/bash
# GPU binding for mpisrv --mem device: run "$@" with only the local rank's GPU
# (or tile) visible, one per rank, like ALCF's gpu_tile_compact.sh. Core
# binding stays with the launcher (config.sh's launch). config.sh puts this
# in front of mpisrv when MEM=device.
#
# A rank it cannot bind still runs "$@", with MPISRV_BIND_ERROR set: mpisrv
# then fails on every rank at once (a rank that exits alone would leave the
# others waiting in MPI_Init).
#
# GPU_RUNTIME (site file): ze sets ZE_AFFINITY_MASK, to the device index
# under ZE_FLAT_DEVICE_HIERARCHY=FLAT (each tile a device; the default
# assumed here) or to GPU.TILE under COMPOSITE (GPU_TILES_PER_GPU tiles per
# GPU); cuda sets CUDA_VISIBLE_DEVICES; fake sets nothing. GPU_DEVICES: the
# devices of a node (local ranks beyond it are an error).
r=${PALS_LOCAL_RANKID:-${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-}}}}
if [ -z "$r" ]; then
    export MPISRV_BIND_ERROR="gpu_bind.sh: no local rank in the environment (PALS, Hydra, Open MPI or Slurm)"
    exec "$@"
fi
if [ -n "${GPU_DEVICES:-}" ] && [ "$r" -ge "$GPU_DEVICES" ]; then
    export MPISRV_BIND_ERROR="gpu_bind.sh: local rank $r, but a node has $GPU_DEVICES GPU devices (one rank each)"
    exec "$@"
fi
case ${GPU_RUNTIME:-ze} in
ze)
    if [ "${ZE_FLAT_DEVICE_HIERARCHY:-FLAT}" = COMPOSITE ]; then
        t=${GPU_TILES_PER_GPU:-1}
        export ZE_AFFINITY_MASK=$((r / t)).$((r % t))
    else
        export ZE_AFFINITY_MASK=$r
    fi ;;
cuda) export CUDA_VISIBLE_DEVICES=$r ;;
fake) ;;
*) export MPISRV_BIND_ERROR="gpu_bind.sh: unknown GPU_RUNTIME=$GPU_RUNTIME" ;;
esac
exec "$@"
