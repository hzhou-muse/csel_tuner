# One machine with one CUDA GPU that every rank shares, and MPICH's fake
# nodes as in local.sh. mpisrv uses device 0 and Hydra restricts no GPUs, so
# without csel_tuner's binding wrapper (gpu_bind.sh gives local rank r GPU r)
# all ranks use the one GPU: device-local transfers, and the ranks contend
# for it. GPU_WRAPPER=env replaces the wrapper (an empty value would fall
# back to it). mpisrv: make -B -C server GPU=cuda CUDA_HOME=... MPICC=...
#   SITE=local-cuda MPICH_PREFIX=/path/to/mpich-with-cuda bin/collmodel --site local-cuda --launch gpu ...
source "$(dirname "${BASH_SOURCE[0]}")/local.sh"
GPU_RUNTIME=cuda; GPU_DEVICES=1; GPU_WRAPPER=env
