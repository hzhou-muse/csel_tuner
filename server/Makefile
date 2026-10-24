# Build the benchmark server with the MPICH under test:
#   make MPICC=$MPICH_PREFIX/bin/mpicc [GPU=ze|cuda|fake]
# GPU selects the device-buffer backend for mpisrv --mem device (gpu.c):
#   none  host buffers only (default)
#   ze    Level Zero: needs level_zero/ze_api.h and libze_loader (ZE_PREFIX if
#         not in the system paths)
#   cuda  CUDA runtime: needs cuda_runtime.h and libcudart (CUDA_HOME)
#   fake  host memory standing in for a device: functional runs without a GPU
MPICC ?= mpicc
CFLAGS ?= -O2 -Wall
GPU ?= none

ifeq ($(GPU),ze)
GPU_CFLAGS = -DMPISRV_GPU_ZE $(if $(ZE_PREFIX),-I$(ZE_PREFIX)/include)
GPU_LIBS = $(if $(ZE_PREFIX),-L$(ZE_PREFIX)/lib64 -L$(ZE_PREFIX)/lib) -lze_loader
else ifeq ($(GPU),cuda)
CUDA_HOME ?= /usr/local/cuda
GPU_CFLAGS = -DMPISRV_GPU_CUDA -I$(CUDA_HOME)/include
GPU_LIBS = -L$(CUDA_HOME)/lib64 -lcudart
else ifeq ($(GPU),fake)
GPU_CFLAGS = -DMPISRV_GPU_FAKE
else ifneq ($(GPU),none)
$(error GPU must be none, ze, cuda or fake)
endif

mpisrv: mpisrv.c gpu.c gpu.h
	$(MPICC) $(CFLAGS) $(GPU_CFLAGS) -o $@ mpisrv.c gpu.c $(GPU_LIBS)

clean:
	rm -f mpisrv
