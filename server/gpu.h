/* Device buffers for mpisrv --mem device: one GPU (or tile) per rank.
 *
 * The runtime is chosen at build time (make GPU=ze|cuda|fake, see the
 * Makefile); without one, gpu_init fails. The rank's GPU is the only device
 * it sees: a launch wrapper (launch/gpu_bind.sh) restricts each rank to one
 * device, and gpu_init checks that it sees exactly one. */
#ifndef MPISRV_GPU_H
#define MPISRV_GPU_H

#include <stddef.h>

#define GPU_UUID_LEN 16

/* The runtime's name: "ze", "cuda", "fake" or "none". */
const char *gpu_runtime(void);

/* Initialize the runtime on the rank's one device and check that MPI
 * supports its memory. Returns 0, or -1 with a message in err. After
 * MPI_Init. */
int gpu_init(char *err, size_t errlen);

/* An identity of the rank's device, to find ranks of a node sharing one. */
void gpu_uuid(unsigned char uuid[GPU_UUID_LEN]);

/* Device memory: n bytes, or NULL. */
void *gpu_alloc(size_t n);
void gpu_free(void *p);

/* Set n bytes of device memory to `byte`. Returns 0 or -1. */
int gpu_fill(void *p, int byte, size_t n);

void gpu_finalize(void);

#endif
