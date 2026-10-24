/* Device buffers for mpisrv --mem device (see gpu.h). One backend is built:
 * MPISRV_GPU_ZE (Level Zero), MPISRV_GPU_CUDA, MPISRV_GPU_FAKE (malloc, for
 * functional runs without a GPU), or none. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "gpu.h"

/* Whether this MPI handles the runtime's device memory (MPICH's
 * MPIX_GPU_query_support; it is off, e.g., with MPIR_CVAR_ENABLE_GPU=0).
 * gpu_type < 0, or an MPI without the call: assumed yes. */
static int mpi_supports(int gpu_type, char *err, size_t errlen, const char *name)
{
#ifdef MPIX_GPU_SUPPORT_ZE
    int ok = 0;
    if (gpu_type >= 0 && (MPIX_GPU_query_support(gpu_type, &ok) != MPI_SUCCESS || !ok)) {
        snprintf(err, errlen, "MPI reports no %s support (MPICH: built --with-%s, "
                 "MPIR_CVAR_ENABLE_GPU=1?)", name, name);
        return 0;
    }
#else
    (void) gpu_type, (void) err, (void) errlen, (void) name;
#endif
    return 1;
}

/* ---- Level Zero -------------------------------------------------------- */
#if defined(MPISRV_GPU_ZE)
#include <level_zero/ze_api.h>

static ze_context_handle_t ctx;
static ze_device_handle_t dev;
static ze_command_list_handle_t cl;     /* immediate, synchronous: for fills */
static unsigned char uuid[GPU_UUID_LEN];

const char *gpu_runtime(void)
{
    return "ze";
}

#define ZE_TRY(call) do { \
        ze_result_t r_ = (call); \
        if (r_ != ZE_RESULT_SUCCESS) { \
            snprintf(err, errlen, "%s failed: 0x%x", #call, (unsigned) r_); \
            return -1; \
        } \
    } while (0)

int gpu_init(char *err, size_t errlen)
{
    ZE_TRY(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t ndrv = 0;
    ZE_TRY(zeDriverGet(&ndrv, NULL));
    ze_driver_handle_t *drv = calloc(ndrv ? ndrv : 1, sizeof(*drv));
    ZE_TRY(zeDriverGet(&ndrv, drv));
    /* the GPUs this rank sees, over all drivers */
    int ngpu = 0;
    ze_driver_handle_t mine = NULL;
    for (uint32_t i = 0; i < ndrv; i++) {
        uint32_t n = 0;
        ZE_TRY(zeDeviceGet(drv[i], &n, NULL));
        ze_device_handle_t *d = calloc(n ? n : 1, sizeof(*d));
        ZE_TRY(zeDeviceGet(drv[i], &n, d));
        for (uint32_t j = 0; j < n; j++) {
            ze_device_properties_t p = { .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES };
            ZE_TRY(zeDeviceGetProperties(d[j], &p));
            if (p.type != ZE_DEVICE_TYPE_GPU)
                continue;
            if (ngpu++ == 0) {
                mine = drv[i];
                dev = d[j];
                memcpy(uuid, p.uuid.id, GPU_UUID_LEN);
            }
        }
        free(d);
    }
    free(drv);
    if (ngpu != 1) {
        snprintf(err, errlen, "the rank sees %d GPU devices, not 1 (launch through the site's GPU "
                 "wrapper, launch/gpu_bind.sh, which sets ZE_AFFINITY_MASK)", ngpu);
        return -1;
    }
    ze_context_desc_t cd = { .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC };
    ZE_TRY(zeContextCreate(mine, &cd, &ctx));

    /* a compute queue group for the fill command list */
    uint32_t ngrp = 0;
    ZE_TRY(zeDeviceGetCommandQueueGroupProperties(dev, &ngrp, NULL));
    ze_command_queue_group_properties_t *gp = calloc(ngrp ? ngrp : 1, sizeof(*gp));
    for (uint32_t i = 0; i < ngrp; i++)
        gp[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ZE_TRY(zeDeviceGetCommandQueueGroupProperties(dev, &ngrp, gp));
    int ord = -1;
    for (uint32_t i = 0; i < ngrp && ord < 0; i++)
        if (gp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
            ord = (int) i;
    free(gp);
    if (ord < 0) {
        snprintf(err, errlen, "no compute queue group on the device");
        return -1;
    }
    ze_command_queue_desc_t qd = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = (uint32_t) ord,
        .mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    ZE_TRY(zeCommandListCreateImmediate(ctx, dev, &qd, &cl));
    return mpi_supports(MPIX_GPU_SUPPORT_ZE, err, errlen, "ze") ? 0 : -1;
}

void gpu_uuid(unsigned char out[GPU_UUID_LEN])
{
    memcpy(out, uuid, GPU_UUID_LEN);
}

void *gpu_alloc(size_t n)
{
    ze_device_mem_alloc_desc_t d = { .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
    void *p = NULL;
    return zeMemAllocDevice(ctx, &d, n, 4096, dev, &p) == ZE_RESULT_SUCCESS ? p : NULL;
}

void gpu_free(void *p)
{
    if (p)
        zeMemFree(ctx, p);
}

int gpu_fill(void *p, int byte, size_t n)
{
    unsigned char b = (unsigned char) byte;
    return zeCommandListAppendMemoryFill(cl, p, &b, 1, n, NULL, 0, NULL) == ZE_RESULT_SUCCESS ? 0 : -1;
}

void gpu_finalize(void)
{
    zeCommandListDestroy(cl);
    zeContextDestroy(ctx);
}

/* ---- CUDA -------------------------------------------------------------- */
#elif defined(MPISRV_GPU_CUDA)
#include <cuda_runtime.h>

static unsigned char uuid[GPU_UUID_LEN];

const char *gpu_runtime(void)
{
    return "cuda";
}

int gpu_init(char *err, size_t errlen)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        snprintf(err, errlen, "cudaGetDeviceCount failed: %s", cudaGetErrorString(e));
        return -1;
    }
    if (n != 1) {
        snprintf(err, errlen, "the rank sees %d GPU devices, not 1 (launch through the site's GPU "
                 "wrapper, launch/gpu_bind.sh, which sets CUDA_VISIBLE_DEVICES)", n);
        return -1;
    }
    struct cudaDeviceProp p;
    if ((e = cudaSetDevice(0)) != cudaSuccess || (e = cudaGetDeviceProperties(&p, 0)) != cudaSuccess) {
        snprintf(err, errlen, "CUDA device 0: %s", cudaGetErrorString(e));
        return -1;
    }
    memcpy(uuid, p.uuid.bytes, GPU_UUID_LEN);
    return mpi_supports(MPIX_GPU_SUPPORT_CUDA, err, errlen, "cuda") ? 0 : -1;
}

void gpu_uuid(unsigned char out[GPU_UUID_LEN])
{
    memcpy(out, uuid, GPU_UUID_LEN);
}

void *gpu_alloc(size_t n)
{
    void *p = NULL;
    return cudaMalloc(&p, n) == cudaSuccess ? p : NULL;
}

void gpu_free(void *p)
{
    cudaFree(p);
}

int gpu_fill(void *p, int byte, size_t n)
{
    return cudaMemset(p, byte, n) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

void gpu_finalize(void)
{
}

/* ---- fake: host memory standing in for a device ------------------------ */
#elif defined(MPISRV_GPU_FAKE)
const char *gpu_runtime(void)
{
    return "fake";
}

int gpu_init(char *err, size_t errlen)
{
    return mpi_supports(-1, err, errlen, "fake") ? 0 : -1;
}

/* every process its own device */
void gpu_uuid(unsigned char out[GPU_UUID_LEN])
{
    long id[2] = { gethostid(), getpid() };
    memset(out, 0, GPU_UUID_LEN);
    memcpy(out, id, sizeof(id) < GPU_UUID_LEN ? sizeof(id) : GPU_UUID_LEN);
}

void *gpu_alloc(size_t n)
{
    void *p = NULL;
    return posix_memalign(&p, 4096, n) ? NULL : p;
}

void gpu_free(void *p)
{
    free(p);
}

int gpu_fill(void *p, int byte, size_t n)
{
    memset(p, byte, n);
    return 0;
}

void gpu_finalize(void)
{
}

/* ---- none -------------------------------------------------------------- */
#else
const char *gpu_runtime(void)
{
    return "none";
}

int gpu_init(char *err, size_t errlen)
{
    (void) mpi_supports;
    snprintf(err, errlen, "mpisrv was built without GPU support (make GPU=ze|cuda|fake)");
    return -1;
}

void gpu_uuid(unsigned char out[GPU_UUID_LEN])
{
    memset(out, 0, GPU_UUID_LEN);
}

void *gpu_alloc(size_t n)
{
    (void) n;
    return NULL;
}

void gpu_free(void *p)
{
    (void) p;
}

int gpu_fill(void *p, int byte, size_t n)
{
    (void) p, (void) byte, (void) n;
    return -1;
}

void gpu_finalize(void)
{
}
#endif
