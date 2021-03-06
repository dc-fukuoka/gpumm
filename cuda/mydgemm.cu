/*
 * nvidia P100
 * Total Shared Memory per Block: 49152
 * Warp Size:                     32
 * Maximum Threads per Block:     1024
 * Maximum Block Dimensions:      1024, 1024, 64
 * Maximum Grid Dimensions:       2147483647 x 65535 x 65535
 */

#include <stdio.h>
#include "mydgemm.h"
#include "reduce_dsum.h"

#define idx(JMAX, I, J) ((JMAX)*(I)+(J))

__device__ static void clearbuf(size_t *dsize, double *p) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= *dsize || j >= *dsize) return;
    p[idx(*dsize, j, i)] = 0.0;
}

/* with shared memory
 * ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory 
 */
#ifdef _USE_SM
__global__ static void _mydgemm(size_t *dsize, double *dA, double *dB, double *dC) {
    unsigned int k, ii;
    unsigned int bi, bj, ti, tj;
    unsigned int subsize, gsize;
#if 0
    extern __shared__ double dAsub[], dBsub[]; // this does not work!
#endif
    /* http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared */
    extern __shared__ double sm[];
    double *dAsub = &sm[0];
    double *dBsub = &sm[blockDim.x*blockDim.y];
    double *pdAsub, *pdBsub, *pdCsub;

    bi = blockIdx.x;
    bj = blockIdx.y;
    ti = threadIdx.x;
    tj = threadIdx.y;

    if (blockDim.x != blockDim.y || gridDim.x != gridDim.y) return;
    subsize = blockDim.x;
    gsize = *dsize;

    clearbuf(dsize, dC);

    /* shared memory version algorithm
     *
     * ex. blockIdx.x = 1, blockIdx.y = 1
     *     gridDim.x  = 3, gridDim.y  = 3
     * a, b and c are local matrices, a and b are stored in the shared memory
     * step 0
     * A: | | | | B: | |b| | C: | | | |
     *    |a| | |    | | | |    | |c| | c += a*b
     *    | | | |    | | | |    | | | |
     *
     * step 1
     * A: | | | | B: | | | | C: | | | |
     *    | |a| |    | |b| |    | |c| | c += a*b
     *    | | | |    | | | |    | | | |
     *
     * step 2
     * A: | | | | B: | | | | C: | | | |
     *    | | |a|    | | | |    | |c| | c += a*b
     *    | | | |    | |b| |    | | | |
     *
     */

    pdCsub = &dC[subsize*idx(gsize, bj, bi)];
    pdCsub[idx(gsize, tj, ti)] = 0.0;
    for (ii=0; ii<gridDim.x; ii++) {
	pdAsub = &dA[subsize*idx(gsize, bj, ii)];
	pdBsub = &dB[subsize*idx(gsize, ii, bi)];
	/* copy the elements to the shared memory */
	dAsub[idx(subsize, tj, ti)] = pdAsub[idx(gsize, tj, ti)];
	dBsub[idx(subsize, tj, ti)] = pdBsub[idx(gsize, tj, ti)];
	__syncthreads();
	for (k=0; k<subsize; k++)
	    pdCsub[idx(gsize, tj, ti)] += dAsub[idx(subsize, tj, k)]*dBsub[idx(subsize, k, ti)];
	__syncthreads();
    }
}
#else
/* no shared memory */
__global__ static void _mydgemm(size_t *dsize, double *dA, double *dB, double *dC) {
    unsigned int i, j, k;
    i = blockIdx.x*blockDim.x + threadIdx.x;
    j = blockIdx.y*blockDim.y + threadIdx.y;
    clearbuf(dsize, dC);
    if (i >= *dsize || j >= *dsize) return;
    for (k=0; k<*dsize; k++)
        dC[idx(*dsize, j, i)] += dA[idx(*dsize, j, k)]*dB[idx(*dsize, k, i)];
}
#endif /* _USE_SM */

__host__ void mydgemm(dim3 &nblocks_per_grid, dim3 &nthreads_per_block, size_t size, double *hA, double *hB, double *hC) {
    double *dA, *dB, *dC;
    size_t *dsize;
    cudaDeviceProp dp;
    unsigned int smsize, smsize_used;

    cudaGetDeviceProperties(&dp, 0);
    cuda_error_check();
    smsize   = dp.sharedMemPerBlock;

    printf("# of blocks per grid:   x:%4u, y:%4u\n", nblocks_per_grid.x,   nblocks_per_grid.y);
    printf("# of threads per block: x:%4u, y:%4u\n", nthreads_per_block.x, nthreads_per_block.y);
    if (nthreads_per_block.x*nthreads_per_block.y > dp.maxThreadsPerBlock)
	printf("warning: nthreads_per_block.x*nthreads_per_block.y exceeds dp.maxThreadsPerBlock, dp.maxThreadsPerBlock: %u\n", dp.maxThreadsPerBlock);
    
    cudaMalloc((void**)&dA,    sizeof(*dA)*size*size); cuda_error_check();
    cudaMalloc((void**)&dB,    sizeof(*dB)*size*size); cuda_error_check();
    cudaMalloc((void**)&dC,    sizeof(*dC)*size*size); cuda_error_check();
    cudaMalloc((void**)&dsize, sizeof(*dsize));        cuda_error_check();
    
    cudaMemcpy(dA,    hA,    sizeof(*dA)*size*size, cudaMemcpyHostToDevice); cuda_error_check();
    cudaMemcpy(dB,    hB,    sizeof(*dA)*size*size, cudaMemcpyHostToDevice); cuda_error_check();
    cudaMemcpy(dsize, &size, sizeof(*dsize),        cudaMemcpyHostToDevice); cuda_error_check();

#ifdef _USE_SM
    smsize_used = sizeof(*dA)*nthreads_per_block.x*nthreads_per_block.y*2;
    if (smsize_used > smsize)
	printf("warning: used shared memory exceeds the limit, used shared memory size[B]:%u limit[B]: %u\n", smsize_used, smsize);
    printf("shared memory version\nsize of shared memory used[B]: %u\n", smsize_used);
    _mydgemm<<<nblocks_per_grid, nthreads_per_block, smsize_used>>>(dsize, dA, dB, dC);
    cuda_error_check();
#else
    printf("no shared memory version\n");
    _mydgemm<<<nblocks_per_grid, nthreads_per_block>>>(dsize, dA, dB, dC);
    cuda_error_check();
#endif /* _USE_SM */
   
    cudaMemcpy(hC, dC, sizeof(*hC)*size*size, cudaMemcpyDeviceToHost); cuda_error_check();
    cudaFree((void*)dA);    cuda_error_check();
    cudaFree((void*)dB);    cuda_error_check();
    cudaFree((void*)dC);    cuda_error_check();
    cudaFree((void*)dsize); cuda_error_check();
}

__host__ void calc_trace(size_t size, double *hC, double *trace) {
    double *diag, *ddiag;
    double *dtrace;
    dim3 nblocks_per_grid, nthreads_per_block;
    unsigned int warpsize, maxthreads;
    cudaDeviceProp dp;

    cudaGetDeviceProperties(&dp, 0);
    cuda_error_check();
    maxthreads = dp.maxThreadsPerBlock;
    warpsize   = dp.warpSize;
    
    if (size >= maxthreads)
	nthreads_per_block.x = maxthreads;
    else
	nthreads_per_block.x = size;
    nblocks_per_grid.x = size/nthreads_per_block.x;
    if (size%nthreads_per_block.x)
	printf("warning: size%nthreads_per_block.x = %d\n", size%nthreads_per_block.x);

    *trace = 0.0;
    diag = (double*)malloc(sizeof(*diag)*size);
    for (int i=0; i<size; i++)
	diag[i] = hC[idx(size, i, i)];
    cudaMalloc((void**)&ddiag,  sizeof(*ddiag)*size); cuda_error_check();
    cudaMalloc((void**)&dtrace, sizeof(*dtrace));     cuda_error_check();
    cudaMemcpy(ddiag, diag, sizeof(*ddiag)*size, cudaMemcpyHostToDevice); cuda_error_check();
    deviceReduceBlockAtomicKernel<<<nblocks_per_grid, nthreads_per_block>>>(warpsize, ddiag, dtrace, size);
    cuda_error_check();
    cudaMemcpy(trace, dtrace, sizeof(*trace), cudaMemcpyDeviceToHost); cuda_error_check();
    cudaFree(ddiag);  cuda_error_check();
    cudaFree(dtrace); cuda_error_check();
    free(diag);
}
