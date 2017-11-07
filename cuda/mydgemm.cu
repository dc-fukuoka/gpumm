/*
 * nvidia K80
 *
 * Warp Size:                     32
 * Maximum Threads per Block:     1024
 * Maximum Block Dimensions:      1024, 1024, 64
 * Maximum Grid Dimensions:       2147483647 x 65535 x 65535
 */

#include <stdio.h>

#define idx(JMAX, I, J) ((JMAX)*(I)+(J))

__device__ static void clearbuf(size_t *dsize, double *p) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= *dsize || j >= *dsize) return;
    p[idx(*dsize, i, j)] = 0.0;
}

/* with shared memory
 * ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory 
 */
#ifdef _USE_SM
__global__ static void _mydgemm(size_t *dsize, double *dA, double *dB, double *dC) {
    unsigned int k, ii;
    unsigned int bi, bj, ti, tj;
    unsigned int subsize, stride;
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
    stride = *dsize;

    clearbuf(dsize, dC);

    pdCsub = &dC[subsize*idx(stride, bi, bj)];
    pdCsub[idx(stride, ti, tj)] = 0.0;
    for (ii=0; ii<gridDim.x; ii++) {
	pdAsub = &dA[subsize*idx(stride, bi, ii)];
	pdBsub = &dB[subsize*idx(stride, ii, bj)];
	/* copy the elements to the shared memory */
	dAsub[idx(subsize, ti, tj)] = pdAsub[idx(stride, ti, tj)];
	dBsub[idx(subsize, ti, tj)] = pdBsub[idx(stride, ti, tj)];
	__syncthreads();
	for (k=0; k<subsize; k++)
	    pdCsub[idx(stride, tj, ti)] += dAsub[idx(subsize, tj, k)]*dBsub[idx(subsize, k, ti)];
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
    unsigned int warpsize;

    cudaGetDeviceProperties(&dp, 0);
    warpsize = dp.warpSize;
//    printf("warp size: %u\n", warpsize);
    
    cudaMalloc((void**)&dA,    sizeof(*dA)*size*size);
    cudaMalloc((void**)&dB,    sizeof(*dB)*size*size);
    cudaMalloc((void**)&dC,    sizeof(*dC)*size*size);
    cudaMalloc((void**)&dsize, sizeof(*dsize));
    
    cudaMemcpy(dA,    hA,    sizeof(*dA)*size*size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB,    hB,    sizeof(*dA)*size*size, cudaMemcpyHostToDevice);
    cudaMemcpy(dsize, &size, sizeof(*dsize),        cudaMemcpyHostToDevice);

#ifdef _USE_SM
    printf("shared memory version\n");
#else
    printf("no shared memory version\n");
#endif /* _USE_SM */
    _mydgemm<<<nblocks_per_grid, nthreads_per_block, sizeof(*dA)*nthreads_per_block.x*nthreads_per_block.y*2>>>(dsize, dA, dB, dC);
    
    cudaMemcpy(hC, dC, sizeof(*hC)*size*size, cudaMemcpyDeviceToHost);
    cudaFree((void*)dA);
    cudaFree((void*)dB);
    cudaFree((void*)dC);
    cudaFree((void*)dsize);
}
