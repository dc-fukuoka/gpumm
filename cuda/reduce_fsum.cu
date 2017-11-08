#include <stdio.h>
#include <stdlib.h>

// in cuda, local variables in the kernel functions are private: https://stackoverflow.com/questions/16959815/declaring-a-private-thread-specific-variable-in-a-kernel-and-then-returning-th
//
// reduction ref: https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

// reduction in each warp
#if 0
// allreduce, all of lanes have the reduction value
__inline__ __device__ static float warpAllReduceSum(unsigned int warpsize, float val) {
    for (int mask = (warpsize>>1); mask; mask >>= 1)
	val += __shfl_xor(val, mask);
    return val;
}
#endif

// lane 0 owns the reduction value
__inline__ __device__ static float warpReduceSum(unsigned int warpsize, float val) {
    for (int delta = (warpsize>>1); delta; delta >>= 1)
	val += __shfl_down(val, delta);
    return val;
}

__device__ static float blockReduceSum(unsigned int warpsize, float val) {
    static __shared__ float sm[32]; // maximum # of threads per block = 1024, warpsize = 32, so maximum # of warps = 32
    int lid = threadIdx.x%warpsize; // 0,1,2,...,warpsize-1,0,1,2,...
    int wid = threadIdx.x/warpsize; // 0,0,0,...          0,1,1,1,...
    int nwarps = blockDim.x/warpsize;

    val = warpReduceSum(warpsize, val); // lane 0 gets the warp local reduction value

    if (!lid) sm[wid] = val; // write the value by lane 0 in each warp to the shared memory
    __syncthreads();

    val = (threadIdx.x < nwarps) ? sm[lid] : 0.0f; // read the value from only existing warps
    // do the final reduction in the 1st warp
    if (!wid)
	val = warpReduceSum(warpsize, val);
    return val; // only thread 0 has the reduction value
}

__global__ void deviceReduceBlockAtomicKernel(unsigned int warpsize, float *in, float *out, size_t N) {
    float sum = 0.0f;
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < N; i += blockDim.x*gridDim.x)
	sum += in[i];
    
    sum = blockReduceSum(warpsize, sum);
    if (!threadIdx.x) {
	*out = 0.0f;
	atomicAdd(out, sum);
    }
}

int main(int argc, char **argv) {
    float *array;
    float *farray, *fsum;
    size_t size = argv[1] ? (size_t)atoi(argv[1]) : 1024;
    dim3 nblocks, nthreads;
    unsigned int warpsize;
    cudaDeviceProp dp;
    float sum;

    cudaGetDeviceProperties(&dp, 0);
    warpsize = dp.warpSize;

    printf("warpsize: %u\n", warpsize);
    
    array = (float*)malloc(sizeof(*array)*size);
    for (int i=0; i<size; i++)
	array[i] = (float)i;
    cudaMalloc((void**)&farray, sizeof(*farray)*size);
    cudaMalloc((void**)&fsum,   sizeof(*fsum));
    cudaMemcpy(farray, array,   sizeof(*farray)*size, cudaMemcpyHostToDevice);
    nthreads.x = 1024;
    nblocks.x  = size/nthreads.x;
    printf("nblocks.x: %u, nthreads.x: %u\n", nblocks.x, nthreads.x);
    deviceReduceBlockAtomicKernel<<<nblocks, nthreads>>>(warpsize, farray, fsum, size);
    cudaDeviceSynchronize();
    cudaMemcpy(&sum, fsum, sizeof(sum), cudaMemcpyDeviceToHost);

    printf("sum: %f\n", sum);
    printf("ans: %f\n", (float)(size*(size-1)/2.0f));
    
    cudaFree(farray);
    cudaFree(fsum);
    free(array);
    return 0;
}
