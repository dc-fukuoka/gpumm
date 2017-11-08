#include <stdio.h>
#include <stdlib.h>
// reduction ref: https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
// reduction in each warp
#if 0
// allreduce
__inline__ __device__ static double warpAllReduceSum(unsigned int warpsize, double val) {
    for (int mask = (warpsize>>1); mask; mask >>= 1)
	val += __shfl_xor(val, mask);
    return val;
}
#endif
// reduce, lane 0 has the reductionvalue
__inline__ __device__ static double warpReduceSum(unsigned int warpsize, double val) {
    for (int delta = (warpsize>>1); delta; delta >>= 1)
	val += __shfl_down(val, delta);
    return val;
}

__device__ static double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
         		__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ static double blockReduceSum(unsigned int warpsize, double val) {
    static __shared__ double sm[32];
    int lid = threadIdx.x%warpsize; // 0,1,2,...,warpsize-1,0,1,2,...
    int wid = threadIdx.x/warpsize; // 0,0,0,...          0,1,1,1,...
    int nwarps = blockDim.x/warpsize;

    val = warpReduceSum(warpsize, val);

    if (!lid) sm[wid] = val; // write the value by lane 0 in each warp to the shared memory
    __syncthreads();

    val = (threadIdx.x < nwarps) ? sm[lid] : 0.0; // read the value from only existing warps

    if (!wid) val = warpReduceSum(warpsize, val); // do the final reduction in the 1st warp
    return val; // only thread 0 has the reduction value
}

__global__ void deviceReduceBlockAtomicKernel(unsigned int warpsize, double *in, double *out, size_t N) {
    double sum = 0.0;

    if (!threadIdx.x && !blockIdx.x)
	*out = 0.0;
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < N; i += blockDim.x*gridDim.x)
	sum += in[i];
    
    sum = blockReduceSum(warpsize, sum);
    if (!threadIdx.x)
	atomicAdd(out, sum); // reduce over blocks
}
