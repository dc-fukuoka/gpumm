#ifndef REDUCE_DSUM_H
#define REDUCE_DSUM_H

__global__ void deviceReduceBlockAtomicKernel(unsigned int warpsize, double *in, double *out, size_t N);

#endif
