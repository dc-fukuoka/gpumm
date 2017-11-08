#ifndef MYDGEMM_H
#define MYDGEMM_H

__host__ void mydgemm(dim3 &nblocks_per_grid, dim3 &nthreads_per_block, size_t size, double *hA, double *hB, double *hC);
__host__ void calc_trace(size_t size, double *hC, double *trace);

#endif
