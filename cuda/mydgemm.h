#ifndef MYDGEMM_H
#define MYDGEMM_H

void mydgemm(dim3 &nblocks_per_grid, dim3 &nthreads_per_block, size_t size, double *hA, double *hB, double *hC);

#endif
