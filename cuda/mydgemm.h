#ifndef MYDGEMM_H
#define MYDGEMM_H

#define cuda_error_check()						\
	do {								\
		cudaError_t err = cudaGetLastError();			\
		if (err != cudaSuccess) {				\
			fprintf(stderr, "%s:%d: in %s(): %s\n", __FILE__, __LINE__, __func__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE);				\
		}							\
	} while(0)
	  
__host__ void mydgemm(dim3 &nblocks_per_grid, dim3 &nthreads_per_block, size_t size, double *hA, double *hB, double *hC);
__host__ void calc_trace(size_t size, double *hC, double *trace);

#endif
