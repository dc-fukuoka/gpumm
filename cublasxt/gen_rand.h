#include <mkl_vsl.h>

#ifndef GEN_RAND_H
#define GEN_RAND_H

void gen_rand(MKL_INT seed, double min, double max, size_t size, double *arr);

#endif
