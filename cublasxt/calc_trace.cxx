#include <omp.h>

#define idx(JMAX, I, J) (JMAX)*(I)+(J)

double calc_trace(size_t size, double *C)
{
        int i;
        double trace = 0.0;

#ifdef _OPENMP
#pragma omp parallel for simd private(i) reduction(+:trace)
#endif
        for (i=0; i<size; i++)
                trace += C[idx(size, i, i)];
        return trace;
}
