#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include <mkl.h>
#include <mkl_vsl.h>

#define idx(JMAX, I, J) ((JMAX)*(I)+(J))

static void gen_rand(MKL_INT seed, double min, double max, size_t size, double *arr)
{
	VSLStreamStatePtr stream;
	MKL_INT brng   = VSL_BRNG_MT19937;
	MKL_INT method = VSL_RNG_METHOD_UNIFORM_STD_ACCURATE;
	MKL_INT ret;

	ret = vslNewStream(&stream, brng, seed);                 assert(ret==VSL_ERROR_OK);
	ret = vdRngUniform(method, stream, size, arr, min, max); assert(ret==VSL_ERROR_OK);
	ret = vslDeleteStream(&stream);                          assert(ret==VSL_ERROR_OK);
}

static double calc_trace(size_t size, double *C)
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

static double dclock(void)
{
	struct timespec tp;
	
	clock_gettime(CLOCK_REALTIME, &tp);
	return (double)(1.0*tp.tv_sec+1.0e-9*tp.tv_nsec);
}

int main(int argc, char **argv)
{

	size_t size;
	double *A = NULL, *B = NULL, *C = NULL;
	double t0, time;
	double trace = 0.0;
	int i, j, k;
	FILE *fp;
	char filename[] = "C.mkl";
	
	size = (argc==1) ? 1024 : (size_t)atoi(argv[1]);

	printf("size: %lu\n", size);
	
	A  = (double*)mkl_malloc(sizeof(*A)*size*size, 4); // AVX, 256/8/8
	B  = (double*)mkl_malloc(sizeof(*B)*size*size, 4);
	C  = (double*)mkl_malloc(sizeof(*C)*size*size, 4);
	for (i=0; i<size*size; i++)
		C[i] = 0.0;

	gen_rand(5555, -1.0, 1.0, size*size, A);
	gen_rand(7777, -1.0, 1.0, size*size, B);

	t0 = dclock();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    size, size, size, 1.0, A, size, B, size, 0.0, C, size);
	time = dclock() - t0;

	trace = calc_trace(size, C);

	printf("time[s]: %lf\n", time);
	printf("trace: %.15le\n", trace);
	fp = fopen(filename, "wb");
	fwrite(C, sizeof(*C), size*size, fp);
	fclose(fp);

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
	return 0;
}
