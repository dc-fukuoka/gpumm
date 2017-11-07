#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <mkl.h>
#include <mkl_vsl.h>

#define idx(JMAX, I, J) (JMAX)*(I)+(J)

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
#if 0
static void mydgemm(size_t size, double *A, double *B, double *C)
{
	int i, j, k;
	
#ifdef _OPENMP
#pragma omp parallel for simd private(i, j, k)
#endif
	for (i=0; i<size; i++) {
		for (k=0; k<size; k++) {
			for (j=0; j<size; j++) {
				C[idx(size, i, j)] += A[idx(size, i, k)]*B[idx(size, k, j)];
			}
		}
	}
}
#endif
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

	printf("time[s]: %lf\n", time);
	fp = fopen(filename, "wb");
	fwrite(C, sizeof(*C), size*size, fp);
	fclose(fp);

	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
	return 0;
}
