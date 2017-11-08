#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
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

static void mydgemm(size_t size, double *A, double *B, double *C)
{
	int i, j, k;
	
#ifdef _OPENACC
#pragma acc enter data copyin(A[0:size*size], B[0:size*size], C[0:size*size])
#pragma acc parallel private(i, j, k) present(A[:], B[:], C[:])
#pragma acc loop independent
#endif
	for (i=0; i<size; i++) {
#ifdef _OPENACC
#pragma acc loop independent
#endif
		for (j=0; j<size; j++) {
			for (k=0; k<size; k++) {
				C[idx(size, i, j)] += A[idx(size, i, k)]*B[idx(size, k, j)];
			}
		}
	}
#ifdef _OPENACC
#pragma acc exit data delete(A[0:size*size], B[0:size*size]) copyout(C[0:size*size])
#endif
}

static double calc_trace(size_t size, double *C)
{
        int i;
        double trace = 0.0;

#ifdef _OPENACC
#pragma acc parallel private(i) copyin(C[0:size*size])
#pragma acc loop reduction(+:trace)
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
	char filename[] = "C.acc";
	
	size = (argc==1) ? 1024 : (size_t)atoi(argv[1]);

	printf("size: %lu\n", size);
	
	A  = (double*)malloc(sizeof(*A)*size*size);
	B  = (double*)malloc(sizeof(*B)*size*size);
	C  = (double*)malloc(sizeof(*C)*size*size);
	for (i=0; i<size*size; i++)
		C[i] = 0.0;

	gen_rand(5555, -1.0, 1.0, size*size, A);
	gen_rand(7777, -1.0, 1.0, size*size, B);

	t0 = dclock();
	mydgemm(size, A, B, C);
	time = dclock() - t0;

	trace = calc_trace(size, C);

	printf("time[s]: %lf\n", time);
	printf("trace: %.15le\n", trace);
	fp = fopen(filename, "wb");
	fwrite(C, sizeof(*C), size*size, fp);
	fclose(fp);

	free(A);
	free(B);
	free(C);
	return 0;
}
