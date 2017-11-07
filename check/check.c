#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>

int main(int argc, char **argv)
{
	FILE *fp1, *fp2;
	double *C1, *C2;
	char *file1, *file2;
	struct stat stat1;
	size_t size = 0;
	double max_err;
	int i;

	if (argc != 3) {
		printf("usage: %s <file1> <file2>\n", argv[0]);
		return -1;
	}
	file1 = argv[1];
	file2 = argv[2];
	
	stat(file1, &stat1);
	size = (size_t)stat1.st_size/8;
	size = (size_t)sqrt((double)size);
	printf("size: %lu\n", size);
	
	C1 = (double*)malloc(sizeof(*C1)*size*size);
	C2 = (double*)malloc(sizeof(*C2)*size*size);
	if (!C1 || !C2) {
		printf("C1: %p, C2: %p\n", C1, C2);
		return -2;
	}
	
	fp1 = fopen(file1, "rb");
	fp2 = fopen(file2, "rb");
	if (!fp1 || !fp2) {
		fprintf(stderr, "fp1: %p\nfp2: %p\n", fp1, fp2);
		return -3;
	}
	fread(C1, sizeof(*C1), size*size, fp1);
	fread(C2, sizeof(*C2), size*size, fp2);
	fclose(fp1);
	fclose(fp2);

	max_err = 0.0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (i=0; i<size*size; i++) {
		max_err = fmax(max_err, fabs(C1[i]-C2[i]));
	}
	printf("maximum error: %e\n", max_err);
	
	free(C1);
	free(C2);
	return 0;
}
