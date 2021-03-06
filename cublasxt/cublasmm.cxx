#include <stdio.h>
#include <stdlib.h>
#include "cublasdgemm.h"
#include "gen_rand.h"
#include "calc_trace.h"
#include "dclock.h"

int main(int argc, char **argv) {
    size_t size;
    double *A = NULL, *B = NULL, *C = NULL;
    double trace = 0.0;
    double t0, time;
    FILE *fp;
    char filename[] = "C.cublasxt";

    size = (argc==1) ? 1024 : (size_t)atoi(argv[1]);

    printf("size: %lu\n", size);

    A  = (double*)malloc(sizeof(*A)*size*size);
    B  = (double*)malloc(sizeof(*B)*size*size);
    C  = (double*)malloc(sizeof(*C)*size*size);

    gen_rand(5555, -1.0, 1.0, size*size, A);
    gen_rand(7777, -1.0, 1.0, size*size, B);

    t0 = dclock();
    cublasdgemm(size, A, B, C);
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
