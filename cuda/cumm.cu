#include <stdio.h>
#include <stdlib.h>
#include "mydgemm.h"
#include "gen_rand.h"
#include "dclock.h"

int main(int argc, char **argv) {
    size_t size, nblocks;
    double *A = NULL, *B = NULL, *C = NULL;
    double trace = 0.0;
    double t0, time;
    FILE *fp;
    char filename[] = "C.cuda";

    size    = (argc==1)            ? 1024                  : (size_t)atoi(argv[1]);
    nblocks = (argv[1] && argv[2]) ? (size_t)atoi(argv[2]) : 16;

    printf("size: %lu\n", size);
    
    if (size%nblocks)
	fprintf(stderr, "warning: mod(size, nblocks)=%d\n", size%nblocks);

    dim3 nthreads_per_block(nblocks, nblocks);
    dim3 nblocks_per_grid(size/nthreads_per_block.x, size/nthreads_per_block.y);
	
    A  = (double*)malloc(sizeof(*A)*size*size);
    B  = (double*)malloc(sizeof(*B)*size*size);
    C  = (double*)malloc(sizeof(*C)*size*size);
	
    gen_rand(5555, -1.0, 1.0, size*size, A);
    gen_rand(7777, -1.0, 1.0, size*size, B);

    t0 = dclock();
    mydgemm(nblocks_per_grid, nthreads_per_block, size, A, B, C);
    time = dclock() - t0;

    calc_trace(size, C, &trace);

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
