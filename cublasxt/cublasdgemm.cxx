#include <cublasXt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#define cublas_assert(expr, fmt, ...)					\
    do {								\
	if (!(expr)) {							\
	    fprintf(stderr, "%s: line %d: in %s(): " fmt, __FILE__, __LINE__, __func__, ## __VA_ARGS__); \
	    fprintf(stderr, "\n");					\
	    exit(EXIT_FAILURE);						\
	}								\
    } while(0)

void cublasdgemm(size_t size, double *hA, double *hB, double *hC) {
    double *dA, *dB, *dC, *dCt;
    int lda, ldb, ldc;
    double alpha = 1.0, beta = 0.0;
    cublasXtHandle_t handle_xt;
    cublasHandle_t handle;
    cudaError_t err;
    cublasStatus_t stat;
    int ndev, *devs;
    int i, j, k;

    lda = ldb = ldc = size;

    cudaGetDeviceCount(&ndev);
    printf("# of devices: %d\n", ndev);
    devs = (int*)malloc(ndev*sizeof(*devs));
    for (i=0; i<ndev; i++)
	devs[i] = i;
    
    stat = cublasXtCreate(&handle_xt);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasXtCreate");
    stat = cublasXtDeviceSelect(handle_xt, ndev, devs);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasXtDeviceSelect");
    
    cudaMalloc((void**)&dA,  size*size*sizeof(*dA));
    cudaMalloc((void**)&dB,  size*size*sizeof(*dB));
    cudaMalloc((void**)&dC,  size*size*sizeof(*dC));
    cudaMalloc((void**)&dCt, size*size*sizeof(*dCt));
    
    stat = cublasSetMatrix(size, size, sizeof(*hA), hA, lda, dA, size);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasSetMatrix hA");
    stat = cublasSetMatrix(size, size, sizeof(*hB), hB, ldb, dB, size);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasSetMatrix hB");
    stat = cublasSetMatrix(size, size, sizeof(*hC), hC, ldc, dC, size);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasSetMatrix hC");

    // note: cublas uses column major(fortran order) matrices
    // http://docs.nvidia.com/cuda/cublas/index.html
    // so trancepose row major matrices
    stat = cublasXtDgemm(handle_xt, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasXtDgemm");
    cudaDeviceSynchronize();
    stat = cublasXtDestroy(handle_xt);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasXtDestroy");
    
    // tracepose C since the matrix is column major
    // there is no cublasXtDgeam()...
    stat = cublasCreate(&handle);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasCreate");
    stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, size, size, &alpha, dC, ldc, &beta, dB, ldb, dCt, ldc);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasDgeam");
    stat = cublasDestroy(handle);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasDestroy");
    
    stat = cublasGetMatrix(size, size, sizeof(*hC), dCt, size, hC, ldc);
    cublas_assert(stat == CUBLAS_STATUS_SUCCESS, "cublasGetMatrix");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dCt);
    
    free(devs);
}
