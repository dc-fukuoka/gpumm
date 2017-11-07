gpumm - matrix-matrix multiplication by using CUDA and cublas.
===
cuda, intel compiler and MKL are needed.
It seems that column major indexing is better for cuda even in C/C++.  
  
The following is a result, GPU used in the test is nvidia K80.  
CPU is Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz, 20 cores.
The size of the matrices is size x size.  
  
* CPU version(OpenMP)
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 cpu/mm 8192
size: 8192
time[s]: 26.832899
~~~
* intel MKL dgemm(thread version)
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 mkl/dgemm 8192
size: 8192
time[s]: 3.540060
~~~
* CUDA with shared memory
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./cuda/cumm 8192 32
# of blocks per grid:   x: 256, y: 256
# of threads per block: x: 32, y: 32
shared memory version
size of shared memory used[B]: 16384
time[s]: 16.133767
~~~
* cublasDgemm() (note: the matrices for cublas are column-major, so transepose is performed.)
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 cublas/cublasmm 8192
size: 8192
time[s]: 2.432141
~~~
* cublasXtDgemm()
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./cublasxt/cublasxtmm 8192
size: 8192
# of devices: 6
time[s]: 3.981712
~~~
* check the results
~~~
$ ./check/check C C.mkl
size: 8192
maximum error: 1.136868e-12

$ ./check/check C C.cuda
size: 8192
maximum error: 0.000000e+00

$ ./check/check C C.cublas
size: 8192
maximum error: 0.000000e+00

$ ./check/check C C.cublasxt
size: 8192
maximum error: 1.278977e-12
~~~
