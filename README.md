gpumm - matrix-matrix multiplication by using CUDA and cublas.
===
cuda, intel compiler and MKL are needed.  
  
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
$ KMP_AFFINITY=compact srun -pGPU -n1 cuda/cumm 8192 256
size: 8192
nblocks per grid: 32, nthreads per block: 256
shared memory version
time[s]: 0.775476
~~~
* cublasDgemm() (note: the matrices for cublas are column-major, so transepose is performed.)
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 cublas/cublasmm 8192
size: 8192
time[s]: 2.432141
~~~
* check the results
~~~
$ ./check/check C C.mkl
size: 8192
maximum error: 1.278977e-12

$ ./check/check C C.cuda
size: 8192
maximum error: 0.000000e+00

$ ./check/check C C.cublas
size: 8192
maximum error: 0.000000e+00
~~~~
