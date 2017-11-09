gpumm - matrix-matrix multiplication by using CUDA, cublas, cublasxt and OpenACC.
===
cuda, intel compiler and MKL are needed.  
for openacc, PGI compiler is needed.  
It seems that column major indexing is better for cuda even in C/C++.  
  
The following is a result, GPU used in the test is nvidia K80.  
CPU is Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz, 20 cores.  
The size of the matrices is size x size.  
  
* CPU version(OpenMP)
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./cpu/mm 8192
size: 8192
time[s]: 24.373878
trace: -4.324045225743850e+03
~~~
* intel MKL dgemm(thread version)
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./mkl/dgemm 8192
size: 8192
time[s]: 3.250625
trace: -4.324045225743848e+03
~~~
* CUDA without shared memory
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./cuda/cumm 8192 16
size: 8192
# of blocks per grid:   x: 512, y: 512
# of threads per block: x: 16, y: 16
no shared memory version
time[s]: 40.464609
trace: -4.324045225743850e+03
~~~
* CUDA with shared memory
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./cuda/cumm 8192 16
size: 8192
# of blocks per grid:   x: 512, y: 512
# of threads per block: x: 16, y: 16
shared memory version
size of shared memory used[B]: 4096
time[s]: 7.600645
trace: -4.324045225743850e+03
~~~
* openacc
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./openacc/accmm 8192
size: 8192
time[s]: 20.087699
trace: -4.324045225743852e+03
~~~
* cublasDgemm() (note: the matrices for cublas are column-major, so transepose is performed.)
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./cublas/cublasmm 8192
size: 8192
time[s]: 2.404458
trace: -4.324045225743852e+03
~~~
* cublasXtDgemm()
~~~
$ KMP_AFFINITY=compact srun -pGPU -n1 ./cublasxt/cublasxtmm 8192
size: 8192
# of devices: 6
time[s]: 3.995293
trace: -4.324045225743849e+03
~~~
* check the results
~~~
$ ./check/check C C.mkl
size: 8192
maximum error: 1.364242e-12

$ ./check/check C C.cuda
size: 8192
maximum error: 0.000000e+00

$ ./check/check C C.acc
size: 8192
maximum error: 0.000000e+00

$ ./check/check C C.cublas
size: 8192
maximum error: 0.000000e+00

$ ./check/check C C.cublasxt
size: 8192
maximum error: 1.278977e-12
~~~
