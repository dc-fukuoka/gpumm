gpumm - matrix-matrix multiplication by using CUDA, cublas, cublasxt and OpenACC.
===
cuda, intel compiler and MKL are needed.  
for openacc, PGI compiler is needed.  
It seems that column major indexing is better for cublas/cuda even in C/C++.  
  
The following is a result, GPU used in the test is nvidia P100.  
CPU is Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz.  
The size of the matrices is size x size.  
  
* CPU version(OpenMP)
~~~
$ KMP_AFFINITY=compact ./cpu/mm 8192
size: 8192
time[s]: 16.613501
trace: -4.324045225743852e+03
~~~
* CPU version(OpenMP, fortran)
~~~
$ KMP_AFFINITY=compact ./cpuf/mmf 8192
size: 8192
time[s]:  3.580926
trace: -4.324045225743852E+03
~~~
* intel MKL dgemm(thread version)
~~~
$ KMP_AFFINITY=compact ./mkl/dgemm 8192
size: 8192
time[s]: 3.223232
trace: -4.324045225743848e+03
~~~
* CUDA without shared memory(remove -D_USE_SM from the Makefile)
~~~
$ ./cuda/cumm 8192 32
size: 8192
# of blocks per grid:   x: 256, y: 256
# of threads per block: x:  32, y:  32
no shared memory version
time[s]: 7.732970
trace: -4.324045225743850e+03
~~~
* CUDA with shared memory
~~~
$ ./cuda/cumm 8192 32
size: 8192
# of blocks per grid:   x: 256, y: 256
# of threads per block: x:  32, y:  32
shared memory version
size of shared memory used[B]: 16384
time[s]: 1.974779
trace: -4.324045225743848e+03
~~~
* openacc
~~~
$ ./openacc/accmm 8192
size: 8192
time[s]: 8.721443
trace: -4.324045225743852e+03
~~~
* cublasDgemm() (note: the matrices for cublas are column-major, so transepose is performed.)
~~~
$ ./cublas/cublasmm 8192
size: 8192
time[s]: 1.583693
trace: -4.324045225743852e+03
~~~
* cublasXtDgemm()
~~~
$ ./cublasxt/cublasxtmm 8192
size: 8192
# of devices: 4
time[s]: 4.540568
trace: -4.324045225743849e+03
~~~
* check the results
~~~
$ ./check/check C C_f
size: 8192
maximum error: 0.000000e+00

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
