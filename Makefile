ALL:
	make -C cpu
	make -C mkl
	make -C cuda
	make -C cublas
	make -C cublasxt
	make -C check

clean:
	make -C cpu clean
	make -C mkl clean
	make -C cuda clean
	make -C cublas clean
	make -C cublasxt clean
	make -C check clean
	rm -f *~ C C.*
