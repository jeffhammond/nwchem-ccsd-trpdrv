OPTFLAGS	 = -O2 -g
CUBLASFLAGS	 = -I/usr/local/cuda-10.0/targets/aarch64-linux/include \
		   -L/usr/local/cuda-10.0/targets/aarch64-linux/lib -lcublas

CC		 = /samsung/nvidia/hpc_sdk/Linux_aarch64/21.2/cuda/11.2/bin/nvcc
CFLAGS		 = $(OPTFLAGS) $(CUDAFLAGS) $(CUBLASFLAGS)

all: test_cuda

ccsd_trpdrv_cuda.o: ccsd_trpdrv_cuda.cu
	$(CC) $(CFLAGS) -c $< -o $@

test_cuda: test.c ccsd_trpdrv_cuda.o gettime.o
	$(CC) $(CFLAGS) $^ -o $@

gettime.o: gettime.c
	gcc -Wall -Os -c $< -o $@

clean:
	rm -f test_cuda
	rm -f *.o
	rm -f *.s

realclean: clean

