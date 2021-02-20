OPTFLAGS	 = -O2
MKLFLAGS	 = -lcublas

CC		 = nvcc
CFLAGS		 = $(OPTFLAGS) $(CUDAFLAGS) $(CUBLASFLAGS)

FC		 = ifort
FFLAGS		 = $(OMPFLAGS) $(OPTFLAGS) $(MKLFLAGS)

all: test_cuda

ccsd_tengy_cuda.o: ccsd_tengy_cuda.cu
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_trpdrv_cuda.o: ccsd_trpdrv_cuda.cu
	$(CC) $(CFLAGS) -c $< -o $@

test_cuda: test.c ccsd_trpdrv_cuda.o ccsd_tengy_cuda.o
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f test_cuda
	rm -f *.o
	rm -f *.s

realclean: clean

