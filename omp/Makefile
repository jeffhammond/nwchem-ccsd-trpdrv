VARIANT		:= 0

OMPFLAGS	 = -qopenmp
OPTFLAGS	 = -O3 -xCORE-AVX2
MKLFLAGS	 = -mkl -DMKL #-DSKIP_DGEMM

CC		 = icc -std=c99
CFLAGS		 = $(OMPFLAGS) $(OPTFLAGS) $(MKLFLAGS)

FC		 = ifort
FFLAGS		 = $(OMPFLAGS) $(OPTFLAGS) $(MKLFLAGS)

all: test_c99 test_f77

ccsd_tengy_c99.o: ccsd_tengy_c99.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_tengy_f77.o: ccsd_tengy_f77.f
	$(FC) $(FFLAGS) -fixed -c $< -o $@

ccsd_trpdrv_omp_cbody.o: ccsd_trpdrv_omp_cbody.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_trpdrv_omp_f77.o: ccsd_trpdrv_omp_f77.f
	$(FC) $(FFLAGS) -fpp -fixed -c $< -o $@

ccsd_trpdrv_omp_f90.o: ccsd_trpdrv_omp_f90.f
	$(FC) $(FFLAGS) -fpp -free -c $< -o $@

test_c99: test.c ccsd_trpdrv_omp_cbody.o ccsd_tengy_c99.o
	$(CC) $(CFLAGS) $^ -o $@

test_f77: test.c ccsd_trpdrv_omp_f77.o ccsd_tengy_f77.o
	$(CC) $(CFLAGS) -DUSE_FORTRAN $^ -o $@

clean:
	rm -f test
	rm -f test_c99
	rm -f test_f77
	rm -f *.o
	rm -f *.s

realclean: clean

