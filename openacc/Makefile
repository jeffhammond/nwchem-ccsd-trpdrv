VARIANT		:= 0

OPTFLAGS	 = -O3 -acc -gpu=managed -cuda -cudalib

CC		 = nvc -c99
CFLAGS		 = $(OPTFLAGS)

FC		 = nvfortran
FFLAGS		 = $(OPTFLAGS)

all: test_c99 test_f77

ccsd_tengy_c99.o: ccsd_tengy_c99.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_tengy_f77.o: ccsd_tengy_f77.f
	$(FC) $(FFLAGS) -c $< -o $@

ccsd_trpdrv_acc_cbody.o: ccsd_trpdrv_acc_cbody.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_trpdrv_acc_f77.o: ccsd_trpdrv_acc_f77.f
	$(FC) $(FFLAGS) -c $< -o $@

ccsd_trpdrv_acc_f90.o: ccsd_trpdrv_acc_f90.f
	$(FC) $(FFLAGS) -c $< -o $@

test_c99: test.c ccsd_trpdrv_acc_cbody.o ccsd_tengy_c99.o
	$(CC) $(CFLAGS) $^ -o $@

test_f77: test.c ccsd_trpdrv_acc_f77.o ccsd_tengy_f77.o
	$(CC) $(CFLAGS) -DUSE_FORTRAN $^ -o $@

clean:
	rm -f test
	rm -f test_c99
	rm -f test_f77
	rm -f *.o
	rm -f *.s

realclean: clean

