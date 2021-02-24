VARIANT		:= 3

OMPFLAGS	 = -fiopenmp
OMPFLAGS	+= -fopenmp-targets=spir64 -DUSE_OPENMP_TARGET -DOPENMP_TARGET_VARIANT=$(VARIANT)
OPTFLAGS	 = -O3
#MKLFLAGS	 = -DSKIP_DGEMM
MKLFLAGS	 = -DMKL -qmkl -DMKL_ILP64
#MKLROOT		:= /opt/intel/oneapi/mkl/latest
#MKL_INCLUDE	+= -I$(MKLROOT)/include
#MKL_LIBS	+= -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lOpenCL -lpthread -ldl -lm
MKL_LIBS        += -fsycl -qmkl

CC		 = icx
CFLAGS		 = -std=c99 $(OMPFLAGS) $(OPTFLAGS) $(MKLFLAGS)
LINKFLAGS        = $(OMPFLAGS) $(OPTFLAGS) $(MKLFLAGS)
#CFLAGS		+= -I/opt/intel/vtune_amplifier_2018/include -DUSE_VTUNE
#VTUNELIB	 = /opt/intel/vtune_amplifier_2018/lib64/libittnotify.a
#CFLAGS		+= -I/swtools/intel/vtune_amplifier_2018/include -DUSE_VTUNE
#VTUNELIB	 = /swtools/intel/vtune_amplifier_2018/lib64/libittnotify.a

FC		 = ifx
FFLAGS		 = $(OMPFLAGS) $(OPTFLAGS) $(MKL_INCLUDE)

all: test_c99 test_f77

ccsd_tengy_c99.o: ccsd_tengy_c99.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_tengy_f77.o: ccsd_tengy_f77.f
	$(FC) $(FFLAGS) -fixed -c $< -o $@

ccsd_trpdrv_omp_cbody.o: ccsd_trpdrv_omp_cbody.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_trpdrv_omp_f77.o: ccsd_trpdrv_omp_f77.f
	$(FC) $(FFLAGS) -fixed -c $< -o $@

ccsd_trpdrv_omp_f90.o: ccsd_trpdrv_omp_f90.f
	$(FC) $(FFLAGS) -free -c $< -o $@

test_c99.o: test.c
	$(CC) $(CFLAGS) -c $< -o $@

test_f77.o: test.c
	$(CC) $(CFLAGS) -DUSE_FORTRAN -c $< -o $@

test_c99: test_c99.o ccsd_trpdrv_omp_cbody.o
	$(CC) $(LINKFLAGS) $^ $(VTUNELIB) $(MKL_LIBS) -o $@

test_f77: test_f77.o ccsd_trpdrv_omp_f77.o ccsd_tengy_f77.o
	$(CC) $(LINKFLAGS) -DUSE_FORTRAN $^ $(VTUNELIB) $(MKL_LIBS) -o $@

clean:
	rm -f test
	rm -f test_c99
	rm -f test_f77
	rm -f *.o
	rm -f *.s

cleansim:

realclean: clean cleansim

