VARIANT		:= 0

#CC		 = icc -std=c99
#OMPFLAGS	 = -qopenmp
#OPTFLAGS	 = -O3 -xCORE-AVX2
#MKLFLAGS	 = -DMKL -DSKIP_DGEMM -mkl

CC		 = clang -std=c11 -Wno-unused-command-line-argument
#OMPFLAGS	 = -fopenmp
OPTFLAGS	 = -O3 -mfma -mavx
MKLFLAGS	 = -DACCELERATE -framework Accelerate

#CFLAGS		+= -I/opt/intel/vtune_amplifier_2018/include -DUSE_VTUNE
#VTUNELIB	 = /opt/intel/vtune_amplifier_2018/lib64/libittnotify.a
#CFLAGS		+= -I/swtools/intel/vtune_amplifier_2018/include -DUSE_VTUNE
#VTUNELIB	 = /swtools/intel/vtune_amplifier_2018/lib64/libittnotify.a

#OCLFLAGS	 = -I. -L. -lOpenCL
OCLFLAGS	 = -framework OpenCL -DUSE_OPENCL

CFLAGS		 = $(OMPFLAGS) $(OPTFLAGS) $(MKLFLAGS)

all: test_c99 test_ocl

ccsd_tengy.o: ccsd_tengy.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_trpdrv_cbody.o: ccsd_trpdrv_cbody.c
	$(CC) $(CFLAGS) -c $< -o $@

ccsd_trpdrv_opencl.o: ccsd_trpdrv_opencl.c
	$(CC) $(CFLAGS) $(OCLFLAGS) -c $< -o $@

test_c99: test.c ccsd_trpdrv_cbody.o ccsd_tengy.o
	$(CC) $(CFLAGS) $^ $(VTUNELIB) -o $@

test_ocl: test.c ccsd_trpdrv_opencl.o
	$(CC) $(CFLAGS) $^ $(OCLFLAGS) -o $@

clean:
	rm -f test_ocl
	rm -f test_c99
	rm -f *.o
	rm -f *.s

cleansim:

realclean: clean cleansim

