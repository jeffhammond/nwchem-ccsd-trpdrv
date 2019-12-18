This code corresponds to https://github.com/jeffhammond/nwchem/tree/ccsd_trpdrv_omp_c99/src/ccsd
(or https://github.com/nwchemgit/nwchem/tree/master/src/ccsd if it is merged).

The evolution of the code was:
1. src/ccsd/ccsd_trpdrv.F
2. src/ccsd/ccsd_trpdrv_nb.F (nonblocking communication)
3. src/ccsd/ccsd_trpdrv_bgp2.F (Blue Gene/P version)
4. src/ccsd/ccsd_trpdrv_omp.F (standard OpenMP)
4. src/ccsd/ccsd_trpdrv_omp_fn.F (calls out to Fortran or C99 for the compute)