#include <stdio.h>
#include <stdlib.h>

#ifdef USE_VTUNE
#include <ittnotify.h>
#endif

/* Do not allow the test to allocate more than MAX_MEM gigabytes. */
#ifndef MAX_MEM
#define MAX_MEM 4
#endif

#define MIN(x,y) (x<y ? x : y)
#define MAX(x,y) (x>y ? x : y)

#ifdef _OPENMP
#include <omp.h>
#else
#warning No timer!
double omp_get_wtime() { return 0.0; }
#endif

#ifdef USE_FORTRAN
void ccsd_trpdrv_omp_fbody_(float * restrict f1n, float * restrict f1t,
                            float * restrict f2n, float * restrict f2t,
                            float * restrict f3n, float * restrict f3t,
                            float * restrict f4n, float * restrict f4t,
                            float * restrict eorb,
                            int    * restrict ncor_, int * restrict nocc_, int * restrict nvir_,
                            float * restrict emp4_, float * restrict emp5_,
                            int    * restrict a_, int * restrict i_, int * restrict j_, int * restrict k_, int * restrict klo_,
                            float * restrict tij, float * restrict tkj, float * restrict tia, float * restrict tka,
                            float * restrict xia, float * restrict xka, float * restrict jia, float * restrict jka,
                            float * restrict kia, float * restrict kka, float * restrict jij, float * restrict jkj,
                            float * restrict kij, float * restrict kkj,
                            float * restrict dintc1, float * restrict dintx1, float * restrict t1v1,
                            float * restrict dintc2, float * restrict dintx2, float * restrict t1v2);
#else
void ccsd_trpdrv_omp_cbody_(float * restrict f1n, float * restrict f1t,
                            float * restrict f2n, float * restrict f2t,
                            float * restrict f3n, float * restrict f3t,
                            float * restrict f4n, float * restrict f4t,
                            float * restrict eorb,
                            int    * restrict ncor_, int * restrict nocc_, int * restrict nvir_,
                            float * restrict emp4_, float * restrict emp5_,
                            int    * restrict a_, int * restrict i_, int * restrict j_, int * restrict k_, int * restrict klo_,
                            float * restrict tij, float * restrict tkj, float * restrict tia, float * restrict tka,
                            float * restrict xia, float * restrict xka, float * restrict jia, float * restrict jka,
                            float * restrict kia, float * restrict kka, float * restrict jij, float * restrict jkj,
                            float * restrict kij, float * restrict kkj,
                            float * restrict dintc1, float * restrict dintx1, float * restrict t1v1,
                            float * restrict dintc2, float * restrict dintx2, float * restrict t1v2);
#endif

float * make_array(int n)
{
    float * a = malloc(n*sizeof(float));
    for (int i=0; i<n; i++) {
        a[i] = 1.0/(100.0+i);
    }
    return a;
}

int main(int argc, char* argv[])
{
    int ncor, nocc, nvir;
    int maxiter = 100;
    int nkpass = 1;

    if (argc<3) {
        printf("Usage: ./test_cbody nocc nvir [maxiter] [nkpass]\n");
        return argc;
    } else {
        ncor = 0;
        nocc = atoi(argv[1]);
        nvir = atoi(argv[2]);
        if (argc>3) {
            maxiter = atoi(argv[3]);
            /* if negative, treat as "infinite" */
            if (maxiter<0) maxiter = 1<<30;
        }
        if (argc>4) {
            nkpass = atoi(argv[4]);
        }
    }

    if (nocc<1 || nvir<1) {
        printf("Arguments must be non-negative!\n");
        return 1;
    }

    printf("Test driver for cbody with nocc=%d, nvir=%d, maxiter=%d, nkpass=%d\n", nocc, nvir, maxiter, nkpass);

    const int nbf = ncor + nocc + nvir;
    const int lnvv = nvir * nvir;
    const int lnov = nocc * nvir;
    const int kchunk = (nocc - 1)/nkpass + 1;

    const float memory = (nbf+8.0*lnvv+
                           lnvv+kchunk*lnvv+lnov*nocc+kchunk*lnov+lnov*nocc+kchunk*lnov+lnvv+
                           kchunk*lnvv+lnvv+kchunk*lnvv+lnov*nocc+kchunk*lnov+lnov*nocc+
                           kchunk*lnov+lnov+nvir*kchunk+nvir*nocc+
                           6.0*lnvv)*sizeof(float);
    printf("This test requires %f GB of memory.\n", 1.0e-9*memory);

    if (1.0e-9*memory > MAX_MEM) {
        printf("You need to increase MAX_MEM (%d)\n", MAX_MEM);
        printf("or set nkpass (%d) to a larger number.\n", nkpass);
        return MAX_MEM;
    }

    float * eorb = make_array(nbf);

    float * f1n = make_array(lnvv);
    float * f2n = make_array(lnvv);
    float * f3n = make_array(lnvv);
    float * f4n = make_array(lnvv);
    float * f1t = make_array(lnvv);
    float * f2t = make_array(lnvv);
    float * f3t = make_array(lnvv);
    float * f4t = make_array(lnvv);

    float * Tij  = make_array(lnvv);
    float * Tkj  = make_array(kchunk*lnvv);
    float * Tia  = make_array(lnov*nocc);
    float * Tka  = make_array(kchunk*lnov);
    float * Xia  = make_array(lnov*nocc);
    float * Xka  = make_array(kchunk*lnov);
    float * Jia  = make_array(lnvv);
    float * Jka  = make_array(kchunk*lnvv);
    float * Kia  = make_array(lnvv);
    float * Kka  = make_array(kchunk*lnvv);
    float * Jij  = make_array(lnov*nocc);
    float * Jkj  = make_array(kchunk*lnov);
    float * Kij  = make_array(lnov*nocc);
    float * Kkj  = make_array(kchunk*lnov);
    float * Dja  = make_array(lnov);
    float * Djka = make_array(nvir*kchunk);
    float * Djia = make_array(nvir*nocc);

    float * dintc1 = make_array(lnvv);
    float * dintc2 = make_array(lnvv);
    float * dintx1 = make_array(lnvv);
    float * dintx2 = make_array(lnvv);
    float * t1v1   = make_array(lnvv);
    float * t1v2   = make_array(lnvv);

    int ntimers = MIN(maxiter,nocc*nocc*nocc*nocc);
    double * timers = calloc(ntimers,sizeof(float));

    float emp4=0.0, emp5=0.0;
    int a=1, i=1, j=1, k=1, klo=1;

    int iter = 0;

#ifdef USE_VTUNE
    __itt_resume();
#endif

    for (int klo=1; klo<=nocc; klo+=kchunk) {
        const int khi = MIN(nocc, klo+kchunk-1);
        int a=1;
        for (int j=1; j<=nocc; j++) {
            for (int i=1; i<=nocc; i++) {
                for (int k=klo; k<=MIN(khi,i); k++) {
                    double t0 = omp_get_wtime();
#ifdef USE_FORTRAN
                    ccsd_trpdrv_omp_fbody_(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, eorb,
                                           &ncor, &nocc, &nvir, &emp4, &emp5, &a, &i, &j, &k, &klo,
                                           Tij, Tkj, Tia, Tka, Xia, Xka, Jia, Jka, Kia, Kka, Jij, Jkj, Kij, Kkj,
                                           dintc1, dintx1, t1v1, dintc2, dintx2, t1v2);
#else
                    ccsd_trpdrv_omp_cbody_(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, eorb,
                                           &ncor, &nocc, &nvir, &emp4, &emp5, &a, &i, &j, &k, &klo,
                                           Tij, Tkj, Tia, Tka, Xia, Xka, Jia, Jka, Kia, Kka, Jij, Jkj, Kij, Kkj,
                                           dintc1, dintx1, t1v1, dintc2, dintx2, t1v2);
#endif
                    double t1 = omp_get_wtime();
                    timers[iter] = (t1-t0);

                    iter++;
                    if (iter==maxiter) {
                        printf("Stopping after %d iterations...\n", iter);
                        goto maxed_out;
                    }

                    /* prevent NAN for large maxiter... */
                    if (emp4 >  1000.0) emp4 -= 1000.0;
                    if (emp4 < -1000.0) emp4 += 1000.0;
                    if (emp5 >  1000.0) emp5 -= 1000.0;
                    if (emp5 < -1000.0) emp5 += 1000.0;
                }
            }
        }
    }

maxed_out:
    printf("");

#ifdef USE_VTUNE
    __itt_pause();
#endif

    double tsum =  0.0;
    double tmax = -1.0e10;
    double tmin =  1.0e10;
    for (int i=0; i<iter; i++) {
        //printf("timers[%d] = %f\n", i, timers[i]);
        tsum += timers[i];
        tmax  = MAX(tmax,timers[i]);
        tmin  = MIN(tmin,timers[i]);
    }
    double tavg = tsum / iter;
    printf("TIMING: min=%f, max=%f, avg=%f\n", tmin, tmax, tavg);

    double sgemm_flops = ((8.0*nvir)*nvir)*(nvir+nocc);
    double sgemm_mops  = 8.0*(4.0*nvir*nvir + 2.0*nvir*nocc);

    /* The inner loop of tengy touches 86 f[1234][nt] elements and 8 other arrays...
     * We will just assume flops=mops even though flops>mops */
    double tengy_ops = ((1.0*nvir)*nvir)*(86+8);

    printf("OPS: sgemm_flops=%10.3e sgemm_mops=%10.3e tengy_ops=%10.3e\n",
            sgemm_flops, sgemm_mops, tengy_ops);
    printf("PERF: GF/s=%10.3e GB/s=%10.3e\n",
            1.0e-9*(sgemm_flops+tengy_ops)/tavg, 8.0e-9*(sgemm_mops+tengy_ops)/tavg);

    printf("These are meaningless but should not vary for a particular input:\n");
    printf("emp4=%f emp5=%f\n", emp4, emp5);

    printf("SUCCESS\n");

    return 0;
}
