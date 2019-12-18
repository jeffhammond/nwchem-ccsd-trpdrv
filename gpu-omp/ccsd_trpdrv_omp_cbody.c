//#include <stdio.h>

#ifndef SKIP_DGEMM
#if defined(MKL)
# include <mkl.h>
# ifndef MKL_INT
#  error MKL_INT not defined!
# endif
  typedef MKL_INT cblas_int;
#elif defined(ACCELERATE)
  /* The location of cblas.h is not in the system include path when -framework Accelerate is provided. */
# include <Accelerate/Accelerate.h>
  typedef int cblas_int;
#else
# include <cblas.h>
  typedef int cblas_int;
#endif
#endif

void ccsd_tengy_omp(const float * restrict f1n,    const float * restrict f1t,
                    const float * restrict f2n,    const float * restrict f2t,
                    const float * restrict f3n,    const float * restrict f3t,
                    const float * restrict f4n,    const float * restrict f4t,
                    const float * restrict dintc1, const float * restrict dintx1, const float * restrict t1v1,
                    const float * restrict dintc2, const float * restrict dintx2, const float * restrict t1v2,
                    const float * restrict eorb,   const float eaijk,
                    float * restrict emp4i, float * restrict emp5i,
                    float * restrict emp4k, float * restrict emp5k,
                    const int ncor, const int nocc, const int nvir);

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
                            float * restrict dintc2, float * restrict dintx2, float * restrict t1v2)
{
    float emp4 = *emp4_;
    float emp5 = *emp5_;

    float emp4i = 0.0;
    float emp5i = 0.0;
    float emp4k = 0.0;
    float emp5k = 0.0;

    const int ncor = *ncor_;
    const int nocc = *nocc_;
    const int nvir = *nvir_;

    const int lnov = nocc * nvir;
    const int lnvv = nvir * nvir;

    /* convert from Fortran to C offset convention... */
    const int k   = *k_ - 1;
    const int klo = *klo_ - 1;

#ifndef SKIP_DGEMM

    const cblas_int nv = nvir;
    const cblas_int no = nocc;

    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f1n, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tia, nv, &kkj[(k-klo)*lnov], no, 1.0, f1n, nv);
    }
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f2n, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xia, nv, &kkj[(k-klo)*lnov], no, 1.0, f2n, nv);
    }
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f3n, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tia, nv, &jkj[(k-klo)*lnov], no, 1.0, f3n, nv);
    }
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f4n, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xia, nv, &jkj[(k-klo)*lnov], no, 1.0, f4n, nv);
    }
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f1t, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, kij, no, 1.0, f1t, nv);
    }
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f2t, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, kij, no, 1.0, f2t, nv);
    }
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f3t, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, jij, no, 1.0, f3t, nv);
    }
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f4t, nv);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, jij, no, 1.0, f4t, nv);
    }

#endif // SKIP_DGEMM

    /* convert from Fortran to C offset convention... */
    const int a   = *a_ - 1;
    const int i   = *i_ - 1;
    const int j   = *j_ - 1;

    const float eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

    ccsd_tengy_omp(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
                   dintc1, dintx1, t1v1, dintc2, dintx2, t1v2,
                   eorb, eaijk, &emp4i, &emp5i, &emp4k, &emp5k,
                   ncor, nocc, nvir);

    emp4 += emp4i;
    emp5 += emp5i;

    if (*i_ != *k_) {
        emp4 += emp4k;
        emp5 += emp5k;
    }

    *emp4_ = emp4;
    *emp5_ = emp5;

    return;
}

