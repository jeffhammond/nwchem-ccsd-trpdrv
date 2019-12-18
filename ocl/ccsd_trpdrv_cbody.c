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

void ccsd_tengy(const double * restrict f1n,    const double * restrict f1t,
                const double * restrict f2n,    const double * restrict f2t,
                const double * restrict f3n,    const double * restrict f3t,
                const double * restrict f4n,    const double * restrict f4t,
                const double * restrict dintc1, const double * restrict dintx1, const double * restrict t1v1,
                const double * restrict dintc2, const double * restrict dintx2, const double * restrict t1v2,
                const double * restrict eorb,   const double eaijk,
                double * restrict emp4i, double * restrict emp5i,
                double * restrict emp4k, double * restrict emp5k,
                const int ncor, const int nocc, const int nvir);

void ccsd_trpdrv_cbody_(double * restrict f1n, double * restrict f1t,
                        double * restrict f2n, double * restrict f2t,
                        double * restrict f3n, double * restrict f3t,
                        double * restrict f4n, double * restrict f4t,
                        double * restrict eorb,
                        int    * restrict ncor_, int * restrict nocc_, int * restrict nvir_,
                        double * restrict emp4_, double * restrict emp5_,
                        int    * restrict a_, int * restrict i_, int * restrict j_, int * restrict k_, int * restrict klo_,
                        double * restrict tij, double * restrict tkj, double * restrict tia, double * restrict tka,
                        double * restrict xia, double * restrict xka, double * restrict jia, double * restrict jka,
                        double * restrict kia, double * restrict kka, double * restrict jij, double * restrict jkj,
                        double * restrict kij, double * restrict kkj,
                        double * restrict dintc1, double * restrict dintx1, double * restrict t1v1,
                        double * restrict dintc2, double * restrict dintx2, double * restrict t1v2)
{
    double emp4 = *emp4_;
    double emp5 = *emp5_;

    double emp4i = 0.0;
    double emp5i = 0.0;
    double emp4k = 0.0;
    double emp5k = 0.0;

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
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f1n, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tia, nv, &kkj[(k-klo)*lnov], no, 1.0, f1n, nv);
    }
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f2n, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xia, nv, &kkj[(k-klo)*lnov], no, 1.0, f2n, nv);
    }
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f3n, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tia, nv, &jkj[(k-klo)*lnov], no, 1.0, f3n, nv);
    }
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f4n, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xia, nv, &jkj[(k-klo)*lnov], no, 1.0, f4n, nv);
    }
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f1t, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, kij, no, 1.0, f1t, nv);
    }
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f2t, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, kij, no, 1.0, f2t, nv);
    }
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f3t, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, jij, no, 1.0, f3t, nv);
    }
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f4t, nv);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, jij, no, 1.0, f4t, nv);
    }

#endif // SKIP_DGEMM

    /* convert from Fortran to C offset convention... */
    const int a   = *a_ - 1;
    const int i   = *i_ - 1;
    const int j   = *j_ - 1;

    const double eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

    ccsd_tengy(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
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

