//#include <stdio.h>

//#define SKIP_DGEMM
#ifndef SKIP_DGEMM
#if defined(MKL)
# include <mkl.h>
# include <mkl_omp_offload.h>
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

#define MIN(x,y) ((x)<(y)?(x):(y))

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
    float emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

    const int ncor = *ncor_;
    const int nocc = *nocc_;
    const int nvir = *nvir_;

    const int lnov = nocc * nvir;
    const int lnvv = nvir * nvir;

    /* convert from Fortran to C offset convention... */
    const int a   = *a_ - 1;
    const int i   = *i_ - 1;
    const int j   = *j_ - 1;
    const int k   = *k_ - 1;
    const int klo = *klo_ - 1;

#ifndef SKIP_DGEMM
    const cblas_int nv = nvir;
    const cblas_int no = nocc;
#endif

    const float eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

    float *tkj_tmp = &tkj[(k-klo)*lnvv];
    float *kkj_tmp = &kkj[(k-klo)*lnov];
    float *jkj_tmp = &jkj[(k-klo)*lnov];
    float *jka_tmp = &jka[(k-klo)*lnvv];
    float *tka_tmp = &tka[(k-klo)*lnov];
    float *kka_tmp = &kka[(k-klo)*lnvv];
    float *xka_tmp = &xka[(k-klo)*lnov];

#ifndef SKIP_DGEMM
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(jia,tkj_tmp,f1n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, jia, nv, tkj_tmp, nv, 0.0, f1n, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(tia,kkj_tmp,f1n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tia, nv, kkj_tmp, no, 1.0, f1n, nv);
    }
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(kia,tkj_tmp,f2n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, kia, nv, tkj_tmp, nv, 0.0, f2n, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(xia,kkj_tmp,f2n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xia, nv, kkj_tmp, no, 1.0, f2n, nv);
    }
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(jia,tkj_tmp,f3n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, jia, nv, tkj_tmp, nv, 0.0, f3n, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(tia,jkj_tmp,f3n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tia, nv, jkj_tmp, no, 1.0, f3n, nv);
    }
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(kia,tkj_tmp,f4n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, kia, nv, tkj_tmp, nv, 0.0, f4n, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(xia,jkj_tmp,f4n)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xia, nv, jkj_tmp, no, 1.0, f4n, nv);
    }
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(jka_tmp,tij,f1t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, jka_tmp, nv, tij, nv, 0.0, f1t, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(tka_tmp,kij,f1t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tka_tmp, nv, kij, no, 1.0, f1t, nv);
    }
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(kka_tmp,tij,f2t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    nv, nv, nv, 1.0, kka_tmp, nv, tij, nv, 0.0, f2t, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(xka_tmp,kij,f2t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xka_tmp, nv, kij, no, 1.0, f2t, nv);
    }
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(jka_tmp,tij,f3t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, jka_tmp, nv, tij, nv, 0.0, f3t, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(tka_tmp,jij,f3t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, tka_tmp, nv, jij, no, 1.0, f3t, nv);
    }
    {
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(kka_tmp,tij,f4t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, nv, 1.0, kka_tmp, nv, tij, nv, 0.0, f4t, nv);
#if USE_OPENMP_TARGET
#pragma omp target variant dispatch use_device_ptr(xka_tmp,jij,f4t)
#endif  // USE_OPENMP_TARGET
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    nv, nv, no, -1.0, xka_tmp, nv, jij, no, 1.0, f4t, nv);
    }
#endif // SKIP_DGEMM

#if USE_OPENMP_TARGET
    #pragma omp target map(tofrom: emp5i, emp4i, emp5k, emp4k)
    #pragma omp teams distribute parallel for collapse(2) reduction(+:emp5i,emp4i,emp5k,emp4k)
#endif  // USE_OPENMP_TARGET
    for (int b = 0; b < nvir; ++b) {
        for (int c = 0; c < nvir; ++c) {

            const float denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

            // nvir < 10000 so this should never overflow
            const int bc = b+c*nvir;
            const int cb = c+b*nvir;

            const float f1nbc = f1n[bc];
            const float f1tbc = f1t[bc];
            const float f1ncb = f1n[cb];
            const float f1tcb = f1t[cb];

            const float f2nbc = f2n[bc];
            const float f2tbc = f2t[bc];
            const float f2ncb = f2n[cb];
            const float f2tcb = f2t[cb];

            const float f3nbc = f3n[bc];
            const float f3tbc = f3t[bc];
            const float f3ncb = f3n[cb];
            const float f3tcb = f3t[cb];

            const float f4nbc = f4n[bc];
            const float f4tbc = f4t[bc];
            const float f4ncb = f4n[cb];
            const float f4tcb = f4t[cb];

            emp4i += denom * (f1tbc+f1ncb+f2tcb+f3nbc+f4ncb) * (f1tbc-f2tbc*2-f3tbc*2+f4tbc)
                   - denom * (f1nbc+f1tcb+f2ncb+f3ncb) * (f1tbc*2-f2tbc-f3tbc+f4tbc*2)
                   + denom * 3 * (f1nbc*(f1nbc+f3ncb+f4tcb*2) +f2nbc*f2tcb+f3nbc*f4tbc);
            emp4k += denom * (f1nbc+f1tcb+f2ncb+f3tbc+f4tcb) * (f1nbc-f2nbc*2-f3nbc*2+f4nbc)
                   - denom * (f1tbc+f1ncb+f2tcb+f3tcb) * (f1nbc*2-f2nbc-f3nbc+f4nbc*2)
                   + denom * 3 * (f1tbc*(f1tbc+f3tcb+f4ncb*2) +f2tbc*f2ncb+f3tbc*f4nbc);

            const float t1v1b = t1v1[b];
            const float t1v2b = t1v2[b];

            const float dintx1c = dintx1[c];
            const float dintx2c = dintx2[c];
            const float dintc1c = dintc1[c];
            const float dintc2c = dintc2[c];

            emp5i += denom * t1v1b * dintx1c * (f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
                                                +(f3nbc+f4tbc+f1ncb)*4)
                   + denom * t1v1b * dintc1c * (f1nbc+f4nbc+f1tcb -(f2nbc+f3nbc+f2tcb)*2);
            emp5k += denom * t1v2b * dintx2c * (f1nbc+f2tbc+f4tcb -(f3nbc+f4tbc+f2tcb +f1tbc+f2nbc+f3tcb)*2
                                                +(f3tbc+f4nbc+f1tcb)*4)
                   + denom * t1v2b * dintc2c * (f1tbc+f4tbc+f1ncb -(f2tbc+f3tbc+f2ncb)*2);
        }
    }

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

