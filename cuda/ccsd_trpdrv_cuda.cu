#include <stdio.h>

#include <cublas_v2.h>

extern "C" {

#define RESTRICT

static inline void trpdrv_error_check(cublasStatus_t s)
{
    if (s == CUBLAS_STATUS_SUCCESS) return;

    if (s == CUBLAS_STATUS_NOT_INITIALIZED) {
        printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
    } else if (s == CUBLAS_STATUS_ALLOC_FAILED) {
        printf("CUBLAS_STATUS_ALLOC_FAILED\n");
    } else if (s == CUBLAS_STATUS_INVALID_VALUE) {
        printf("CUBLAS_STATUS_INVALID_VALUE\n");
    } else if (s == CUBLAS_STATUS_ARCH_MISMATCH) {
        printf("CUBLAS_STATUS_ARCH_MISMATCH\n");
    } else if (s == CUBLAS_STATUS_MAPPING_ERROR) {
        printf("CUBLAS_STATUS_MAPPING_ERROR\n");
    } else if (s == CUBLAS_STATUS_EXECUTION_FAILED) {
        printf("CUBLAS_STATUS_EXECUTION_FAILED\n");
    } else if (s == CUBLAS_STATUS_INTERNAL_ERROR) {
        printf("CUBLAS_STATUS_INTERNAL_ERROR\n");
    } else if (s == CUBLAS_STATUS_NOT_SUPPORTED) {
        printf("CUBLAS_STATUS_NOT_SUPPORTED\n");
    } else if (s == CUBLAS_STATUS_LICENSE_ERROR) {
        printf("CUBLAS_STATUS_LICENSE_ERROR\n");
    } else {
        printf("CUBLAS Unknown Error\n");
    }
}

void ccsd_tengy_cuda(const double * RESTRICT f1n,    const double * RESTRICT f1t,
                     const double * RESTRICT f2n,    const double * RESTRICT f2t,
                     const double * RESTRICT f3n,    const double * RESTRICT f3t,
                     const double * RESTRICT f4n,    const double * RESTRICT f4t,
                     const double * RESTRICT dintc1, const double * RESTRICT dintx1, const double * RESTRICT t1v1,
                     const double * RESTRICT dintc2, const double * RESTRICT dintx2, const double * RESTRICT t1v2,
                     const double * RESTRICT eorb,   const double eaijk,
                     double * RESTRICT emp4i, double * RESTRICT emp5i,
                     double * RESTRICT emp4k, double * RESTRICT emp5k,
                     const int ncor, const int nocc, const int nvir);

void ccsd_trpdrv_cuda_(double * RESTRICT f1n, double * RESTRICT f1t,
                       double * RESTRICT f2n, double * RESTRICT f2t,
                       double * RESTRICT f3n, double * RESTRICT f3t,
                       double * RESTRICT f4n, double * RESTRICT f4t,
                       double * RESTRICT eorb,
                       int    * RESTRICT ncor_, int * RESTRICT nocc_, int * RESTRICT nvir_,
                       double * RESTRICT emp4_, double * RESTRICT emp5_,
                       int    * RESTRICT a_, int * RESTRICT i_, int * RESTRICT j_, int * RESTRICT k_, int * RESTRICT klo_,
                       double * RESTRICT tij, double * RESTRICT tkj, double * RESTRICT tia, double * RESTRICT tka,
                       double * RESTRICT xia, double * RESTRICT xka, double * RESTRICT jia, double * RESTRICT jka,
                       double * RESTRICT kia, double * RESTRICT kka, double * RESTRICT jij, double * RESTRICT jkj,
                       double * RESTRICT kij, double * RESTRICT kkj,
                       double * RESTRICT dintc1, double * RESTRICT dintx1, double * RESTRICT t1v1,
                       double * RESTRICT dintc2, double * RESTRICT dintx2, double * RESTRICT t1v2)
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

    const int nv = nvir;
    const int no = nocc;

    double one =  1.0;
    double neg = -1.0;
    double nul =  0.0;

    cublasStatus_t s = CUBLAS_STATUS_SUCCESS;
    cublasHandle_t h;
    cublasOperation_t notr = CUBLAS_OP_N;
    cublasOperation_t yatr = CUBLAS_OP_T;

    s = cublasCreate(&h);
    trpdrv_error_check(s);

    {
        double * p_tkj = &tkj[(k-klo)*lnvv];
        double * p_kkj = &kkj[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f1n, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, tia, nv, &kkj[(k-klo)*lnov], no, 1.0, f1n, nv);
        s = cublasDgemm(h, notr, yatr, nv, nv, nv, &one, jia, nv, p_tkj, nv, &nul, f1n, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, tia, nv, p_kkj, nv, &one, f1n, nv); trpdrv_error_check(s);
        // cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        //                            int m, int n, int k,
        //                            const double          *alpha,
        //                            const double          *A, int lda,
        //                            const double          *B, int ldb,
        //                            const double          *beta,
        //                                  double          *C, int ldc)
    }
    {
        double * p_tkj = &tkj[(k-klo)*lnvv];
        double * p_kkj = &kkj[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f2n, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, xia, nv, &kkj[(k-klo)*lnov], no, 1.0, f2n, nv);
        s = cublasDgemm(h, notr, yatr, nv, nv, nv, &one, kia, nv, p_tkj, nv, &nul, f2n, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, xia, nv, p_kkj, nv, &one, f2n, nv); trpdrv_error_check(s);
    }
    {
        double * p_tkj = &tkj[(k-klo)*lnvv];
        double * p_jkj = &jkj[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f3n, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, tia, nv, &jkj[(k-klo)*lnov], no, 1.0, f3n, nv);
        s = cublasDgemm(h, notr, notr, nv, nv, nv, &one, jia, nv, p_tkj, nv, &nul, f3n, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, tia, nv, p_jkj, nv, &one, f3n, nv); trpdrv_error_check(s);
    }
    {
        double * p_tkj = &tkj[(k-klo)*lnvv];
        double * p_jkj = &jkj[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f4n, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, xia, nv, &jkj[(k-klo)*lnov], no, 1.0, f4n, nv);
        s = cublasDgemm(h, notr, notr, nv, nv, nv, &one, kia, nv, p_tkj, nv, &nul, f4n, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, xia, nv, p_jkj, nv, &one, f4n, nv); trpdrv_error_check(s);
    }
    {
        double * p_jka = &jka[(k-klo)*lnvv];
        double * p_tka = &tka[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f1t, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, kij, no, 1.0, f1t, nv);
        s = cublasDgemm(h, notr, yatr, nv, nv, nv, &one, p_jka, nv, tij, nv, &nul, f1t, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, p_tka, nv, kij, nv, &one, f1t, nv); trpdrv_error_check(s);
    }
    {
        double * p_kka = &kka[(k-klo)*lnvv];
        double * p_xka = &xka[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f2t, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, kij, no, 1.0, f2t, nv);
        s = cublasDgemm(h, notr, yatr, nv, nv, nv, &one, p_kka, nv, tij, nv, &nul, f2t, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, p_xka, nv, kij, nv, &one, f2t, nv); trpdrv_error_check(s);
    }
    {
        double * p_jka = &jka[(k-klo)*lnvv];
        double * p_tka = &tka[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f3t, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, jij, no, 1.0, f3t, nv);
        s = cublasDgemm(h, notr, notr, nv, nv, nv, &one, p_jka, nv, tij, nv, &nul, f3t, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, p_tka, nv, jij, nv, &one, f3t, nv); trpdrv_error_check(s);
    }
    {
        double * p_kka = &kka[(k-klo)*lnvv];
        double * p_xka = &xka[(k-klo)*lnov];
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f4t, nv);
        //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, jij, no, 1.0, f4t, nv);
        s = cublasDgemm(h, notr, notr, nv, nv, nv, &one, p_kka, nv, tij, nv, &nul, f4t, nv); trpdrv_error_check(s);
        s = cublasDgemm(h, notr, notr, nv, nv, no, &neg, p_xka, nv, jij, nv, &one, f4t, nv); trpdrv_error_check(s);
    }

    /* convert from Fortran to C offset convention... */
    const int a   = *a_ - 1;
    const int i   = *i_ - 1;
    const int j   = *j_ - 1;

    const double eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

    ccsd_tengy_cuda(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
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

} // extern "C"
