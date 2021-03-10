#include <stdio.h>

#include <cublas_v2.h>

extern "C" {

#define RESTRICT

#define MIN(x,y) ((x)<(y)?(x):(y))

#define TILESIZE 32

static inline int divceil(int numerator, int denominator) {
    return ( numerator / denominator + (numerator % denominator > 0) );
}

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

__global__
void ccsd_tengy_cuda(const double * RESTRICT f1n,    const double * RESTRICT f1t,
                     const double * RESTRICT f2n,    const double * RESTRICT f2t,
                     const double * RESTRICT f3n,    const double * RESTRICT f3t,
                     const double * RESTRICT f4n,    const double * RESTRICT f4t,
                     const double * RESTRICT dintc1, const double * RESTRICT dintx1, const double * RESTRICT t1v1,
                     const double * RESTRICT dintc2, const double * RESTRICT dintx2, const double * RESTRICT t1v2,
                     const double * RESTRICT eorb, const int a, const int i, const int j, const int k,
                     double * RESTRICT emp4i_, double * RESTRICT emp5i_,
                     double * RESTRICT emp4k_, double * RESTRICT emp5k_,
                     const int ncor, const int nocc, const int nvir)
{
    // need to implement a proper reduction on these
    double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

    //for (int b = 0; b < nvir; ++b)
    //  for (int c = 0; c < nvir; ++c) {

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if ((b<nvir) && (c<nvir)) {

        // hoist later
        const double eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);
        const double denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

        // nvir < 10000 so this should never overflow
        const int bc = b+c*nvir;
        const int cb = c+b*nvir;

        const double f1nbc = f1n[bc];
        const double f1tbc = f1t[bc];
        const double f1ncb = f1n[cb];
        const double f1tcb = f1t[cb];

        const double f2nbc = f2n[bc];
        const double f2tbc = f2t[bc];
        const double f2ncb = f2n[cb];
        const double f2tcb = f2t[cb];

        const double f3nbc = f3n[bc];
        const double f3tbc = f3t[bc];
        const double f3ncb = f3n[cb];
        const double f3tcb = f3t[cb];

        const double f4nbc = f4n[bc];
        const double f4tbc = f4t[bc];
        const double f4ncb = f4n[cb];
        const double f4tcb = f4t[cb];

        emp4i += denom * (f1tbc+f1ncb+f2tcb+f3nbc+f4ncb) * (f1tbc-f2tbc*2-f3tbc*2+f4tbc)
               - denom * (f1nbc+f1tcb+f2ncb+f3ncb) * (f1tbc*2-f2tbc-f3tbc+f4tbc*2)
               + denom * 3 * (f1nbc*(f1nbc+f3ncb+f4tcb*2) +f2nbc*f2tcb+f3nbc*f4tbc);
        emp4k += denom * (f1nbc+f1tcb+f2ncb+f3tbc+f4tcb) * (f1nbc-f2nbc*2-f3nbc*2+f4nbc)
               - denom * (f1tbc+f1ncb+f2tcb+f3tcb) * (f1nbc*2-f2nbc-f3nbc+f4nbc*2)
               + denom * 3 * (f1tbc*(f1tbc+f3tcb+f4ncb*2) +f2tbc*f2ncb+f3tbc*f4nbc);

        const double t1v1b = t1v1[b];
        const double t1v2b = t1v2[b];

        const double dintx1c = dintx1[c];
        const double dintx2c = dintx2[c];
        const double dintc1c = dintc1[c];
        const double dintc2c = dintc2[c];

        emp5i += denom * t1v1b * dintx1c * (f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
                                            +(f3nbc+f4tbc+f1ncb)*4)
               + denom * t1v1b * dintc1c * (f1nbc+f4nbc+f1tcb -(f2nbc+f3nbc+f2tcb)*2);
        emp5k += denom * t1v2b * dintx2c * (f1nbc+f2tbc+f4tcb -(f3nbc+f4tbc+f2tcb +f1tbc+f2nbc+f3tcb)*2
                                            +(f3tbc+f4nbc+f1tcb)*4)
               + denom * t1v2b * dintc2c * (f1tbc+f4tbc+f1ncb -(f2tbc+f3tbc+f2ncb)*2);
    }

    *emp4i_ = emp4i;
    *emp4k_ = emp4k;
    *emp5i_ = emp5i;
    *emp5k_ = emp5k;
}

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

    const int tile_size = 32;
    dim3 dimGrid(divceil(nvir,tile_size),divceil(nvir,tile_size),1);
    dim3 dimBlock(tile_size, tile_size, 1);

    ccsd_tengy_cuda<<<dimGrid, dimBlock>>>(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t,
                                           dintc1, dintx1, t1v1, dintc2, dintx2, t1v2,
                                           eorb, a, i, j, k,
                                           &emp4i, &emp5i, &emp4k, &emp5k,
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
