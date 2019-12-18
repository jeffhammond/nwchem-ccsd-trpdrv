#include <stdio.h>

#define MIN(x,y) ((x)<(y)?(x):(y))

void ccsd_tengy_omp(const float * restrict f1n,    const float * restrict f1t,
                    const float * restrict f2n,    const float * restrict f2t,
                    const float * restrict f3n,    const float * restrict f3t,
                    const float * restrict f4n,    const float * restrict f4t,
                    const float * restrict dintc1, const float * restrict dintx1, const float * restrict t1v1,
                    const float * restrict dintc2, const float * restrict dintx2, const float * restrict t1v2,
                    const float * restrict eorb,   const float eaijk,
                    float * restrict emp4i_, float * restrict emp5i_,
                    float * restrict emp4k_, float * restrict emp5k_,
                    const int ncor, const int nocc, const int nvir)
{
    float emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

#ifdef USE_OPENMP_TARGET
    printf("ccsd_tengy_omp using OpenMP target for - variant %d\n", OPENMP_TARGET_VARIANT );
    #pragma omp target map(to: f1n[0:nvir*nvir], f1t[0:nvir*nvir], \
                               f2n[0:nvir*nvir], f2t[0:nvir*nvir], \
                               f3n[0:nvir*nvir], f3t[0:nvir*nvir], \
                               f4n[0:nvir*nvir], f4t[0:nvir*nvir] ) \
                       map(to: dintc1[0:nvir], dintc2[0:nvir], \
                               dintx1[0:nvir], dintx2[0:nvir], \
                               t1v1[0:nvir],   t1v2[0:nvir] ) \
                       map(to: eorb[0:ncor+nocc+nvir] ) \
                       map(to: ncor, nocc, nvir, eaijk) \
                       map(tofrom: emp5i, emp4i, emp5k, emp4k)
# if OPENMP_TARGET_VARIANT == 0
#warning variant 0
    // default version
    #pragma omp parallel for reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
    for (int b = 0; b < nvir; ++b) {{
        #pragma omp parallel for reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
        for (int c = 0; c < nvir; ++c) {{
# elif OPENMP_TARGET_VARIANT == 2
#warning variant 2
#define TILESIZE 64
    #pragma omp parallel for reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
    for (int bt = 0; bt < nvir; bt+=TILESIZE) {
      #pragma omp parallel for reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
      for (int ct = 0; ct < nvir; ct+=TILESIZE) {
        #pragma omp parallel for reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
        for (int b = bt; b < MIN(bt+TILESIZE,nvir); ++b) {
          #pragma omp parallel for reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
          for (int c = ct; c < MIN(ct+TILESIZE,nvir); ++c) {
# elif OPENMP_TARGET_VARIANT == 3
#warning variant 3
    #pragma omp parallel for collapse(2) reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
    for (int b = 0; b < nvir; ++b) {{
        for (int c = 0; c < nvir; ++c) {{
# elif OPENMP_TARGET_VARIANT == 4
#warning variant 4
#define TILESIZE 32
    #pragma omp parallel for collapse(2) reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
    for (int bt = 0; bt < nvir; bt+=TILESIZE) {
      for (int ct = 0; ct < nvir; ct+=TILESIZE) {
        for (int b = bt; b < MIN(bt+TILESIZE,nvir); ++b) {
          for (int c = ct; c < MIN(ct+TILESIZE,nvir); ++c) {
# else
#  error No variant selected!
# endif
#else // !USE_OPENMP_TARGET
#define TILESIZE 32
# if 0
    #pragma omp parallel for reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
    for (int b = 0; b < nvir; ++b) {{
        for (int c = 0; c < nvir; ++c) {{
# else
    #pragma omp parallel for collapse(2) reduction(+:emp5i,emp4i) reduction(+:emp5k,emp4k)
    for (int bt = 0; bt < nvir; bt+=TILESIZE) {
      for (int ct = 0; ct < nvir; ct+=TILESIZE) {
        for (int b = bt; b < MIN(bt+TILESIZE,nvir); ++b) {
          for (int c = ct; c < MIN(ct+TILESIZE,nvir); ++c) {
# endif
#endif // USE_OPENMP_TARGET
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
        }}
    }}

    *emp4i_ = emp4i;
    *emp4k_ = emp4k;
    *emp5i_ = emp5i;
    *emp5k_ = emp5k;
}
