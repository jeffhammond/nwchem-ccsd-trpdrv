#define MIN(x,y) ((x)<(y)?(x):(y))

void ccsd_tengy(const double * restrict f1n,    const double * restrict f1t,
                const double * restrict f2n,    const double * restrict f2t,
                const double * restrict f3n,    const double * restrict f3t,
                const double * restrict f4n,    const double * restrict f4t,
                const double * restrict dintc1, const double * restrict dintx1, const double * restrict t1v1,
                const double * restrict dintc2, const double * restrict dintx2, const double * restrict t1v2,
                const double * restrict eorb,   const double eaijk,
                double * restrict emp4i_, double * restrict emp5i_,
                double * restrict emp4k_, double * restrict emp5k_,
                const int ncor, const int nocc, const int nvir)
{
    double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

#ifdef TILESIZE
    for (int bt = 0; bt < nvir; bt+=TILESIZE)
      for (int ct = 0; ct < nvir; ct+=TILESIZE)
        for (int b = bt; b < MIN(bt+TILESIZE,nvir); ++b) {
          for (int c = ct; c < MIN(ct+TILESIZE,nvir); ++c) {
#else
    for (int b = 0; b < nvir; b++) {
      for (int c = 0; c < nvir; c++) {
#endif

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
    }

    *emp4i_ = emp4i;
    *emp4k_ = emp4k;
    *emp5i_ = emp5i;
    *emp5k_ = emp5k;
}
