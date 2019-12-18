      subroutine ccsd_tengy_omp(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,
     &                          dintc1,dintc2,dintx1,dintx2,
     &                          t1v1,t1v2,eorb,eaijk,
     &                          emp4i,emp5i,emp4k,emp5k,
     &                          ncor,nocc,nvir)
      implicit none
      integer nocc,ncor,nvir
      integer b,c,bb,cc
      double precision emp4i,emp5i
      double precision emp4k,emp5k
      double precision f1n(nvir,nvir),f1t(nvir,nvir)
      double precision f2n(nvir,nvir),f2t(nvir,nvir)
      double precision f3n(nvir,nvir),f3t(nvir,nvir)
      double precision f4n(nvir,nvir),f4t(nvir,nvir)
      double precision dintc1(nvir),dintx1(nvir)
      double precision dintc2(nvir),dintx2(nvir)
      double precision t1v1(nvir),t1v2(nvir)
      double precision eorb(ncor+nocc+nvir)
      double precision eaijk,denom
      integer chunking
      chunking = 32
!$omp target map(to: f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t )
!$omp&       map(to: dintc1, dintc2, dintx1, dintx2, t1v1, t1v2 )
!$omp&       map(to: eorb, ncor, nocc, nvir, eaijk)
!$omp&       map(tofrom: emp5i, emp4i, emp5k, emp4k)
!$omp parallel do collapse(2)
!$omp& schedule(static)
!$omp& shared(eorb,eaijk)
!$omp& shared(f1n,f2n,f3n,f4n,f1t,f2t,f3t,f4t)
!$omp& shared(t1v1,dintc1,dintx1)
!$omp& shared(t1v2,dintc2,dintx2)
!$omp& private(denom)
!$omp& firstprivate(ncor,nocc,nvir)
!$omp& reduction(+:emp5i,emp4i)
!$omp& reduction(+:emp5k,emp4k)
      do b=1,nvir
        do c=1,nvir
          denom=-1.0d0/(eorb(ncor+nocc+b)+eorb(ncor+nocc+c)+eaijk)
          emp4i=emp4i+denom*
     &         (f1t(b,c)+f1n(c,b)+f2t(c,b)+f3n(b,c)+f4n(c,b))*
     &         (f1t(b,c)-2*f2t(b,c)-2*f3t(b,c)+f4t(b,c))
     &               -denom*
     &         (f1n(b,c)+f1t(c,b)+f2n(c,b)+f3n(c,b))*
     &         (2*f1t(b,c)-f2t(b,c)-f3t(b,c)+2*f4t(b,c))
     &               +3*denom*(
     &         f1n(b,c)*(f1n(b,c)+f3n(c,b)+2*f4t(c,b))+
     &         f2n(b,c)*f2t(c,b)+f3n(b,c)*f4t(b,c))
          emp4k=emp4k+denom*
     &         (f1n(b,c)+f1t(c,b)+f2n(c,b)+f3t(b,c)+f4t(c,b))*
     &         (f1n(b,c)-2*f2n(b,c)-2*f3n(b,c)+f4n(b,c))
     &               -denom*
     &         (f1t(b,c)+f1n(c,b)+f2t(c,b)+f3t(c,b))*
     &         (2*f1n(b,c)-f2n(b,c)-f3n(b,c)+2*f4n(b,c))
     &               +3*denom*(
     &         f1t(b,c)*(f1t(b,c)+f3t(c,b)+2*f4n(c,b))+
     &         f2t(b,c)*f2n(c,b)+f3t(b,c)*f4n(b,c))
          emp5i=emp5i+denom*t1v1(b)*dintx1(c)*
     &        (    f1t(b,c)+f2n(b,c)+f4n(c,b)
     &         -2*(f3t(b,c)+f4n(b,c)+f2n(c,b)+
     &             f1n(b,c)+f2t(b,c)+f3n(c,b))
     &         +4*(f3n(b,c)+f4t(b,c)+f1n(c,b)))
     &               +denom*t1v1(b)*dintc1(c)*
     &        (     f1n(b,c)+f4n(b,c)+f1t(c,b)
     &          -2*(f2n(b,c)+f3n(b,c)+f2t(c,b)))
          emp5k=emp5k+denom*t1v2(b)*dintx2(c)*
     &        (    f1n(b,c)+f2t(b,c)+f4t(c,b)
     &         -2*(f3n(b,c)+f4t(b,c)+f2t(c,b)+
     &             f1t(b,c)+f2n(b,c)+f3t(c,b))
     &         +4*(f3t(b,c)+f4n(b,c)+f1t(c,b)))
     &               +denom*t1v2(b)*dintc2(c)*
     &        (     f1t(b,c)+f4t(b,c)+f1n(c,b)
     &          -2*(f2t(b,c)+f3t(b,c)+f2n(c,b)))
        enddo
      enddo
!$omp end parallel do
!$omp end target
      return
      end
