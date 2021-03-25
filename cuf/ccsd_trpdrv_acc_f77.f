      subroutine ccsd_trpdrv_acc_fbody(
     &     f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,eorb,
     &     ncor,nocc,nvir, emp4,emp5,
     &     a,i,j,k,klo,
     &     Tij, Tkj, Tia, Tka, Xia, Xka,
     &     Jia, Jka, Kia, Kka,
     &     Jij, Jkj, Kij, Kkj,
     &     dintc1,dintx1,t1v1,
     &     dintc2,dintx2,t1v2)
      implicit none
      double precision emp4,emp5
      integer ncor,nocc,nvir
      integer a,i,j,k, klo
      double precision f1n(nvir,nvir),f1t(nvir,nvir)
      double precision f2n(nvir,nvir),f2t(nvir,nvir)
      double precision f3n(nvir,nvir),f3t(nvir,nvir)
      double precision f4n(nvir,nvir),f4t(nvir,nvir)
      double precision eorb(*)
      double precision Tij(*), Tkj(*), Tia(*), Tka(*)
      double precision Xia(*), Xka(*)
      double precision Jia(*), Jka(*), Jij(*), Jkj(*)
      double precision Kia(*), Kka(*), Kij(*), Kkj(*)
      double precision dintc1(nvir),dintx1(nvir)
      double precision dintc2(nvir),dintx2(nvir)
      double precision t1v1(nvir),t1v2(nvir)
      double precision emp4i,emp5i,emp4k,emp5k
      double precision eaijk,denom
      integer lnov,lnvv
      ! chunking is the loop blocking size in the loop nest
      ! formerly associated with the tengy routine.
      ! we have not explored this paramater space but 32 is
      ! optimal for TLB blocking in matrix transpose on most
      ! architectures (especially x86).
      integer b,c,bb,cc
      integer chunking
      chunking = 32
      lnov=nocc*nvir
      lnvv=nvir*nvir
      emp4i = 0.0d0
      emp5i = 0.0d0
      emp4k = 0.0d0
      emp5k = 0.0d0

      !print*,'SKIPPING DGEMM'
      !if (.false.) then
      if (.true.) then

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Jia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f1n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tia,nvir,Kkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f1n,nvir)

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Kia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f2n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xia,nvir,Kkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f2n,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Jia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f3n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tia,nvir,Jkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f3n,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Kia,nvir,Tkj(1+(k-klo)*lnvv),nvir,0.0d0,
     2     f4n,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xia,nvir,Jkj(1+(k-klo)*lnov),nocc,1.0d0,
     2     f4n,nvir)

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Jka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f1t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tka(1+(k-klo)*lnov),nvir,Kij,nocc,1.0d0,
     2     f1t,nvir)

      call dgemm('n','t',nvir,nvir,nvir,1.0d0,
     1     Kka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f2t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xka(1+(k-klo)*lnov),nvir,Kij,nocc,1.0d0,
     2     f2t,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Jka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f3t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Tka(1+(k-klo)*lnov),nvir,Jij,nocc,1.0d0,
     2     f3t,nvir)

      call dgemm('n','n',nvir,nvir,nvir,1.0d0,
     1     Kka(1+(k-klo)*lnvv),nvir,Tij,nvir,0.0d0,
     2     f4t,nvir)
      call dgemm('n','n',nvir,nvir,nocc,-1.0d0,
     1     Xka(1+(k-klo)*lnov),nvir,Jij,nocc,1.0d0,
     2     f4t,nvir)

      endif

      eaijk = eorb(a) - ( eorb(ncor+i)+eorb(ncor+j)+eorb(ncor+k) )

      call ccsd_tengy_omp(f1n,f1t,f2n,f2t,f3n,f3t,f4n,f4t,
     &                    dintc1,dintc2,dintx1,dintx2,
     &                    t1v1,t1v2,eorb,eaijk,
     &                    emp4i,emp5i,emp4k,emp5k,
     &                    ncor,nocc,nvir)

      emp4 = emp4 + emp4i
      emp5 = emp5 + emp5i
      if (i.ne.k) then
          emp4 = emp4 + emp4k
          emp5 = emp5 + emp5k
      end if ! (i.ne.k)
      end
