!
!   -- MAGMA (version 1.0.0) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      October 2012
!
!   @generated d Wed Oct 24 00:32:43 2012
!

#define PRECISION_d

module magma_dfortran

  use magma_param, only: sizeof_double

  implicit none

  !---- Fortran interfaces to MAGMA subroutines ----
  interface

     subroutine magmaf_dgetptr( m, n, A, lda, d, e,tauq, taup, work, lwork, info)
       integer       :: m
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       double precision:: d(*)
       double precision:: e(*)
       double precision    :: tauq(*)
       double precision    :: taup(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgetptr


     subroutine magmaf_dgebrd( m, n, A, lda, d, e,tauq, taup, work, lwork, info)
       integer       :: m
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       double precision:: d(*)
       double precision:: e(*)
       double precision    :: tauq(*)
       double precision    :: taup(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgebrd

     subroutine magmaf_dgehrd2(n, ilo, ihi,A, lda, tau, work, lwork, info)
       integer       :: n
       integer       :: ilo
       integer       :: ihi
       double precision    :: A(*)
       integer       :: lda
       double precision    :: tau(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgehrd2

     subroutine magmaf_dgehrd(n, ilo, ihi,A, lda, tau, work, lwork, d_T, info)
       integer       :: n
       integer       :: ilo
       integer       :: ihi
       double precision    :: A(*)
       integer       :: lda
       double precision    :: tau(*)
       double precision    :: work(*)
       integer       :: lwork
       double precision    :: d_T(*)
       integer       :: info
     end subroutine magmaf_dgehrd

     subroutine magmaf_dgelqf( m, n, A,    lda,   tau, work, lwork, info)
       integer       :: m
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       double precision    :: tau(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgelqf

     subroutine magmaf_dgeqlf( m, n, A,    lda,   tau, work, lwork, info)
       integer       :: m
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       double precision    :: tau(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgeqlf

     subroutine magmaf_dgeqrf( m, n, A, lda, tau, work, lwork, info)
       integer       :: m
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       double precision    :: tau(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgeqrf

     subroutine magmaf_dgesv(  n, nrhs, A, lda, ipiv, B, ldb, info)
       integer       :: n
       integer       :: nrhs
       double precision    :: A
       integer       :: lda
       integer       :: ipiv(*)
       double precision    :: B
       integer       :: ldb
       integer       :: info
     end subroutine magmaf_dgesv

     subroutine magmaf_dgetrf( m, n, A, lda, ipiv, info)
       integer       :: m
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       integer       :: ipiv(*)
       integer       :: info
     end subroutine magmaf_dgetrf

     subroutine magmaf_dposv(  uplo, n, nrhs, dA, ldda, dB, lddb, info)
       character     :: uplo
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       magma_devptr_t:: dB
       integer       :: lddb
       integer       :: info
     end subroutine magmaf_dposv
     
     subroutine magmaf_dpotrf( uplo, n, A, lda, info)
       character          :: uplo
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       integer       :: info
     end subroutine magmaf_dpotrf

     subroutine magmaf_dsytrd( uplo, n, A, lda, d, e, tau, work, lwork, info)
       character          :: uplo
       integer       :: n
       double precision    :: A(*)
       integer       :: lda
       double precision:: d(*)
       double precision:: e(*)
       double precision    :: tau(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dsytrd

     subroutine magmaf_dormqr( side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info)
       character          :: side
       character          :: trans
       integer       :: m
       integer       :: n
       integer       :: k
       double precision    :: a(*)
       integer       :: lda
       double precision    :: tau(*)
       double precision    :: c(*)
       integer       :: ldc
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dormqr

     subroutine magmaf_dormtr( side, uplo, trans, m, n, a, lda,tau,c,    ldc,work, lwork,info)
       character          :: side
       character          :: uplo
       character          :: trans
       integer       :: m
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision    :: tau(*)
       double precision    :: c(*)
       integer       :: ldc
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dormtr
#if defined(PRECISION_z) || defined(PRECISION_c)

     subroutine magmaf_dgeev( jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info)
       character          :: jobvl
       character          :: jobvr
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision    :: w(*)
       double precision    :: vl(*)
       integer       :: ldvl
       double precision    :: vr(*)
       integer       :: ldvr
       double precision    :: work(*)
       integer       :: lwork
       double precision:: rwork(*)
       integer       :: info
     end subroutine magmaf_dgeev

     subroutine magmaf_dgesvd( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info)
       character          :: jobu
       character          :: jobvt
       integer       :: m
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision:: s(*)
       double precision    :: u(*)
       integer       :: ldu
       double precision    :: vt(*)
       integer       :: ldvt
       double precision    :: work(*)
       integer       :: lwork
       double precision:: rwork(*)
       integer       :: info
     end subroutine magmaf_dgesvd

     subroutine magmaf_dsyevd( jobz, uplo, n, a, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)
       character     :: jobz
       character     :: uplo
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision:: w(*)
       double precision    :: work(*)
       integer       :: lwork
       double precision:: rwork(*)
       integer       :: lrwork
       integer       :: iwork(*)
       integer       :: liwork
       integer       :: info
     end subroutine magmaf_dsyevd

     subroutine magmaf_dsygvd( itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, lrwork, iwork, liwork, info)
       integer       :: itype
       character     :: jobz
       character     :: uplo
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision    :: b(*)
       integer       :: ldb
       double precision:: w(*)
       double precision    :: work(*)
       integer       :: lwork
       double precision:: rwork(*)
       integer       :: lrwork
       integer       :: iwork(*)
       integer       :: liwork
       integer       :: info
     end subroutine magmaf_dsygvd

#else
     subroutine magmaf_dgeev( jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
       character          :: jobvl
       character          :: jobvr
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision    :: wr(*)
       double precision    :: wi(*)
       double precision    :: vl(*)
       integer       :: ldvl
       double precision    :: vr(*)
       integer       :: ldvr
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgeev

     subroutine magmaf_dgesvd( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
       character          :: jobu
       character          :: jobvt
       integer       :: m
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision:: s(*)
       double precision    :: u(*)
       integer       :: ldu
       double precision    :: vt(*)
       integer       :: ldvt
       double precision    :: work(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgesvd

     subroutine magmaf_dsyevd( jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
       character          :: jobz
       character          :: uplo
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision:: w(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: iwork(*)
       integer       :: liwork
       integer       :: info
     end subroutine magmaf_dsyevd

     subroutine magmaf_dsygvd( itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, iwork, liwork, info)
       integer       :: itype
       character     :: jobz
       character     :: uplo
       integer       :: n
       double precision    :: a(*)
       integer       :: lda
       double precision    :: b(*)
       integer       :: ldb
       double precision:: w(*)
       double precision    :: work(*)
       integer       :: lwork
       integer       :: iwork(*)
       integer       :: liwork
       integer       :: info
     end subroutine magmaf_dsygvd
#endif

     subroutine magmaf_dgels_gpu(  trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info)
       character          :: trans
       integer       :: m
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       magma_devptr_t:: dB
       integer       :: lddb
       double precision    :: hwork(*)
       integer       :: lwork
       integer       :: info
     end subroutine magmaf_dgels_gpu

     subroutine magmaf_dgeqrf_gpu( m, n, dA, ldda, tau, dT, info)
       integer       :: m
       integer       :: n
       magma_devptr_t:: dA
       integer       :: ldda
       double precision    :: tau(*)
       magma_devptr_t:: dT
       integer       :: info
     end subroutine magmaf_dgeqrf_gpu

     subroutine magmaf_dgeqrf2_gpu(m, n, dA, ldda, tau, info)
       integer       :: m
       integer       :: n
       magma_devptr_t:: dA
       integer       :: ldda
       double precision    :: tau(*)
       integer       :: info
     end subroutine magmaf_dgeqrf2_gpu

     subroutine magmaf_dgeqrf3_gpu(m, n, dA, ldda, tau, dT, info)
       integer       :: m
       integer       :: n
       magma_devptr_t:: dA
       integer       :: ldda
       double precision    :: tau(*)
       magma_devptr_t:: dT
       integer       :: info
     end subroutine magmaf_dgeqrf3_gpu

     subroutine magmaf_dgeqrs_gpu( m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lhwork, info)
       integer       :: m
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       double precision    :: tau
       magma_devptr_t:: dT
       magma_devptr_t:: dB
       integer       :: lddb
       double precision    :: hwork(*)
       integer       :: lhwork
       integer       :: info
     end subroutine magmaf_dgeqrs_gpu

     subroutine magmaf_dgeqrs3_gpu( m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lhwork, info)
       integer       :: m
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       double precision    :: tau
       magma_devptr_t:: dT
       magma_devptr_t:: dB
       integer       :: lddb
       double precision    :: hwork(*)
       integer       :: lhwork
       integer       :: info
     end subroutine magmaf_dgeqrs3_gpu

     subroutine magmaf_dgessm_gpu( storev, m, n, k, ib, ipiv, dL1, lddl1, dL,  lddl, dA,  ldda, info)
       character          :: storev
       integer       :: m
       integer       :: n
       integer       :: k
       integer       :: ib
       integer       :: ipiv(*)
       magma_devptr_t:: dL1
       integer       :: lddl1
       magma_devptr_t:: dL
       integer       :: lddl
       magma_devptr_t:: dA
       integer       :: ldda
       integer       :: info
     end subroutine magmaf_dgessm_gpu

     subroutine magmaf_dgesv_gpu(  n, nrhs, dA, ldda, ipiv, dB, lddb, info)
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       integer       :: ipiv(*)
       magma_devptr_t:: dB
       integer       :: lddb
       integer       :: info
     end subroutine magmaf_dgesv_gpu

     subroutine magmaf_dgetrf_gpu( m, n, dA, ldda, ipiv, info)
       integer       :: m
       integer       :: n
       magma_devptr_t:: dA
       integer       :: ldda
       integer       :: ipiv(*)
       integer       :: info
     end subroutine magmaf_dgetrf_gpu

     subroutine magmaf_dgetrs_gpu( trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
       character          :: trans
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       integer       :: ipiv(*)
       magma_devptr_t:: dB
       integer       :: lddb
       integer       :: info
     end subroutine magmaf_dgetrs_gpu

     subroutine magmaf_dlabrd_gpu( m, n, nb, a, lda, da, ldda, d, e, tauq, taup, x, ldx, dx, lddx, y, ldy, dy, lddy)
       integer       :: m
       integer       :: n
       integer       :: nb
       double precision    :: a(*)
       integer       :: lda
       magma_devptr_t:: da
       integer       :: ldda
       double precision:: d(*)
       double precision:: e(*)
       double precision    :: tauq(*)
       double precision    :: taup(*)
       double precision    :: x(*)
       integer       :: ldx
       magma_devptr_t:: dx
       integer       :: lddx
       double precision    :: y(*)
       integer       :: ldy
       magma_devptr_t:: dy
       integer       :: lddy
     end subroutine magmaf_dlabrd_gpu

     subroutine magmaf_dlarfb_gpu( side, trans, direct, storev, m, n, k, dv, ldv, dt, ldt, dc, ldc, dowrk, ldwork)
       character          :: side
       character          :: trans
       character          :: direct
       character          :: storev
       integer       :: m
       integer       :: n
       integer       :: k
       magma_devptr_t:: dv
       integer       :: ldv
       magma_devptr_t:: dt
       integer       :: ldt
       magma_devptr_t:: dc
       integer       :: ldc
       magma_devptr_t:: dowrk
       integer       :: ldwork
     end subroutine magmaf_dlarfb_gpu

     subroutine magmaf_dposv_gpu(  uplo, n, nrhs, dA, ldda, dB, lddb, info)
       character          :: uplo
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       magma_devptr_t:: dB
       integer       :: lddb
       integer       :: info
     end subroutine magmaf_dposv_gpu

     subroutine magmaf_dpotrf_gpu( uplo, n, dA, ldda, info)
       character          :: uplo
       integer       :: n
       magma_devptr_t:: dA
       integer       :: ldda
       integer       :: info
     end subroutine magmaf_dpotrf_gpu

     subroutine magmaf_dpotrs_gpu( uplo,  n, nrhs, dA, ldda, dB, lddb, info)
       character          :: uplo
       integer       :: n
       integer       :: nrhs
       magma_devptr_t:: dA
       integer       :: ldda
       magma_devptr_t:: dB
       integer       :: lddb
       integer       :: info
     end subroutine magmaf_dpotrs_gpu

     subroutine magmaf_dssssm_gpu( storev, m1, n1, m2, n2, k, ib, dA1, ldda1, dA2, ldda2, dL1, lddl1, dL2, lddl2, IPIV, info)
       character          :: storev
       integer       :: m1
       integer       :: n1
       integer       :: m2
       integer       :: n2
       integer       :: k
       integer       :: ib
       magma_devptr_t:: dA1
       integer       :: ldda1
       magma_devptr_t:: dA2
       integer       :: ldda2
       magma_devptr_t:: dL1
       integer       :: lddl1
       magma_devptr_t:: dL2
       integer       :: lddl2
       integer       :: IPIV(*)
       integer       :: info
     end subroutine magmaf_dssssm_gpu

     subroutine magmaf_dorgqr_gpu( m, n, k, da, ldda, tau, dwork, nb, info)
       integer       :: m
       integer       :: n
       integer       :: k
       magma_devptr_t:: da
       integer       :: ldda
       double precision    :: tau(*)
       magma_devptr_t:: dwork
       integer       :: nb
       integer       :: info
     end subroutine magmaf_dorgqr_gpu

     subroutine magmaf_dormqr_gpu( side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, td, nb, info)
       character          :: side
       character          :: trans
       integer       :: m
       integer       :: n
       integer       :: k
       magma_devptr_t:: a
       integer       :: lda
       double precision    :: tau(*)
       magma_devptr_t:: c
       integer       :: ldc
       magma_devptr_t:: work
       integer       :: lwork
       magma_devptr_t:: td
       integer       :: nb
       integer       :: info
     end subroutine magmaf_dormqr_gpu

  end interface

contains
  
  subroutine magmaf_doff1d( ptrNew, ptrOld, inc, i)
    magma_devptr_t :: ptrNew
    magma_devptr_t :: ptrOld
    integer        :: inc, i
    
    ptrNew = ptrOld + (i-1) * inc * sizeof_double
    
  end subroutine magmaf_doff1d
  
  subroutine magmaf_doff2d( ptrNew, ptrOld, lda, i, j)
    magma_devptr_t :: ptrNew
    magma_devptr_t :: ptrOld
    integer        :: lda, i, j
    
    ptrNew = ptrOld + ((j-1) * lda + (i-1)) * sizeof_double
    
  end subroutine magmaf_doff2d
  
end module magma_dfortran
