/*
    -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

       @generated c Wed Oct 24 00:32:48 2012

*/

#include <stdio.h>
#include "common_magma.h"

magma_err_t
magma_cgetrf_gpu(magma_int_t m, magma_int_t n, 
                 magmaFloatComplex_ptr dA, size_t dA_offset, magma_int_t ldda,
                 magma_int_t *ipiv, magma_int_t *info,
                 magma_queue_t queue )
{
/*  -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

    Purpose
    =======
    CGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDDA     (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -7, internal GPU memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.
    =====================================================================    */

#define inAT(i,j) dAT, dAT_offset + (i)*nb*lddat + (j)*nb

    magmaFloatComplex c_one     = MAGMA_C_MAKE(  1.0, 0.0 );
    magmaFloatComplex c_neg_one = MAGMA_C_MAKE( -1.0, 0.0 );

    magma_int_t iinfo, nb;
    magma_int_t maxm, maxn, mindim;
    magma_int_t i, rows, cols, s, lddat, lddwork;
    
    magmaFloatComplex_ptr dAT, dAP;
    magmaFloatComplex *work;

    magma_err_t err;

    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (m == 0 || n == 0)
        return MAGMA_SUCCESS;

    mindim = min(m, n);
    nb     = magma_get_cgetrf_nb(m);
    s      = mindim / nb;

    if (nb <= 1 || nb >= min(m,n)) 
      {
        // use CPU code
        err = magma_malloc_host( (void**) &work, m*n*sizeof(magmaFloatComplex) );
        if ( err != MAGMA_SUCCESS ) {
          *info = MAGMA_ERR_HOST_ALLOC;
          return *info;
        }

        chk( magma_cgetmatrix( m, n, dA, dA_offset, ldda, work, 0, m, queue ));
        lapackf77_cgetrf(&m, &n, work, &m, ipiv, info);
        chk( magma_csetmatrix( m, n, work, 0, m, dA, dA_offset, ldda, queue ));

        magma_free_host(work);
      }
    else 
      {
        size_t dAT_offset;

        // use hybrid blocked code
        maxm = ((m + 31)/32)*32;
        maxn = ((n + 31)/32)*32;

        lddat   = maxn;
        lddwork = maxm;

        if ( MAGMA_SUCCESS != magma_malloc( &dAP, nb*maxm*sizeof(magmaFloatComplex))) {
          *info = MAGMA_ERR_DEVICE_ALLOC;
          return *info;
        }

        if ((m == n) && (m % 32 == 0) && (ldda%32 == 0))
          {
            dAT = dA;
            dAT_offset = dA_offset;
            magma_cinplace_transpose( dAT, dAT_offset, ldda, lddat, queue );
          }
        else 
          {
            dAT_offset = 0;
            if ( MAGMA_SUCCESS != magma_malloc( &dAT, maxm*maxn*sizeof(magmaFloatComplex))) {
              magma_free( dAP );
              *info = MAGMA_ERR_DEVICE_ALLOC;
              return *info;
            }

            magma_ctranspose2( dAT, dAT_offset, lddat, dA, dA_offset,  ldda, m, n, queue );
        }

        if ( MAGMA_SUCCESS != magma_malloc_host((void**)&work, maxm*nb*sizeof(magmaFloatComplex)) ) {
          magma_free( dAP );
          if (! ((m == n) && (m % 32 == 0) && (ldda%32 == 0)) )
            magma_free( dAT );

          *info = MAGMA_ERR_HOST_ALLOC;
          return *info;
        }

        for( i=0; i<s; i++ )
            {
                // download i-th panel
                cols = maxm - i*nb;
                magma_ctranspose( dAP, 0, cols, inAT(i,i), lddat, nb, cols, queue );
                magma_cgetmatrix(m-i*nb, nb, dAP, 0, cols, work, 0, lddwork, queue);

                if ( i>0 ){
                    magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                                 n - (i+1)*nb, nb, 
                                 c_one, inAT(i-1,i-1), lddat, 
                                 inAT(i-1,i+1), lddat, queue );
                    magma_cgemm( MagmaNoTrans, MagmaNoTrans, 
                                 n-(i+1)*nb, m-i*nb, nb, 
                                 c_neg_one, inAT(i-1,i+1), lddat, 
                                            inAT(i,  i-1), lddat, 
                                 c_one,     inAT(i,  i+1), lddat, queue );
                }

                // do the cpu part
                rows = m - i*nb;
                lapackf77_cgetrf( &rows, &nb, work, &lddwork, ipiv+i*nb, &iinfo);
                if ( (*info == 0) && (iinfo > 0) )
                    *info = iinfo + i*nb;

                magma_cpermute_long2(n, dAT, dAT_offset, lddat, ipiv, nb, i*nb, queue );

                // upload i-th panel
                magma_csetmatrix(m-i*nb, nb, work, 0, lddwork, dAP, 0, maxm, queue);
                magma_ctranspose(inAT(i,i), lddat, dAP, 0, maxm, cols, nb, queue );

                // do the small non-parallel computations
                if ( s > (i+1) ) {
                    magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                                 nb, nb, 
                                 c_one, inAT(i, i  ), lddat,
                                 inAT(i, i+1), lddat, queue);
                    magma_cgemm( MagmaNoTrans, MagmaNoTrans, 
                                 nb, m-(i+1)*nb, nb, 
                                 c_neg_one, inAT(i,   i+1), lddat,
                                            inAT(i+1, i  ), lddat, 
                                 c_one,     inAT(i+1, i+1), lddat, queue );
                }
                else {
                    magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                                 n-s*nb, nb, 
                                 c_one, inAT(i, i  ), lddat,
                                 inAT(i, i+1), lddat, queue);
                    magma_cgemm( MagmaNoTrans, MagmaNoTrans, 
                                 n-(i+1)*nb, m-(i+1)*nb, nb,
                                 c_neg_one, inAT(i,   i+1), lddat,
                                            inAT(i+1, i  ), lddat, 
                                 c_one,     inAT(i+1, i+1), lddat, queue );
                }
            }

        magma_int_t nb0 = min(m - s*nb, n - s*nb);
        rows = m - s*nb;
        cols = maxm - s*nb;

        magma_ctranspose2( dAP, 0, maxm, inAT(s,s), lddat, nb0, rows, queue);
        magma_cgetmatrix(rows, nb0, dAP, 0, maxm, work, 0, lddwork, queue);

        // do the cpu part
        lapackf77_cgetrf( &rows, &nb0, work, &lddwork, ipiv+s*nb, &iinfo);
        if ( (*info == 0) && (iinfo > 0) )
            *info = iinfo + s*nb;
        magma_cpermute_long2(n, dAT, dAT_offset, lddat, ipiv, nb0, s*nb, queue );

        // upload i-th panel
        magma_csetmatrix(rows, nb0, work, 0, lddwork, dAP, 0, maxm, queue);
        magma_ctranspose2( inAT(s,s), lddat, dAP, 0, maxm, rows, nb0, queue );

        magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                     n-s*nb-nb0, nb0,
                     c_one, inAT(s,s),     lddat, 
                     inAT(s,s)+nb0, lddat, queue);

        if ((m == n) && (m % 32 == 0) && (ldda%32 == 0)) {
          magma_cinplace_transpose( dAT, dAT_offset, lddat, ldda, queue );
        }
        else {
          magma_ctranspose2( dA, dA_offset, ldda, dAT, dAT_offset, lddat, n, m, queue );
          magma_free( dAT );
        }

        magma_free( dAP );
        magma_free_host( work );
    }

    return *info;
    /* End of MAGMA_CGETRF_GPU */
}

#undef inAT
