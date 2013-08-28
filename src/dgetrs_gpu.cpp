/*  
    -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville                                                                           
       Univ. of California, Berkeley                                                                           
       Univ. of Colorado, Denver                                                                               
       April 2012                                                                                              
                                                                                                              
       @generated d Wed Oct 24 00:32:49 2012
*/

#include <stdio.h>
#include "common_magma.h"


extern "C" magma_err_t
magma_dgetrs_gpu(magma_trans_t trans, magma_int_t n, magma_int_t nrhs, 
                 magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
                 magma_int_t *ipiv, 
                 magmaDouble_ptr dB, size_t dB_offset, magma_int_t lddb, 
                 magma_int_t *info, magma_queue_t queue)
{
/*  -- clMagma (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

    Purpose
    =======

    Solves a system of linear equations
      A * X = B  or  A' * X = B
    with a general N-by-N matrix A using the LU factorization computed by DGETRF_GPU.

    Arguments
    =========

    TRANS   (input) CHARACTER*1
            Specifies the form of the system of equations:
            = 'N':  A * X = B  (No transpose)
            = 'T':  A'* X = B  (Transpose)
            = 'C':  A'* X = B  (Conjugate transpose = Transpose)

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    A       (input) DOUBLE_PRECISION array on the GPU, dimension (LDA,N)
            The factors L and U from the factorization A = P*L*U as computed
            by DGETRF_GPU.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    IPIV    (input) INTEGER array, dimension (N)
            The pivot indices from DGETRF; for 1<=i<=N, row i of the
            matrix was interchanged with row IPIV(i).

    B       (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    HWORK   (workspace) DOUBLE_PRECISION array, dimension N*NRHS
    =====================================================================    */


    double z_one = MAGMA_D_MAKE( 1.0, 0.0 );
    double *work = NULL;
    magma_trans_t            trans_ = trans;
    long int    notran = lapackf77_lsame(lapack_const(trans_), lapack_const(MagmaNoTrans));
    magma_int_t i1, i2, inc;

    *info = 0;
    if ( (! notran) && 
         (! lapackf77_lsame(lapack_const(trans_), lapack_const(MagmaTrans))) && 
         (! lapackf77_lsame(lapack_const(trans_), lapack_const(MagmaTrans))) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (lddb < max(1,n)) {
        *info = -8;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }
	
	work=(double*)malloc( n*nrhs*sizeof(double) );
    if ( !work ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
      
    i1 = 1;
    i2 = n;
    if (notran) {
        inc = 1;

        /* Solve A * X = B. */
		magma_dgetmatrix( n, nrhs, dB, dB_offset, lddb, work, 0, n, queue );
        lapackf77_dlaswp(&nrhs, work, &n, &i1, &i2, ipiv, &inc);
		magma_dsetmatrix( n, nrhs, work, 0, n, dB, dB_offset, lddb, queue );

        if ( nrhs == 1) {
            chk(magma_dtrsv(MagmaLower, MagmaNoTrans, MagmaUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue));
            chk(magma_dtrsv(MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue));
        } else {
            chk(magma_dtrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
            chk(magma_dtrsm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
        }
    } else {
        inc = -1;

        /* Solve A' * X = B. */
        if ( nrhs == 1) {
            chk(magma_dtrsv(MagmaUpper, trans, MagmaNonUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue ));
            chk(magma_dtrsv(MagmaLower, trans, MagmaUnit, n, dA, dA_offset, ldda, dB, dB_offset, 1, queue ));
        } else {
			chk(magma_dtrsm(MagmaLeft, MagmaUpper, trans, MagmaNonUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
            chk(magma_dtrsm(MagmaLeft, MagmaLower, trans, MagmaUnit, n, nrhs, z_one, dA, dA_offset, ldda, dB, dB_offset, lddb, queue));
        }

		magma_dgetmatrix( n, nrhs, dB, dB_offset, lddb, work, 0, n, queue );
        lapackf77_dlaswp(&nrhs, work, &n, &i1, &i2, ipiv, &inc);
        magma_dsetmatrix( n, nrhs, work, 0, n, dB, dB_offset, lddb, queue );
    }
    free(work);

    return *info;
}

