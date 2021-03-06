/*
     -- clMAGMA (version 1.0.0) --
        Univ. of Tennessee, Knoxville
        Univ. of California, Berkeley
        Univ. of Colorado, Denver
        April 2012
  
        @generated s Wed Oct 24 00:32:47 2012

*/

#include <stdio.h>
#include "common_magma.h"

magma_err_t
magma_spotrf_gpu( magma_uplo_t   uplo, magma_int_t    n,
		  magmaFloat_ptr dA, size_t dA_offset, magma_int_t ldda,
		  magma_err_t*   info, magma_queue_t queue )
{
/*  -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

    Purpose   
    =======   
    SPOTRF computes the Cholesky factorization of a real symmetric   
    positive definite matrix dA.   

    The factorization has the form   
       dA = U\*\*H * U,  if UPLO = 'U', or   
       dA = L  * L\*\*H,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   
    UPLO    (input) INTEGER
            = MagmaUpper:  Upper triangle of dA is stored;   
            = MagmaLower:  Lower triangle of dA is stored.   

    N       (input) INTEGER   
            The order of the matrix dA.  N >= 0.   

    dA      (input/output) REAL array on the GPU, dimension (LDDA,N)   
            On entry, the symmetric matrix dA.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of dA contains the upper   
            triangular part of the matrix dA, and the strictly lower   
            triangular part of dA is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of dA contains the lower   
            triangular part of the matrix dA, and the strictly upper   
            triangular part of dA is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization dA = U\*\*H*U or dA = L*L\*\*H.   

    LDDA    (input) INTEGER   
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   
    =====================================================================   */

// produces pointer and offset as two args to magmaBLAS routines 
#define dA(i,j)  dA, ( (dA_offset) + (i) + (j)*ldda )

// produces pointer as single arg to BLAS routines   
#define A(i,j)  &A[ (i) + (j)*lda ]

    magma_int_t j, jb, nb;
    float  z_one = MAGMA_S_MAKE(  1.0, 0.0 );
    float mz_one = MAGMA_S_MAKE( -1.0, 0.0 );
    float    one =  1.0;
    float  m_one = -1.0;
    float* work;
    magma_err_t err;
    
    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        *info = -1;
    } else if ( n < 0 ) {
        *info = -2;
    } else if ( ldda < max(1,n) ) {
        *info = -4;
    }
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    nb = magma_get_spotrf_nb( n );
    
    err = magma_malloc_host( (void**) &work, nb*nb*sizeof(float) );
    if ( err != MAGMA_SUCCESS ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    magma_event_t event = NULL;
    
    if ((nb <= 1) || (nb >= n)) {
        // use unblocked code
        chk( magma_sgetmatrix( n, n, dA, dA_offset, ldda, work, 0, n, queue ));
        lapackf77_spotrf( lapack_uplo_const(uplo), &n, work, &n, info );
        chk( magma_ssetmatrix( n, n, work, 0, n, dA, dA_offset, ldda, queue ));
    }
    else {
        if ( uplo == MagmaUpper ) {
            // --------------------
            // compute Cholesky factorization A = U'*U
            // using the left looking algorithm
            for( j = 0; j < n; j += nb ) {
                // apply all previous updates to diagonal block
                jb = min( nb, n-j );
                if ( j > 0 ) {
                    chk( magma_ssyrk( MagmaUpper, MagmaTrans, jb, j,
                        m_one, dA(0,j), ldda,
                          one, dA(j,j), ldda, queue ));
                }
                
		// start asynchronous data transfer 
		chk( magma_sgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, &event ));

                // apply all previous updates to block row right of diagonal block
                if ( j+jb < n ) {
                    chk( magma_sgemm( MagmaTrans, MagmaNoTrans,
                        jb, n-j-jb, j,
                        mz_one, dA(0, j   ), ldda,
                                dA(0, j+jb), ldda,
                        z_one,  dA(j, j+jb), ldda, queue ));
                }
                
                // simultaneous with above sgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                chk( magma_event_sync( event )); 
                lapackf77_spotrf( MagmaUpperStr, &jb, work, &jb, info );
                if ( *info != 0 ) {
                    assert( *info > 0 );
                    *info += j;
                    break;
                }
                chk( magma_ssetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, &event ));
                
                // apply diagonal block to block row right of diagonal block
                if ( j+jb < n ) {
  		    chk( magma_event_sync( event ));
                    chk( magma_strsm(
                        MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                        jb, n-j-jb,
                        z_one, dA(j, j),    ldda,
                               dA(j, j+jb), ldda, queue ));
                }
            }
        }
        else {
            // --------------------
            // compute Cholesky factorization A = L*L'
            // using the left looking algorithm
            for( j = 0; j < n; j += nb ) {
                // apply all previous updates to diagonal block
                jb = min( nb, n-j );
                chk( magma_ssyrk( MagmaLower, MagmaNoTrans, jb, j,
                    m_one, dA(j, 0), ldda,
                      one, dA(j, j), ldda, queue ));
                
		// start asynchronous data transfer
                chk( magma_sgetmatrix_async( jb, jb, dA(j,j), ldda, work, 0, jb, queue, &event ));

                // apply all previous updates to block column below diagonal block
                if ( j+jb < n ) {
                    chk( magma_sgemm( MagmaNoTrans, MagmaTrans,
                        n-j-jb, jb, j,
                        mz_one, dA(j+jb, 0), ldda,
                                dA(j,    0), ldda,
                        z_one,  dA(j+jb, j), ldda, queue ));
                }
                
                // simultaneous with above sgemm, transfer data, factor
                // diagonal block on CPU, and test for positive definiteness
                chk( magma_event_sync( event ));
                lapackf77_spotrf( MagmaLowerStr, &jb, work, &jb, info );
                if ( *info != 0 ) {
                    assert( *info > 0 );
                    *info += j;
                    break;
                }
                chk( magma_ssetmatrix_async( jb, jb, work, 0, jb, dA(j,j), ldda, queue, &event ));
                
                // apply diagonal block to block column below diagonal
                if ( j+jb < n ) {
	 	    chk( magma_event_sync( event ));
                    chk( magma_strsm(
                        MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                        n-j-jb, jb,
                        z_one, dA(j, j   ), ldda,
                               dA(j+jb, j), ldda, queue ));
                }
            }
        }
    }
    
    chk( magma_queue_sync( queue ));
    chk( magma_free_host( work ));
    
    return *info;
}
