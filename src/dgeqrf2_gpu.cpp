/*
    -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

       @generated d Wed Oct 24 00:32:47 2012

*/

#include <stdio.h>
#include "common_magma.h"

magma_err_t
magma_dgeqrf2_gpu( magma_int_t m, magma_int_t n,
                   magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
                   double *tau, magma_err_t *info,
		   magma_queue_t queue)
{
/*  -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

    Purpose
    =======
    DGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    dA      (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix dA.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    TAU     (output) DOUBLE_PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -9, internal GPU memory allocation failed.

    Further Details
    ===============

    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).
    =====================================================================    */

    #define dA(a_1,a_2)    dA, (dA_offset + (a_1) + (a_2)*(ldda))
    #define work_ref(a_1)  ( work + (a_1))
    #define hwork          ( work + (nb)*(m))

    magmaDouble_ptr dwork;
    double  *work;

    magma_int_t i, k, ldwork, lddwork, old_i, old_ib, rows;
    magma_int_t nbmin, nx, ib, nb;
    magma_int_t lhwork, lwork;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    k = min(m,n);
    if (k == 0)
        return MAGMA_SUCCESS;

    nb = magma_get_dgeqrf_nb(m);

    lwork  = (m+n) * nb;
    lhwork = lwork - (m)*nb;

    
    if ( MAGMA_SUCCESS != magma_malloc( &dwork, n*nb*sizeof(double))) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    if ( MAGMA_SUCCESS != magma_malloc_host((void**)&work, lwork*sizeof(double)) ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        magma_free( dwork );
        return *info;
    }
    
    magma_event_t event[2] = {NULL, NULL};                                                            

    nbmin = 2;
    nx    = nb;
    ldwork = m;
    lddwork= n;

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code initially */
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nx; i += nb) {
            ib = min(k-i, nb);
            rows = m -i;
	    
	    magma_queue_sync( queue );
	    magma_dgetmatrix_async(rows, ib, dA(i, i), ldda, work_ref(i), 0, ldwork, queue, &event[0]);
          
            if (i>0){
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_dlarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  dA(old_i, old_i         ), ldda, dwork,0,      lddwork,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork,old_ib, lddwork, queue);

		magma_dsetmatrix_async( old_ib, old_ib, work_ref(old_i), 0, ldwork,
					dA(old_i, old_i), ldda, queue, &event[1]); 
            }

	    magma_event_sync(event[0]);
            lapackf77_dgeqrf(&rows, &ib, work_ref(i), &ldwork, tau+i, hwork, &lhwork, info);
   
	    if (i > 0) {
	      magma_event_sync(event[1]);
	    }

	    /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr, 
                              &rows, &ib, 
                              work_ref(i), &ldwork, tau+i, hwork, &ib);

            dpanel_to_q( MagmaUpper, ib, work_ref(i), ldwork, hwork+ib*ib );
	    magma_dsetmatrix(rows, ib, work_ref(i), 0, ldwork, dA(i,i), ldda, queue);
            dq_to_panel( MagmaUpper, ib, work_ref(i), ldwork, hwork+ib*ib );
	    
            if (i + ib < n) 
	      {
		magma_dsetmatrix(ib, ib, hwork, 0, ib, dwork, 0, lddwork, queue);
		
                if (i+nb < k-nx)
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    magma_dlarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib, 
                                      dA(i, i   ), ldda, dwork,0,  lddwork, 
                                      dA(i, i+ib), ldda, dwork,ib, lddwork, queue);
                else {
                    magma_dlarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib, 
                                      dA(i, i   ), ldda, dwork,0,  lddwork, 
                                      dA(i, i+ib), ldda, dwork,ib, lddwork, queue);
		    magma_dsetmatrix(ib, ib, work_ref(i), 0, ldwork, dA(i,i), ldda, queue);
                }
                old_i  = i;
                old_ib = ib;
	      }
        }
    } else {
        i = 0;
    }

    magma_free(dwork);

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
	magma_dgetmatrix(rows, ib, dA(i, i), ldda, work, 0, rows, queue);

        lhwork = lwork - rows*ib;
        lapackf77_dgeqrf(&rows, &ib, work, &rows, tau+i, work+ib*rows, &lhwork, info);
        
	magma_dsetmatrix(rows, ib, work, 0, rows, dA(i, i), ldda, queue);
    }

    magma_free_host(work);

    return *info;
} /* magma_dgeqrf2_gpu */
