/*
   -- clMAGMA (version 1.0.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   August 2012

   @generated d Wed Oct 24 00:32:51 2012

 */

#include <stdio.h>
#include "common_magma.h"

#define A(i, j)  a, (offset_a + (j)*lda  + (i))

extern "C" magma_int_t
magma_dpotri_gpu(magma_uplo_t uplo, magma_int_t n,
		magmaDouble_ptr a, size_t offset_a, magma_int_t lda, magma_int_t *info, magma_queue_t queue)
{
	/*  -- MAGMA (version 1.0.0) --
		Univ. of Tennessee, Knoxville
		Univ. of California, Berkeley
		Univ. of Colorado, Denver
		August 2012

		Purpose
		=======

		DPOTRI computes the inverse of a real symmetric positive definite
		matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
		computed by DPOTRF.

		Arguments
		=========

		UPLO    (input) CHARACTER*1
		= 'U':  Upper triangle of A is stored;
		= 'L':  Lower triangle of A is stored.

		N       (input) INTEGER
		The order of the matrix A.  N >= 0.

		A       (input/output) DOUBLE_PRECISION array, dimension (LDA,N)
		On entry, the triangular factor U or L from the Cholesky
		factorization A = U**T*U or A = L*L**T, as computed by
		DPOTRF.
		On exit, the upper or lower triangle of the (symmetric)
		inverse of A, overwriting the input factor U or L.

		LDA     (input) INTEGER
		The leading dimension of the array A.  LDA >= max(1,N).
		INFO    (output) INTEGER
		= 0:  successful exit
		< 0:  if INFO = -i, the i-th argument had an illegal value
		> 0:  if INFO = i, the (i,i) element of the factor U or L is
		zero, and the inverse could not be computed.

		===================================================================== */

	/* Local variables */
	magma_uplo_t uplo_ = uplo;

	*info = 0;
	if ((! lapackf77_lsame(lapack_const(uplo_), lapack_const(MagmaUpper))) && (! lapackf77_lsame(lapack_const(uplo_), lapack_const(MagmaLower))))
		*info = -1;
	else if (n < 0)
		*info = -2;
	else if (lda < max(1,n))
		*info = -4;

	if (*info != 0) {
		magma_xerbla( __func__, -(*info) );
		return *info;
	}

	/* Quick return if possible */
	if ( n == 0 )
		return *info;

	/* Invert the triangular Cholesky factor U or L */
	magma_dtrtri_gpu( uplo, MagmaNonUnit, n, a, offset_a, lda, info );
	
	if ( *info == 0 ) {
		/* Form inv(U) * inv(U)**T or inv(L)**T * inv(L) */
		magma_dlauum_gpu( uplo, n, a, offset_a, lda, info, queue );
	}

	return *info;
} /* magma_dpotri */
