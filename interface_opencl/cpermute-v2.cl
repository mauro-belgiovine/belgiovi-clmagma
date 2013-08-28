/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @generated c Wed Oct 24 00:32:58 2012
 */
//#include "common_magma.h"

#define PRECISION_c
#define BLOCK_SIZE 32
#define __mul24( x, y )  ((x)*(y))

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float2 magmaFloatComplex;
#endif

typedef struct {
        int n, lda, j0;
        short ipiv[BLOCK_SIZE];
} claswp_params_t;

typedef struct {
        int n, lda, j0, npivots;
        short ipiv[BLOCK_SIZE];
} claswp_params_t2;

/*
 * Old version
 */
__kernel void myclaswp2(__global magmaFloatComplex *Ain, int offset, claswp_params_t2 params)
{
	unsigned int tid = get_local_id(0) + 
		__mul24(get_local_size (0), get_group_id(0));

	if( tid < params.n )
	{
		int lda = params.lda;
		__global magmaFloatComplex *A = Ain + offset + tid + lda * params.j0;

		for( int i = 0; i < params.npivots; i++ )
		{
			int j = params.ipiv[i];
			__global magmaFloatComplex *p1 = A + i*lda;
			__global magmaFloatComplex *p2 = A + j*lda;
			magmaFloatComplex temp = *p1;
			*p1 = *p2;
			*p2 = temp;
		}
	}
}
