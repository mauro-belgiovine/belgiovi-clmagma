/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      May 2012
 *
 * @generated s Wed Oct 24 00:32:58 2012
 */

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from sgehrd.  The routine is called 
	  in 16 blocks, 32 thread per block and initializes to zero the 1st 
	  32x32 block of A.
*/

#define PRECISION_s
#if defined(PRECISION_c) || defined(PRECISION_z)
typedef float float;
#endif

#define slaset_threads 64
#define __mul24( x, y )  ((x)*(y))

__kernel void sset_nbxnb_to_zero(int nb, __global float *A, int offset, int lda){
	//int ind = blockIdx.x*lda + threadIdx.x, i, j;
	int ind = get_group_id(0)*lda+get_local_id(0); 
	int i, j;
	A += (ind+offset);
	float MAGMA_S_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
	MAGMA_S_ZERO = (float)(0.0, 0.0);
#else
	MAGMA_S_ZERO = 0.0;
#endif
	for(i=0; i<nb; i+=32){
		for(j=0; j<nb; j+=32)
			A[j] = MAGMA_S_ZERO;
		A += 32*lda;
	}
}

__kernel void slaset_upper(int m, int n, __global float *A, int offset, int lda)
{
	//int ibx = blockIdx.x * slaset_threads;
	int ibx = get_group_id(0)*slaset_threads;
	//int iby = blockIdx.y * 32;
	int iby = get_group_id(1)*32;

	//int ind = ibx + threadIdx.x;
	int ind = ibx + get_local_id(0);
	A += offset + ind + __mul24(iby, lda);
	float MAGMA_S_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
	MAGMA_S_ZERO = (float)(0.0, 0.0);
#else
	MAGMA_S_ZERO = 0.0;
#endif
	for(int i=0; i<32; i++)
		if (iby+i < n && ind < m && ind < i+iby)
			A[i*lda] = MAGMA_S_ZERO;
}

__kernel void slaset_lower(int m, int n, __global float *A, int offset, int lda)
{
	//int ibx = blockIdx.x * slaset_threads;
	int ibx = get_group_id(0)*slaset_threads;
	//int iby = blockIdx.y * 32;
	int iby = get_group_id(1)*32;

	//int ind = ibx + threadIdx.x;
	int ind = ibx + get_local_id(0);
	A += offset + ind + __mul24(iby, lda);
	float MAGMA_S_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
	MAGMA_S_ZERO = (float)(0.0, 0.0);
#else
	MAGMA_S_ZERO = 0.0;
#endif
	for(int i=0; i<32; i++){
		if (iby+i < n && ind < m && ind > i+iby)
			A[i*lda] = MAGMA_S_ZERO;
	}
}

__kernel void slaset(int m, int n, __global float *A, int offset, int lda)
{
	//int ibx = blockIdx.x * slaset_threads;
	int ibx = get_group_id(0)*slaset_threads;
	//int iby = blockIdx.y * 32;
	int iby = get_group_id(1)*32;

	//int ind = ibx + threadIdx.x;
	int ind = ibx + get_local_id(0);
	A += offset + ind + __mul24(iby, lda);
	float MAGMA_S_ZERO;
#if defined(PRECISION_c) || defined(PRECISION_z)
	MAGMA_S_ZERO = (float)(0.0, 0.0);
#else
	MAGMA_S_ZERO = 0.0;
#endif
	for(int i=0; i<32; i++)
		if (iby+i < n && ind < m)
			A[i*lda] = MAGMA_S_ZERO;
}
