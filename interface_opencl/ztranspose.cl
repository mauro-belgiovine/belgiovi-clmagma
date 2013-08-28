/*
    -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012

       @precisions normal z -> s d c
*/

//#include "common_magma.h"
//#include "commonblas.h"

#define PRECISION_z

#define ZSIZE_1SHARED 32
#define __mul24( x, y )  ((x)*(y))

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double2 magmaDoubleComplex;
#endif

__kernel void ztranspose_32( __global magmaDoubleComplex *B, int offsetB, int ldb, __global magmaDoubleComplex *A, int offsetA,  int lda )
{
		__local magmaDoubleComplex a[32][ZSIZE_1SHARED+1];
        
        int inx = get_local_id(0);
        int iny = get_local_id(1);
        int ibx = get_group_id(0)*32;
        int iby = get_group_id(1)*32;

		A += offsetA;
		B += offsetB;
        
        A += ibx + inx + __mul24( iby + iny, lda );
        B += iby + inx + __mul24( ibx + iny, ldb );
        
        a[iny+0][inx] = A[0*lda];
        a[iny+8][inx] = A[8*lda];
        a[iny+16][inx] = A[16*lda];
        a[iny+24][inx] = A[24*lda];
        
        barrier(CLK_LOCAL_MEM_FENCE);

#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) || defined(PRECISION_z) 
        B[0*ldb] = a[inx][iny+0];
        B[8*ldb] = a[inx][iny+8];
        B[16*ldb] = a[inx][iny+16];
        B[24*ldb] = a[inx][iny+24];
#else 
        B[0*ldb]    = a[inx][iny+0];
        B[8*ldb]    = a[inx][iny+8];
        B[0*ldb+16] = a[inx+16][iny+0];
        B[8*ldb+16] = a[inx+16][iny+8];

        barrier(CLK_LOCAL_MEM_FENCE);
        A += ZSIZE_1SHARED;
        B += __mul24( 16, ldb);

        a[iny+0][inx] = A[0*lda];
        a[iny+8][inx] = A[8*lda];
        a[iny+16][inx] = A[16*lda];
        a[iny+24][inx] = A[24*lda];

        barrier(CLK_LOCAL_MEM_FENCE);
        B[0*ldb] = a[inx][iny+0];
		B[8*ldb] = a[inx][iny+8];
		B[0*ldb+16] = a[inx+16][iny+0];
        B[8*ldb+16] = a[inx+16][iny+8];
#endif
} 
