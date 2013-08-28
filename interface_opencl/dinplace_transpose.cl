/*
    -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
	   Univ. of Colorado, Denver
       May 2012

       @generated d Wed Oct 24 00:32:58 2012

*/
// #include "common_magma.h"
// #define PRECISION_d
// #include "commonblas.h"

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define PRECISION_d
//#define NB 16
#define DSIZE_2SHARED 16

//#define NBhalf (NB/2)
#define NBhalf (DSIZE_2SHARED/2)
#define __mul24( x, y )  ((x)*(y))

#if defined(PRECISION_c) || defined(PRECISION_z)
typedef double double;
#endif

//#if NB == 32
#define COPY_1D_TO_2D( a, lda, b, inx, iny )        \
    b[iny][inx]			= a[0];                        \
    b[iny+NBhalf][inx]	= a[NBhalf*lda];

#define COPY_2D_TO_1D( b, inx, iny, a, lda )        \
    a[0]			= b[inx][iny];                        \
    a[NBhalf*lda]	= b[inx][iny+NBhalf];

/*
#else

#define COPY_1D_TO_2D( a, lda, b, inx, iny )        \
    b[iny][inx] = a[0];

#define COPY_2D_TO_1D( b, inx, iny, a, lda )        \
    a[0]        = b[inx][iny];

#endif
*/


__kernel void dinplace_T_even_kernel(__global double *matrix, int offset, int lda, int half )
{     
//	__local double a[NB][NB+1];
//	__local double b[NB][NB+1];

	__local double a[DSIZE_2SHARED][DSIZE_2SHARED+1];
	__local double b[DSIZE_2SHARED][DSIZE_2SHARED+1];

	int inx = get_local_id(0);
	int iny = get_local_id(1);

	bool bottom = ( get_group_id(0) > get_group_id(1) );
	int ibx = bottom ? (get_group_id(0) - 1) : (get_group_id(1) + half);
	int iby = bottom ? get_group_id(1)       : (get_group_id(0) + half);

//	ibx *= NB;
//	iby *= NB;

	ibx *= DSIZE_2SHARED;
	iby *= DSIZE_2SHARED;

	matrix += offset;

	__global double *A = matrix + ibx + inx + (iby + iny)*lda;
	COPY_1D_TO_2D( A, lda, a, inx, iny);

	if( ibx == iby )
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		COPY_2D_TO_1D( a, inx, iny, A, lda);
	}
	else
	{
		__global double *B = matrix + iby + inx + __mul24( ibx + iny, lda );

		COPY_1D_TO_2D( B, lda, b, inx, iny);
		barrier(CLK_LOCAL_MEM_FENCE);

		COPY_2D_TO_1D( b, inx, iny, A, lda);
		COPY_2D_TO_1D( a, inx, iny, B, lda);
	}
} 

__kernel void dinplace_T_odd_kernel (__global double *matrix, int offset, int lda, int half )
{        
       //__local double a[NB][NB+1];
       //__local double b[NB][NB+1];
        
		__local double a[DSIZE_2SHARED][DSIZE_2SHARED+1];
		__local double b[DSIZE_2SHARED][DSIZE_2SHARED+1];

        int inx = get_local_id(0);
        int iny = get_local_id(1);

        bool bottom = ( get_group_id(0) >= get_group_id(1) );
        int ibx = bottom ? get_group_id(0)  : (get_group_id(1) + half - 1);
        int iby = bottom ? get_group_id(1)  : (get_group_id(0) + half);

        //ibx *= NB;
        //iby *= NB;

		ibx *= DSIZE_2SHARED;
		iby *= DSIZE_2SHARED;

		matrix += offset;

        __global double *A = matrix + ibx + inx + __mul24( iby + iny, lda );

        COPY_1D_TO_2D( A, lda, a, inx, iny);

        if( ibx == iby )
        {
                barrier(CLK_LOCAL_MEM_FENCE);
                COPY_2D_TO_1D( a, inx, iny, A, lda);
        }
        else
        {
                __global double *B = matrix + iby + inx + __mul24( ibx + iny, lda );

                COPY_1D_TO_2D( B, lda, b, inx, iny);
                barrier(CLK_LOCAL_MEM_FENCE);

                COPY_2D_TO_1D( b, inx, iny, A, lda);
                COPY_2D_TO_1D( a, inx, iny, B, lda);
        }
} 
