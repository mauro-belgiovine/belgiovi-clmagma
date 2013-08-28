/*
 *  -- clMAGMA (version 1.0.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     April 2012
 *
 * @generated d Wed Oct 24 00:33:01 2012
 *
 **/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_d
// Flops formula
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_POTRF(n) + 2. * FADDS_POTRF(n) )
#else
#define FLOPS(n) (      FMULS_POTRF(n) +      FADDS_POTRF(n) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dpotrf
*/
#define hA(i,j) hA[ i + j*lda ]

int main( int argc, char** argv) 
{
    real_Double_t gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double *hA, *hR;
    magmaDouble_ptr dA;
    magma_int_t N = 0, n2, lda, ldda;
    magma_int_t size[10] =
       // { 1024, 2048, 3072, 4032, 5184, 6048, 7200, 8064, 8928, 10560 };
    	{1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112};
    magma_int_t i, info;
    double mz_one = MAGMA_D_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], matnorm, diffnorm;
    
    if (argc != 1){
        for(i = 1; i<argc; i++){        
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if (N>0) size[0] = size[9] = N;
        else exit(1);
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_dpotrf_gpu -N %d\n\n", 1024);
    }

    /* Initialize */
    magma_queue_t  queue;
    magma_device_t device;
    int num = 0;
    magma_err_t err;
    magma_init();
    err = magma_get_devices( &device, 1, &num );
    if ( err != 0 || num < 1 ) {
        fprintf( stderr, "magma_get_devices failed: %d\n", err );
        exit(-1);
    }
    err = magma_queue_create( device, &queue );
    if ( err != 0 ) {
        fprintf( stderr, "magma_queue_create failed: %d\n", err );
        exit(-1);
    }

    /* Allocate memory for the largest matrix */
    N    = size[9];
    n2   = N * N;
    ldda = ((N+31)/32) * 32;
    TESTING_MALLOC(      hA, double, n2 );
    TESTING_MALLOC_HOST( hR, double, n2 );
    TESTING_MALLOC_DEV(  dA, double, ldda*N );
    
    printf("\n\n");
    printf("  N    CPU GFlop/s (sec)    GPU GFlop/s (sec)    ||R_magma-R_lapack||_F / ||R_lapack||_F\n");
    printf("========================================================================================\n");
    for(i=0; i<10; i++){
        N   = size[i];
        lda = N; 
        n2  = lda*N;
        ldda = ((N+31)/32)*32;
        gflops = FLOPS( (double)N ) * 1e-9;
        
        /* Initialize the matrix */
        lapackf77_dlarnv( &ione, ISEED, &n2, hA );
        /* Symmetrize and increase the diagonal */
        for( int i = 0; i < N; ++i ) {
            MAGMA_D_SET2REAL( hA(i,i), MAGMA_D_REAL(hA(i,i)) + N );
            for( int j = 0; j < i; ++j ) {
	      hA(i, j) = MAGMA_D_CNJG( hA(j,i) );
            }
        }
        lapackf77_dlacpy( MagmaFullStr, &N, &N, hA, &lda, hR, &lda );

	/* Warm up to measure the performance */
	magma_dsetmatrix( N, N, hA, 0, lda, dA, 0, ldda, queue );
	magma_dpotrf_gpu( MagmaUpper, N, dA, 0, ldda, &info, queue );

        /* ====================================================================
           Performs operation using MAGMA 
           =================================================================== */
        magma_dsetmatrix( N, N, hA, 0, lda, dA, 0, ldda, queue );
        gpu_time = get_time();
        magma_dpotrf_gpu( MagmaUpper, N, dA, 0, ldda, &info, queue );
        gpu_time = get_time() - gpu_time;
        if (info != 0)
            printf( "magma_dpotrf had error %d.\n", info );

        gpu_perf = gflops / gpu_time;
        
        /* =====================================================================
           Performs operation using LAPACK 
           =================================================================== */
        cpu_time = get_time();
        lapackf77_dpotrf( MagmaUpperStr, &N, hA, &lda, &info );
        cpu_time = get_time() - cpu_time;
        if (info != 0)
            printf( "lapackf77_dpotrf had error %d.\n", info );
        
        cpu_perf = gflops / cpu_time;
        
        /* =====================================================================
           Check the result compared to LAPACK
           |R_magma - R_lapack| / |R_lapack|
           =================================================================== */
        magma_dgetmatrix( N, N, dA, 0, ldda, hR, 0, lda, queue );
        matnorm = lapackf77_dlange("f", &N, &N, hA, &lda, work);
        blasf77_daxpy(&n2, &mz_one, hA, &ione, hR, &ione);
        diffnorm = lapackf77_dlange("f", &N, &N, hR, &lda, work);
        printf( "%5d     %6.2f (%6.2f)     %6.2f (%6.2f)         %e\n", 
                N, cpu_perf, cpu_time, gpu_perf, gpu_time, diffnorm / matnorm );
        
        if (argc != 1)
            break;
    }

    /* clean up */
    TESTING_FREE( hA );
    TESTING_FREE_HOST( hR );
    TESTING_FREE_DEV( dA );
    magma_queue_destroy( queue );
    magma_finalize();
}
