/*
 *  -- clMAGMA (version 1.0.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     April 2012
 *
 * @generated s Wed Oct 24 00:33:02 2012
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

#define PRECISION_s
// Flops formula
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS_POTRF(n)       ( 6.*FMULS_POTRF(n)       + 2.*FADDS_POTRF(n)       )
#define FLOPS_POTRS(n, nrhs) ( 6.*FMULS_POTRS(n, nrhs) + 2.*FADDS_POTRS(n, nrhs) )
#else
#define FLOPS_POTRF(n)       (    FMULS_POTRF(n)       +    FADDS_POTRF(n)       )
#define FLOPS_POTRS(n, nrhs) (    FMULS_POTRS(n, nrhs) +    FADDS_POTRS(n, nrhs) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sposv
*/
#define hA(i,j) hA[ i + j*lda ]

int main(int argc, char **argv)
{
    real_Double_t gflops, gpu_perf, gpu_time;
	float	Rnorm, Anorm, Xnorm, *work;
    float *hA, *hB, *hX;
    magmaFloat_ptr dA, dB;
    magma_int_t N = 0, n2, lda, ldb, ldda, lddb;
    magma_int_t size[7] =
        { 1024, 2048, 3072, 4032, 5184, 6048, 7000};
    
    magma_int_t i, info, szeB;
    float z_one = MAGMA_S_ONE;
    float mz_one = MAGMA_S_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
	magma_int_t NRHS = 100;

    if (argc != 1){
        for(i = 1; i<argc; i++){        
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
			if (strcmp("-R", argv[i])==0)
				NRHS = atoi(argv[++i]);
        }
        if (N>0) size[0] = size[6] = N;
        else exit(1);
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_sposv_gpu -N <matrix size> -R <right hand sides>\n\n");
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
    N    = size[6];
    n2   = N * N;
    ldda = ((N+31)/32) * 32;
   // ldda = N;
	lddb = ldda;
    TESTING_MALLOC_HOST( hA, float, n2 );
    TESTING_MALLOC_HOST( hB, float, N*NRHS );
    TESTING_MALLOC_HOST( hX, float, N*NRHS );
    TESTING_MALLOC_HOST( work, float, N );
    TESTING_MALLOC_DEV(  dA, float, ldda*N );
    TESTING_MALLOC_DEV(  dB, float, lddb*NRHS );
    
    printf("\n\n");
	printf("    N   NRHS   GPU GFlop/s (sec)   ||B - AX|| / ||A||*||X||\n");
    printf("===========================================================\n");
    for(i=0; i<7; i++){
        N   = size[i];
        lda = N;
		ldb = lda;
        n2  = lda*N;
		szeB = ldb*NRHS;
        ldda = ((N+31)/32)*32;
	//	ldda = N;
		lddb = ldda;
		gflops = ( FLOPS_POTRF( (float)N ) +
		                   FLOPS_POTRS( (float)N, (float)NRHS ) ) / 1e9;

        /* Initialize the matrix */
        lapackf77_slarnv( &ione, ISEED, &n2, hA );
        lapackf77_slarnv( &ione, ISEED, &szeB, hB );
        /* Symmetrize and increase the diagonal */
        for( int i = 0; i < N; ++i ) {
            MAGMA_S_SET2REAL( hA(i,i), MAGMA_S_REAL(hA(i,i)) + N );
            for( int j = 0; j < i; ++j ) {
	      hA(i, j) = MAGMA_S_CNJG( hA(j,i) );
            }
        }
        
		/* Warm up to measure the performance */
		magma_ssetmatrix( N, N, hA, 0, lda, dA, 0, ldda, queue );
		magma_ssetmatrix( N, NRHS, hB, 0, lda, dB, 0, lddb, queue );
		magma_sposv_gpu( MagmaUpper, N, NRHS, dA, 0, ldda, dB, 0, lddb, &info, queue );
        
        /* ====================================================================
           Performs operation using MAGMA 
           =================================================================== */
        magma_ssetmatrix( N, N, hA, 0, lda, dA, 0, ldda, queue );
		magma_ssetmatrix( N, NRHS, hB, 0, lda, dB, 0, lddb, queue );
		gpu_time = get_time();
		magma_sposv_gpu( MagmaUpper, N, NRHS, dA, 0, ldda, dB, 0, lddb, &info, queue );
        gpu_time = get_time() - gpu_time;
        if (info != 0)
            printf( "magma_sposv had error %d.\n", info );

        gpu_perf = gflops / gpu_time;
        
        /* =====================================================================
           Residual
           =================================================================== */
        magma_sgetmatrix( N, NRHS, dB, 0, lddb, hX, 0, ldb, queue );
		Anorm = lapackf77_slange("I", &N, &N,    hA, &lda, work);
        Xnorm = lapackf77_slange("I", &N, &NRHS, hX, &ldb, work);

		blasf77_sgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &NRHS, &N,
						&z_one,  hA, &lda,
						hX, &ldb,
						&mz_one, hB, &ldb );

		Rnorm = lapackf77_slange("I", &N, &NRHS, hB, &ldb, work);

             printf( "%5d  %5d   %7.2f (%7.2f)   %8.2e\n",
			                 N, NRHS, gpu_perf, gpu_time, Rnorm/(Anorm*Xnorm) );

        if (argc != 1)
            break;
    }

    /* clean up */
    TESTING_FREE_HOST( hA );
    TESTING_FREE_HOST( hB );
    TESTING_FREE_HOST( hX );
    TESTING_FREE_HOST( work );
    TESTING_FREE_DEV( dA );
    TESTING_FREE_DEV( dB );
    magma_queue_destroy( queue );
    magma_finalize();
}
