/*
 *  -- clMAGMA (version 1.0.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     April 2011
 *
 * @generated c Wed Oct 24 00:33:02 2012
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

// Flops formula
#define PRECISION_c
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6. * FMULS_GETRF(m, n) + 2. * FADDS_GETRF(m, n) )
#else
#define FLOPS(m, n) (      FMULS_GETRF(m, n) +      FADDS_GETRF(m, n) )
#endif

float get_LU_error(magma_int_t M, magma_int_t N, 
                    magmaFloatComplex *A,  magma_int_t lda, 
                    magmaFloatComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaFloatComplex alpha = MAGMA_C_ONE;
    magmaFloatComplex beta  = MAGMA_C_ZERO;
    magmaFloatComplex *L, *U;
    float work[1], matnorm, residual;
                       
    TESTING_MALLOC( L, magmaFloatComplex, M*min_mn);
    TESTING_MALLOC( U, magmaFloatComplex, min_mn*N);
    memset( L, 0, M*min_mn*sizeof(magmaFloatComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaFloatComplex) );

    lapackf77_claswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_clacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_clacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for(j=0; j<min_mn; j++)
        L[j+j*M] = MAGMA_C_MAKE( 1., 0. );
    
    matnorm = lapackf77_clange("f", &M, &N, A, &lda, work);

    blasf77_cgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_C_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_clange("f", &M, &N, LU, &lda, work);

    TESTING_FREE(L);
    TESTING_FREE(U);

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgetrf
*/
int main( int argc, char** argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time, error;
    magmaFloatComplex *h_A, *h_R;
    magmaFloatComplex_ptr d_A;
    magma_int_t     *ipiv;

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, lda, ldda;
#if defined (PRECISION_z)
    magma_int_t size[10] = {1024,2048,3072,4032,4992,5952,7000,7000,7000,7000};
#else
    magma_int_t size[10] = {1024,2048,3072,4032,4992,5952,7104,8064,9000,10000};
#endif
    magma_int_t i, info, min_mn;
    //magma_int_t nb, maxn, ret;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
        }
        if (M>0 && N>0)
            printf("  testing_cgetrf -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_cgetrf -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_cgetrf_gpu -M %d -N %d\n\n", 1024, 1024);
        M = N = size[9];
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
    
    ldda   = ((M+31)/32)*32;
    //maxn   = ((N+31)/32)*32;
    n2     = M * N;
    min_mn = min(M, N);
    //nb     = magma_get_cgetrf_nb(min_mn);

    /* Allocate host memory for the matrix */
    TESTING_MALLOC(     ipiv, magma_int_t, min_mn);
    TESTING_MALLOC(      h_A, magmaFloatComplex, n2     );
    TESTING_MALLOC_HOST( h_R, magmaFloatComplex, n2     );
    TESTING_MALLOC_DEV(  d_A, magmaFloatComplex, ldda*N );

    printf("\n\n");
    printf("  M     N    CPU GFlop/ (sec)s   GPU GFlop/s (sec)   ||PA-LU||/(||A||*N)\n");
    printf("========================================================================\n");
    for(i=0; i<10; i++){
        if (argc == 1){
            M = N = size[i];
        }
        min_mn= min(M, N);
        lda   = M;
        n2    = lda*N;
        ldda  = ((M+31)/32)*32;
        gflops = FLOPS( (float)M, (float)N ) *1e-9;

        /* Initialize the matrix */
        lapackf77_clarnv( &ione, ISEED, &n2, h_A );
        lapackf77_clacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = get_time();
        lapackf77_cgetrf(&M, &N, h_A, &lda, ipiv, &info);
        cpu_time = get_time() - cpu_time;
        if (info < 0)
            printf("Argument %d of cgetrf had an illegal value.\n", -info);

        cpu_perf = gflops / cpu_time;

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
		magma_csetmatrix( M, N, h_R, 0, lda, d_A, 0, ldda, queue );
		magma_cgetrf_gpu( M, N, d_A, 0, ldda, ipiv, &info, queue );

		magma_csetmatrix( M, N, h_R, 0, lda, d_A, 0, ldda, queue );
		gpu_time = get_time();
		magma_cgetrf_gpu( M, N, d_A, 0, ldda, ipiv, &info, queue );
		gpu_time = get_time() - gpu_time;
        if (info < 0)
            printf("Argument %d of cgetrf had an illegal value.\n", -info);

        gpu_perf = gflops / gpu_time;

        /* =====================================================================
           Check the factorization
           =================================================================== */
		magma_cgetmatrix( M, N, d_A, 0, ldda, h_A, 0, lda, queue );
        error = get_LU_error(M, N, h_R, lda, h_A, ipiv);
        
        printf("%5d %5d  %6.2f (%6.2f)     %6.2f (%6.2f)      %e\n",
               M, N, cpu_perf, cpu_time, gpu_perf, gpu_time, error);

        if (argc != 1)
            break;
    }

    /* clean up */
    TESTING_FREE( ipiv );
    TESTING_FREE( h_A );
    TESTING_FREE_HOST( h_R );
    TESTING_FREE_DEV( d_A );

    magma_queue_destroy( queue );
    magma_finalize();
}
