/*
 *  -- clMAGMA (version 1.0.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     April 2011
 *
 * @precisions normal z -> c d s
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
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS_GEQRF(m, n      ) ( 6.*FMULS_GEQRF(m, n      ) + 2.*FADDS_GEQRF(m, n      ) )
#define FLOPS_GEQRS(m, n, nrhs) ( 6.*FMULS_GEQRS(m, n, nrhs) + 2.*FADDS_GEQRS(m, n, nrhs) )
#else
#define FLOPS_GEQRF(m, n      ) (    FMULS_GEQRF(m, n      ) +    FADDS_GEQRF(m, n      ) )
#define FLOPS_GEQRS(m, n, nrhs) (    FMULS_GEQRS(m, n, nrhs) +    FADDS_GEQRS(m, n, nrhs) )
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrs_gpu
*/
int main( int argc, char** argv)
{
//#if defined(PRECISION_s)
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
  
	real_Double_t gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           matnorm, work[1];
    magmaDoubleComplex  c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_A2, *h_B, *h_X, *h_R, *tau, *hwork, tmp[1];
    magmaDoubleComplex_ptr d_A, d_B;

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2;
    magma_int_t lda, ldb, ldda, lddb, lworkgpu, lhwork;
    magma_int_t size[7] = {1024,2048,3072,4032,5184,6016,7000};

    magma_int_t i, info, min_mn, nb, l1, l2;
    magma_int_t ione     = 1;
    magma_int_t nrhs     = 3;
    magma_int_t ISEED[4] = {0,0,0,1};

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            else if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
            else if (strcmp("-nrhs", argv[i])==0)
                nrhs = atoi(argv[++i]);
        }
        if (N>0 && M>0 && M >= N)
            printf("  testing_zgeqrs_gpu -nrhs %d -M %d -N %d\n\n", nrhs, M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_zgeqrs_gpu -nrhs %d  -M %d  -N %d\n\n", nrhs, M, N);
                printf("  M has to be >= N, exit.\n");
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgeqrs_gpu -nrhs %d  -M %d  -N %d\n\n", nrhs, 1024, 1024);
        M = N = size[6];
    }

    ldda   = ((M+31)/32)*32;
    lddb   = ldda;
    n2     = M * N;
    min_mn = min(M, N);
    nb     = magma_get_zgeqrf_nb(M);
    lda = ldb = M;
    lworkgpu = (M-N + nb)*(nrhs+2*nb);

    /* Allocate host memory for the matrix */
    TESTING_MALLOC_HOST( tau,  magmaDoubleComplex, min_mn   );
    TESTING_MALLOC_HOST( h_A,  magmaDoubleComplex, lda*N    );
    TESTING_MALLOC_HOST( h_A2, magmaDoubleComplex, lda*N    );
    TESTING_MALLOC_HOST( h_B,  magmaDoubleComplex, ldb*nrhs );
    TESTING_MALLOC_HOST( h_X,  magmaDoubleComplex, ldb*nrhs );
    TESTING_MALLOC_HOST( h_R,  magmaDoubleComplex, ldb*nrhs );

    TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N      );
    TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, lddb*nrhs   );

    /*
     * Get size for host workspace
     */
    lhwork = -1;
    lapackf77_zgeqrf(&M, &N, h_A, &M, tau, tmp, &lhwork, &info);
    l1 = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
    lhwork = -1;
    lapackf77_zunmqr( MagmaLeftStr, MagmaConjTransStr, 
                      &M, &nrhs, &min_mn, h_A, &lda, tau,
                      h_X, &ldb, tmp, &lhwork, &info);
    l2 = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
    lhwork = max( max( l1, l2 ), lworkgpu );

    TESTING_MALLOC_HOST( hwork, magmaDoubleComplex, lhwork );

    printf("\n");
    printf("                                         ||b-Ax|| / (N||A||)\n");
    printf("  M     N    CPU GFlop/s   GPU GFlop/s      CPU      GPU    \n");
    printf("============================================================\n");
    for(i=0; i<7; i++){
        if (argc == 1){
            M = N = size[i];
        }
        min_mn= min(M, N);
        ldb = lda = M;
        n2    = lda*N;
        ldda  = ((M+31)/32)*32;
        gflops = (FLOPS_GEQRF( (double)M, (double)N ) 
                 + FLOPS_GEQRS( (double)M, (double)N, (double)nrhs )) / 1e9;

        /* Initialize the matrices */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_A2, &lda );

        n2 = M*nrhs;
        lapackf77_zlarnv( &ione, ISEED, &n2, h_B );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
		/* Warm up to measure the performance */
		magma_zsetmatrix( M, N,    h_A, 0, lda, d_A, 0, ldda, queue );
        magma_zsetmatrix( M, nrhs, h_B, 0, ldb, d_B, 0, lddb, queue );
        magma_zgels_gpu( MagmaNoTrans, M, N, nrhs, d_A, 0, ldda, 
                         d_B, 0, lddb, hwork, lworkgpu, &info, queue);
        
		magma_zsetmatrix( M, N,    h_A, 0, lda, d_A, 0, ldda, queue );
        magma_zsetmatrix( M, nrhs, h_B, 0, ldb, d_B, 0, lddb, queue );
        
		gpu_time = get_time();
        magma_zgels_gpu( MagmaNoTrans, M, N, nrhs, d_A, 0, ldda, 
                         d_B, 0, lddb, hwork, lworkgpu, &info, queue);
        gpu_time = get_time() - gpu_time;
        if (info < 0)
            printf("Argument %d of magma_zgels had an illegal value.\n", -info);
        
        gpu_perf = gflops / gpu_time;

        // Get the solution in h_X
        magma_zgetmatrix( N, nrhs, d_B, 0, lddb, h_X, 0, ldb, queue );

        // compute the residual
        blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N, 
                       &c_neg_one, h_A, &lda, 
                                   h_X, &ldb, 
                       &c_one,     h_R, &ldb);
        matnorm = lapackf77_zlange("f", &M, &N, h_A, &lda, work);

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &nrhs, h_B, &ldb, h_X, &ldb );

        cpu_time = get_time();
        lapackf77_zgels( MagmaNoTransStr, &M, &N, &nrhs,
                         h_A, &lda, h_X, &ldb, hwork, &lhwork, &info);
        cpu_time = get_time()-cpu_time;
        cpu_perf = gflops / cpu_time;
        if (info < 0)
          printf("Argument %d of lapackf77_zgels had an illegal value.\n", -info);

        blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N, 
                       &c_neg_one, h_A2, &lda, 
                                   h_X,  &ldb, 
                       &c_one,     h_B,  &ldb);

        printf("%5d %5d   %6.1f       %6.1f       %7.2e   %7.2e\n",
               M, N, cpu_perf, gpu_perf,
               lapackf77_zlange("f", &M, &nrhs, h_B, &M, work)/(min_mn*matnorm),
               lapackf77_zlange("f", &M, &nrhs, h_R, &M, work)/(min_mn*matnorm) );

        if (argc != 1)
            break;
    }

    /* Memory clean up */
    TESTING_FREE_HOST( tau );
    TESTING_FREE_HOST( h_A );
    TESTING_FREE_HOST( h_A2 );
    TESTING_FREE_HOST( h_B );
    TESTING_FREE_HOST( h_X );
    TESTING_FREE_HOST( h_R );
    TESTING_FREE_HOST( hwork );
    TESTING_FREE_DEV( d_A );
    TESTING_FREE_DEV( d_B );

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
}
