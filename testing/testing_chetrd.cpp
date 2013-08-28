/*
 *  -- clMAGMA (version 1.0.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     April 2011
 *
 * @generated c Wed Oct 24 00:33:03 2012
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
#define FLOPS(n) ( 6. * FMULS_HETRD(n) + 2. * FADDS_HETRD(n))
#else
#define FLOPS(n) (      FMULS_HETRD(n) +      FADDS_HETRD(n))
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chetrd
*/
int main( int argc, char** argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    
    float           eps;
    magmaFloatComplex *h_A, *h_R, *h_Q, *h_work, *work;
	magmaFloatComplex *h_R1, *h_work1;
    magmaFloatComplex *tau;
    float          *diag, *offdiag, *rwork;
    float           result[2] = {0., 0.};

    /* Matrix size */
    magma_int_t N = 0, n2, lda, lwork;
#if defined(PRECISION_z)
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,7040,7040,7040};
#else
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};
#endif

    magma_int_t i, info, nb, checkres, once = 0;
    magma_int_t ione     = 1;
    magma_int_t itwo     = 2;
    magma_int_t ithree   = 3;
    magma_int_t ISEED[4] = {0,0,0,1};
    char *uplo = (char *)MagmaLowerStr;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
                once = 1;
            }
            else if (strcmp("-U", argv[i])==0)
                uplo = (char *)MagmaUpperStr;
            else if (strcmp("-L", argv[i])==0)
                uplo = (char *)MagmaLowerStr;
        }
        if ( N > 0 )
            printf("  testing_chetrd -L|U -N %d\n\n", (int) N);
        else
        {
            printf("\nUsage: \n");
            printf("  testing_chetrd -L|U -N %d\n\n", 1024);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_chetrd -L|U -N %d\n\n", 1024);
        N = size[9];
    }

    checkres  = getenv("MAGMA_TESTINGS_CHECK") != NULL;

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
    
	eps = lapackf77_slamch( "E" );
    lda = N;
    n2  = lda * N;
    nb  = magma_get_chetrd_nb(N);
    /* We suppose the magma nb is bigger than lapack nb */
    lwork = N*nb; 

    /* Allocate host memory for the matrix */
    TESTING_MALLOC_HOST( h_A,    magmaFloatComplex, lda*N );
    TESTING_MALLOC_HOST( h_R1,    magmaFloatComplex, lda*N );
    TESTING_MALLOC_HOST( h_R,    magmaFloatComplex, lda*N );
    TESTING_MALLOC_HOST( h_work, magmaFloatComplex, lwork );
    TESTING_MALLOC_HOST( h_work1, magmaFloatComplex, lwork );
    TESTING_MALLOC_HOST( tau,    magmaFloatComplex, N     );
    TESTING_MALLOC_HOST( diag,    float, N   );
    TESTING_MALLOC_HOST( offdiag, float, N-1 );

    /* To avoid uninitialized variable warning */
    h_Q   = NULL;
    work  = NULL;
    rwork = NULL; 

    if ( checkres ) {
        TESTING_MALLOC( h_Q,  magmaFloatComplex, lda*N );
        TESTING_MALLOC( work, magmaFloatComplex, 2*N*N );
#if defined(PRECISION_z) || defined(PRECISION_c) 
        TESTING_MALLOC( rwork, float, N );
#endif
    }

    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
        if ( !once ) {
            N = size[i];
        }
        lda  = N;
        n2   = N*lda;
        gflops = FLOPS( (float)N ) / 1e9;

        /* ====================================================================
           Initialize the matrix
           =================================================================== */
        lapackf77_clarnv( &ione, ISEED, &n2, h_A );
        /* Make the matrix hermitian */
        {
            magma_int_t i, j;
            for(i=0; i<N; i++) {
                MAGMA_C_SET2REAL( h_A[i*lda+i], ( MAGMA_C_REAL(h_A[i*lda+i]) ) );
                for(j=0; j<i; j++)
                    h_A[i*lda+j] = MAGMA_C_CNJG(h_A[j*lda+i]);
            }
        }
        lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
        lapackf77_clacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R1, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
		// warm-up
        magma_chetrd(uplo[0], N, h_R1, lda, diag, offdiag, 
                     tau, h_work1, lwork, &info, queue);

        gpu_time = get_time();
        magma_chetrd(uplo[0], N, h_R, lda, diag, offdiag, 
                     tau, h_work, lwork, &info, queue);
        gpu_time = get_time() - gpu_time;
        if ( info < 0 )
            printf("Argument %d of magma_chetrd had an illegal value\n", (int) -info);

        gpu_perf = gflops / gpu_time;

        /* =====================================================================
           Check the factorization
           =================================================================== */
        if ( checkres ) {

            lapackf77_clacpy(uplo, &N, &N, h_R, &lda, h_Q, &lda);
            lapackf77_cungtr(uplo, &N, h_Q, &lda, tau, h_work, &lwork, &info);

#if defined(PRECISION_z) || defined(PRECISION_c) 
            lapackf77_chet21(&itwo, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, rwork, &result[0]);

            lapackf77_chet21(&ithree, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, rwork, &result[1]);

#else

            lapackf77_chet21(&itwo, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, &result[0]);

            lapackf77_chet21(&ithree, uplo, &N, &ione, 
                             h_A, &lda, diag, offdiag,
                             h_Q, &lda, h_R, &lda, 
                             tau, work, &result[1]);

#endif
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = get_time();
        lapackf77_chetrd(uplo, &N, h_A, &lda, diag, offdiag, tau, 
                         h_work, &lwork, &info);
        cpu_time = get_time() - cpu_time;

        if (info < 0)
            printf("Argument %d of lapackf77_chetrd had an illegal value.\n", (int) -info);

        cpu_perf = gflops / cpu_time;

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        if ( checkres ) {
            printf("%5d   %6.2f        %6.2f       %e %e\n",
                   (int) N, cpu_perf, gpu_perf,
                   result[0]*eps, result[1]*eps );
        } else {
            printf("%5d   %6.2f        %6.2f\n",
                   (int) N, cpu_perf, gpu_perf );
        }

        if ( once )
            break;
    }

    /* Memory clean up */
    TESTING_FREE( h_A );
    TESTING_FREE( tau );
    TESTING_FREE( diag );
    TESTING_FREE( offdiag );
    TESTING_FREE_HOST( h_R );
    TESTING_FREE_HOST( h_R1 );
    TESTING_FREE_HOST( h_work );
    TESTING_FREE_HOST( h_work1 );

    if ( checkres ) {
        TESTING_FREE( h_Q );
        TESTING_FREE( work );
#if defined(PRECISION_z) || defined(PRECISION_c) 
        TESTING_FREE( rwork );
#endif
    }

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    return EXIT_SUCCESS;
}
