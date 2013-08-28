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
#define CHECK_ERROR
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_GEHRD(n) + 2. * FADDS_GEHRD(n))
#else
#define FLOPS(n) (      FMULS_GEHRD(n) +      FADDS_GEHRD(n))
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgehrd2
*/
int main( int argc, char** argv)
{
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
	//*h_R1 is used for warm-up
    magmaDoubleComplex *h_A, *h_R, *h_Q, *h_work, *tau, *twork, *h_R1;
	magmaDoubleComplex_ptr dT;
    double          *rwork;
    double           result[2] = {0., 0.};
	double	eps;
	int checkres;
	checkres = getenv("MAGMA_TESTINGS_CHECK") != NULL;
    /* Matrix size */
    int N=0, n2, lda, nb, lwork, ltwork, once = 0;
#if defined (PRECISION_z)
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7000,7000,7000,7000};
#else
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,9900};
#endif

    int i, info;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};
    
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
        }
        if ( N > 0 ){
            printf("  testing_zgehrd -N %d\n\n", N);
			once = 1;
		}
        else
        {
            printf("\nUsage: \n");
            printf("  testing_zgehrd -N %d\n\n", 1024);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zgehrd -N %d\n\n", 1024);
        N = size[9];
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

    eps   = lapackf77_dlamch( "E" );
    lda   = N;
    n2    = N*lda;
    nb    = magma_get_zgehrd_nb(N);
    /* We suppose the magma nb is bigger than lapack nb */
    lwork = N*nb;
    
    TESTING_MALLOC_HOST( h_A   , magmaDoubleComplex, n2    );
    TESTING_MALLOC_HOST( tau   , magmaDoubleComplex, N     );
    TESTING_MALLOC_HOST( h_R   , magmaDoubleComplex, n2    );
    TESTING_MALLOC_HOST( h_R1   , magmaDoubleComplex, n2    );
    TESTING_MALLOC_HOST( h_work, magmaDoubleComplex, lwork );
    TESTING_MALLOC_DEV ( dT    , magmaDoubleComplex, nb*N  );

    /* To avoid uninitialized variable warning */
    h_Q   = NULL;
    twork = NULL;
    rwork = NULL; 

    if ( checkres ) {
        ltwork = 2*(N*N);
        TESTING_MALLOC_HOST( h_Q,   magmaDoubleComplex, lda*N  );
        TESTING_MALLOC_HOST( twork, magmaDoubleComplex, ltwork );
#if defined(PRECISION_z) || defined(PRECISION_c) 
        TESTING_MALLOC_HOST( rwork, double,          N      );
#endif
    }

    printf("\n\n");
    printf("  N    CPU GFlop/s    GPU GFlop/s   |A-QHQ'|/N|A|  |I-QQ'|/N \n");
    printf("=============================================================\n");
    for(i=0; i<10; i++){
        if ( !once ) {
            N = size[i];
        }
        lda = N;
        n2  = lda*N;
        gflops = FLOPS( (double)N ) / 1e9;

        /* Initialize the matrices */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R1, &lda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_zgehrd ( N, ione, N, h_R1, lda, tau, h_work, lwork, dT, 0, &info, queue);
        if ( info < 0 )
            printf("Argument %d of magma_zgehrd had an illegal value\n", -info);
        clFinish(queue);
		gpu_time = get_time();
        magma_zgehrd ( N, ione, N, h_R, lda, tau, h_work, lwork, dT, 0, &info, queue);
        gpu_time = get_time() - gpu_time;
        if ( info < 0 )
            printf("Argument %d of magma_zgehrd had an illegal value\n", -info);

        gpu_perf = gflops / gpu_time;

        /* =====================================================================
           Check the factorization
           =================================================================== */
        if ( checkres ) {

            lapackf77_zlacpy(MagmaUpperLowerStr, &N, &N, h_R, &lda, h_Q, &lda);
            { 
                int i, j;
                for(j=0; j<N-1; j++)
                    for(i=j+2; i<lda; i++)
                        h_R[i+j*lda] = MAGMA_Z_ZERO;
            }

            nb = magma_get_zgehrd_nb(N);
            magma_zunghr(N, ione, N, h_Q, lda, tau, dT, 0, nb, &info, queue);
#if defined(PRECISION_z) || defined(PRECISION_c) 
            lapackf77_zhst01(&N, &ione, &N, h_A, &lda, h_R, &lda, h_Q, &lda, twork, &ltwork, rwork, result);
#else
            lapackf77_zhst01(&N, &ione, &N, h_A, &lda, h_R, &lda, h_Q, &lda, twork, &ltwork, result);
#endif
        }

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        cpu_time = get_time();
        lapackf77_zgehrd(&N, &ione, &N, h_A, &lda, tau, h_work, &lwork, &info);
        cpu_time = get_time() - cpu_time;
        if (info < 0)
            printf("Argument %d of lapack_zgehrd had an illegal value.\n", -info);

        cpu_perf = gflops / cpu_time;

        /* =====================================================================
           Print performance and error.
           =================================================================== */
        if ( checkres ) {
            printf("%5d    %6.2f         %6.2f      %e %e\n",
                   N, cpu_perf, gpu_perf,
                   result[0]*eps, result[1]*eps );
        } else {
            printf("%5d    %6.2f         %6.2f\n",
                   N, cpu_perf, gpu_perf );
        }

        if ( once )
            break;
    }

    /* Memory clean up */
    TESTING_FREE    ( h_A  );
    TESTING_FREE    ( tau  );
    TESTING_FREE_HOST( h_work);
    TESTING_FREE_HOST( h_R  );
    TESTING_FREE_HOST( h_R1  );
    TESTING_FREE_DEV ( dT   );

    if ( checkres ) {
        TESTING_FREE_HOST( h_Q );
        TESTING_FREE( twork );
#if defined(PRECISION_z) || defined(PRECISION_c) 
        TESTING_FREE( rwork );
#endif
    }

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    return EXIT_SUCCESS;
}
