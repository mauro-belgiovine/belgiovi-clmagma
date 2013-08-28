/*
 *  -- clMAGMA (version 1.0.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     April 2012
 *
 * @generated d Wed Oct 24 00:33:05 2012
 *
 **/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_d

static int diffMatrix( double *A, double *B, int m, int n, int lda){
    int i, j;
    for(i=0; i<m; i++) {
        for(j=0; j<n; j++)
            if ( !(MAGMA_D_EQUAL( A[lda*j+i], B[lda*j+i] )) ){
				printf("Error in row %d, col %d\n",i, j );
                return 1;
			}
    }
    return 0;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dlaswp
*/
int main( int argc, char** argv) 
{
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

    double *h_A1, *h_A2, *h_A3, *h_AT;
    magmaDouble_ptr d_A1;

    real_Double_t gpu_time, cpu_time1, cpu_time2;

    /* Matrix size */
    int M=0, N=0, n2, lda, ldat;
    int size[7] = {1000,2000,3000,4000,5000,6000,7000};
    int i, j;
    int ione     = 1;
    int ISEED[4] = {0,0,0,1};
    int *ipiv;

	int k1, k2, r, c, incx;

    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0)
                N = atoi(argv[++i]);
            if (strcmp("-M", argv[i])==0)
                M = atoi(argv[++i]);
        }
        if (M>0 && N>0)
            printf("  testing_dlaswp -M %d -N %d\n\n", M, N);
        else
            {
                printf("\nUsage: \n");
                printf("  testing_dlaswp -M %d -N %d\n\n", 1024, 1024);
                exit(1);
            }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_dlaswp -M %d -N %d\n\n", 1024, 1024);
        M = N = size[6];
	}

    lda = M;
    n2 = M*N;

    /* Allocate host memory for the matrix */
	TESTING_MALLOC(h_A1, double, n2);
	TESTING_MALLOC(h_A2, double, n2);
	TESTING_MALLOC(h_A3, double, n2);
	TESTING_MALLOC(h_AT, double, n2);
    
    TESTING_MALLOC_DEV(  d_A1, double, n2 );

    ipiv = (int*)malloc(M * sizeof(int));
    if (ipiv == 0) {
        fprintf (stderr, "!!!! host memory allocation error (ipiv)\n");
    }
  
    printf("\n\n");
    printf("  M     N    CPU_BLAS (sec)  CPU_LAPACK (sec) GPU (sec)	                     \n");
    printf("=============================================================================\n");
    for(i=0; i<7; i++) {
		if(argc == 1){
			M = N = size[i];
		}
        lda = M;
		ldat = N;
        n2 = M*N;
        
		/* Initialize the matrix */
		lapackf77_dlarnv( &ione, ISEED, &n2, h_A1 );
        lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A1, &lda, h_A2, &lda );
		for(r=0;r<M;r++){
			for(c=0;c<N;c++){
				h_AT[c+r*ldat] = h_A1[r+c*lda];	
			}
		}

		magma_dsetmatrix( N, M, h_AT, 0, ldat, d_A1, 0, ldat, queue);

        for(j=0; j<M; j++) {
          ipiv[j] = (int)((rand()*1.*M) / (RAND_MAX * 1.)) + 1;
        }

        /*
         *  BLAS swap
         */
        /* Column Major */
        cpu_time1 = get_time();
        for ( j=0; j<M; j++) {
            if ( j != (ipiv[j]-1)) {
				blasf77_dswap( &N, h_A1+j, &lda, h_A1+(ipiv[j]-1), &lda);
            }
        }
        cpu_time1 = get_time() - cpu_time1;

        /*
         *  LAPACK laswp
         */
		cpu_time2 = get_time();
		k1 = 1;
		k2 = M;
		incx = 1;
		lapackf77_dlaswp(&N, h_A2, &lda, &k1, &k2, ipiv, &incx);
		cpu_time2 = get_time() - cpu_time2;
        
		/*
         *  GPU swap
         */
        /* Col swap on transpose matrix*/
        gpu_time = get_time();
		magma_dpermute_long2(N, d_A1, 0, ldat, ipiv, M, 0, queue);
        gpu_time = get_time() - gpu_time;
		
		/* Check Result */
		magma_dgetmatrix( N, M, d_A1, 0, ldat, h_AT, 0, ldat, queue);
		for(r=0;r<N;r++){
			for(c=0;c<M;c++){
				h_A3[c+r*lda] = h_AT[r+c*ldat];	
			}
		}
		
		int check_bl, check_bg, check_lg;

		check_bl = diffMatrix( h_A1, h_A2, M, N, lda );
		check_bg = diffMatrix( h_A1, h_A3, M, N, lda );
		check_lg = diffMatrix( h_A2, h_A3, M, N, lda );
        
        printf("%5d %5d  %6.2f		%6.2f		%6.2f	%s	%s	%s\n",
                M, N, cpu_time1, cpu_time2, gpu_time,
               (check_bl == 0) ? "SUCCESS" : "FAILED",
               (check_bg == 0) ? "SUCCESS" : "FAILED",
               (check_lg == 0) ? "SUCCESS" : "FAILED");

		if(check_lg !=0){
			printf("lapack swap results:\n");
			magma_dprint(M, N, h_A1, lda);
			printf("gpu swap transpose matrix result:\n");
			magma_dprint(M, N, h_A3, lda);
		}

        if (argc != 1)
          break;
    }
    
    /* clean up */
    TESTING_FREE( ipiv );
    TESTING_FREE( h_A1 );
    TESTING_FREE( h_A2 );
    TESTING_FREE( h_A3 );
    TESTING_FREE( h_AT );
    TESTING_FREE_DEV( d_A1 );

    magma_queue_destroy( queue );
    magma_finalize();
}
