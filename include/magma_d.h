/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @author Mark Gates
 * @generated d Wed Oct 24 00:32:41 2012
 */

#ifndef MAGMA_D_H
#define MAGMA_D_H

#include "magma_types.h"

#define PRECISION_d

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/

magma_err_t
magma_dgebrd(magma_int_t m, magma_int_t n,
             double *a, magma_int_t lda, double *d, double *e,
             double *tauq, double *taup,
             double *work, magma_int_t lwork,
             magma_int_t *info, magma_queue_t queue);

magma_int_t
magma_dormtr(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, 
			magma_int_t m, magma_int_t n,
			double *a,    magma_int_t lda,
			double *tau,
			double *c,    magma_int_t ldc,
			double *work, magma_int_t lwork,
			magma_int_t *info, magma_queue_t queue);

magma_int_t 
magma_dormqr(magma_side_t side, magma_trans_t trans,
			magma_int_t m, magma_int_t n, magma_int_t k,
			double *a,    magma_int_t lda,
			double *tau,
			double *c,    magma_int_t ldc,
			double *work, magma_int_t lwork,
			magma_int_t *info, magma_queue_t queue);

magma_int_t
magma_dormql(magma_side_t side, magma_trans_t trans,
			magma_int_t m, magma_int_t n, magma_int_t k,
			double *a, magma_int_t lda,
			double *tau,
			double *c, magma_int_t ldc,
			double *work, magma_int_t lwork,
			magma_int_t *info, magma_queue_t queue);

#if defined(PRECISION_z) || defined(PRECISION_c)

magma_int_t 
magma_dgeev(magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
			double *a, magma_int_t lda,
			double *geev_w_array,
			double *vl, magma_int_t ldvl,
			double *vr, magma_int_t ldvr,
			double *work, magma_int_t lwork,
			double *rwork, magma_int_t *info, magma_queue_t queue);

magma_int_t 
magma_dgesvd(char jobu, char jobvt, magma_int_t m_, magma_int_t n_, 
			double *a,    magma_int_t lda_, double *s,
			double *u,    magma_int_t ldu_,
			double *vt,   magma_int_t ldvt_,
			double *work, magma_int_t lwork_,
			double *rwork, magma_int_t *info, magma_queue_t queue);

magma_int_t 
magma_dsyevd(magma_vec_t jobz, magma_uplo_t uplo,
			magma_int_t n,
			double *a, magma_int_t lda,
			double *w,
			double *work, magma_int_t lwork,
			double *rwork, magma_int_t lrwork,
			magma_int_t *iwork, magma_int_t liwork,
			magma_int_t *info, magma_queue_t queue);

magma_int_t 
magma_dstedx(magma_vec_t range, magma_int_t n, double vl, double vu,
			magma_int_t il, magma_int_t iu, double* d, double* e,
			double* z, magma_int_t ldz,
			double* rwork, magma_int_t lrwork,
			magma_int_t* iwork, magma_int_t liwork,
			magmaDouble_ptr dwork, magma_int_t* info, magma_queue_t queue);

#else

magma_int_t 
magma_dgeev(magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
			double *a, magma_int_t lda,
			double *WR, double *WI,
			double *vl, magma_int_t ldvl,
			double *vr, magma_int_t ldvr,
			double *work, magma_int_t lwork,
			magma_int_t *info, magma_queue_t queue);

magma_int_t 
magma_dgesvd(char jobu, char jobvt, magma_int_t m_, magma_int_t n_,
			double *a,    magma_int_t lda_, double *s,
			double *u,    magma_int_t ldu_,
			double *vt,   magma_int_t ldvt_,
			double *work, magma_int_t lwork_,
			magma_int_t *info, magma_queue_t queue);

magma_int_t 
magma_dsyevd(magma_vec_t jobz, magma_uplo_t uplo,
			magma_int_t n,
			double *a, magma_int_t lda,
			double *w,
			double *work, magma_int_t lwork,
			magma_int_t *iwork, magma_int_t liwork,
			magma_int_t *info, magma_queue_t queue);

magma_int_t
magma_dstedx(magma_vec_t range, magma_int_t n, double vl, double vu, 
			magma_int_t il, magma_int_t iu, double* d, double* e,
			double* z, magma_int_t ldz,
			double* rwork, magma_int_t lrwork,
			magma_int_t* iwork, magma_int_t liwork,
			magmaDouble_ptr dwork, magma_int_t* info, magma_queue_t queue);

magma_int_t
magma_dlaex0(magma_int_t n, double* d, double* e, double* q, magma_int_t ldq,
			double* work, magma_int_t* iwork, magmaDouble_ptr dwork, 
			magma_vec_t range, double vl, double vu, 
			magma_int_t il, magma_int_t iu, magma_int_t* info, magma_queue_t queue);

magma_int_t 
magma_dlaex1(magma_int_t n, double* d, double* q, magma_int_t ldq,
			magma_int_t* indxq, double rho, magma_int_t cutpnt,
			double* work, magma_int_t* iwork, magmaDouble_ptr dwork,
			magma_vec_t range, double vl, double vu,
			magma_int_t il, magma_int_t iu, magma_int_t* info, magma_queue_t queue);


magma_int_t 
magma_dlaex3(magma_int_t k, magma_int_t n, magma_int_t n1, double* d, 
			double* q, magma_int_t ldq, double rho, 
			double* dlamda, double* q2, magma_int_t* indx,
			magma_int_t* ctot, double* w, double* s, magma_int_t* indxq,
			magmaDouble_ptr dwork,
			magma_vec_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
			magma_int_t* info, magma_queue_t queue );

#endif
/* //////////////////////////////////////////////////////////////////////////// 
 -- MAGMA function definitions / Data on GPU
*/

magma_err_t
magma_dgeqrf2_gpu(
        magma_int_t m, magma_int_t n, 
        magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, 
        double *tau, magma_int_t *info, magma_queue_t queue);

magma_err_t
magma_dgetrf_gpu(
        magma_int_t m, magma_int_t n, 
        magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, 
        magma_int_t *ipiv, magma_int_t *info, magma_queue_t queue);

magma_err_t
magma_dlarfb_gpu( 
        int side, int trans, int direct, int storev, 
        magma_int_t m, magma_int_t n, magma_int_t k,
        magmaDouble_ptr dV, size_t dV_offset, magma_int_t ldv,
        magmaDouble_ptr dT, size_t dT_offset, magma_int_t ldt, 
        magmaDouble_ptr dC, size_t dC_offset, magma_int_t ldc,
        magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t ldwork,
        magma_queue_t queue);

magma_err_t
magma_dpotrf_gpu(
        int uplo,
        magma_int_t n, 
        magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, 
        magma_int_t *info, magma_queue_t queue);

magma_err_t
magma_dpotrs_gpu(
		magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
        magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
		magmaDouble_ptr dB, size_t dB_offset, magma_int_t lddb,
		magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_dposv_gpu( 
		magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
        magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
		magmaDouble_ptr dB, size_t dB_offset, magma_int_t lddb,
		magma_err_t *info, magma_queue_t queue );

magma_err_t
magma_dgetrs_gpu(magma_trans_t trans, magma_int_t n, magma_int_t nrhs, 
		magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
		magma_int_t *ipiv, 
		magmaDouble_ptr dB, size_t dB_offset, magma_int_t lddb, 
		magma_int_t *info, magma_queue_t queue);

magma_err_t
magma_dgesv_gpu( magma_int_t n, magma_int_t nrhs,
                 magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
				 magma_int_t *ipiv,
				 magmaDouble_ptr dB, size_t dB_offset, magma_int_t lddb,
				 magma_err_t *info, magma_queue_t queue );

magma_int_t
magma_dormqr_gpu(magma_side_t side, magma_trans_t trans,
                 magma_int_t m, magma_int_t n, magma_int_t k,
                 magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, 
                 double *tau,
                 magmaDouble_ptr dC, size_t dC_offset, magma_int_t lddc,
                 double *hwork, magma_int_t lwork,
                 magmaDouble_ptr dT, size_t dT_offset, magma_int_t nb, 
                 magma_int_t *info, magma_queue_t queue);

magma_err_t
magma_dgeqrs_gpu(magma_int_t m, magma_int_t n, magma_int_t nrhs,
                 magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, 
                 double *tau,   magmaDouble_ptr dT, size_t dT_offset, 
				 magmaDouble_ptr dB, size_t dB_offset, magma_int_t lddb, 
                 double *hwork, magma_int_t lwork, 
                 magma_int_t *info, magma_queue_t queue);

magma_err_t
magma_dgeqrf_gpu( magma_int_t m, magma_int_t n, 
                  magmaDouble_ptr dA, size_t dA_offset,  magma_int_t ldda,
                  double *tau, magmaDouble_ptr dT, size_t dT_offset, 
                  magma_int_t *info, magma_queue_t queue);

magma_int_t
magma_dgels_gpu( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                 magmaDouble_ptr dA, size_t dA_offset,  magma_int_t ldda, 
                 magmaDouble_ptr dB, size_t dB_offset,  magma_int_t lddb, 
                 double *hwork, magma_int_t lwork, 
                 magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_dgehrd(	magma_int_t n, magma_int_t ilo, magma_int_t ihi, 
		double *a, magma_int_t lda, 
		double *tau, 
		double *work, magma_int_t lwork, 
		magmaDouble_ptr dT, size_t dT_offset, 
		magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_dlabrd_gpu( magma_int_t m, magma_int_t n, magma_int_t nb,
                  double *a, magma_int_t lda,
                  magmaDouble_ptr da, size_t da_offset, magma_int_t ldda,
                  double *d, double *e, double *tauq, double *taup,
                  double *x, magma_int_t ldx,
                  magmaDouble_ptr dx, size_t dx_offset, magma_int_t lddx,
                  double *y, magma_int_t ldy,
                  magmaDouble_ptr dy, size_t dy_offset, magma_int_t lddy,
                  magma_queue_t queue );

magma_err_t
magma_dlahr2(	magma_int_t n, magma_int_t k, magma_int_t nb,
		magmaDouble_ptr da, size_t da_offset, magmaDouble_ptr dv, size_t dv_offset, 
		double *a, magma_int_t lda, 
		double *tau, double *t, magma_int_t ldt, 
		double *y, magma_int_t ldy, 
		magma_queue_t queue	);

magma_err_t
magma_dlahru(	magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb, 
		double *a, magma_int_t lda, 
		magmaDouble_ptr d_a, size_t d_a_offset, magmaDouble_ptr y, size_t y_offset, 
		magmaDouble_ptr v, size_t v_offset, magmaDouble_ptr d_t, size_t dt_offset, 
		magmaDouble_ptr d_work, size_t d_work_offset, magma_queue_t queue );

magma_err_t
magma_dorghr(	magma_int_t n, magma_int_t ilo, magma_int_t ihi, 
		double *a, magma_int_t lda, 
		double *tau, 
		magmaDouble_ptr dT, size_t dT_offset, magma_int_t nb, 
		magma_int_t *info, magma_queue_t queue );

magma_err_t
magma_dorgqr(	magma_int_t m, magma_int_t n, magma_int_t k,
		double *a, magma_int_t lda,
		double *tau, magmaDouble_ptr dT, size_t dT_offset,
		magma_int_t nb, magma_int_t *info, magma_queue_t queue );

magma_err_t 
magma_dlatrd(	char uplo, magma_int_t n, magma_int_t nb, 
		double *a,  magma_int_t lda, 
		double *e, double *tau, 
		double *w,  magma_int_t ldw, 
		magmaDouble_ptr da, size_t da_offset, magma_int_t ldda, 
		magmaDouble_ptr dw, size_t dw_offset, magma_int_t lddw, magma_queue_t queue);

magma_err_t
magma_dsytrd(	char uplo, magma_int_t n,
		double *a, magma_int_t lda,
		double *d, double *e, double *tau,
		double *work, magma_int_t lwork,
		magma_int_t *info, magma_queue_t queue);

magma_int_t 
magma_dtrtri_gpu(magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
		magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, magma_int_t *info);

magma_int_t
magma_dgetri_gpu( magma_int_t n, magmaDouble_ptr dA, size_t dA_offset, magma_int_t lda,
		magma_int_t *ipiv, magmaDouble_ptr dwork, size_t dwork_offset, magma_int_t lwork,
		magma_int_t *info, magma_queue_t queue );

magma_int_t
magma_dlauum_gpu(magma_uplo_t uplo, magma_int_t n,
		magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda,
		magma_int_t *info, magma_queue_t queue);

magma_int_t
magma_dpotri_gpu(magma_uplo_t uplo, magma_int_t n,
		magmaDouble_ptr a, size_t offset_a, magma_int_t lda,
		magma_int_t *info, magma_queue_t queue);

/* //////////////////////////////////////////////////////////////////////////// 
 -- MAGMA utility function definitions
*/

void magma_dprint    ( magma_int_t m, magma_int_t n, double     *A, magma_int_t lda  );
void magma_dprint_gpu( magma_int_t m, magma_int_t n, magmaDouble_ptr dA, size_t dA_offset, magma_int_t ldda, magma_queue_t queue );

#ifdef __cplusplus
}
#endif

#undef PRECISION_d
#endif /* MAGMA_D_H */
