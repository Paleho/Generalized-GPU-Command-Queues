///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <cblas.h>

#include "pthreads_backend_wrappers.hpp"

void cblas_wrap_daxpy(void* backend_data){
  axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) backend_data;
  cblas_daxpy(ptr_ker_translate->N, ptr_ker_translate->alpha,
    (double*) *ptr_ker_translate->x, ptr_ker_translate->incx, (double*)
    *ptr_ker_translate->y, ptr_ker_translate->incy);
}

void cblas_wrap_saxpy(void* backend_data){
  axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) backend_data;
  cblas_saxpy(ptr_ker_translate->N, ptr_ker_translate->alpha,
    (float*) *ptr_ker_translate->x, ptr_ker_translate->incx, (float*)
    *ptr_ker_translate->y, ptr_ker_translate->incy);
}

void cblas_wrap_dgemm(void* backend_data){
  short lvl = 6;
  gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) backend_data;
#ifdef DDEBUG
  if (ptr_ker_translate->dev_id != -1)
    warning("cblas_wrap_dgemm: Suspicious device %d instead of -1\n", ptr_ker_translate->dev_id);
#endif
#ifdef DDEBUG
  lprintf(lvl, "cblas_wrap_dgemm: cblas_dgemm(dev_id = %d, TransA = %c, TransB = %c,\
    M = %d, N = %d, K = %d, alpha = %lf, A = %p, lda = %d, \n\
    B = %p, ldb = %d, beta = %lf, C = %p, ldC = %d)\n",
    ptr_ker_translate->dev_id, ptr_ker_translate->TransA, ptr_ker_translate->TransB,
    ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha,
    (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
    (VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB,
    ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
#endif
  cblas_dgemm(CblasColMajor,
    OpCharToCblas(ptr_ker_translate->TransA), OpCharToCblas(ptr_ker_translate->TransB),
    ptr_ker_translate->M, ptr_ker_translate->N, ptr_ker_translate->K, ptr_ker_translate->alpha,
    (VALUE_TYPE*) *ptr_ker_translate->A, ptr_ker_translate->ldA,
    (VALUE_TYPE*) *ptr_ker_translate->B, ptr_ker_translate->ldB,
    ptr_ker_translate->beta, (VALUE_TYPE*) *ptr_ker_translate->C, ptr_ker_translate->ldC);
}

void cblas_wrap_sgemm(void* backend_data){
    error("cblas_wrap_sgemm: never let empty unimplimented wrapped functions, moron\n");
}
