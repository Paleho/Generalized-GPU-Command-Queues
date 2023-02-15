///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <pthread.h>
#include <cblas.h>

#include "pthreads_backend_wrappers.hpp"

#define THREAD_POOL_SZ 128
pthread_t thread_pool[THREAD_POOL_SZ];
pthread_attr_t thread_pool_attr[THREAD_POOL_SZ];
int thread_ctr = 0, thread_lock = 0 ;

void backend_run_operation(void* backend_data, const char* opname, CQueue_p run_queue){
  short lvl = 5;
  if (!strcmp(opname, "gemm")){
    gemm_backend_in_p ptr_ker_translate = (gemm_backend_in_p) backend_data;
    if (std::is_same<VALUE_TYPE, double>::value){

      run_queue->add_host_func((void*)&cblas_wrap_dgemm, backend_data);
    }
    else if (std::is_same<VALUE_TYPE, float>::value){

      run_queue->add_host_func((void*)&cblas_wrap_sgemm, backend_data);
    }
    else error("backend_run_operation(gemm): Not implemented for VALUETYPE\n");
  }
  else if(!strcmp(opname, "axpy")){
    axpy_backend_in_p ptr_ker_translate = (axpy_backend_in_p) backend_data;
    if (std::is_same<VALUE_TYPE, double>::value){
      
      run_queue->add_host_func((void*)&cblas_wrap_daxpy, backend_data);
    }
    else if (std::is_same<VALUE_TYPE, float>::value){
      
      run_queue->add_host_func((void*)&cblas_wrap_saxpy, backend_data);
    }
    else error("backend_run_operation(axpy): Not implemented for VALUETYPE\n");
  }
  else error("backend_run_operation: unkown/not implemented opname=%s\n", opname);
}

