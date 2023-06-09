///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <pthread.h>
#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "backend_wrappers.hpp"

#define THREAD_POOL_SZ 128
pthread_t thread_pool[THREAD_POOL_SZ];
pthread_attr_t thread_pool_attr[THREAD_POOL_SZ];
int thread_ctr = 0, thread_lock = 0 ;

void backend_run_operation(void* backend_data, const char* opname, CQueue_p run_queue){

  short lvl = 5;
  if (!strcmp(opname, "Dgemm")){
    gemm_backend_in<double>* ptr_ker_translate = (gemm_backend_in<double>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_dgemm, backend_data, "cblas_wrap_dgemm");
    else if(ptr_ker_translate->dev_id >= 0){
      // prepare data (add queue data)
      wider_backend_in_p wider_data = new wider_backend_in;
      wider_data->q_data = (queue_data_p) run_queue->cqueue_backend_data;
      wider_data->backend_data = backend_data;

      run_queue->add_host_func((void*)&cublas_wrap_dgemm, (void*)wider_data, "cublas_wrap_dgemm");
    }
    else error("backend_run_operation(gemm,double): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if (!strcmp(opname, "Dgemv")){
    gemv_backend_in<double>* ptr_ker_translate = (gemv_backend_in<double>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_dgemv, backend_data, "cblas_wrap_dgemv");
    else if(ptr_ker_translate->dev_id >= 0){
      // prepare data (add queue data)
      wider_backend_in_p wider_data = new wider_backend_in;
      wider_data->q_data = (queue_data_p) run_queue->cqueue_backend_data;
      wider_data->backend_data = backend_data;

      run_queue->add_host_func((void*)&cublas_wrap_dgemv, (void*)wider_data, "cublas_wrap_dgemv");
    }
    else error("backend_run_operation(dgemv): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Sgemm")){
    gemm_backend_in<float>* ptr_ker_translate = (gemm_backend_in<float>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_sgemm, backend_data, "cblas_wrap_sgemm");
    else if(ptr_ker_translate->dev_id >= 0){
      // prepare data (add queue data)
      wider_backend_in_p wider_data = new wider_backend_in;
      wider_data->q_data = (queue_data_p) run_queue->cqueue_backend_data;
      wider_data->backend_data = backend_data;

      run_queue->add_host_func((void*)&cublas_wrap_sgemm, (void*)wider_data, "cublas_wrap_sgemm");
    }
    else error("backend_run_operation(sgemm): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Daxpy")){
    axpy_backend_in<double>* ptr_ker_translate = (axpy_backend_in<double>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_daxpy, backend_data, "cblas_wrap_daxpy");
    else if(ptr_ker_translate->dev_id >= 0){
      // prepare data (add queue data)
      wider_backend_in_p wider_data = new wider_backend_in;
      wider_data->q_data = (queue_data_p) run_queue->cqueue_backend_data;
      wider_data->backend_data = backend_data;

      run_queue->add_host_func((void*)&cublas_wrap_daxpy, (void*)wider_data, "cublas_wrap_daxpy");
    }
    else error("backend_run_operation(axpy,double): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Saxpy")){
    axpy_backend_in<float>* ptr_ker_translate = (axpy_backend_in<float>*) backend_data;
    error("backend_run_operation(axpy,float): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Ddot")){
    dot_backend_in<double>* ptr_ker_translate = (dot_backend_in<double>*) backend_data;
    if(ptr_ker_translate->dev_id == -1) run_queue->add_host_func((void*)&cblas_wrap_ddot, backend_data, "cblas_wrap_ddot");
    else if(ptr_ker_translate->dev_id >= 0){
      // prepare data (add queue data)
      wider_backend_in_p wider_data = new wider_backend_in;
      wider_data->q_data = (queue_data_p) run_queue->cqueue_backend_data;
      wider_data->backend_data = backend_data;

      run_queue->add_host_func((void*)&cublas_wrap_ddot, (void*)wider_data, "cublas_wrap_ddot");
    }
    else error("backend_run_operation(ddot): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else if(!strcmp(opname, "Sdot")){
    dot_backend_in<float>* ptr_ker_translate = (dot_backend_in<float>*) backend_data;
    error("backend_run_operation(sdot): Not implemented for dev_id = %d\n", ptr_ker_translate->dev_id);
  }
  else error("backend_run_operation: unkown/not implemented opname=%s\n", opname);
}

