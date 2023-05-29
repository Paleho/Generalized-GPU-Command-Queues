///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <cuda.h>
#include <pthread.h>
#include <cblas.h>
#include "cublas_v2.h"
#include <unihelpers.hpp>


using namespace std;

#define STREAM_POOL_SZ 16

// Queue lock
inline void get_lock_q(int * lock){
	while(__sync_lock_test_and_set (lock, 1));
}
inline void release_lock_q(int * lock){
	__sync_lock_release(lock);
}

typedef struct pthread_task { 
    void* func; 
    void* data; 
}* pthread_task_p; 

typedef struct queue_data { 
    void * taskQueue;
    pthread_t threadId;
    int queueLock; 
    bool terminate;
    cudaStream_t * stream_pool;
    int stream_ctr;
    cublasHandle_t* handle_p;
}* queue_data_p; 

typedef struct pthread_event {
    void * taskQueue;
    pthread_t threadId;
    event_status estate;
}* pthread_event_p; 

typedef struct gemm_backend_in{
	char TransA,  TransB;
	int M, N, K, ldA, ldB, ldC;
	VALUE_TYPE alpha,beta;
	void **A, **B, **C;
	short dev_id;
}* gemm_backend_in_p;

typedef struct axpy_backend_in{
	int N, incx, incy;
	VALUE_TYPE alpha;
	void **x, **y;
	short dev_id;
}* axpy_backend_in_p;

typedef struct wider_backend_in{
    queue_data_p q_data;
    void* backend_data;
}* wider_backend_in_p;

/// Select and run a wrapped operation (e.g. gemm, axpy) depending on opname
void backend_run_operation(void* backend_data, const char* opname, CQueue_p run_queue);

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, long int* ldim, long int dim1, long int dim2);

cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src);
CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src);
cublasOperation_t OpCharToCublas(char src);
CBLAS_TRANSPOSE OpCharToCblas(char src);
char PrintCublasOp(cublasOperation_t src);

/// Internally used utils TODO: Is this the correct way softeng wise?
void cudaCheckErrors();

// Lock wrapped_lock. This functions is fired in a queue to lock when it reaches that point.
void CoCoQueueLock(void* wrapped_lock);
// Unlock wrapped_lock. This functions is fired in a queue to unlock when it reaches that point.
void CoCoQueueUnlock(void* wrapped_lock);

// Struct containing an int pointer
typedef struct Ptr_atomic_int{
	std::atomic<int>* ato_int_ptr;
}* Ptr_atomic_int_p;
void CoCoIncAsync(void* wrapped_ptr_int);
void CoCoDecAsync(void* wrapped_ptr_int);

// Struct containing an int pointer and an int for Asynchronous set
typedef struct Ptr_and_int{
	int* int_ptr;
	int val;
}* Ptr_and_int_p;
void CoCoSetInt(void* wrapped_ptr_and_val);

// Struct containing a void pointer and a void for Asynchronous set
typedef struct Ptr_and_parent{
	void** ptr_parent;
	void* ptr_val;
}* Ptr_and_parent_p;
void CoCoSetPtr(void* wrapped_ptr_and_parent);

void CoCoSetTimerAsync(void* wrapped_timer_Ptr);

void CoCoFreeAllocAsync(void* backend_data);

void cublas_wrap_daxpy(void* backend_data);
void cublas_wrap_saxpy(void* backend_data, void* queue_wrap_p);
void cublas_wrap_dgemm(void* backend_data);
void cublas_wrap_sgemm(void* backend_data, void* queue_wrap_p);

void cblas_wrap_daxpy(void* backend_data);
void cblas_wrap_saxpy(void* backend_data);
void cblas_wrap_dgemm(void* backend_data);
void cblas_wrap_sgemm(void* backend_data);