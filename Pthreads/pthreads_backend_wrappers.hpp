///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief 
///

#include <pthread.h>
#include <cblas.h>
#include "cublas_v2.h"
#include <unihelpers.hpp>


using namespace std;

typedef struct pthread_task { 
    void* func; 
    void* data; 
}* pthread_task_p; 

typedef struct queue_data { 
    void * taskQueue;
    pthread_t threadId;
    int queueLock; 
    bool terminate;
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

/// Select and run a wrapped operation (e.g. gemm, axpy) depending on opname
void backend_run_operation(void* backend_data, const char* opname, CQueue_p run_queue);

void cblas_wrap_daxpy(void* backend_data);
void cblas_wrap_saxpy(void* backend_data);
void cblas_wrap_dgemm(void* backend_data);
void cblas_wrap_sgemm(void* backend_data);

void TransposeTranslate(char TransChar, CBLAS_TRANSPOSE* cblasFlag, cublasOperation_t* cuBLASFlag, long int* ldim, long int dim1, long int dim2);

cublasOperation_t OpCblasToCublas(CBLAS_TRANSPOSE src);
CBLAS_TRANSPOSE OpCublasToCblas(cublasOperation_t src);
cublasOperation_t OpCharToCublas(char src);
CBLAS_TRANSPOSE OpCharToCblas(char src);
char PrintCublasOp(cublasOperation_t src);
