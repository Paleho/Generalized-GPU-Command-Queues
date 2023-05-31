///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Test the use of a single Command Queue. Use the queue to transfer data and perform the dgemm operation 
///			alpha*A * B with the backend_run_operation function

#include <cstdio>
#include <typeinfo>
#include <iostream>

#include "unihelpers.hpp"
#include "pthreads_backend_wrappers.hpp"

#define IDX2F(i,j,ld) (((j)*(ld)) + (i))

const double epsilon = 0.00001;

// Column - major matrix initialization
void matrixInit(double * M, int rows, int cols)
{
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
			M[IDX2F(i,j, rows)] = (double) (rand() % 100);
}

// A = m x k
// B = k x n
// C = m x n
int verifyRes(double* A, double* B, double* C, int m, int n, int k, double alpha, double beta, double* Res)
{
	int rows = m;
	int cols = n;
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++){
			double mulRes = 0;
			for(int x = 0; x < k; x++)
				mulRes += A[IDX2F(i,x, m)]*B[IDX2F(x,j, k)];
			double correctResult = alpha * mulRes + beta * C[IDX2F(i,j, m)];
			double dif = Res[IDX2F(i,j, m)] - correctResult;
			bool expr1 = dif > epsilon;
			bool expr2 = dif < (-1)*epsilon;
			if(expr1 || expr2){
				printf("verifyRes: incorrect at (%d, %d) -- should be = %0.5lf BUT Res[i, j] = %0.5lf -- Error = %0.5lf\n", i, j, correctResult, Res[IDX2F(i,j, m)], dif);
				return 0;
			} 
		}
	return 1;
}

void printMat(double * M, int rows, int cols);

int main(int argc, char ** argv){
	int returnFlag = 0;
	int M = 512;
	int K = 128;
	int N = 256;
 	size_t size_m_by_k = M * K * sizeof(double);
	size_t size_k_by_n = K * N * sizeof(double);
	size_t size_m_by_n = M * N * sizeof(double);
	double alpha = 1.0;

	// Allocate matrices in host memory
	double *h_A = (double *) CoCoMalloc(size_m_by_k, -1); // -1 in loc indicates Host pinned mem
	double *h_B = (double *) CoCoMalloc(size_k_by_n, -1);
	double *h_Res = (double *) CoCoMalloc(size_m_by_n, -1);

	// Initialize input matrices
	matrixInit(h_A, N, K);
	matrixInit(h_B, K, N);

	// Get dev_id
	int dev_id = CoCoPeLiaGetDevice();
	// std::cout << "device id = " << dev_id << std::endl;

	// Allocate matrices in device memory
 	double *d_A = (double *) CoCoMalloc(size_m_by_k, dev_id);
	double *d_B = (double *) CoCoMalloc(size_k_by_n, dev_id);
	double *d_C = (double *) CoCoMalloc(size_m_by_n, dev_id);

	// Test CommandQueue
	CQueue_p MyQueue_p = new CommandQueue(dev_id);
	
	// Copy data from host memory to device memory
	CoCoMemcpyAsync(d_A, h_A, size_m_by_k, dev_id, -1, MyQueue_p); // -1 in loc_src indicates Host
 	CoCoMemcpyAsync(d_B, h_B, size_k_by_n, dev_id, -1, MyQueue_p);

	// Prepare blas operation
	// C = alpha * A * B + beta * C
	// A = m x k
	// B = k x n
	// C = m x n
	gemm_backend_in<double>* gemmData_p = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemmData_p->TransA = 'N';	// normal matrix A
	gemmData_p->TransB = 'N';	// normal matrix B
	gemmData_p->M = M;
	gemmData_p->N = N;
	gemmData_p->K = K;
	gemmData_p->ldA = M;	// in column - major format ldA = rows(A) = M 
	gemmData_p->ldB = K;
	gemmData_p->ldC = M;
	gemmData_p->alpha = alpha;
	gemmData_p->beta = (double) 0.0;
	gemmData_p->A = (void **) &d_A;
	gemmData_p->B = (void **) &d_B;
	gemmData_p->C = (void **) &d_C;
	gemmData_p->dev_id = dev_id;

	// Run blas operation
	backend_run_operation(gemmData_p, "Dgemm", MyQueue_p);
	// gemm stores result matrix d_C
 
	// Copy result from device memory to host memory
 	// h_Res contains the result in host memory
 	CoCoMemcpyAsync(h_Res, d_C, size_m_by_n, -1, dev_id, MyQueue_p);

	// Wait for MyQueue tasks to complete
	MyQueue_p->sync_barrier();

	// // Print matrices
	// printf("h_A = ");
	// printMat(h_A, M, K);
	// printf("h_B = ");
	// printMat(h_B, K, N);
	// printf("h_Res = ");
	// printMat(h_Res, M, N);

	// Verify result
	if(!verifyRes(h_A, h_B, h_Res, M, N, K, alpha, 0.0, h_Res)){
		std::cout << "Fail: The result of gemm is incorrect!\n";
		returnFlag = 1;
	}
	else{
		std::cout << "Success: The result of gemm is correct!\n";
		returnFlag = 0;
	}
	
	// Free device memory
	CoCoFree(d_A, dev_id);
	CoCoFree(d_B, dev_id);
	CoCoFree(d_C, dev_id);

	// Free host memory
	CoCoFree(h_A, -1); // -1 in loc indicates Host pinned mem
	CoCoFree(h_B, -1);
	CoCoFree(h_Res, -1);
	CoCoFree(gemmData_p, -1);
	delete(MyQueue_p);

	return returnFlag;
}

void printMat(double * M, int rows, int cols)
{

	for(int i = 0; i < rows; i++){
		printf("[ ");
		for (int j = 0; j < cols-1; j++)
			printf("%f, ", M[IDX2F(i,j, rows)]);
		printf("%f]\n", M[IDX2F(i, cols-1, rows)]);
	}
}