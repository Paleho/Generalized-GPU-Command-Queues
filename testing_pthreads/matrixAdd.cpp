///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Test the use of a single Command Queue. Use the queue to transfer data and perform the dgemm operation 
///			alpha*A * B with the backend_run_operation function

#include <cstdio>
#include <typeinfo>
#include <iostream>

#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

#define IDX2F(i,j,ld) (((j)*(ld)) + (i))

const double epsilon = 0.00001;

// Column - major matrix initialization
void matrixInit(double * M, int rows, int cols)
{
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
			M[IDX2F(i,j, rows)] = (double) (rand() % 100);
}

void identityMatrixInit(double * M, int rows, int cols)
{
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++){
			if(i == j) M[IDX2F(i,j, rows)] = 1.0;
			else M[IDX2F(i,j, rows)] = 0.0;
		}
}

void printMat(double * M, int rows, int cols);

int verifyRes(double* A, double* B, double* C, int n, double* Res)
{
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++){
			double correctResult = A[IDX2F(i,j, n)] + B[IDX2F(i,j, n)] + C[IDX2F(i,j, n)];
			double dif = Res[IDX2F(i,j, n)] - correctResult;
			bool expr1 = dif > epsilon;
			bool expr2 = dif < (-1)*epsilon;
			if(expr1 || expr2){
				printf("verifyRes: incorrect at (%d, %d) -- should be = %0.5lf BUT Res[i, j] = %0.5lf -- Error = %0.5lf\n", i, j, correctResult, Res[IDX2F(i,j, n)], dif);
				return 0;
			} 
		}
	return 1;
}

int main(int argc, char ** argv){
	int returnFlag = 0;
	int N = 256;
	size_t size_n_by_n = N * N * sizeof(double);

	// Allocate matrices in host memory
	double *h_A = (double *) CoCoMalloc(size_n_by_n, -1); // -1 in loc indicates Host pinned mem
	double *h_B = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_C = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_I = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_Res = (double *) CoCoMalloc(size_n_by_n, -1);

	// Initialize input matrices
	matrixInit(h_A, N, N);
	matrixInit(h_B, N, N);
	matrixInit(h_C, N, N);
	identityMatrixInit(h_I, N, N);

	// Allocate matrices in device memory
 	double *d_A = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d_B = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d_C = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d_I = (double *) CoCoMalloc(size_n_by_n, 0);

	// Test CommandQueue
	CQueue_p MyQueue_p = new CommandQueue(0);
	
	// Copy data from host memory to device memory
	CoCoMemcpyAsync(d_A, h_A, size_n_by_n, 0, -1, MyQueue_p); // -1 in loc_src indicates Host
 	CoCoMemcpyAsync(d_B, h_B, size_n_by_n, 0, -1, MyQueue_p);
	CoCoMemcpyAsync(d_C, h_C, size_n_by_n, 0, -1, MyQueue_p);
	CoCoMemcpyAsync(d_I, h_I, size_n_by_n, 0, -1, MyQueue_p);

	// Prepare blas operation
	// C = A*I + C = A + C
	gemm_backend_in<double>* gemmData_p = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemmData_p->TransA = 'N';	// normal matrix A
	gemmData_p->TransB = 'N';	// normal matrix B
	gemmData_p->M = N;
	gemmData_p->N = N;
	gemmData_p->K = N;
	gemmData_p->ldA = N;	// in column - major format ldA = rows(A) = N
	gemmData_p->ldB = N;
	gemmData_p->ldC = N;
	gemmData_p->alpha = 1.0;
	gemmData_p->beta = 1.0;
	gemmData_p->A = (void **) &d_A;
	gemmData_p->B = (void **) &d_I;
	gemmData_p->C = (void **) &d_C;
	gemmData_p->dev_id = 0;

	// Run blas operation
	backend_run_operation(gemmData_p, "gemm", MyQueue_p);
	// gemm stores result matrix d_C

	// Prepare blas operation
	// C = B*I + C = B + C
	gemm_backend_in<double>* gemmData_p_2 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemmData_p_2->TransA = 'N';	// normal matrix A
	gemmData_p_2->TransB = 'N';	// normal matrix B
	gemmData_p_2->M = N;
	gemmData_p_2->N = N;
	gemmData_p_2->K = N;
	gemmData_p_2->ldA = N;	// in column - major format ldA = rows(A) = N
	gemmData_p_2->ldB = N;
	gemmData_p_2->ldC = N;
	gemmData_p_2->alpha = 1.0;
	gemmData_p_2->beta = 1.0;
	gemmData_p_2->A = (void **) &d_B;
	gemmData_p_2->B = (void **) &d_I;
	gemmData_p_2->C = (void **) &d_C;
	gemmData_p_2->dev_id = 0;

	// Run blas operation
	backend_run_operation(gemmData_p_2, "gemm", MyQueue_p);
	// gemm stores result matrix d_C
 
	// Copy result from device memory to host memory
 	// h_Res contains the result in host memory
 	CoCoMemcpyAsync(h_Res, d_C, size_n_by_n, -1, 0, MyQueue_p);

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
	if(!verifyRes(h_A, h_B, h_C, N, h_Res)){
		std::cout << "Fail: The result of gemm is incorrect!\n";
		returnFlag = 1;
	}
	else{
		std::cout << "Success: The result of gemm is correct!\n";
		returnFlag = 0;
	}
	
	// Free device memory
	CoCoFree(d_A, 0);
	CoCoFree(d_B, 0);
	CoCoFree(d_C, 0);
	CoCoFree(d_I, 0);

	// Free host memory
	CoCoFree(h_A, -1); // -1 in loc indicates Host pinned mem
	CoCoFree(h_B, -1);
	CoCoFree(h_C, -1);
	CoCoFree(h_I, -1);
	CoCoFree(h_Res, -1);
	CoCoFree(gemmData_p, -1);
	CoCoFree(gemmData_p_2, -1);
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