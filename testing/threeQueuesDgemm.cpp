///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Test the use of three Command Queues along with Events.  
///			Q1: c1 * A*B = res1
///			Q2: c2 * C*D = res2
///			Q3: c3 * res1*res2 + A = Result
///			A: M x K
///			B: K x N
///			res1: M x N
///			C: N x L
///			D: L x K
///			res2: N x K
///			res1*res2: M x K

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

// Res = A + B
// A, B: m x n
void matrixAdd(double* A, double* B, int m, int n, double* Res);

// Res = alpha * A*B
// A: m x k
// B: k x n
void matrixMultiply(double* A, double* B, int m, int k, int n, double alpha, double* Res);

int verifyRes(double* hostComputed_Res, double* gpuComputed_Res, int m, int n)
{
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++){
			double correctResult = hostComputed_Res[IDX2F(i,j, m)];
			double dif = gpuComputed_Res[IDX2F(i,j, m)] - correctResult;
			bool expr1 = dif > epsilon;
			bool expr2 = dif < (-1)*epsilon;
			if(expr1 || expr2){
				printf("verifyRes: incorrect at (%d, %d) -- should be = %0.5lf BUT Res[i, j] = %0.5lf -- Error = %0.5lf\n", i, j, correctResult, gpuComputed_Res[IDX2F(i,j, m)], dif);
				return 0;
			} 
		}
	return 1;
}

void printMat(double * M, int rows, int cols);

int main(int argc, char ** argv){
	int returnFlag = 0;
	int M = 1024;
	int K = 1024;
	int N = 1024;
	int L = 1024;
 	size_t size_m_by_k = M * K * sizeof(double);
	size_t size_k_by_n = K * N * sizeof(double);
	size_t size_m_by_n = M * N * sizeof(double);
	size_t size_n_by_l = N * L * sizeof(double);
	size_t size_l_by_k = L * K * sizeof(double);
	size_t size_n_by_k = N * K * sizeof(double);
	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 1.0;

	// Allocate matrices in host memory
	double *h_A = (double *) CoCoMalloc(size_m_by_k, -1); // -1 in loc indicates Host pinned mem
	double *h_B = (double *) CoCoMalloc(size_k_by_n, -1);
	double *h_res1 = (double *) CoCoMalloc(size_m_by_n, -1);
	double *h_C = (double *) CoCoMalloc(size_n_by_l, -1);
	double *h_D = (double *) CoCoMalloc(size_l_by_k, -1);
	double *h_res2 = (double *) CoCoMalloc(size_n_by_k, -1);
	double *h_hostComputed_Res = (double *) CoCoMalloc(size_m_by_k, -1);
	double *h_gpuComputed_Res = (double *) CoCoMalloc(size_m_by_k, -1);

	// Initialize input matrices
	matrixInit(h_A, N, K);
	matrixInit(h_B, K, N);
	matrixInit(h_C, N, L);
	matrixInit(h_D, L, K);

	// Get dev_id
	int dev_id = CoCoPeLiaGetDevice();
	// std::cout << "device id = " << dev_id << std::endl;

	// Allocate matrices in device memory
 	double *d_A = (double *) CoCoMalloc(size_m_by_k, dev_id);
	double *d_B = (double *) CoCoMalloc(size_k_by_n, dev_id);
	double *d_res1 = (double *) CoCoMalloc(size_m_by_n, dev_id);
	double *d_C = (double *) CoCoMalloc(size_n_by_l, dev_id);
	double *d_D = (double *) CoCoMalloc(size_l_by_k, dev_id);
	double *d_res2 = (double *) CoCoMalloc(size_n_by_k, dev_id);

	// Initialize Command Queues
	CQueue_p Q1_p = new CommandQueue(dev_id);
	CQueue_p Q2_p = new CommandQueue(dev_id);
	CQueue_p Q3_p = new CommandQueue(dev_id);
	
	// Copy matrices from host memory to device memory
	CoCoMemcpyAsync(d_A, h_A, size_m_by_k, dev_id, -1, Q1_p); // -1 in loc_src indicates Host
	CoCoMemcpyAsync(d_B, h_B, size_k_by_n, dev_id, -1, Q1_p);

	// Use queue 2 for C and D transfers
	CoCoMemcpyAsync(d_C, h_C, size_n_by_l, dev_id, -1, Q2_p); // -1 in loc_src indicates Host
	CoCoMemcpyAsync(d_D, h_D, size_l_by_k, dev_id, -1, Q2_p);

	// Prepare blas operation
	// C = alpha * A * B + beta * C
	// A = m x k
	// B = k x n
	// C = m x n
	// Q1: c1 *  A*B -> store in d_res1
	gemm_backend_in<double>* gemmData_p_1 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemmData_p_1->TransA = 'N';	// normal matrix A
	gemmData_p_1->TransB = 'N';	// normal matrix B
	gemmData_p_1->M = M;
	gemmData_p_1->N = N;
	gemmData_p_1->K = K;
	gemmData_p_1->ldA = M;	// in column - major format ldA = rows(A) = M 
	gemmData_p_1->ldB = K;
	gemmData_p_1->ldC = M;
	gemmData_p_1->alpha = c1;
	gemmData_p_1->beta = (double) 0.0;
	gemmData_p_1->A = (void **) &d_A;
	gemmData_p_1->B = (void **) &d_B;
	gemmData_p_1->C = (void **) &d_res1;
	gemmData_p_1->dev_id = dev_id;

	// Run blas operation
	backend_run_operation(gemmData_p_1, "Dgemm", Q1_p);
	// gemm stores result matrix in C = d_res1

	// Enqueue event
	Event_p event_p_1 = new Event(dev_id);
	event_p_1->record_to_queue(Q1_p);

	// Prepare blas operation
	// C = alpha * A * B + beta * C
	// A = C = n x l
	// B = D = l x k
	// C = res2 = n x k
	// Q2: c2 * C*D -> store in d_res2
	gemm_backend_in<double>* gemmData_p_2 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemmData_p_2->TransA = 'N';	// normal matrix A
	gemmData_p_2->TransB = 'N';	// normal matrix B
	gemmData_p_2->M = N;
	gemmData_p_2->N = K;
	gemmData_p_2->K = L;
	gemmData_p_2->ldA = N;	// in column - major format ldA = rows(C) = N 
	gemmData_p_2->ldB = L;
	gemmData_p_2->ldC = N;
	gemmData_p_2->alpha = c2;
	gemmData_p_2->beta = (double) 0.0;
	gemmData_p_2->A = (void **) &d_C;
	gemmData_p_2->B = (void **) &d_D;
	gemmData_p_2->C = (void **) &d_res2;
	gemmData_p_2->dev_id = dev_id;

	// Run blas operation
	backend_run_operation(gemmData_p_2, "Dgemm", Q2_p);
	// gemm stores result matrix in C = d_res2

	// Enqueue event
	Event_p event_p_2 = new Event(dev_id);
	event_p_2->record_to_queue(Q2_p);

	// Prepare blas operation
	// C = alpha * A * B + beta * C
	// A = res1 = m x n
	// B = res2 = n x k
	// C = A = m x k
	// Q3: c3 * res1*res2 + A -> store in A
	gemm_backend_in<double>* gemmData_p_3 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemmData_p_3->TransA = 'N';	// normal matrix A
	gemmData_p_3->TransB = 'N';	// normal matrix B
	gemmData_p_3->M = M;
	gemmData_p_3->N = K;
	gemmData_p_3->K = N;
	gemmData_p_3->ldA = M;	// in column - major format ldA = rows(res1) = M
	gemmData_p_3->ldB = N;
	gemmData_p_3->ldC = M;
	gemmData_p_3->alpha = c3;
	gemmData_p_3->beta = (double) 1.0;
	gemmData_p_3->A = (void **) &d_res1;
	gemmData_p_3->B = (void **) &d_res2;
	gemmData_p_3->C = (void **) &d_A;
	gemmData_p_3->dev_id = dev_id;

	// Q3 has to wait Q1 (using event_p_1) and Q2 (using event_p_2)
	Q3_p->wait_for_event(event_p_1);
	Q3_p->wait_for_event(event_p_2);

	// Run blas operation
	backend_run_operation(gemmData_p_3, "Dgemm", Q3_p);
	// gemm stores result matrix in C = d_A
 
	// Copy result from device memory to host memory
	// h_gpuComputed_Res contains the result in host memory
	CoCoMemcpyAsync(h_gpuComputed_Res, d_A, size_m_by_k, -1, dev_id, Q3_p);

	// Host performs the same computation
	matrixMultiply(h_A, h_B, M, K, N, c1, h_res1);
	matrixMultiply(h_C, h_D, N, L, K, c2, h_res2);
	matrixMultiply(h_res1, h_res2, M, N, K, c3, h_hostComputed_Res);
	matrixAdd(h_hostComputed_Res, h_A, M, K, h_hostComputed_Res);

	// Wait for MyQueue tasks to complete
	Q3_p->sync_barrier();

	// // Print matrices
	// printf("h_A = ");
	// printMat(h_A, M, K);
	// printf("h_B = ");
	// printMat(h_B, K, N);
	// printf("h_Res = ");
	// printMat(h_Res, M, N);

	// Verify result
	if(!verifyRes(h_hostComputed_Res, h_gpuComputed_Res, M, K)){
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
	CoCoFree(d_res1, dev_id);
	CoCoFree(d_C, dev_id);
	CoCoFree(d_D, dev_id);
	CoCoFree(d_res2, dev_id);

	// Free host memory
	CoCoFree(h_A, -1); // -1 in loc indicates Host pinned mem
	CoCoFree(h_B, -1);
	CoCoFree(h_res1, -1);
	CoCoFree(h_C, -1);
	CoCoFree(h_D, -1);
	CoCoFree(h_res2, -1);
	CoCoFree(h_hostComputed_Res, -1);
	CoCoFree(h_gpuComputed_Res, -1);
	CoCoFree(gemmData_p_1, -1);
	CoCoFree(gemmData_p_2, -1);
	CoCoFree(gemmData_p_3, -1);
	delete(Q1_p);
	delete(Q2_p);
	delete(Q3_p);
	delete(event_p_1);
	delete(event_p_2);

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

void matrixAdd(double* A, double* B, int m, int n, double* Res)
{
	for(int i = 0; i < m; i++)
		for(int j = 0; j < n; j++)
			Res[IDX2F(i,j,m)] = A[IDX2F(i,j,m)] + B[IDX2F(i,j,m)];
}

void matrixMultiply(double* A, double* B, int m, int k, int n, double alpha, double* Res)
{
	int rows = m;
	int cols = n;
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++){
			double sum = 0;
			for(int x = 0; x < k; x++)
				sum += A[IDX2F(i,x,m)]*B[IDX2F(x,j,k)];

			Res[IDX2F(i,j,m)] = alpha * sum;
		}
}