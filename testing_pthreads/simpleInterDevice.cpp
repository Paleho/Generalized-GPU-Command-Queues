///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Simple test for interdevice dependecies. 
///			Device 0 (GPU) computes the NxN matrix ABCDE
///			Device 1 (GPU) waits for result and computes MABCDE.

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

const double epsilon = 0.00001;

#define IDX2F(i,j,ld) (((j)*(ld)) + (i))

// Column - major matrix initialization
void matrixInit(double * M, int rows, int cols)
{
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
			M[IDX2F(i,j, rows)] = (double) (rand() % 10) - 5;
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

void stallComputations(double* A, double* B, int n, int times, gemm_backend_in<double>* gemmData_stall, CQueue_p Q, int device){
	for(int i = 0; i < times; i++){
		gemmData_stall->TransA = 'N';	// normal matrix A
		gemmData_stall->TransB = 'N';	// normal matrix B
		gemmData_stall->M = n;
		gemmData_stall->N = n;
		gemmData_stall->K = n;
		gemmData_stall->ldA = n;
		gemmData_stall->ldB = n;
		gemmData_stall->ldC = n;
		gemmData_stall->alpha = 1.0;
		gemmData_stall->beta = 0.0;
		gemmData_stall->A = (void **) &A;
		gemmData_stall->B = (void **) &B;
		gemmData_stall->C = (void **) &A;
		gemmData_stall->dev_id = device;

		// Run blas operation
		backend_run_operation(gemmData_stall, "Dgemm", Q);
	}
}

int main(int argc, char ** argv){
	if(argc < 2){
		std::cout << "Invalid number of command line arguments\n";
		std::cout << "Proper usage: " << argv[0] << " [N] [synched/desynched]\n";
		return 2;
	}

	bool synched;
	if(!strcmp(argv[2], "synched")) synched = true;
	else if(!strcmp(argv[2], "desynched")) synched = false;
	else{
		std::cout << "Invalid argv[2]\n";
		std::cout << "Proper usage: " << argv[0] << " [N] [synched/desynched]\n";
		return 2;
	}

	int returnFlag = 0;
	int N = atoi(argv[1]);
	int M = 4096;
 	size_t size_n_by_n = N * N * sizeof(double);
	size_t size_m_by_m = M * M * sizeof(double);

	if(N <= 0){
		std::cout << "Invalid command line arg (N)\n";
		return 2;
	}

	// Allocate input matrices in host memory
	double *h_A = (double *) CoCoMalloc(size_n_by_n, -1); // -1 in loc indicates Host pinned mem
	double *h_B = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_C = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_D = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_E = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_M = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_Res = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_AB = (double *) CoCoMalloc(size_n_by_n, -2);
	double *h_ABC = (double *) CoCoMalloc(size_n_by_n, -2);
	double *h_ABCD = (double *) CoCoMalloc(size_n_by_n, -2);
	double *h_ABCDE = (double *) CoCoMalloc(size_n_by_n, -2);
	double *h_MABCDE = (double *) CoCoMalloc(size_n_by_n, -2);

	double *h_stall_X = (double *) CoCoMalloc(size_m_by_m, -1);
	double *h_stall_Y = (double *) CoCoMalloc(size_m_by_m, -1);

	// Allocate matrices in device 0
	double *d0_A = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_B = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_C = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_D = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_E = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_AB = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_ABC = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_ABCD = (double *) CoCoMalloc(size_n_by_n, 0);
	double *d0_ABCDE = (double *) CoCoMalloc(size_n_by_n, 0);

	double *d0_stall_X = (double *) CoCoMalloc(size_m_by_m, 0);
	double *d0_stall_Y = (double *) CoCoMalloc(size_m_by_m, 0);

	// Allocate matrices in device 1
	double *d1_M = (double *) CoCoMalloc(size_n_by_n, 1);
	double *d1_ABCDE = (double *) CoCoMalloc(size_n_by_n, 1);
	double *d1_MABCDE = (double *) CoCoMalloc(size_n_by_n, 1);

	// Initialize input matrices
	matrixInit(h_A, N, N);
	matrixInit(h_B, N, N);
	matrixInit(h_C, N, N);
	matrixInit(h_D, N, N);
	matrixInit(h_E, N, N);
	matrixInit(h_M, N, N);

	matrixInit(h_stall_X, M, M);
	matrixInit(h_stall_Y, M, M);

	// Device 1 -- only memcpy
	gemm_backend_in<double>* gemmData_p_4 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	CQueue_p Q1 = new CommandQueue(1); 
	CoCoMemcpy2DAsync(d1_M, N, h_M, N, N, N, sizeof(double), 1, -1, Q1);

	/* Device 0 */
	// Q0[]: array of 4 queues in device 0
	CQueue_p * Q0 = (CQueue_p *) CoCoMalloc(4 * sizeof(CQueue_p), -2); // regular (not pinned) host malloc

	gemm_backend_in<double>* gemmData_p_0_stall = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemm_backend_in<double>* gemmData_p_0 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemm_backend_in<double>* gemmData_p_1 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemm_backend_in<double>* gemmData_p_2 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	gemm_backend_in<double>* gemmData_p_3 = (gemm_backend_in<double>*) CoCoMalloc(sizeof(gemm_backend_in<double>), -1);
	Event_p d0_AB_ready = new Event(0);
	Event_p d0_ABC_ready = new Event(0);
	Event_p d0_ABCD_ready = new Event(0);
	Event_p d0_ABCDE_ready = new Event(0);

	for(int i = 0; i < 4; i++){
		Q0[i] = new CommandQueue(0);
	}

	// Q0[0]
	CoCoMemcpy2DAsync(d0_stall_X, M, h_stall_X, M, M, M, sizeof(double), 0, -1, Q0[0]);
	CoCoMemcpy2DAsync(d0_stall_Y, M, h_stall_Y, M, M, M, sizeof(double), 0, -1, Q0[0]);

	// Some random computations just to keep device 0 (Q[0]) busy
	stallComputations(d0_stall_X, d0_stall_Y, M, 10, gemmData_p_0_stall, Q0[0], 0);
	// stall enqueued

	CoCoMemcpy2DAsync(d0_A, N, h_A, N, N, N, sizeof(double), 0, -1, Q0[0]);
	CoCoMemcpy2DAsync(d0_B, N, h_B, N, N, N, sizeof(double), 0, -1, Q0[0]);

	// Compute A*B 
	gemmData_p_0->TransA = 'N';	// normal matrix A
	gemmData_p_0->TransB = 'N';	// normal matrix B
	gemmData_p_0->M = N;
	gemmData_p_0->N = N;
	gemmData_p_0->K = N;
	gemmData_p_0->ldA = N;
	gemmData_p_0->ldB = N;
	gemmData_p_0->ldC = N;
	gemmData_p_0->alpha = 1.0;
	gemmData_p_0->beta = 0.0;
	gemmData_p_0->A = (void **) &d0_A;
	gemmData_p_0->B = (void **) &d0_B;
	gemmData_p_0->C = (void **) &d0_AB;
	gemmData_p_0->dev_id = 0;

	// Run blas operation
	backend_run_operation(gemmData_p_0, "Dgemm", Q0[0]);
	// gemm stores result matrix in C = d0_AB

	d0_AB_ready->record_to_queue(Q0[0]);

	// Q0[1]
	CoCoMemcpy2DAsync(d0_C, N, h_C, N, N, N, sizeof(double), 0, -1, Q0[1]);

	Q0[1]->wait_for_event(d0_AB_ready);

	// Compute A*B*C
	gemmData_p_1->TransA = 'N';	// normal matrix A
	gemmData_p_1->TransB = 'N';	// normal matrix B
	gemmData_p_1->M = N;
	gemmData_p_1->N = N;
	gemmData_p_1->K = N;
	gemmData_p_1->ldA = N;
	gemmData_p_1->ldB = N;
	gemmData_p_1->ldC = N;
	gemmData_p_1->alpha = 1.0;
	gemmData_p_1->beta = 0.0;
	gemmData_p_1->A = (void **) &d0_AB;
	gemmData_p_1->B = (void **) &d0_C;
	gemmData_p_1->C = (void **) &d0_ABC;
	gemmData_p_1->dev_id = 0;

	// Run blas operation
	backend_run_operation(gemmData_p_1, "Dgemm", Q0[1]);
	// gemm stores result matrix in C = d0_ABC

	d0_ABC_ready->record_to_queue(Q0[1]);

	// Q0[2]
	CoCoMemcpy2DAsync(d0_D, N, h_D, N, N, N, sizeof(double), 0, -1, Q0[2]);

	Q0[2]->wait_for_event(d0_ABC_ready);

	// Compute A*B*C*D
	gemmData_p_2->TransA = 'N';	// normal matrix A
	gemmData_p_2->TransB = 'N';	// normal matrix B
	gemmData_p_2->M = N;
	gemmData_p_2->N = N;
	gemmData_p_2->K = N;
	gemmData_p_2->ldA = N;
	gemmData_p_2->ldB = N;
	gemmData_p_2->ldC = N;
	gemmData_p_2->alpha = 1.0;
	gemmData_p_2->beta = 0.0;
	gemmData_p_2->A = (void **) &d0_ABC;
	gemmData_p_2->B = (void **) &d0_D;
	gemmData_p_2->C = (void **) &d0_ABCD;
	gemmData_p_2->dev_id = 0;

	// Run blas operation
	backend_run_operation(gemmData_p_2, "Dgemm", Q0[2]);
	// gemm stores result matrix in C = d0_ABCD

	d0_ABCD_ready->record_to_queue(Q0[2]);

	// Q0[3]
	CoCoMemcpy2DAsync(d0_E, N, h_E, N, N, N, sizeof(double), 0, -1, Q0[3]);

	Q0[3]->wait_for_event(d0_ABCD_ready);

	// Compute A*B*C*D*E
	gemmData_p_3->TransA = 'N';	// normal matrix A
	gemmData_p_3->TransB = 'N';	// normal matrix B
	gemmData_p_3->M = N;
	gemmData_p_3->N = N;
	gemmData_p_3->K = N;
	gemmData_p_3->ldA = N;
	gemmData_p_3->ldB = N;
	gemmData_p_3->ldC = N;
	gemmData_p_3->alpha = 1.0;
	gemmData_p_3->beta = 0.0;
	gemmData_p_3->A = (void **) &d0_ABCD;
	gemmData_p_3->B = (void **) &d0_E;
	gemmData_p_3->C = (void **) &d0_ABCDE;
	gemmData_p_3->dev_id = 0;

	// Run blas operation
	backend_run_operation(gemmData_p_3, "Dgemm", Q0[3]);
	// gemm stores result matrix in C = d0_ABCDE

	// Transfer result to device 1
	CoCoMemcpy2DAsync(d1_ABCDE, N, d0_ABCDE, N, N, N, sizeof(double), 1, 0, Q0[3]);

	d0_ABCDE_ready->record_to_queue(Q0[3]);

	/* Device 1 */
	if(synched)
		Q1->wait_for_event(d0_ABCDE_ready);

	// Compute M*A*B*C*D*E
	gemmData_p_4->TransA = 'N';	// normal matrix A
	gemmData_p_4->TransB = 'N';	// normal matrix B
	gemmData_p_4->M = N;
	gemmData_p_4->N = N;
	gemmData_p_4->K = N;
	gemmData_p_4->ldA = N;
	gemmData_p_4->ldB = N;
	gemmData_p_4->ldC = N;
	gemmData_p_4->alpha = 1.0;
	gemmData_p_4->beta = 0.0;
	gemmData_p_4->A = (void **) &d1_M;
	gemmData_p_4->B = (void **) &d1_ABCDE;
	gemmData_p_4->C = (void **) &d1_MABCDE;
	gemmData_p_4->dev_id = 1;

	// Run blas operation
	backend_run_operation(gemmData_p_4, "Dgemm", Q1);
	// gemm stores result matrix in C = d1_MABCDE

	CoCoMemcpy2DAsync(h_Res, N, d1_MABCDE, N, N, N, sizeof(double), -1, 1, Q1);

	/* HOST */
	// Sync with device 1
	Q1->sync_barrier();

	matrixMultiply(h_A, h_B, N, N, N, 1.0, h_AB);
	matrixMultiply(h_AB, h_C, N, N, N, 1.0, h_ABC);
	matrixMultiply(h_ABC, h_D, N, N, N, 1.0, h_ABCD);
	matrixMultiply(h_ABCD, h_E, N, N, N, 1.0, h_ABCDE);
	matrixMultiply(h_M, h_ABCDE, N, N, N, 1.0, h_MABCDE);

	// Verify result
	if(!verifyRes(h_MABCDE, h_Res, N, N)){
		std::cout << "Fail: The result of " << argv[2] << " computations is incorrect!\n";
		returnFlag = 1;

		std::cout << "Some more elements: \n";

		printf("h_MABCDE[0, 1] = %0.5lf BUT Res[0, 1] = %0.5lf\n", h_MABCDE[IDX2F(0, 1, N)], h_Res[IDX2F(0, 1, N)]);
		printf("h_MABCDE[1, 0] = %0.5lf BUT Res[1, 0] = %0.5lf\n", h_MABCDE[IDX2F(1, 0, N)], h_Res[IDX2F(1, 0, N)]);
		printf("h_MABCDE[2, 1] = %0.5lf BUT Res[2, 1] = %0.5lf\n", h_MABCDE[IDX2F(2, 1, N)], h_Res[IDX2F(2, 1, N)]);
	}
	else{
		std::cout << "Success: The result of " << argv[2] << " computations is correct!\n";
		returnFlag = 0;
	}

	for(int i = 0; i < 4; i++) delete(Q0[i]);
	CoCoFree(Q0, -2);

	delete(d0_AB_ready);
	delete(d0_ABC_ready);
	delete(d0_ABCD_ready);
	delete(d0_ABCDE_ready);

	delete(Q1);

	// Free device 0 mem
	CoCoFree(d0_A, 0);
	CoCoFree(d0_B, 0);
	CoCoFree(d0_C, 0);
	CoCoFree(d0_D, 0);
	CoCoFree(d0_E, 0);

	CoCoFree(d0_AB, 0);
	CoCoFree(d0_ABC, 0);
	CoCoFree(d0_ABCD, 0);
	CoCoFree(d0_ABCDE, 0);

	CoCoFree(d0_stall_X, 0);
	CoCoFree(d0_stall_Y, 0);

	CoCoFree(gemmData_p_0, -1);
	CoCoFree(gemmData_p_1, -1);
	CoCoFree(gemmData_p_2, -1);
	CoCoFree(gemmData_p_3, -1);
	CoCoFree(gemmData_p_0_stall, -1);

	// Free device 1 mem
	CoCoFree(d1_M, 1);
	CoCoFree(d1_ABCDE, 1);
	CoCoFree(d1_MABCDE, 1);

	CoCoFree(gemmData_p_4, -1);

	// Free host mem
	CoCoFree(h_A, -1);
	CoCoFree(h_B, -1);
	CoCoFree(h_C, -1);
	CoCoFree(h_D, -1);
	CoCoFree(h_E, -1);
	CoCoFree(h_M, -1);
	CoCoFree(h_Res, -1);

	CoCoFree(h_AB, -2);
	CoCoFree(h_ABC, -2);
	CoCoFree(h_ABCD, -2);
	CoCoFree(h_ABCDE, -2);
	CoCoFree(h_MABCDE, -2);

	CoCoFree(h_stall_X, -1);
	CoCoFree(h_stall_Y, -1);

	return returnFlag;
}