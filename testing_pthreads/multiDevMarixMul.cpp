///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Test the use of 24 Command Queues in 4 devices (3 GPUs + Host).  
///			
///			Perform tiled matrix multiplication M*A*B = R
///				(all matrices are NxN)

#include <cstdio>
#include <iostream>

#include "unihelpers.hpp"
#include "pthreads_backend_wrappers.hpp"

const double epsilon = 0.00001;

#define IDX2F(i,j,ld) (((j)*(ld)) + (i))

// Column - major matrix initialization
void matrixInit(double * M, int rows, int cols)
{
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
			M[IDX2F(i,j, rows)] = (double) (rand() % 100);
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

int main(int argc, char ** argv){
	int returnFlag = 0;
	int N = 1024;
 	size_t size_n_by_n = N * N * sizeof(double);
	size_t quarter_size = N/2 * N/2 * sizeof(double);

	// Allocate input matrices h_A, h_B, h_M in host memory
	double *h_A = (double *) CoCoMalloc(size_n_by_n, -1); // -1 in loc indicates Host pinned mem
	double *h_B = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_M = (double *) CoCoMalloc(size_n_by_n, -1);
	double *h_Res = (double *) CoCoMalloc(size_n_by_n, -1);

	// Initialize input matrices
	matrixInit(h_A, N, N);
	matrixInit(h_B, N, N);
	matrixInit(h_M, N, N);

	// copy-back matrices for device 0 
	double *h_M11A11B11 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M21A11B11 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M11A11B12 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M21A11B12 = (double *) CoCoMalloc(quarter_size, -1);

	// copy-back matrices for device 1
	double *h_M11A12B21 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M21A12B21 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M11A12B22 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M21A12B22 = (double *) CoCoMalloc(quarter_size, -1);

	// copy-back matrices for device 2
	double *h_M12A21B11 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M22A21B11 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M12A21B12 = (double *) CoCoMalloc(quarter_size, -1);
	double *h_M22A21B12 = (double *) CoCoMalloc(quarter_size, -1);

	// matrices for host
	double *host_A22B21 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_A22B22 = (double *) CoCoMalloc(quarter_size, -2);

	double *host_M12A22B21 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_M22A22B21 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_M12A22B22 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_M22A22B22 = (double *) CoCoMalloc(quarter_size, -2);

	// Q1[]: array of queues in device 0
	CQueue_p * Q1 = (CQueue_p *) CoCoMalloc(6 * sizeof(CQueue_p), -2); // regular (not pinned) host malloc
	// Q2[]: array of queues in device 1
	CQueue_p * Q2 = (CQueue_p *) CoCoMalloc(6 * sizeof(CQueue_p), -2); // regular (not pinned) host malloc
	// Q3[]: array of queues in device 2
	CQueue_p * Q3 = (CQueue_p *) CoCoMalloc(6 * sizeof(CQueue_p), -2); // regular (not pinned) host malloc
	// Qhost[]: array of queues in host
	CQueue_p * Qhost = (CQueue_p *) CoCoMalloc(6 * sizeof(CQueue_p), -2); // regular (not pinned) host malloc

	// std::cout << "DEBUG: Initialization complete\n";

	/* Device 0: GPU 1*/
	double *d0_A11 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_B11 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_A11B11 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_M11 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_M21 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_B12 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_A11B12 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_M11A11B11 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_M21A11B11 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_M11A11B12 = (double *) CoCoMalloc(quarter_size, 0);
	double *d0_M21A11B12 = (double *) CoCoMalloc(quarter_size, 0);

	Event_p transfer_A11_d0 = new Event(0);
	gemm_backend_in_p gemmData_p_1_A11B11 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p compute_A11B11_d0 = new Event(0);

	gemm_backend_in_p gemmData_p_1_A11B12 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p compute_A11B12_d0 = new Event(0);
	Event_p transfer_M11_d0 = new Event(0);

	gemm_backend_in_p gemmData_p_1_M11A11B11 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p transfer_M21_d0 = new Event(0);

	gemm_backend_in_p gemmData_p_1_M21A11B11 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	gemm_backend_in_p gemmData_p_1_M11A11B12 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	gemm_backend_in_p gemmData_p_1_M21A11B12 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	{
		for(int i = 0; i < 6; i++){
			Q1[i] = new CommandQueue(0);
		}

		// Q1[0]
		CoCoMemcpy2DAsync(d0_A11, N/2, h_A, N, N/2, N/2, sizeof(double), 0, -1, Q1[0]);

		// Enqueue event
		transfer_A11_d0->record_to_queue(Q1[0]);

		CoCoMemcpy2DAsync(d0_B11, N/2, h_B, N, N/2, N/2, sizeof(double), 0, -1, Q1[0]);

		
		gemmData_p_1_A11B11->TransA = 'N';	// normal matrix A
		gemmData_p_1_A11B11->TransB = 'N';	// normal matrix B
		gemmData_p_1_A11B11->M = N/2;
		gemmData_p_1_A11B11->N = N/2;
		gemmData_p_1_A11B11->K = N/2;
		gemmData_p_1_A11B11->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_1_A11B11->ldB = N/2;
		gemmData_p_1_A11B11->ldC = N/2;
		gemmData_p_1_A11B11->alpha = 1.0;
		gemmData_p_1_A11B11->beta = 0.0;
		gemmData_p_1_A11B11->A = (void **) &d0_A11;
		gemmData_p_1_A11B11->B = (void **) &d0_B11;
		gemmData_p_1_A11B11->C = (void **) &d0_A11B11;
		gemmData_p_1_A11B11->dev_id = 0;

		// Run blas operation
		backend_run_operation(gemmData_p_1_A11B11, "gemm", Q1[0]);
		// gemm stores result matrix in C = d0_A11B11

		// Enqueue event
		compute_A11B11_d0->record_to_queue(Q1[0]);


		// Q1[1]
		CoCoMemcpy2DAsync(d0_B12, N/2, &h_B[IDX2F(0, N/2, N)], N, N/2, N/2, sizeof(double), 0, -1, Q1[1]);
		Q1[1]->wait_for_event(transfer_A11_d0);

		gemmData_p_1_A11B12->TransA = 'N';	// normal matrix A
		gemmData_p_1_A11B12->TransB = 'N';	// normal matrix B
		gemmData_p_1_A11B12->M = N/2;
		gemmData_p_1_A11B12->N = N/2;
		gemmData_p_1_A11B12->K = N/2;
		gemmData_p_1_A11B12->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_1_A11B12->ldB = N/2;
		gemmData_p_1_A11B12->ldC = N/2;
		gemmData_p_1_A11B12->alpha = 1.0;
		gemmData_p_1_A11B12->beta = 0.0;
		gemmData_p_1_A11B12->A = (void **) &d0_A11;
		gemmData_p_1_A11B12->B = (void **) &d0_B12;
		gemmData_p_1_A11B12->C = (void **) &d0_A11B12;
		gemmData_p_1_A11B12->dev_id = 0;

		// Run blas operation
		backend_run_operation(gemmData_p_1_A11B12, "gemm", Q1[1]);
		// gemm stores result matrix in C = d0_A11B12

		// Enqueue event
		compute_A11B12_d0->record_to_queue(Q1[1]);

		// Q1[2]
		CoCoMemcpy2DAsync(d0_M11, N/2, &h_M[IDX2F(0, 0, N)], N, N/2, N/2, sizeof(double), 0, -1, Q1[2]);

		// Enqueue event
		transfer_M11_d0->record_to_queue(Q1[2]);

		Q1[2]->wait_for_event(compute_A11B11_d0);

		gemmData_p_1_M11A11B11->TransA = 'N';	// normal matrix A
		gemmData_p_1_M11A11B11->TransB = 'N';	// normal matrix B
		gemmData_p_1_M11A11B11->M = N/2;
		gemmData_p_1_M11A11B11->N = N/2;
		gemmData_p_1_M11A11B11->K = N/2;
		gemmData_p_1_M11A11B11->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_1_M11A11B11->ldB = N/2;
		gemmData_p_1_M11A11B11->ldC = N/2;
		gemmData_p_1_M11A11B11->alpha = 1.0;
		gemmData_p_1_M11A11B11->beta = 0.0;
		gemmData_p_1_M11A11B11->A = (void **) &d0_M11;
		gemmData_p_1_M11A11B11->B = (void **) &d0_A11B11;
		gemmData_p_1_M11A11B11->C = (void **) &d0_M11A11B11;
		gemmData_p_1_M11A11B11->dev_id = 0;

		// Run blas operation
		backend_run_operation(gemmData_p_1_M11A11B11, "gemm", Q1[2]);
		// gemm stores result matrix in C = d0_M11A11B11

		// Copy back result M11A11B11
		CoCoMemcpy2DAsync(h_M11A11B11, N/2, d0_M11A11B11, N/2, N/2, N/2, sizeof(double), -1, 0, Q1[2]);

		// Q1[3]
		CoCoMemcpy2DAsync(d0_M21, N/2, &h_M[IDX2F(N/2, 0, N)], N, N/2, N/2, sizeof(double), 0, -1, Q1[3]);

		// Enqueue event
		transfer_M21_d0->record_to_queue(Q1[3]);

		Q1[3]->wait_for_event(compute_A11B11_d0);

		gemmData_p_1_M21A11B11->TransA = 'N';	// normal matrix A
		gemmData_p_1_M21A11B11->TransB = 'N';	// normal matrix B
		gemmData_p_1_M21A11B11->M = N/2;
		gemmData_p_1_M21A11B11->N = N/2;
		gemmData_p_1_M21A11B11->K = N/2;
		gemmData_p_1_M21A11B11->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_1_M21A11B11->ldB = N/2;
		gemmData_p_1_M21A11B11->ldC = N/2;
		gemmData_p_1_M21A11B11->alpha = 1.0;
		gemmData_p_1_M21A11B11->beta = 0.0;
		gemmData_p_1_M21A11B11->A = (void **) &d0_M21;
		gemmData_p_1_M21A11B11->B = (void **) &d0_A11B11;
		gemmData_p_1_M21A11B11->C = (void **) &d0_M21A11B11;
		gemmData_p_1_M21A11B11->dev_id = 0;

		// Run blas operation
		backend_run_operation(gemmData_p_1_M21A11B11, "gemm", Q1[3]);
		// gemm stores result matrix in C = d0_M21A11B11

		// Copy back result M21A11B11
		CoCoMemcpy2DAsync(h_M21A11B11, N/2, d0_M21A11B11, N/2, N/2, N/2, sizeof(double), -1, 0, Q1[3]);

		// Q1[4]
		Q1[4]->wait_for_event(transfer_M11_d0);
		Q1[4]->wait_for_event(compute_A11B12_d0);

		gemmData_p_1_M11A11B12->TransA = 'N';	// normal matrix A
		gemmData_p_1_M11A11B12->TransB = 'N';	// normal matrix B
		gemmData_p_1_M11A11B12->M = N/2;
		gemmData_p_1_M11A11B12->N = N/2;
		gemmData_p_1_M11A11B12->K = N/2;
		gemmData_p_1_M11A11B12->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_1_M11A11B12->ldB = N/2;
		gemmData_p_1_M11A11B12->ldC = N/2;
		gemmData_p_1_M11A11B12->alpha = 1.0;
		gemmData_p_1_M11A11B12->beta = 0.0;
		gemmData_p_1_M11A11B12->A = (void **) &d0_M11;
		gemmData_p_1_M11A11B12->B = (void **) &d0_A11B12;
		gemmData_p_1_M11A11B12->C = (void **) &d0_M11A11B12;
		gemmData_p_1_M11A11B12->dev_id = 0;

		// Run blas operation
		backend_run_operation(gemmData_p_1_M11A11B12, "gemm", Q1[4]);
		// gemm stores result matrix in C = d0_M11A11B12

		// Copy back result M11A11B12
		CoCoMemcpy2DAsync(h_M11A11B12, N/2, d0_M11A11B12, N/2, N/2, N/2, sizeof(double), -1, 0, Q1[4]);

		// Q1[5]
		Q1[5]->wait_for_event(transfer_M21_d0);
		Q1[5]->wait_for_event(compute_A11B12_d0);

		gemmData_p_1_M21A11B12->TransA = 'N';	// normal matrix A
		gemmData_p_1_M21A11B12->TransB = 'N';	// normal matrix B
		gemmData_p_1_M21A11B12->M = N/2;
		gemmData_p_1_M21A11B12->N = N/2;
		gemmData_p_1_M21A11B12->K = N/2;
		gemmData_p_1_M21A11B12->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_1_M21A11B12->ldB = N/2;
		gemmData_p_1_M21A11B12->ldC = N/2;
		gemmData_p_1_M21A11B12->alpha = 1.0;
		gemmData_p_1_M21A11B12->beta = 0.0;
		gemmData_p_1_M21A11B12->A = (void **) &d0_M21;
		gemmData_p_1_M21A11B12->B = (void **) &d0_A11B12;
		gemmData_p_1_M21A11B12->C = (void **) &d0_M21A11B12;
		gemmData_p_1_M21A11B12->dev_id = 0;

		// Run blas operation
		backend_run_operation(gemmData_p_1_M21A11B12, "gemm", Q1[5]);
		// gemm stores result matrix in C = d0_M21A11B12

		// Copy back result M21A11B12
		CoCoMemcpy2DAsync(h_M21A11B12, N/2, d0_M21A11B12, N/2, N/2, N/2, sizeof(double), -1, 0, Q1[5]);
	}
	// std::cout << "DEBUG: Device 0 code complete\n";

	/* Device 1: GPU 2*/
	double *d1_A12 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_B21 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_B22 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_A12B21 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_M11 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_M21 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_A12B22 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_M11A12B21 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_M21A12B21 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_M11A12B22 = (double *) CoCoMalloc(quarter_size, 1);
	double *d1_M21A12B22 = (double *) CoCoMalloc(quarter_size, 1);

	Event_p transfer_A12_d1 = new Event(1);
	gemm_backend_in_p gemmData_p_2_A12B21 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);

	Event_p compute_A12B21_d1 = new Event(1);
	Event_p transfer_B22_d1 = new Event(1);

	gemm_backend_in_p gemmData_p_2_A12B22 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p compute_A12B22_d1 = new Event(1);
	Event_p transfer_M11_d1 = new Event(1);

	gemm_backend_in_p gemmData_p_2_M11A12B21 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p transfer_M21_d1 = new Event(1);
	gemm_backend_in_p gemmData_p_2_M21A12B21 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);

	gemm_backend_in_p gemmData_p_2_M11A12B22 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	gemm_backend_in_p gemmData_p_2_M21A12B22 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	{
		for(int i = 0; i < 6; i++){
			Q2[i] = new CommandQueue(1);
		}

		// Q2[0]
		CoCoMemcpy2DAsync(d1_A12, N/2, &h_A[IDX2F(0, N/2, N)], N, N/2, N/2, sizeof(double), 1, -1, Q2[0]);

		// Enqueue event
		transfer_A12_d1->record_to_queue(Q2[0]);

		CoCoMemcpy2DAsync(d1_B21, N/2, &h_B[IDX2F(N/2, 0, N)], N, N/2, N/2, sizeof(double), 1, -1, Q2[0]);

		gemmData_p_2_A12B21->TransA = 'N';	// normal matrix A
		gemmData_p_2_A12B21->TransB = 'N';	// normal matrix B
		gemmData_p_2_A12B21->M = N/2;
		gemmData_p_2_A12B21->N = N/2;
		gemmData_p_2_A12B21->K = N/2;
		gemmData_p_2_A12B21->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_2_A12B21->ldB = N/2;
		gemmData_p_2_A12B21->ldC = N/2;
		gemmData_p_2_A12B21->alpha = 1.0;
		gemmData_p_2_A12B21->beta = 0.0;
		gemmData_p_2_A12B21->A = (void **) &d1_A12;
		gemmData_p_2_A12B21->B = (void **) &d1_B21;
		gemmData_p_2_A12B21->C = (void **) &d1_A12B21;
		gemmData_p_2_A12B21->dev_id = 1;

		// Run blas operation
		backend_run_operation(gemmData_p_2_A12B21, "gemm", Q2[0]);
		// gemm stores result matrix in C = d1_A12B21

		// Enqueue event
		compute_A12B21_d1->record_to_queue(Q2[0]);

		// Q2[1]
		CoCoMemcpy2DAsync(d1_B22, N/2, &h_B[IDX2F(N/2, N/2, N)], N, N/2, N/2, sizeof(double), 1, -1, Q2[1]);

		// Enqueue event
		transfer_B22_d1->record_to_queue(Q2[1]);

		Q2[1]->wait_for_event(transfer_A12_d1);

		gemmData_p_2_A12B22->TransA = 'N';	// normal matrix A
		gemmData_p_2_A12B22->TransB = 'N';	// normal matrix B
		gemmData_p_2_A12B22->M = N/2;
		gemmData_p_2_A12B22->N = N/2;
		gemmData_p_2_A12B22->K = N/2;
		gemmData_p_2_A12B22->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_2_A12B22->ldB = N/2;
		gemmData_p_2_A12B22->ldC = N/2;
		gemmData_p_2_A12B22->alpha = 1.0;
		gemmData_p_2_A12B22->beta = 0.0;
		gemmData_p_2_A12B22->A = (void **) &d1_A12;
		gemmData_p_2_A12B22->B = (void **) &d1_B22;
		gemmData_p_2_A12B22->C = (void **) &d1_A12B22;
		gemmData_p_2_A12B22->dev_id = 1;

		// Run blas operation
		backend_run_operation(gemmData_p_2_A12B22, "gemm", Q2[1]);
		// gemm stores result matrix in C = d1_A12B22

		// Enqueue event
		compute_A12B22_d1->record_to_queue(Q2[1]);

		// Q2[2]
		CoCoMemcpy2DAsync(d1_M11, N/2, &h_M[IDX2F(0, 0, N)], N, N/2, N/2, sizeof(double), 1, -1, Q2[2]);
		// Enqueue event
		transfer_M11_d1->record_to_queue(Q2[2]);

		Q2[2]->wait_for_event(compute_A12B21_d1);

		gemmData_p_2_M11A12B21->TransA = 'N';	// normal matrix A
		gemmData_p_2_M11A12B21->TransB = 'N';	// normal matrix B
		gemmData_p_2_M11A12B21->M = N/2;
		gemmData_p_2_M11A12B21->N = N/2;
		gemmData_p_2_M11A12B21->K = N/2;
		gemmData_p_2_M11A12B21->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_2_M11A12B21->ldB = N/2;
		gemmData_p_2_M11A12B21->ldC = N/2;
		gemmData_p_2_M11A12B21->alpha = 1.0;
		gemmData_p_2_M11A12B21->beta = 0.0;
		gemmData_p_2_M11A12B21->A = (void **) &d1_M11;
		gemmData_p_2_M11A12B21->B = (void **) &d1_A12B21;
		gemmData_p_2_M11A12B21->C = (void **) &d1_M11A12B21;
		gemmData_p_2_M11A12B21->dev_id = 1;

		// Run blas operation
		backend_run_operation(gemmData_p_2_M11A12B21, "gemm", Q2[2]);
		// gemm stores result matrix in C = d1_M11A12B21

		// Copy back result M11A12B21
		CoCoMemcpy2DAsync(h_M11A12B21, N/2, d1_M11A12B21, N/2, N/2, N/2, sizeof(double), -1, 1, Q2[2]);

		// Q2[3]
		CoCoMemcpy2DAsync(d1_M21, N/2, &h_M[IDX2F(N/2, 0, N)], N, N/2, N/2, sizeof(double), 1, -1, Q2[3]);
		// Enqueue event
		transfer_M21_d1->record_to_queue(Q2[3]);

		Q2[3]->wait_for_event(compute_A12B21_d1);

		gemmData_p_2_M21A12B21->TransA = 'N';	// normal matrix A
		gemmData_p_2_M21A12B21->TransB = 'N';	// normal matrix B
		gemmData_p_2_M21A12B21->M = N/2;
		gemmData_p_2_M21A12B21->N = N/2;
		gemmData_p_2_M21A12B21->K = N/2;
		gemmData_p_2_M21A12B21->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_2_M21A12B21->ldB = N/2;
		gemmData_p_2_M21A12B21->ldC = N/2;
		gemmData_p_2_M21A12B21->alpha = 1.0;
		gemmData_p_2_M21A12B21->beta = 0.0;
		gemmData_p_2_M21A12B21->A = (void **) &d1_M21;
		gemmData_p_2_M21A12B21->B = (void **) &d1_A12B21;
		gemmData_p_2_M21A12B21->C = (void **) &d1_M21A12B21;
		gemmData_p_2_M21A12B21->dev_id = 1;

		// Run blas operation
		backend_run_operation(gemmData_p_2_M21A12B21, "gemm", Q2[3]);
		// gemm stores result matrix in C = d1_M21A12B21

		// Copy back result M21A12B21
		CoCoMemcpy2DAsync(h_M21A12B21, N/2, d1_M21A12B21, N/2, N/2, N/2, sizeof(double), -1, 1, Q2[3]);

		// Q2[4]
		Q2[4]->wait_for_event(compute_A12B22_d1);
		Q2[4]->wait_for_event(transfer_M11_d1);

		gemmData_p_2_M11A12B22->TransA = 'N';	// normal matrix A
		gemmData_p_2_M11A12B22->TransB = 'N';	// normal matrix B
		gemmData_p_2_M11A12B22->M = N/2;
		gemmData_p_2_M11A12B22->N = N/2;
		gemmData_p_2_M11A12B22->K = N/2;
		gemmData_p_2_M11A12B22->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_2_M11A12B22->ldB = N/2;
		gemmData_p_2_M11A12B22->ldC = N/2;
		gemmData_p_2_M11A12B22->alpha = 1.0;
		gemmData_p_2_M11A12B22->beta = 0.0;
		gemmData_p_2_M11A12B22->A = (void **) &d1_M11;
		gemmData_p_2_M11A12B22->B = (void **) &d1_A12B22;
		gemmData_p_2_M11A12B22->C = (void **) &d1_M11A12B22;
		gemmData_p_2_M11A12B22->dev_id = 1;

		// Run blas operation
		backend_run_operation(gemmData_p_2_M11A12B22, "gemm", Q2[4]);
		// gemm stores result matrix in C = d1_M11A12B22

		// Copy back result M11A12B22
		CoCoMemcpy2DAsync(h_M11A12B22, N/2, d1_M11A12B22, N/2, N/2, N/2, sizeof(double), -1, 1, Q2[4]);

		// Q2[5]
		Q2[5]->wait_for_event(compute_A12B22_d1);
		Q2[5]->wait_for_event(transfer_M21_d1);

		gemmData_p_2_M21A12B22->TransA = 'N';	// normal matrix A
		gemmData_p_2_M21A12B22->TransB = 'N';	// normal matrix B
		gemmData_p_2_M21A12B22->M = N/2;
		gemmData_p_2_M21A12B22->N = N/2;
		gemmData_p_2_M21A12B22->K = N/2;
		gemmData_p_2_M21A12B22->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_2_M21A12B22->ldB = N/2;
		gemmData_p_2_M21A12B22->ldC = N/2;
		gemmData_p_2_M21A12B22->alpha = 1.0;
		gemmData_p_2_M21A12B22->beta = 0.0;
		gemmData_p_2_M21A12B22->A = (void **) &d1_M21;
		gemmData_p_2_M21A12B22->B = (void **) &d1_A12B22;
		gemmData_p_2_M21A12B22->C = (void **) &d1_M21A12B22;
		gemmData_p_2_M21A12B22->dev_id = 1;

		// Run blas operation
		backend_run_operation(gemmData_p_2_M21A12B22, "gemm", Q2[5]);
		// gemm stores result matrix in C = d1_M21A12B22

		// Copy back result M21A12B22
		CoCoMemcpy2DAsync(h_M21A12B22, N/2, d1_M21A12B22, N/2, N/2, N/2, sizeof(double), -1, 1, Q2[5]);
	}
	// std::cout << "DEBUG: Device 1 code complete\n";

	/* Device 2: GPU 3*/
	double *d2_A21 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_B11 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_B12 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_A21B11 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_M12 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_M22 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_A21B12 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_M12A21B11 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_M22A21B11 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_M12A21B12 = (double *) CoCoMalloc(quarter_size, 2);
	double *d2_M22A21B12 = (double *) CoCoMalloc(quarter_size, 2);

	Event_p transfer_A21_d2 = new Event(2);
	gemm_backend_in_p gemmData_p_3_A21B11 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);

	Event_p compute_A21B11_d2 = new Event(2);
	gemm_backend_in_p gemmData_p_3_A21B12 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);

	Event_p compute_A21B12_d2 = new Event(2);
	Event_p transfer_M12_d2 = new Event(2);

	gemm_backend_in_p gemmData_p_3_M12A21B11 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p transfer_M22_d2 = new Event(2);

	gemm_backend_in_p gemmData_p_3_M22A21B11 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	gemm_backend_in_p gemmData_p_3_M12A21B12 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	gemm_backend_in_p gemmData_p_3_M22A21B12 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	{
		for(int i = 0; i < 6; i++){
			Q3[i] = new CommandQueue(2);
		}

		// Q3[0]
		CoCoMemcpy2DAsync(d2_A21, N/2, &h_A[IDX2F(N/2, 0, N)], N, N/2, N/2, sizeof(double), 2, -1, Q3[0]);

		// Enqueue event
		transfer_A21_d2->record_to_queue(Q3[0]);

		CoCoMemcpy2DAsync(d2_B11, N/2, &h_B[IDX2F(0, 0, N)], N, N/2, N/2, sizeof(double), 2, -1, Q3[0]);

		
		gemmData_p_3_A21B11->TransA = 'N';	// normal matrix A
		gemmData_p_3_A21B11->TransB = 'N';	// normal matrix B
		gemmData_p_3_A21B11->M = N/2;
		gemmData_p_3_A21B11->N = N/2;
		gemmData_p_3_A21B11->K = N/2;
		gemmData_p_3_A21B11->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_3_A21B11->ldB = N/2;
		gemmData_p_3_A21B11->ldC = N/2;
		gemmData_p_3_A21B11->alpha = 1.0;
		gemmData_p_3_A21B11->beta = 0.0;
		gemmData_p_3_A21B11->A = (void **) &d2_A21;
		gemmData_p_3_A21B11->B = (void **) &d2_B11;
		gemmData_p_3_A21B11->C = (void **) &d2_A21B11;
		gemmData_p_3_A21B11->dev_id = 2;

		// Run blas operation
		backend_run_operation(gemmData_p_3_A21B11, "gemm", Q3[0]);
		// gemm stores result matrix in C = d2_A21B11

		// Enqueue event
		compute_A21B11_d2->record_to_queue(Q3[0]);

		// Q3[1]
		CoCoMemcpy2DAsync(d2_B12, N/2, &h_B[IDX2F(0, N/2, N)], N, N/2, N/2, sizeof(double), 2, -1, Q3[1]);

		Q3[1]->wait_for_event(transfer_A21_d2);

		gemmData_p_3_A21B12->TransA = 'N';	// normal matrix A
		gemmData_p_3_A21B12->TransB = 'N';	// normal matrix B
		gemmData_p_3_A21B12->M = N/2;
		gemmData_p_3_A21B12->N = N/2;
		gemmData_p_3_A21B12->K = N/2;
		gemmData_p_3_A21B12->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_3_A21B12->ldB = N/2;
		gemmData_p_3_A21B12->ldC = N/2;
		gemmData_p_3_A21B12->alpha = 1.0;
		gemmData_p_3_A21B12->beta = 0.0;
		gemmData_p_3_A21B12->A = (void **) &d2_A21;
		gemmData_p_3_A21B12->B = (void **) &d2_B12;
		gemmData_p_3_A21B12->C = (void **) &d2_A21B12;
		gemmData_p_3_A21B12->dev_id = 2;

		// Run blas operation
		backend_run_operation(gemmData_p_3_A21B12, "gemm", Q3[1]);
		// gemm stores result matrix in C = d2_A21B12

		// Enqueue event
		compute_A21B12_d2->record_to_queue(Q3[1]);

		// Q3[2]
		CoCoMemcpy2DAsync(d2_M12, N/2, &h_M[IDX2F(0, N/2, N)], N, N/2, N/2, sizeof(double), 2, -1, Q3[2]);

		// Enqueue event
		transfer_M12_d2->record_to_queue(Q3[2]);

		Q3[2]->wait_for_event(compute_A21B11_d2);

		gemmData_p_3_M12A21B11->TransA = 'N';	// normal matrix A
		gemmData_p_3_M12A21B11->TransB = 'N';	// normal matrix B
		gemmData_p_3_M12A21B11->M = N/2;
		gemmData_p_3_M12A21B11->N = N/2;
		gemmData_p_3_M12A21B11->K = N/2;
		gemmData_p_3_M12A21B11->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_3_M12A21B11->ldB = N/2;
		gemmData_p_3_M12A21B11->ldC = N/2;
		gemmData_p_3_M12A21B11->alpha = 1.0;
		gemmData_p_3_M12A21B11->beta = 0.0;
		gemmData_p_3_M12A21B11->A = (void **) &d2_M12;
		gemmData_p_3_M12A21B11->B = (void **) &d2_A21B11;
		gemmData_p_3_M12A21B11->C = (void **) &d2_M12A21B11;
		gemmData_p_3_M12A21B11->dev_id = 2;

		// Run blas operation
		backend_run_operation(gemmData_p_3_M12A21B11, "gemm", Q3[2]);
		// gemm stores result matrix in C = d2_M12A21B11

		// Copy back result M12A21B11
		CoCoMemcpy2DAsync(h_M12A21B11, N/2, d2_M12A21B11, N/2, N/2, N/2, sizeof(double), -1, 2, Q3[2]);

		// Q3[3]
		CoCoMemcpy2DAsync(d2_M22, N/2, &h_M[IDX2F(N/2, N/2, N)], N, N/2, N/2, sizeof(double), 2, -1, Q3[3]);

		// Enqueue event
		transfer_M22_d2->record_to_queue(Q3[3]);

		Q3[3]->wait_for_event(compute_A21B11_d2);

		gemmData_p_3_M22A21B11->TransA = 'N';	// normal matrix A
		gemmData_p_3_M22A21B11->TransB = 'N';	// normal matrix B
		gemmData_p_3_M22A21B11->M = N/2;
		gemmData_p_3_M22A21B11->N = N/2;
		gemmData_p_3_M22A21B11->K = N/2;
		gemmData_p_3_M22A21B11->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_3_M22A21B11->ldB = N/2;
		gemmData_p_3_M22A21B11->ldC = N/2;
		gemmData_p_3_M22A21B11->alpha = 1.0;
		gemmData_p_3_M22A21B11->beta = 0.0;
		gemmData_p_3_M22A21B11->A = (void **) &d2_M22;
		gemmData_p_3_M22A21B11->B = (void **) &d2_A21B11;
		gemmData_p_3_M22A21B11->C = (void **) &d2_M22A21B11;
		gemmData_p_3_M22A21B11->dev_id = 2;

		// Run blas operation
		backend_run_operation(gemmData_p_3_M22A21B11, "gemm", Q3[3]);
		// gemm stores result matrix in C = d2_M22A21B11

		// Copy back result M22A21B11
		CoCoMemcpy2DAsync(h_M22A21B11, N/2, d2_M22A21B11, N/2, N/2, N/2, sizeof(double), -1, 2, Q3[3]);

		// Q3[4]
		Q3[4]->wait_for_event(transfer_M12_d2);
		Q3[4]->wait_for_event(compute_A21B12_d2);

		gemmData_p_3_M12A21B12->TransA = 'N';	// normal matrix A
		gemmData_p_3_M12A21B12->TransB = 'N';	// normal matrix B
		gemmData_p_3_M12A21B12->M = N/2;
		gemmData_p_3_M12A21B12->N = N/2;
		gemmData_p_3_M12A21B12->K = N/2;
		gemmData_p_3_M12A21B12->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_3_M12A21B12->ldB = N/2;
		gemmData_p_3_M12A21B12->ldC = N/2;
		gemmData_p_3_M12A21B12->alpha = 1.0;
		gemmData_p_3_M12A21B12->beta = 0.0;
		gemmData_p_3_M12A21B12->A = (void **) &d2_M12;
		gemmData_p_3_M12A21B12->B = (void **) &d2_A21B12;
		gemmData_p_3_M12A21B12->C = (void **) &d2_M12A21B12;
		gemmData_p_3_M12A21B12->dev_id = 2;

		// Run blas operation
		backend_run_operation(gemmData_p_3_M12A21B12, "gemm", Q3[4]);
		// gemm stores result matrix in C = d2_M12A21B12

		// Copy back result M12A21B12
		CoCoMemcpy2DAsync(h_M12A21B12, N/2, d2_M12A21B12, N/2, N/2, N/2, sizeof(double), -1, 2, Q3[4]);

		// Q3[5]
		Q3[5]->wait_for_event(transfer_M22_d2);
		Q3[5]->wait_for_event(compute_A21B12_d2);

		gemmData_p_3_M22A21B12->TransA = 'N';	// normal matrix A
		gemmData_p_3_M22A21B12->TransB = 'N';	// normal matrix B
		gemmData_p_3_M22A21B12->M = N/2;
		gemmData_p_3_M22A21B12->N = N/2;
		gemmData_p_3_M22A21B12->K = N/2;
		gemmData_p_3_M22A21B12->ldA = N/2;	// in column - major format ldA = rows(A) = N/2 
		gemmData_p_3_M22A21B12->ldB = N/2;
		gemmData_p_3_M22A21B12->ldC = N/2;
		gemmData_p_3_M22A21B12->alpha = 1.0;
		gemmData_p_3_M22A21B12->beta = 0.0;
		gemmData_p_3_M22A21B12->A = (void **) &d2_M22;
		gemmData_p_3_M22A21B12->B = (void **) &d2_A21B12;
		gemmData_p_3_M22A21B12->C = (void **) &d2_M22A21B12;
		gemmData_p_3_M22A21B12->dev_id = 2;

		// Run blas operation
		backend_run_operation(gemmData_p_3_M22A21B12, "gemm", Q3[5]);
		// gemm stores result matrix in C = d2_M22A21B12

		// Copy back result M22A21B12
		CoCoMemcpy2DAsync(h_M22A21B12, N/2, d2_M22A21B12, N/2, N/2, N/2, sizeof(double), -1, 2, Q3[5]);
	}
	// std::cout << "DEBUG: Device 2 code complete\n";

	/* Device -1: CPU*/
	gemm_backend_in_p gemmData_p_4_A22B21 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p compute_A22B21_host = new Event(-1);
	gemm_backend_in_p gemmData_p_4_A22B22 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p compute_A22B22_host = new Event(-1);
	gemm_backend_in_p gemmData_p_4_M12A22B21 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p compute_M12A22B21_host = new Event(-1);
	gemm_backend_in_p gemmData_p_4_M22A22B21 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	Event_p compute_M22A22B21_host = new Event(-1);
	gemm_backend_in_p gemmData_p_4_M12A22B22 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);
	gemm_backend_in_p gemmData_p_4_M22A22B22 = (gemm_backend_in_p) CoCoMalloc(sizeof(struct gemm_backend_in), -1);

	double *host_A22 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_B21 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_B22 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_M12 = (double *) CoCoMalloc(quarter_size, -2);
	double *host_M22 = (double *) CoCoMalloc(quarter_size, -2);

	Event_p transfer_A22_host = new Event(-1);
	Event_p transfer_M12_host = new Event(-1);
	Event_p transfer_M22_host = new Event(-1);
	{
		for(int i = 0; i < 6; i++){
			Qhost[i] = new CommandQueue(-1);
		}

		// Qhost[0]
		CoCoMemcpy2DAsync(host_A22, N/2, &h_A[IDX2F(N/2, N/2, N)], N, N/2, N/2, sizeof(double), -1, -1, Qhost[0]);
		transfer_A22_host->record_to_queue(Qhost[0]);

		CoCoMemcpy2DAsync(host_B21, N/2, &h_B[IDX2F(N/2, 0, N)], N, N/2, N/2, sizeof(double), -1, -1, Qhost[0]);

		gemmData_p_4_A22B21->TransA = 'N';	// normal matrix A
		gemmData_p_4_A22B21->TransB = 'N';	// normal matrix B
		gemmData_p_4_A22B21->M = N/2;
		gemmData_p_4_A22B21->N = N/2;
		gemmData_p_4_A22B21->K = N/2;
		gemmData_p_4_A22B21->ldA = N/2;	// in column - major format ldA = rows(A) = N/2
		gemmData_p_4_A22B21->ldB = N/2;
		gemmData_p_4_A22B21->ldC = N/2;
		gemmData_p_4_A22B21->alpha = 1.0;
		gemmData_p_4_A22B21->beta = 0.0;
		gemmData_p_4_A22B21->A = (void **) &host_A22;
		gemmData_p_4_A22B21->B = (void **) &host_B21;
		gemmData_p_4_A22B21->C = (void **) &host_A22B21;
		gemmData_p_4_A22B21->dev_id = -1;

		// Run blas operation
		backend_run_operation(gemmData_p_4_A22B21, "gemm", Qhost[0]);
		// gemm stores result matrix in C = host_A22B21

		// Enqueue event
		compute_A22B21_host->record_to_queue(Qhost[0]);

		// Qhost[1]
		CoCoMemcpy2DAsync(host_B22, N/2, &h_B[IDX2F(N/2, N/2, N)], N, N/2, N/2, sizeof(double), -1, -1, Qhost[1]);

		Qhost[1]->wait_for_event(transfer_A22_host);

		gemmData_p_4_A22B22->TransA = 'N';	// normal matrix A
		gemmData_p_4_A22B22->TransB = 'N';	// normal matrix B
		gemmData_p_4_A22B22->M = N/2;
		gemmData_p_4_A22B22->N = N/2;
		gemmData_p_4_A22B22->K = N/2;
		gemmData_p_4_A22B22->ldA = N/2;	// in column - major format ldA = rows(A) = N/2
		gemmData_p_4_A22B22->ldB = N/2;
		gemmData_p_4_A22B22->ldC = N/2;
		gemmData_p_4_A22B22->alpha = 1.0;
		gemmData_p_4_A22B22->beta = 0.0;
		gemmData_p_4_A22B22->A = (void **) &host_A22;
		gemmData_p_4_A22B22->B = (void **) &host_B22;
		gemmData_p_4_A22B22->C = (void **) &host_A22B22;
		gemmData_p_4_A22B22->dev_id = -1;

		// Run blas operation
		backend_run_operation(gemmData_p_4_A22B22, "gemm", Qhost[1]);
		// gemm stores result matrix in C = host_A22B21

		// Enqueue event
		compute_A22B22_host->record_to_queue(Qhost[1]);

		// Qhost[2]
		CoCoMemcpy2DAsync(host_M12, N/2, &h_M[IDX2F(0, N/2, N)], N, N/2, N/2, sizeof(double), -1, -1, Qhost[2]);
		transfer_M12_host->record_to_queue(Qhost[2]);

		Qhost[2]->wait_for_event(compute_A22B21_host);

		gemmData_p_4_M12A22B21->TransA = 'N';	// normal matrix A
		gemmData_p_4_M12A22B21->TransB = 'N';	// normal matrix B
		gemmData_p_4_M12A22B21->M = N/2;
		gemmData_p_4_M12A22B21->N = N/2;
		gemmData_p_4_M12A22B21->K = N/2;
		gemmData_p_4_M12A22B21->ldA = N/2;	// in column - major format ldA = rows(A) = N
		gemmData_p_4_M12A22B21->ldB = N/2;
		gemmData_p_4_M12A22B21->ldC = N/2;
		gemmData_p_4_M12A22B21->alpha = 1.0;
		gemmData_p_4_M12A22B21->beta = 0.0;
		gemmData_p_4_M12A22B21->A = (void **) &host_M12;
		gemmData_p_4_M12A22B21->B = (void **) &host_A22B21;
		gemmData_p_4_M12A22B21->C = (void **) &host_M12A22B21;
		gemmData_p_4_M12A22B21->dev_id = -1;

		// Run blas operation
		backend_run_operation(gemmData_p_4_M12A22B21, "gemm", Qhost[2]);
		// gemm stores result matrix in C = host_M12A22B21

		// Enqueue event
		compute_M12A22B21_host->record_to_queue(Qhost[2]);

		// Qhost[3]
		CoCoMemcpy2DAsync(host_M22, N/2, &h_M[IDX2F(N/2, N/2, N)], N, N/2, N/2, sizeof(double), -1, -1, Qhost[3]);
		transfer_M22_host->record_to_queue(Qhost[3]);

		Qhost[3]->wait_for_event(compute_A22B21_host);

		gemmData_p_4_M22A22B21->TransA = 'N';	// normal matrix A
		gemmData_p_4_M22A22B21->TransB = 'N';	// normal matrix B
		gemmData_p_4_M22A22B21->M = N/2;
		gemmData_p_4_M22A22B21->N = N/2;
		gemmData_p_4_M22A22B21->K = N/2;
		gemmData_p_4_M22A22B21->ldA = N/2;	// in column - major format ldA = rows(A) = N
		gemmData_p_4_M22A22B21->ldB = N/2;
		gemmData_p_4_M22A22B21->ldC = N/2;
		gemmData_p_4_M22A22B21->alpha = 1.0;
		gemmData_p_4_M22A22B21->beta = 0.0;
		gemmData_p_4_M22A22B21->A = (void **) &host_M22;
		gemmData_p_4_M22A22B21->B = (void **) &host_A22B21;
		gemmData_p_4_M22A22B21->C = (void **) &host_M22A22B21;
		gemmData_p_4_M22A22B21->dev_id = -1;

		// Run blas operation
		backend_run_operation(gemmData_p_4_M22A22B21, "gemm", Qhost[3]);
		// gemm stores result matrix in C = host_M22A22B21

		// Enqueue event
		compute_M22A22B21_host->record_to_queue(Qhost[3]);

		// Qhost[4]
		Qhost[4]->wait_for_event(compute_A22B22_host);
		Qhost[4]->wait_for_event(transfer_M12_host);

		gemmData_p_4_M12A22B22->TransA = 'N';	// normal matrix A
		gemmData_p_4_M12A22B22->TransB = 'N';	// normal matrix B
		gemmData_p_4_M12A22B22->M = N/2;
		gemmData_p_4_M12A22B22->N = N/2;
		gemmData_p_4_M12A22B22->K = N/2;
		gemmData_p_4_M12A22B22->ldA = N/2;	// in column - major format ldA = rows(A) = N
		gemmData_p_4_M12A22B22->ldB = N/2;
		gemmData_p_4_M12A22B22->ldC = N/2;
		gemmData_p_4_M12A22B22->alpha = 1.0;
		gemmData_p_4_M12A22B22->beta = 0.0;
		gemmData_p_4_M12A22B22->A = (void **) &host_M12;
		gemmData_p_4_M12A22B22->B = (void **) &host_A22B22;
		gemmData_p_4_M12A22B22->C = (void **) &host_M12A22B22;
		gemmData_p_4_M12A22B22->dev_id = -1;

		// Run blas operation
		backend_run_operation(gemmData_p_4_M12A22B22, "gemm", Qhost[4]);
		// gemm stores result matrix in C = host_M12A22B22

		// Qhost[5]
		Qhost[5]->wait_for_event(compute_A22B22_host);
		Qhost[4]->wait_for_event(transfer_M22_host);

		gemmData_p_4_M22A22B22->TransA = 'N';	// normal matrix A
		gemmData_p_4_M22A22B22->TransB = 'N';	// normal matrix B
		gemmData_p_4_M22A22B22->M = N/2;
		gemmData_p_4_M22A22B22->N = N/2;
		gemmData_p_4_M22A22B22->K = N/2;
		gemmData_p_4_M22A22B22->ldA = N/2;	// in column - major format ldA = rows(A) = N
		gemmData_p_4_M22A22B22->ldB = N/2;
		gemmData_p_4_M22A22B22->ldC = N/2;
		gemmData_p_4_M22A22B22->alpha = 1.0;
		gemmData_p_4_M22A22B22->beta = 0.0;
		gemmData_p_4_M22A22B22->A = (void **) &host_M22;
		gemmData_p_4_M22A22B22->B = (void **) &host_A22B22;
		gemmData_p_4_M22A22B22->C = (void **) &host_M22A22B22;
		gemmData_p_4_M22A22B22->dev_id = -1;

		// Run blas operation
		backend_run_operation(gemmData_p_4_M22A22B22, "gemm", Qhost[5]);
		// gemm stores result matrix in C = host_M22A22B22
	}
	// std::cout << "DEBUG: Host code complete\n";

	// Synchronize with streams that write back results
	for(int i = 2; i < 6; i++){
		Q1[i]->sync_barrier();
		Q2[i]->sync_barrier();
		Q3[i]->sync_barrier();
		Qhost[i]->sync_barrier();
	}

	// std::cout << "DEBUG: Sync streams complete\n";

	// R11
	double *h_res1 = (double *) CoCoMalloc(quarter_size, -2);
	matrixAdd(h_M11A11B11, h_M11A12B21, N/2, N/2, h_res1);
	double *h_res2 = (double *) CoCoMalloc(quarter_size, -2);
	matrixAdd(h_M12A21B11, host_M12A22B21, N/2, N/2, h_res2);

	double *h_R11 = (double *) CoCoMalloc(quarter_size, -2);
	matrixAdd(h_res1, h_res2, N/2, N/2, h_R11);

	// std::cout << "DEBUG: h_R11 complete\n";

	// R12
	matrixAdd(h_M11A11B12, h_M11A12B22, N/2, N/2, h_res1);
	matrixAdd(h_M12A21B12, host_M12A22B22, N/2, N/2, h_res2);

	double *h_R12 = (double *) CoCoMalloc(quarter_size, -2);
	matrixAdd(h_res1, h_res2, N/2, N/2, h_R12);

	// std::cout << "DEBUG: h_R12 complete\n";

	// R21
	matrixAdd(h_M21A11B11, h_M21A12B21, N/2, N/2, h_res1);
	matrixAdd(h_M22A21B11, host_M22A22B21, N/2, N/2, h_res2);

	double *h_R21 = (double *) CoCoMalloc(quarter_size, -2);
	matrixAdd(h_res1, h_res2, N/2, N/2, h_R21);

	// std::cout << "DEBUG: h_R21 complete\n";

	// R22
	matrixAdd(h_M21A11B12, h_M21A12B22, N/2, N/2, h_res1);
	matrixAdd(h_M22A21B12, host_M22A22B22, N/2, N/2, h_res2);

	double *h_R22 = (double *) CoCoMalloc(quarter_size, -2);
	matrixAdd(h_res1, h_res2, N/2, N/2, h_R22);

	// std::cout << "DEBUG: h_R22 complete\n";

	CoCoFree(h_res1, -2);
	CoCoFree(h_res2, -2);

	// Combine R11, R12, R21, R22 into R matrix
	for(int i = 0; i < N/2; i++){
		for(int j = 0; j < N/2; j++){
			h_Res[IDX2F(i, j, N)] = h_R11[IDX2F(i, j, N/2)];
		}
	}

	for(int i = 0; i < N/2; i++){
		for(int j = 0; j < N/2; j++){
			h_Res[IDX2F(i, j+N/2, N)] = h_R12[IDX2F(i, j, N/2)];
		}
	}

	for(int i = 0; i < N/2; i++){
		for(int j = 0; j < N/2; j++){
			h_Res[IDX2F(i+N/2, j, N)] = h_R21[IDX2F(i, j, N/2)];
		}
	}

	for(int i = 0; i < N/2; i++){
		for(int j = 0; j < N/2; j++){
			h_Res[IDX2F(i+N/2, j+N/2, N)] = h_R22[IDX2F(i, j, N/2)];
		}
	}

	// std::cout << "DEBUG: Host combination into R complete\n";

	// Compute M*A*B in CPU to check result
	double *check_Res = (double *) CoCoMalloc(size_n_by_n, -2);
	double *check_C = (double *) CoCoMalloc(size_n_by_n, -2);
	matrixMultiply(h_A, h_B, N, N, N, 1.0, check_C);
	matrixMultiply(h_M, check_C, N, N, N, 1.0, check_Res);

	// std::cout << "DEBUG: Host matrix multiply complete\n";

	// Verify result
	if(!verifyRes(check_Res, h_Res, N, N)){
		std::cout << "Fail: The result of gemm is incorrect!\n";
		returnFlag = 1;
	}
	else{
		std::cout << "Success: The result of gemm is correct!\n";
		returnFlag = 0;
	}

	CoCoFree(h_R11, -2);
	CoCoFree(h_R12, -2);
	CoCoFree(h_R21, -2);
	CoCoFree(h_R22, -2);

	// free copy-back matrices for device 0 
	CoCoFree(h_M11A11B11, -1);
	CoCoFree(h_M21A11B11, -1);
	CoCoFree(h_M11A11B12, -1);
	CoCoFree(h_M21A11B12, -1);

	// free copy-back matrices for device 1
	CoCoFree(h_M11A12B21, -1);
	CoCoFree(h_M21A12B21, -1);
	CoCoFree(h_M11A12B22, -1);
	CoCoFree(h_M21A12B22, -1);

	// free copy-back matrices for device 2
	CoCoFree(h_M12A21B11, -1);
	CoCoFree(h_M22A21B11, -1);
	CoCoFree(h_M12A21B12, -1);
	CoCoFree(h_M22A21B12, -1);

	// free matrices for host
	CoCoFree(host_A22B21, -1);
	CoCoFree(host_A22B22, -1);

	CoCoFree(host_M12A22B21, -1);
	CoCoFree(host_M22A22B21, -1);
	CoCoFree(host_M12A22B22, -1);
	CoCoFree(host_M22A22B22, -1);

	// delete and free all queues
	for(int i = 0; i < 6; i++) delete(Q1[i]);
	CoCoFree(Q1, -2);
	for(int i = 0; i < 6; i++) delete(Q2[i]);
	CoCoFree(Q2, -2);
	for(int i = 0; i < 6; i++) delete(Q3[i]);
	CoCoFree(Q3, -2);
	for(int i = 0; i < 6; i++) delete(Qhost[i]);
	CoCoFree(Qhost, -2);

	// Free device 0 mem
	CoCoFree(d0_A11, 0);
	CoCoFree(d0_B11, 0);
	CoCoFree(d0_A11B11, 0);
	CoCoFree(d0_M11, 0);
	CoCoFree(d0_M21, 0);
	CoCoFree(d0_B12, 0);
	CoCoFree(d0_A11B12, 0);
	CoCoFree(d0_M11A11B11, 0);
	CoCoFree(d0_M21A11B11, 0);
	CoCoFree(d0_M11A11B12, 0);
	CoCoFree(d0_M21A11B12, 0);

	delete(transfer_A11_d0);
	CoCoFree(gemmData_p_1_A11B11, -1);
	delete(compute_A11B11_d0);

	CoCoFree(gemmData_p_1_A11B12, -1);
	delete(compute_A11B12_d0);
	delete(transfer_M11_d0);

	CoCoFree(gemmData_p_1_M11A11B11, -1);
	delete(transfer_M21_d0);

	CoCoFree(gemmData_p_1_M21A11B11, -1);
	CoCoFree(gemmData_p_1_M11A11B12, -1);
	CoCoFree(gemmData_p_1_M21A11B12, -1);

	// Free device 1 mem
	CoCoFree(d1_A12, 1);
	CoCoFree(d1_B21, 1);
	CoCoFree(d1_B22, 1);
	CoCoFree(d1_A12B21, 1);
	CoCoFree(d1_M11, 1);
	CoCoFree(d1_M21, 1);
	CoCoFree(d1_A12B22, 1);
	CoCoFree(d1_M11A12B21, 1);
	CoCoFree(d1_M21A12B21, 1);
	CoCoFree(d1_M11A12B22, 1);
	CoCoFree(d1_M21A12B22, 1);

	delete(transfer_A12_d1);
	CoCoFree(gemmData_p_2_A12B21, -1);

	delete(compute_A12B21_d1);
	delete(transfer_B22_d1);

	CoCoFree(gemmData_p_2_A12B22, -1);
	delete(compute_A12B22_d1);
	delete(transfer_M11_d1);

	CoCoFree(gemmData_p_2_M11A12B21, -1);
	delete(transfer_M21_d1);
	CoCoFree(gemmData_p_2_M21A12B21, -1);

	CoCoFree(gemmData_p_2_M11A12B22, -1);
	CoCoFree(gemmData_p_2_M21A12B22, -1);

	// Free device 2 mem
	CoCoFree(d2_A21, 2);
	CoCoFree(d2_B11, 2);
	CoCoFree(d2_B12, 2);
	CoCoFree(d2_A21B11, 2);
	CoCoFree(d2_M12, 2);
	CoCoFree(d2_M22, 2);
	CoCoFree(d2_A21B12, 2);
	CoCoFree(d2_M12A21B11, 2);
	CoCoFree(d2_M22A21B11, 2);
	CoCoFree(d2_M12A21B12, 2);
	CoCoFree(d2_M22A21B12, 2);

	delete(transfer_A21_d2);
	CoCoFree(gemmData_p_3_A21B11, -1);

	delete(compute_A21B11_d2);
	CoCoFree(gemmData_p_3_A21B12, -1);

	delete(compute_A21B12_d2);
	delete(transfer_M12_d2);

	CoCoFree(gemmData_p_3_M12A21B11, -1);
	delete(transfer_M22_d2);

	CoCoFree(gemmData_p_3_M22A21B11, -1);
	CoCoFree(gemmData_p_3_M12A21B12, -1);
	CoCoFree(gemmData_p_3_M22A21B12, -1);

	// Free secondary host mem
	CoCoFree(gemmData_p_4_A22B21, -1);
	delete(compute_A22B21_host);
	CoCoFree(gemmData_p_4_A22B22, -1);
	delete(compute_A22B22_host);
	CoCoFree(gemmData_p_4_M12A22B21, -1);
	delete(compute_M12A22B21_host);
	CoCoFree(gemmData_p_4_M22A22B21, -1);
	delete(compute_M22A22B21_host);
	CoCoFree(gemmData_p_4_M12A22B22, -1);
	CoCoFree(gemmData_p_4_M22A22B22, -1);

	delete(transfer_A22_host);
	delete(transfer_M12_host);
	delete(transfer_M22_host);

	CoCoFree(host_A22, -2);
	CoCoFree(host_B21, -2);
	CoCoFree(host_B22, -2);
	CoCoFree(host_M12, -2);
	CoCoFree(host_M22, -2);

	// Free host memory
	CoCoFree(h_A, -1); // -1 in loc indicates Host pinned mem
	CoCoFree(h_B, -1);
	CoCoFree(h_M, -1);
	CoCoFree(h_Res, -1);

	return returnFlag;
}