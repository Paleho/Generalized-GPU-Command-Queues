///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Simple test for CoCoMemcpy2DAsync

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
	if(argc < 2){
		std::cout << "Invalid number of command line arguments\n";
		std::cout << "Proper usage: " << argv[0] << " [N]\n";
		return 2;
	}

	int returnFlag = 0;
	int N = atoi(argv[1]);
 	size_t size_n_by_n = N * N * sizeof(double);

	if(N <= 0){
		std::cout << "Invalid command line arg (N)\n";
		return 2;
	}

	// Allocate input matrices in host memory
	double *h_A = (double *) CoCoMalloc(size_n_by_n, -1); // -1 in loc indicates Host pinned mem
    double *h_Res = (double *) CoCoMalloc(size_n_by_n, -1); // -1 in loc indicates Host pinned mem

	// Allocate matrices in device 0
	double *d0_A = (double *) CoCoMalloc(size_n_by_n, 0);

	// Allocate matrices in device 1
	double *d1_A = (double *) CoCoMalloc(size_n_by_n, 1);

	// Initialize input matrices
	matrixInit(h_A, N, N);

	/* Device 0 */
	CQueue_p Q0 = new CommandQueue(1);

	// Device 1 
	CQueue_p Q1 = new CommandQueue(1);

	// Q0
	CoCoMemcpy2DAsync(d0_A, N, h_A, N, N, N, sizeof(double), 0, -1, Q0);

	// wait
    Q0->sync_barrier();

    // Q1
	CoCoMemcpy2DAsync(d1_A, N, d0_A, N, N, N, sizeof(double), 1, 0, Q1);
    CoCoMemcpy2DAsync(h_Res, N, d1_A, N, N, N, sizeof(double), -1, 1, Q1);

    // wait
    Q1->sync_barrier();

	// Verify result
	if(!verifyRes(h_A, h_Res, N, N)){
		std::cout << "Fail: The result is incorrect!\n";
		returnFlag = 1;

		std::cout << "Some more elements: \n";

		printf("h_A[0, 1] = %0.5lf BUT Res[0, 1] = %0.5lf\n", h_A[IDX2F(0, 1, N)], h_Res[IDX2F(0, 1, N)]);
		printf("h_A[1, 0] = %0.5lf BUT Res[1, 0] = %0.5lf\n", h_A[IDX2F(1, 0, N)], h_Res[IDX2F(1, 0, N)]);
		printf("h_A[2, 1] = %0.5lf BUT Res[2, 1] = %0.5lf\n", h_A[IDX2F(2, 1, N)], h_Res[IDX2F(2, 1, N)]);
	}
	else{
		std::cout << "Success: The result is correct!\n";
		returnFlag = 0;
	}

	// Free device 0 mem
	CoCoFree(d0_A, 0);

	// Free device 1 mem
	CoCoFree(d1_A, 1);
	delete(Q1);
    delete(Q0);

	// Free host mem
	CoCoFree(h_A, -1);
	CoCoFree(h_Res, -1);

	return returnFlag;
}