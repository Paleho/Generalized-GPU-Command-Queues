#include <cstdio>
#include <typeinfo>
#include <iostream>

#include "unihelpers.hpp"
#include "backend_wrappers.hpp"

const double epsilon = 0.00001;

void vectorInit(double * V, int n)
{
	for(int i = 0; i < n; i++)
		V[i] = (double) (rand() % 100);
}

int verifyRes(double* A, double* B, double* C, double alpha, int n)
{
	for(int i = 0; i < n; i++){
		double correctResult = alpha * A[i] + B[i];
		double dif = C[i] - correctResult;
		bool expr1 = dif > epsilon;
		bool expr2 = dif < (-1)*epsilon;
		if(expr1 || expr2){
			printf("verifyRes: incorrect at index %d -- alpha * A[i] + B[i] = %0.5lf BUT C[i] = %0.5lf -- Error = %0.5lf\n", i, correctResult, C[i], dif);
			return 0;
		} 
	}
	return 1;
}

void printVec(double* V, int N);

int main(int argc, char ** argv){
	int returnFlag = 0;
	int N = 10000;
 	size_t size = N * sizeof(double);
	double alpha = 1.0;

	// Allocate input vectors h_A and h_B in host memory
	double *h_A = (double *) CoCoMalloc(size, -1); // -1 in loc indicates Host pinned mem
	double *h_B = (double *) CoCoMalloc(size, -1);
	double *h_Res = (double *) CoCoMalloc(size, -1);

	// Initialize input vectors
	vectorInit(h_A, N);
	vectorInit(h_B, N);

	// Get dev_id
	int dev_id = CoCoPeLiaGetDevice();
	// std::cout << "device id = " << dev_id << std::endl;

	// Allocate vectors in device memory
 	double *d_A = (double *) CoCoMalloc(size, dev_id);
	double *d_B = (double *) CoCoMalloc(size, dev_id);

	// Test CommandQueue
	CQueue_p MyQueue_p = new CommandQueue(dev_id);
	
	// Copy vectors from host memory to device memory
 	// cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
 	// cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	CoCoMemcpyAsync(d_A, h_A, size, dev_id, -1, MyQueue_p); // -1 in loc_src indicates Host
 	CoCoMemcpyAsync(d_B, h_B, size, dev_id, -1, MyQueue_p);

	// Prepare blas operation
	axpy_backend_in_p axpyData_p = (axpy_backend_in_p) CoCoMalloc(sizeof(struct axpy_backend_in), -1);
	axpyData_p->N = N;
	axpyData_p->incx = 1;
	axpyData_p->incy = 1;
	axpyData_p->alpha = alpha;
	axpyData_p->x = (void **) &d_A;
	axpyData_p->y = (void **) &d_B;
	axpyData_p->dev_id = dev_id;

	// Run blas operation
	backend_run_operation(axpyData_p, "axpy", MyQueue_p);
	// axpy stores result in y vector = d_B
 
	// Copy result from device memory to host memory
 	// h_Res contains the result in host memory
 	CoCoMemcpyAsync(h_Res, d_B, size, -1, dev_id, MyQueue_p);

	// Wait for MyQueue tasks to complete
	MyQueue_p->sync_barrier();

	// // Print vectors
	// printf("h_A = ");
	// printVec(h_A, N);
	// printf("h_B = ");
	// printVec(h_B, N);
	// printf("h_Res = ");
	// printVec(h_Res, N);

	// Verify result
	if(!verifyRes(h_A, h_B, h_Res, alpha, N)){
		std::cout << "Fail: The result of axpy is incorrect!\n";
		returnFlag = 1;
	}
	else{
		std::cout << "Success: The result of axpy is correct!\n";
		returnFlag = 0;
	}
	
	// Free device memory
	CoCoFree(d_A, dev_id);
	CoCoFree(d_B, dev_id);

	// Free host memory
	CoCoFree(h_A, -1); // -1 in loc indicates Host pinned mem
	CoCoFree(h_B, -1);
	CoCoFree(h_Res, -1);
	CoCoFree(axpyData_p, -1);
	delete(MyQueue_p);

	return returnFlag;
}

void printVec(double* V, int N)
{
	printf("[ ");
	for (int i = 0; i < N-1; i++)
		printf("%f, ", V[i]);
	printf("%f]\n", V[N-1]);
}