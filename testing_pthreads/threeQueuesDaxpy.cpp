///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Test the use of three Command Queues along with Events.  
///			Q1: c1*A + B = res1
///			Q2: c2*C + D = res2
///			Q3: c3*res1 + res2 = Result

#include <cstdio>
#include <typeinfo>
#include <iostream>

#include "unihelpers.hpp"
#include "pthreads_backend_wrappers.hpp"

const double epsilon = 0.00001;

void vectorInit(double * V, int n)
{
	for(int i = 0; i < n; i++)
		V[i] = (double) (rand() % 100);
}

int verifyRes(double* A, double* B, double c1, double* C, double* D, double c2, double c3, double* Res, int n)
{
	for(int i = 0; i < n; i++){
		double correctResult = c3 * (c1*A[i] + B[i]) + (c2*C[i] + D[i]);
		double dif = Res[i] - correctResult;
		bool expr1 = dif > epsilon;
		bool expr2 = dif < (-1)*epsilon;
		if(expr1 || expr2){
			printf("verifyRes: incorrect at index %d -- should be = %0.5lf BUT Res[i] = %0.5lf -- Error = %0.5lf\n", i, correctResult, Res[i], dif);
			return 0;
		} 
	}
	return 1;
}

void printVec(double* V, int N);

int main(int argc, char ** argv){
	int returnFlag = 0;
	int N = 1000;
 	size_t size = N * sizeof(double);
	double c1 = 2.5;
	double c2 = 1.0;
	double c3 = 10.0;

	// Allocate input vectors h_A and h_B in host memory
	double *h_A = (double *) CoCoMalloc(size, -1); // -1 in loc indicates Host pinned mem
	double *h_B = (double *) CoCoMalloc(size, -1);
	double *h_C = (double *) CoCoMalloc(size, -1);
	double *h_D = (double *) CoCoMalloc(size, -1);
	double *h_Res = (double *) CoCoMalloc(size, -1);

	// Initialize input vectors
	vectorInit(h_A, N);
	vectorInit(h_B, N);
	vectorInit(h_C, N);
	vectorInit(h_D, N);

	// Get dev_id
	int dev_id = CoCoPeLiaGetDevice();
	// std::cout << "device id = " << dev_id << std::endl;

	// Allocate vectors in device memory
 	double *d_A = (double *) CoCoMalloc(size, dev_id);
	double *d_B = (double *) CoCoMalloc(size, dev_id);
	double *d_C = (double *) CoCoMalloc(size, dev_id);
	double *d_D = (double *) CoCoMalloc(size, dev_id);

	// Initialize Command Queues
	CQueue_p Q1_p = new CommandQueue(dev_id);
	CQueue_p Q2_p = new CommandQueue(dev_id);
	CQueue_p Q3_p = new CommandQueue(dev_id);
	
	// Copy vectors from host memory to device memory
	// cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	CoCoMemcpyAsync(d_A, h_A, size, dev_id, -1, Q1_p); // -1 in loc_src indicates Host
	CoCoMemcpyAsync(d_B, h_B, size, dev_id, -1, Q1_p);

	// Use queue 2 for C and D transfers
	CoCoMemcpyAsync(d_C, h_C, size, dev_id, -1, Q2_p); // -1 in loc_src indicates Host
	CoCoMemcpyAsync(d_D, h_D, size, dev_id, -1, Q2_p);

	// Prepare blas operation
	// Q1: c1*A + B
	axpy_backend_in_p axpyData_p_1 = (axpy_backend_in_p) CoCoMalloc(sizeof(struct axpy_backend_in), -1);
	axpyData_p_1->N = N;
	axpyData_p_1->incx = 1;
	axpyData_p_1->incy = 1;
	axpyData_p_1->alpha = c1;
	axpyData_p_1->x = (void **) &d_A;
	axpyData_p_1->y = (void **) &d_B;
	axpyData_p_1->dev_id = dev_id;

	// Run blas operation
	backend_run_operation(axpyData_p_1, "axpy", Q1_p);
	// axpy stores result in y vector = d_B

	// Enqueue event
	Event_p event_p_1 = new Event(dev_id);
	event_p_1->record_to_queue(Q1_p);

	// Prepare blas operation
	// Q2: c2*C + D
	axpy_backend_in_p axpyData_p_2 = (axpy_backend_in_p) CoCoMalloc(sizeof(struct axpy_backend_in), -1);
	axpyData_p_2->N = N;
	axpyData_p_2->incx = 1;
	axpyData_p_2->incy = 1;
	axpyData_p_2->alpha = c2;
	axpyData_p_2->x = (void **) &d_C;
	axpyData_p_2->y = (void **) &d_D;
	axpyData_p_2->dev_id = dev_id;

	// Run blas operation
	backend_run_operation(axpyData_p_2, "axpy", Q2_p);
	// axpy stores result in y vector = d_D
	// Enqueue event
	Event_p event_p_2 = new Event(dev_id);
	event_p_2->record_to_queue(Q2_p);

	// Prepare blas operation
	// Q3: c3*Res_Q1 + Res_Q2
	axpy_backend_in_p axpyData_p_3 = (axpy_backend_in_p) CoCoMalloc(sizeof(struct axpy_backend_in), -1);
	axpyData_p_3->N = N;
	axpyData_p_3->incx = 1;
	axpyData_p_3->incy = 1;
	axpyData_p_3->alpha = c3;
	axpyData_p_3->x = (void **) &d_B;
	axpyData_p_3->y = (void **) &d_D;
	axpyData_p_3->dev_id = dev_id;

	// Q3 has to wait Q1 (using event_p_1) and Q2 (using event_p_2)
	Q3_p->wait_for_event(event_p_1);
	Q3_p->wait_for_event(event_p_2);

	// Run blas operation
	backend_run_operation(axpyData_p_3, "axpy", Q3_p);
	// axpy stores result in y vector = d_B
 
	// Copy result from device memory to host memory
	// h_Res contains the result in host memory
	CoCoMemcpyAsync(h_Res, d_D, size, -1, dev_id, Q3_p);

	// Wait for MyQueue tasks to complete
	Q3_p->sync_barrier();

	// // Print vectors
	// printf("h_A = ");
	// printVec(h_A, N);
	// printf("h_B = ");
	// printVec(h_B, N);
	// printf("h_C = ");
	// printVec(h_C, N);
	// printf("h_D = ");
	// printVec(h_D, N);
	// printf("h_Res = ");
	// printVec(h_Res, N);

	// Verify result
	if(!verifyRes(h_A, h_B, c1, h_C, h_D, c2, c3, h_Res, N)){
		std::cout << "Fail: The result is incorrect!\n";
		returnFlag = 1;
	}
	else{
		std::cout << "Success: The result is correct!\n";
		returnFlag = 0;
	}
	
	// Free device memory
	CoCoFree(d_A, dev_id);
	CoCoFree(d_B, dev_id);
	CoCoFree(d_C, dev_id);
	CoCoFree(d_D, dev_id);

	// Free host memory
	CoCoFree(h_A, -1); // -1 in loc indicates Host pinned mem
	CoCoFree(h_B, -1);
	CoCoFree(h_C, -1);
	CoCoFree(h_D, -1);
	CoCoFree(h_Res, -1);
	CoCoFree(axpyData_p_1, -1);
	CoCoFree(axpyData_p_2, -1);
	CoCoFree(axpyData_p_3, -1);
	delete(Q1_p);
	delete(Q2_p);
	delete(Q3_p);
	delete(event_p_1);
	delete(event_p_2);

	return returnFlag;
}

void printVec(double* V, int N)
{
	printf("[ ");
	for (int i = 0; i < N-1; i++)
		printf("%f, ", V[i]);
	printf("%f]\n", V[N-1]);
}