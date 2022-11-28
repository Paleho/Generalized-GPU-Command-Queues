///
/// \author Poutas Sokratis (sokratispoutas@gmail.com)
///
/// \brief Test the use of CoCoMemcpy2DAsync.  
///			

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

int main(int argc, char ** argv){
	int returnFlag = 0;
	int N = 512;
 	size_t size_n_by_n = N * N * sizeof(double);

	double *h_A = (double *) CoCoMalloc(4 * 4 * sizeof(double), -1);
	double *h_A_part = (double *) CoCoMalloc(2 * 2 * sizeof(double), -1);
	double *d_A_part = (double *) CoCoMalloc(2 * 2 * sizeof(double), 0);

	double counter = 1.0;
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			h_A[IDX2F(i,j, 4)] = counter;
			counter += 1;
		}
	}

	printf("h_A = \n");
	printMat(h_A, 4, 4);

	CQueue_p Q1_p = new CommandQueue(0);

	CoCoMemcpy2DAsync(d_A_part, 2, h_A, 4, 2, 2, sizeof(double), 0, -1, Q1_p);
	CoCoMemcpy2DAsync(h_A_part, 2, d_A_part, 2, 2, 2, sizeof(double), -1, 0, Q1_p);

	Q1_p->sync_barrier();

	printf("h_A_part = \n");
	printMat(h_A_part, 2, 2);

	CoCoMemcpy2DAsync(d_A_part, 2, &h_A[IDX2F(3,3, 4)], 4, 2, 2, sizeof(double), 0, -1, Q1_p);
	CoCoMemcpy2DAsync(h_A_part, 2, d_A_part, 2, 2, 2, sizeof(double), -1, 0, Q1_p);

	Q1_p->sync_barrier();

	printf("h_A_part2 = \n");
	printMat(h_A_part, 2, 2);

	CoCoFree(h_A, -1);
	CoCoFree(h_A_part, -1);
	CoCoFree(d_A_part, 0);
	delete(Q1_p);

	return returnFlag;
}